import os
import torch
import configs.parameters as cf
import glob
# from models.mpnn import Net
from models.gps import GPSPlusPlus
from contextlib import nullcontext
from torch.amp import autocast
from utils.helpers import load_from_zip, CustomDataLoader, HierarchicalDataLoader
from torch_geometric.data import Data
import wandb
from torch.profiler import profile, record_function, ProfilerActivity

cfg   = cf.get_config()
wandb.init(
    project="gnn-memory-profile",
    name="one_pass_memory_profiler",
    config={k: v for k, v in vars(cfg).items() if not k.startswith('_')},
    dir=f"/vol/bitbucket/{cf.USERNAME}/belka_dti/wanlog"
)
zip_paths = glob.glob(f"{cfg.zip_dir}/features*.zip")
if len(zip_paths) == 1:
    train_data = load_from_zip(zip_paths[0])
    train_graph = [Data(**d) for d in train_data]
    train_loader = CustomDataLoader(train_graph, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)
    # 2) Instantiate the hierarchical loader
    train_loader = HierarchicalDataLoader(
        train_graph,
        pack_size=(640, 1280, 16),          # max nodes, edges, graphs per pack
        packs_per_microbatch=8,          # number of packs per micro-batch
        shuffle=True,                   # whether to shuffle packs each epoch
        num_workers=4,                   # multi-worker sharding
        pin_memory=True
    )

for m_idx, microbatch in enumerate(train_loader):
    print(m_idx)
    for pack in microbatch:
        print(pack.edge_index.shape)
        print(pack.face_index.shape)
# first_pass = next(iter(train_loader))
# print(first_pass)
# print(first_pass.edge_index)
# print(first_pass.face_index)
# for passes in train_loader:
#     face_idx, bond_idx = passes.face_index
#     if max(bond_idx).item() >= passes.edge_attr.shape[0]:
#         print(passes.face_index)
#         print(passses.edge_index)
#         print(passes.edge_attr)

for passes in train_loader:
    face_idx, bond_idx = passes.face_index
    edge_size = passes.edge_index.shape[1]
    if max(bond_idx).item() >= edge_size:
        print(passes.face_index)
        print(passes.edge_index)
        print(passes.edge_attr)

for passes in train_loader:
    face_idx, bond_idx = passes.face_index
    edge_size = passes.edge_index.shape[1]
    if max(bond_idx).item() >= edge_size:
        print(passes.face_index)
        print(passes.edge_index)
        print(passes.edge_attr)

node_feature_map = {'n_atom_types': len(cf.ATOM_NUM_LIST), 'n_period': len(cf.PERIOD_LIST), 
                    'n_group': 18, 'n_degree': len(cf.DEGREE_LIST), 'n_valence': len(cf.IMPLICIT_VALENCE_LIST), 
                    'n_h_counts': len(cf.POSSIBLE_NUMH_LIST), 'n_radical_charges': len(cf.POSSIBLE_NUMBER_RADICAL_E_LIST), 
                    'n_formal_charges': len(cf.POSSIBLE_FORMAL_CHARGE_LIST), 'n_hybs': len(cf.POSSIBLE_HYBRIDISATION_LIST)}
edge_feature_map = {'n_bond_type': len(cf.POSSIBLE_BOND_TYPE_LIST), 'n_stereo_type': len(cf.POSSIBLE_BOND_STEREO_LIST)}
net = GPSPlusPlus(cf.NODE_DIM, cf.EDGE_DIM, cf.FACE_DIM, cf.GLOBAL_DIM, node_feature_map, edge_feature_map, cfg,
                  max_degree=5)
net.to(cf.DEVICE)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.peak_lr)


def lr_multiplier(epoch):
    warmup = cfg.warmup_epochs
    total  = cfg.total_epochs
    base   = cfg.peak_lr
    floor  = cfg.min_lr

    if epoch < warmup:
        # ramp-up from 0 → base
        return float(epoch+1) / float(warmup)
    else:
        # linear decay from base → floor
        p = float(epoch - warmup) / float(total - warmup)
        target_lr = p * floor + (1.0 - p) * base
        # convert to a fraction of base
        return target_lr / base

ctx = autocast(device_type=cf.DEVICE) if cf.DEVICE == 'cuda' else nullcontext()
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
scaler = torch.amp.GradScaler('cuda', enabled=True)

# Global dict to accumulate memory deltas
mem_usage = {}

def start_memory_tracking(model):
    """
    Registers hooks on every submodule of `model` that record
    their forward-pass GPU memory delta into `mem_usage`.
    Returns a list of all hook handles (so you can remove them later).
    """
    mem_usage.clear()

    def pre_hook(module, inp):
        # make sure all async CUDA work is finished
        torch.cuda.synchronize()
        # stash the baseline for this particular call
        module.__mem_before = torch.cuda.memory_allocated()

    def post_hook(module, inp, out):
        torch.cuda.synchronize()
        delta = torch.cuda.memory_allocated() - module.__mem_before
        name = module.__class__.__name__
        mem_usage[name] = mem_usage.get(name, 0) + delta

    handles = []
    for m in model.modules():
        # skip counting the top‐level container twice
        handles.append(m.register_forward_pre_hook(pre_hook))
        handles.append(m.register_forward_hook(post_hook))
    return handles

def stop_memory_tracking(handles):
    """Remove all the hooks you registered."""
    for h in handles:
        h.remove()

with profile(
    activities=[ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,            # <— turn on memory profiling
    with_stack=True,
) as prof:
    net.train()
    first_pass.to(cf.DEVICE)
    optimizer.zero_grad()
    handles = start_memory_tracking(net)
    with ctx:
            output = net(first_pass)
            loss = output['losses']
    stop_memory_tracking(handles)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
    scaler.step(optimizer)
    scaler.update()

wandb.log({
    "train/loss": loss.item(),
    "gpu/mem_allocated_MB": torch.cuda.memory_allocated() / 1e6,
    "gpu/max_mem_allocated_MB": torch.cuda.max_memory_allocated() / 1e6,
})

for module_name, bytes_used in sorted(mem_usage.items(),
                                     key=lambda kv: kv[1], reverse=True):
    print(f"{module_name:<20s}  {bytes_used/1024**2:6.2f} MB")


# # Print top memory-consuming ops
# print(prof.key_averages(group_by_stack_n=5).table(
#     sort_by="self_cuda_memory_usage", row_limit=15
# ))
# trace_path = "one_pass_trace.json"
# prof.export_chrome_trace(trace_path)
# wandb.save(trace_path)  # will upload the trace as a W&B artifact file

# wb_table = wandb.Table(columns=columns, data=rows)
# wandb.log({"profiler/memory_table": wb_table})
wandb.finish()