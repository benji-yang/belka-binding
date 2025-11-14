import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import logging
import configs.parameters as cf
import random
import glob
from models.gps import GPSPlusPlus
from contextlib import nullcontext
from torch.amp import autocast
from utils.helpers import load_from_zip, CustomDataLoader
from torch_geometric.data import Data
import wandb
torch.set_float32_matmul_precision('high') 

cfg   = cf.get_config()

wandb.login()
run = wandb.init(
    project="belka-binding",
    config={k: v for k, v in vars(cfg).items() if not k.startswith('_')},
    dir="wanlog" if cfg.cluster == "HPC" else f"/vol/bitbucket/{cf.USERNAME}/belka_dti/wanlog"
)
config = wandb.config

size_suffix = f"_{cfg.experiment_size}" if cfg.experiment_size is not None else ""
barycentric_suffix = "_barycentric" if cfg.barycentric else "_normal"
file_suffix = size_suffix + barycentric_suffix
zip_paths = glob.glob(f"{cfg.zip_dir}/features{file_suffix}*.zip")
if len(zip_paths) == 1:
    # train_data = load_from_zip(zip_paths[0])
    train_graph = [Data(**d) for d in load_from_zip(zip_paths[0])]
    train_loader = CustomDataLoader(
    train_graph,
    bary_config=cfg.barycentric,
    batch_size = cfg.batch_size,
    use_token_bucket=cfg.use_bucket,
    max_nodes_per_batch=cfg.max_nodes,  # pick a safe budget for your GPU
    # max_edges_per_batch=40_000,  # optional
    # max_graphs_per_batch=64,     # optional
    sortish_window=2048,         # bigger → better packing, a bit more CPU
    shuffle=True,
    # num_workers=2,
    pin_memory=True,
    # persistent_workers=True,
)
    # train_loader = CustomDataLoader(train_graph, cfg.barycentric, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)

# train_data = torch.load(cf.TRAIN_GRAPH_PATH, map_location='cpu', weights_only=True)
val_data = load_from_zip(cfg.zip_val_path)
pos_weight = torch.load(cfg.train_weight_path, map_location='cpu', weights_only=True)
val_graph = [Data(**d) for d in val_data]
val_loader   = CustomDataLoader(val_graph, cfg.barycentric, batch_size=cfg.batch_size, use_token_bucket=cfg.use_bucket, 
                                max_nodes_per_batch=cfg.max_nodes, shuffle=False, pin_memory=True)

## example training loop
node_feature_map = {'n_atom_types': len(cf.ATOM_NUM_LIST), 'n_period': len(cf.PERIOD_LIST), 
                    'n_group': 18, 'n_degree': len(cf.DEGREE_LIST), 'n_valence': len(cf.IMPLICIT_VALENCE_LIST), 
                    'n_h_counts': len(cf.POSSIBLE_NUMH_LIST), 'n_radical_charges': len(cf.POSSIBLE_NUMBER_RADICAL_E_LIST), 
                    'n_formal_charges': len(cf.POSSIBLE_FORMAL_CHARGE_LIST), 'n_hybs': len(cf.POSSIBLE_HYBRIDISATION_LIST)}
edge_feature_map = {'n_bond_type': len(cf.POSSIBLE_BOND_TYPE_LIST), 'n_stereo_type': len(cf.POSSIBLE_BOND_STEREO_LIST)}
ring_feature_map = {'n_ring_size': len(cf.RING_SIZE_LIST), 'n_het_counts': len(cf.HET_COUNT_LIST), 'n_avg_en': len(cf.ELECTRONEGATIVITY_LIST)}
net = GPSPlusPlus(cf.NODE_DIM, cf.EDGE_DIM, cf.FACE_DIM, cf.GLOBAL_DIM, node_feature_map, edge_feature_map, cfg,
                  max_degree=5, face_chem_dict=ring_feature_map, pos_weight=pos_weight.to(cf.DEVICE))
net.to(cf.DEVICE)

def resume_from_checkpoint(checkpoint_path, net, optimizer, scheduler=None, scaler=None, device="cuda"):
    if not os.path.exists(checkpoint_path):
        # no checkpoint; start fresh
        return 0, float("inf"), 0, 0, None

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Model first (already on device)
    net.load_state_dict(ckpt["model_state_dict"])

    # Optimizer
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    # Move optimizer state tensors to the right device (sometimes needed)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # Scheduler (if present)
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # AMP scaler (if present)
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    # Early-stopping / counters
    start_epoch        = int(ckpt.get("epoch", 0))
    best_val_loss      = float(ckpt.get("best_val_loss", float("inf")))
    epochs_no_improve  = int(ckpt.get("epochs_no_improve", 0))
    passes             = int(ckpt.get("passes", 0))

    # W&B (optional): resume same run if you want
    wandb_run_id = ckpt.get("wandb_run_id")
    return start_epoch, best_val_loss, epochs_no_improve, passes, wandb_run_id

# torch.autograd.set_detect_anomaly(True)
# activation_stats = {}

# def get_activation_hook(name):
#     def hook(module, inp, out):
#         # Only care about tensor outputs
#         if isinstance(out, torch.Tensor):
#             activation_stats[name] = {
#                 'min': out.min().item(),
#                 'max': out.max().item(),
#                 'mean': out.mean().item(),
#             }
#     return hook

# # === 1) Register hooks on every sub‑module ===
# for name, module in net.named_modules():
#     module.register_forward_hook(get_activation_hook(name))

wandb.watch(net, log="all", log_freq=100)

params = filter(lambda p: p.requires_grad, net.parameters())
try:
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.peak_lr,
        fused=torch.cuda.is_available(),   # use fused CUDA AdamW if possible
    )
except (TypeError, RuntimeError):
    optimizer = torch.optim.AdamW(params, lr=cfg.peak_lr)  # older PyTorch/CPU


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

# Early stopping state
best_val_loss = float('inf')
epochs_no_improve = 0
checkpoint_path = cfg.checkpoint_path

def train_one_epoch_batch(epoch, iteration_start):
    net.train()
    iteration = iteration_start
    random.shuffle(zip_paths)
    for zip_path in zip_paths:
        chunk_data = load_from_zip(zip_path)
    
        # build PyG Data objects + loader
        graphs = [Data(**d) for d in chunk_data]
        train_loader = CustomDataLoader(graphs, cfg.barycentric, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)
        for graph_batch in train_loader:
            graph_batch = graph_batch.to(cf.DEVICE)
            optimizer.zero_grad()
            with ctx:
                output    = net(graph_batch)
                bce_loss  = output['losses']

            scaler.scale(bce_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if hasattr(net, "svdd_criterion") and hasattr(net.svdd_criterion, "renorm_centers_"):
                net.svdd_criterion.renorm_centers_()

            if iteration % 100 == 0:
                log_lines.append(f"[Train] Epoch {epoch:03d} Iter {iteration:06d}  Loss: {bce_loss.item():.4f}")
            iteration += 1
        torch.cuda.empty_cache()

    return iteration

def train_one_epoch(epoch, iteration_start, t_loader):
    net.train()
    iteration = iteration_start
    for graph_batch in t_loader:
        graph_batch = graph_batch.to(cf.DEVICE)
        optimizer.zero_grad()
        with ctx:
            output    = net(graph_batch)
            loss  = output['losses']
            svdd = output.get("svdd", 0)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if hasattr(net, "svdd_criterion") and hasattr(net.svdd_criterion, "renorm_centers_"):
            net.svdd_criterion.renorm_centers_()

        if iteration % 100 == 0:
            lr = scheduler.get_last_lr()[0]
            wandb.log({
                "train/loss": loss.item(),
                "train/learning_rate": lr,
                "train/iteration": iteration,
                "train/epoch": epoch,
                "train/svdd" : svdd.item() if hasattr(svdd, "item") else svdd
            })
        iteration += 1
    torch.cuda.empty_cache()

    return iteration


# def train_one_epoch(epoch, iteration_start, t_loader):
#     net.train()
#     iteration = iteration_start

#     for graph_batch in t_loader:
#         graph_batch = graph_batch.to(cf.DEVICE)
#         metrics = {}

#         # ——— INPUT STATS & NaN/Inf CHECK ———
#         for attr, val in graph_batch.__dict__.items():
#             if isinstance(val, torch.Tensor):
#                 mn, mx, mean = val.min().item(), val.max().item(), val.mean().item()
#                 metrics[f"input/{attr}_min"]  = mn
#                 metrics[f"input/{attr}_max"]  = mx
#                 metrics[f"input/{attr}_mean"] = mean

#                 if torch.any(torch.isnan(val)) or torch.any(torch.isinf(val)):
#                     # create a 1‑row table for the error text
#                     error_table = wandb.Table(columns=["Iteration", "Error"])
#                     error_table.add_data(
#                         iteration,
#                         f"NaN/Inf in input '{attr}'"
#                     )
#                     # log both your existing scalars *and* the table
#                     wandb.log({**metrics, "errors": error_table}, step=iteration)
#                     return iteration, True

#         optimizer.zero_grad()
#         try:
#             # ——— FORWARD & ACTIVATIONS ———
#             with ctx:
#                 output = net(graph_batch)
#             for mod_name, stats in activation_stats.items():
#                 metrics[f"act/{mod_name}_min"]  = stats['min']
#                 metrics[f"act/{mod_name}_max"]  = stats['max']
#                 metrics[f"act/{mod_name}_mean"] = stats['mean']
#             activation_stats.clear()

#             # ——— OUTPUT STATS & NaN/Inf CHECK ———
#             for k, v in output.items():
#                 if isinstance(v, torch.Tensor):
#                     mn, mx, mean = v.min().item(), v.max().item(), v.mean().item()
#                     metrics[f"out/{k}_min"]  = mn
#                     metrics[f"out/{k}_max"]  = mx
#                     metrics[f"out/{k}_mean"] = mean

#                     if torch.any(torch.isnan(v)) or torch.any(torch.isinf(v)):
#                         error_table = wandb.Table(columns=["Iteration", "Error"])
#                         error_table.add_data(
#                             iteration,
#                             f"NaN/Inf in output '{k}'"
#                         )
#                         wandb.log({**metrics, "errors": error_table}, step=iteration)
#                         torch.save(v, f"bad_output_{k}_ep{epoch}_it{iteration}.pt")
#                         return iteration, True

#             # ——— LOSS CHECK ———
#             bce_loss = output['losses']
#             metrics["loss"] = bce_loss.item()
#             if torch.isnan(bce_loss) or torch.isinf(bce_loss):
#                 error_table = wandb.Table(columns=["Iteration", "Error"])
#                 error_table.add_data(
#                     iteration,
#                     "NaN/Inf in loss"
#                 )
#                 wandb.log({**metrics, "errors": error_table}, step=iteration)
#                 return iteration, True

#             # ——— BACKWARD & GRADS ———
#             with torch.autograd.detect_anomaly():
#                 scaler.scale(bce_loss).backward()
#             for name, p in net.named_parameters():
#                 if p.grad is not None:
#                     g = p.grad
#                     metrics[f"grad/{name}_min"]  = g.min().item()
#                     metrics[f"grad/{name}_max"]  = g.max().item()
#                     metrics[f"grad/{name}_mean"] = g.mean().item()
#                     metrics[f"grad/{name}_norm"] = g.norm().item()

#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
#             scaler.step(optimizer)
#             scaler.update()

#             # ——— PERIODIC SUMMARY ———
#             if iteration % 100 == 0:
#                 total_norm = sum(
#                     (p.grad.data.norm(2).item() ** 2)
#                     for p in net.parameters() if p.grad is not None
#                 ) ** 0.5
#                 learning_rate = scheduler.get_last_lr()[0]
#                 metrics["grad_norm"] = total_norm
#                 metrics["lr"] = learning_rate

#         except Exception as e:
#             error_table = wandb.Table(columns=["Iteration", "Error"])
#             error_table.add_data(iteration, str(e))
#             wandb.log({**metrics, "errors": error_table}, step=iteration)
#             return iteration, True

#         # ——— NORMAL ITERATION LOGGING ———
#         wandb.log(metrics, step=iteration)
#         iteration += 1

#     torch.cuda.empty_cache()
#     return iteration, False


@torch.no_grad()
def validate(epoch):
    net.eval()
    total_loss = 0.0
    total_svdd = 0.0
    n_batches = 0
    for graph_batch in val_loader:
        graph_batch = graph_batch.to(cf.DEVICE)
        with ctx:
            output   = net(graph_batch)
        v_loss = output['losses']
        svdd = output.get("svdd", 0)
        total_loss += v_loss.item()
        total_svdd += svdd.item() if hasattr(svdd, "item") else svdd
        n_batches += 1
    torch.cuda.empty_cache()

    avg_loss = total_loss / max(1, n_batches)
    avg_svdd = total_svdd / max(1, n_batches)
    wandb.log({
        "val/loss": avg_loss,
        "val/epoch": epoch,
        "val/svdd" : avg_svdd
    })
    return avg_loss


# === Main training loop ===
log_lines = []
passes = 0
start_epoch = 0
early_stop_trigger = False
last_epoch_done = -1
# start_epoch, best_val_loss, epochs_no_improve, passes, wandb_run_id = \
#     resume_from_checkpoint(checkpoint_path, net, optimizer, scheduler, scaler, device=cf.DEVICE)
for step in range(start_epoch, cfg.stopping_epochs):
    if len(zip_paths) == 1:
        train_loader.set_epoch(step)
        passes = train_one_epoch(step, passes, train_loader)
        # passes, failed = train_one_epoch(step, passes, train_loader)
        # if failed:
        #     msg = f"Training stopped at epoch {step}, iteration {passes} due to NaN/Inf"

        #     # build a 1‑row table for the error info
        #     error_table = wandb.Table(columns=["Epoch", "Iteration", "Error"])
        #     error_table.add_data(step, passes, msg)

        #     # log your stop flags + the error table all at once
        #     wandb.log({
        #         "train/stopped_due_to_nan":  1,
        #         "train/stop_epoch":          step,
        #         "train/stop_iteration":      passes,
        #         "train/errors":              error_table,
        #     }, step=passes)

        #     break

    else:
        passes = train_one_epoch_batch(step, passes)
    scheduler.step()
    val_loss = validate(step)

    min_delta = 1e-6
    warmup = cfg.warmup_epochs
    patience = cfg.patience

    # assume step is 0-based epoch index; start counting AFTER warmup-1, i.e., at step >= warmup
    in_warmup = step < warmup

    # define improvement
    improved = (val_loss < best_val_loss - min_delta)

    if in_warmup:
        # Option A: track best for info but DO NOT count patience or save
        if improved or best_val_loss == float("inf"):
            best_val_loss = val_loss
        epochs_no_improve = 0  # never accumulate during warmup

        # Optional: if you *do* want to checkpoint even during warmup, move the save block here.
    else:
        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0

            checkpoint = {
                "epoch": step + 1,  # next epoch to run
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state_dict": scaler.state_dict() if "scaler" in globals() and scaler is not None else None,
                "best_val_loss": float(best_val_loss),
                "epochs_no_improve": int(epochs_no_improve),
                "passes": int(passes),
                "wandb_run_id": wandb.run.id if wandb.run is not None else None,
                "tag": "best",
            }
            torch.save(checkpoint, checkpoint_path)

            # Prefer artifacts over wandb.save in modern W&B, but keep if you're using it:
            wandb.save(checkpoint_path)
            wandb.log({"checkpoint/updated": True, "checkpoint/val_loss": best_val_loss})
        else:
            epochs_no_improve += 1
            wandb.log({"early_stop/count_no_improve": epochs_no_improve})

    last_epoch_done = step
    # (Optional) hard reset at the *end* of warmup to use the post-warmup loss as baseline
    if step == warmup:
        best_val_loss = val_loss
        epochs_no_improve = 0

    # Early stop check (only meaningful after warmup)
    if (step >= warmup) and (epochs_no_improve >= patience):
        early_stop_trigger = True
        log_lines.append(f"Stopping early at epoch {step}")
        break

if not early_stop_trigger:
    checkpoint = {
    "epoch": last_epoch_done + 1,  # next epoch to run
    "model_state_dict": net.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    "scaler_state_dict": scaler.state_dict() if "scaler" in globals() and scaler is not None else None,
    "best_val_loss": float(best_val_loss),
    "epochs_no_improve": int(epochs_no_improve),
    "passes": int(passes),
    "wandb_run_id": wandb.run.id if wandb.run is not None else None,
    "tag": "last",
    }
    torch.save(checkpoint, checkpoint_path)