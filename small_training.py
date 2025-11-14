import torch
import configs.parameters as cf
import numpy as np
from models.gps import GPSPlusPlus
from utils.helpers import calculate_score, load_from_zip, CustomDataLoader
from torch_geometric.data import Data

# #PBS -J 1-6
cfg   = cf.get_config()

node_feature_map = {'n_atom_types': len(cf.ATOM_NUM_LIST), 'n_period': len(cf.PERIOD_LIST), 
                    'n_group': 18, 'n_degree': len(cf.DEGREE_LIST), 'n_valence': len(cf.IMPLICIT_VALENCE_LIST), 
                    'n_h_counts': len(cf.POSSIBLE_NUMH_LIST), 'n_radical_charges': len(cf.POSSIBLE_NUMBER_RADICAL_E_LIST), 
                    'n_formal_charges': len(cf.POSSIBLE_FORMAL_CHARGE_LIST), 'n_hybs': len(cf.POSSIBLE_HYBRIDISATION_LIST)}
edge_feature_map = {'n_bond_type': len(cf.POSSIBLE_BOND_TYPE_LIST), 'n_stereo_type': len(cf.POSSIBLE_BOND_STEREO_LIST)}
ring_feature_map = {'n_ring_size': len(cf.RING_SIZE_LIST), 'n_het_counts': len(cf.HET_COUNT_LIST), 'n_avg_en': len(cf.ELECTRONEGATIVITY_LIST)}

checkpoint_path = cfg.checkpoint_path

def testing():
    all_preds = []
    all_trues = []
    split_groups = []
    net.eval()
    for graph_batch in test_loader:
        graph_batch = graph_batch.to(cf.DEVICE)
        with torch.no_grad():
            output   = net(graph_batch)
        pred_labels = output['bind']
        all_preds.append(pred_labels.cpu())
        t_labels = graph_batch.y
        all_trues.append(t_labels.cpu())
        s_group = graph_batch.split_group
        split_groups += s_group

    all_preds = torch.cat(all_preds, dim=0).numpy()  # now shape [N, …], N = sum of all B’s
    all_trues = torch.cat(all_trues, dim=0).numpy()
    split_groups = np.array(split_groups)
    return all_preds, all_trues, split_groups

pos_weight = torch.load(cfg.train_weight_path, map_location='cpu', weights_only=True)
test_data = load_from_zip(cfg.zip_test_path)
test_graph = [Data(**d) for d in test_data]
test_loader = CustomDataLoader(test_graph, cfg.barycentric, batch_size=cfg.batch_size, shuffle=False, pin_memory=True)
net = GPSPlusPlus(cf.NODE_DIM, cf.EDGE_DIM, cf.FACE_DIM, cf.GLOBAL_DIM, node_feature_map, edge_feature_map, cfg,
                  max_degree=5, face_chem_dict=ring_feature_map, pos_weight=pos_weight.to(cf.DEVICE))
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.peak_lr)
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
net.to(cf.DEVICE)
predicted_labels, true_labels, split_group_array = testing()
model_score, model_scores = calculate_score(predicted_labels, true_labels, split_group_array)
print(model_score, model_scores)


import re
def load_pruned_state(ckpt_path, device="cpu", key_in="model_state_dict"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get(key_in, ckpt)  # handle {"model": ..., ...} or raw state_dict

    # Drop any layers.*.attn._spd_bias_ref.hop_bias.weight
    patt = re.compile(r"^layers\.\d+\.attn\._spd_bias_ref\.hop_bias\.weight$")
    to_drop = [k for k in list(state.keys()) if patt.match(k)]
    for k in to_drop:
        state.pop(k)

    # # Also handle optional "module." prefix mismatches
    # model_keys = model.state_dict().keys()
    # if not any(k.startswith("module.") for k in model_keys) and any(k.startswith("module.") for k in state.keys()):
    #     state = {k.replace("module.", "", 1): v for k, v in state.items()}

    # missing, unexpected = model.load_state_dict(state, strict=False)

    # if verbose:
    #     print(f"Dropped {len(to_drop)} SPD-bias keys.")
    #     print(f"Missing in checkpoint (ok): {len(missing)} -> first few:", missing[:10])
    #     print(f"Unexpected in checkpoint (ok): {len(unexpected)} -> first few:", unexpected[:10])

    return ckpt, state

variance = [4, 5]
for var in variance:
    cfg.variance = var
    checkpoint_path = cfg.checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    ckpt, pruned_state = load_pruned_state(checkpoint_path)
    if var == 5:
        cfg.spd_bias = False
    net = GPSPlusPlus(cf.NODE_DIM, cf.EDGE_DIM, cf.FACE_DIM, cf.GLOBAL_DIM, node_feature_map, edge_feature_map, cfg,
                  max_degree=5, face_chem_dict=ring_feature_map, pos_weight=pos_weight.to(cf.DEVICE))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.peak_lr)
    net.load_state_dict(pruned_state)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    net.to(cf.DEVICE)
    predicted_labels, true_labels, split_group_array = testing()
    print(f"Variance: {var}, checkpoint: {cfg.checkpoint_path}")
    model_score, model_scores = calculate_score(predicted_labels, true_labels, split_group_array)
    print(model_score, model_scores)

cfg.variance = 0