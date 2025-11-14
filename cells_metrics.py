import torch
import configs.parameters as cf
import numpy as np
from models.gps import GPSPlusPlus
from utils.helpers import calculate_score, load_from_zip, CustomDataLoader
from torch_geometric.data import Data
import json
from pathlib import Path

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
test_loader = CustomDataLoader(test_graph, cfg.barycentric, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)

results = []

variance_list = [0, 1, 2, 3, 4]
sigma_list    = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
alpha_list    = [1.0, 0.5, 0.0]

for var in variance_list:
    cfg.variance = var
    checkpoint_path = cfg.checkpoint_path
    net = GPSPlusPlus(cf.NODE_DIM, cf.EDGE_DIM, cf.FACE_DIM, cf.GLOBAL_DIM, node_feature_map, edge_feature_map, cfg,
                    max_degree=5, face_chem_dict=ring_feature_map, pos_weight=pos_weight.to(cf.DEVICE))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.peak_lr)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    net.to(cf.DEVICE)

    for sigma in sigma_list:
        net.eval()
        net.eval_face_noise_sigma = sigma
        predicted_labels, true_labels, split_group_array = testing()
        model_score, model_scores = calculate_score(predicted_labels, true_labels, split_group_array)
        print(f"[variance={var}] Sigma={sigma} {model_score}")
        results.append({
            "variance": int(var),
            "sigma": float(sigma),
            "alpha": float(getattr(net, "face_alpha", 1.0)),
            "model_score": float(model_score),
            "model_scores": [float(x) for x in model_scores],
            "checkpoint_path": str(checkpoint_path),
            "mean_curves": []
        })

    net.eval_face_noise_sigma = 0.0

    for alpha in alpha_list:
        net.set_face_alpha(alpha)
        predicted_labels, true_labels, split_group_array = testing()
        model_score, model_scores = calculate_score(predicted_labels, true_labels, split_group_array)
        print(f"[variance={var}] Alpha={alpha} {model_score}")
        results.append({
            "variance": int(var),
            "sigma": float(getattr(net, "eval_face_noise_sigma", 0.0)),
            "alpha": float(alpha),
            "model_score": float(model_score),
            "model_scores": [float(x) for x in model_scores],
            "checkpoint_path": str(checkpoint_path),
            "mean_curves": []
        })

    net.set_face_alpha(1.0)

    curves = [] 
    net.enable_cosine_logging(True)
    for bi, g_batch in enumerate(test_loader):
        if bi >= 5:
            break
        g_batch = g_batch.to(cf.DEVICE)
        with torch.no_grad():
            out   = net(g_batch)
        curve = net.get_cosine_logs()       # e.g., length = num_layers
        curves.append(torch.tensor(curve, dtype=torch.float32))
    stacked = torch.stack(curves, dim=0)      # [B', L]
    mean_curve = stacked.mean(0).tolist()     # [L] list of floats
    print(f"[variance={var}] mean_curve={mean_curve}")
    results.append({
        "variance": int(var),
        "sigma": float(getattr(net, "eval_face_noise_sigma", 0.0)),
        "alpha": float(getattr(net, "face_alpha", 1.0)),
        "model_score": None,
        "model_scores": [],
        "checkpoint_path": str(checkpoint_path),
        "mean_curves": [float(x) for x in mean_curve]
    })

outdir = Path(f"/vol/bitbucket/{cf.USERNAME}/belka_dti/logs")
outfile = outdir / f"face_ablation_{int(cfg.barycentric)}_{int(cfg.face)}.json"

with open(outfile, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"[LOG] saved results to {outfile}")