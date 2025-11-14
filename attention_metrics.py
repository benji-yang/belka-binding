# logging metrics
import torch
import configs.parameters as cf
import numpy as np
from models.gps import GPSPlusPlus
from utils.helpers import calculate_score, load_from_zip, CustomDataLoader, count_parameters
from utils.plotting import plot_all, plot_delta
from torch_geometric.data import Data
import glob
import math
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

# checkpoint_path = cfg.checkpoint_path

# pos_weight = torch.load(cfg.train_weight_path, map_location='cpu', weights_only=True)

# size_suffix = f"_{cfg.experiment_size}" if cfg.experiment_size is not None else ""
# barycentric_suffix = "_barycentric" if cfg.barycentric else "_normal"
# file_suffix = size_suffix + barycentric_suffix
# zip_paths = glob.glob(f"{cfg.zip_dir}/features{file_suffix}*.zip")
# if len(zip_paths) == 1:
#     # train_data = load_from_zip(zip_paths[0])
#     train_graph = [Data(**d) for d in load_from_zip(zip_paths[0])]
#     train_loader = CustomDataLoader(
#     train_graph,
#     bary_config=cfg.barycentric,
#     batch_size = 500,
#     use_token_bucket=cfg.use_bucket,
#     max_nodes_per_batch=cfg.max_nodes,  # pick a safe budget for your GPU
#     # max_edges_per_batch=40_000,  # optional
#     # max_graphs_per_batch=64,     # optional
#     sortish_window=2048,         # bigger → better packing, a bit more CPU
#     shuffle=True,
#     # num_workers=2,
#     pin_memory=True,
#     # persistent_workers=True,
#     )
# val_data = load_from_zip(cfg.zip_val_path)
# val_graph = [Data(**d) for d in val_data]
# val_loader   = CustomDataLoader(val_graph, cfg.barycentric, batch_size=500, use_token_bucket=cfg.use_bucket, 
#                                 max_nodes_per_batch=cfg.max_nodes, shuffle=True, pin_memory=True)
# test_path = f"/vol/bitbucket/{cf.USERNAME}/belka_dti/compressed/testing_features_2000_normal.zip"
# test_data = load_from_zip(test_path)
test_data = load_from_zip(cfg.zip_test_path)
test_graph = [Data(**d) for d in test_data]
test_loader   = CustomDataLoader(test_graph, cfg.barycentric, batch_size=cfg.batch_size, use_token_bucket=cfg.use_bucket, 
                                max_nodes_per_batch=cfg.max_nodes, shuffle=True, pin_memory=True)
net = GPSPlusPlus(cf.NODE_DIM, cf.EDGE_DIM, cf.FACE_DIM, cf.GLOBAL_DIM, node_feature_map, edge_feature_map, cfg,
                  max_degree=5, face_chem_dict=ring_feature_map)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.peak_lr)

# Individual modules (pick the first instances from the model)
mpnn_params = count_parameters(net.layers[0].mpn)
attn_params = count_parameters(net.layers[0].attn) if cfg.attention else 0
ffn_params  = count_parameters(net.layers[0].fnn) if cfg.attention else 0
decoder_params = count_parameters(net.decoder)

# Whole model
total_params = count_parameters(net)

print(f"MPNNLayer params: {mpnn_params:,}")
print(f"BiasedSelfAttentionLayer params: {attn_params:,}")
print(f"GPSFeedForward params: {ffn_params:,}")
print(f"OutputDecoder params: {decoder_params:,}")
print(f"Total GPSPlusPlus params: {total_params:,}")

# checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
# net.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# net.to(cf.DEVICE)

# turn on prints for all attn layers
# net.enable_attn_debug_print(True)
# net.eval()
# with torch.no_grad():
#     for i, batch in enumerate(test_loader):
#         if i >= 2: break            # only a few passes
#         _ = net(batch.to(cf.DEVICE))

# # turn them off again
# net.disable_attn_debug_print()

def _sanitize_for_json(x):
    """Recursively replace NaN/Inf with None so JSON is valid."""
    if isinstance(x, dict):
        return {k: _sanitize_for_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_sanitize_for_json(v) for v in x]
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    return x


# --- helpers ---
def load_stats_json(path: str | Path):
    """
    Returns (stats_dict, label). Works for files saved like:
      {"label": "...", "variance": [...], "stats": {<aggregated keys>}}
    or for a raw aggregated dict directly.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "stats" in data:     # our saved payload_on/off
        label = data.get("label", path.stem)
        return data["stats"], label
    else:                                              # already the aggregated stats dict
        return data, path.stem

@torch.no_grad()
def analyze_attention(model, loader, max_batches=5, device="cuda"):
    """
    Collect attention diagnostics over a few batches and return per-layer means
    (over heads) compatible with plot_all / plot_delta.

    Expects model.get_attn_log_summary() -> dict with lists (len = n_layers),
    each item a [H]-tensor, with keys:
        - "entropy"
        - "long_ratio_8"   (new; optional)
        - "long_ratio_16"  (new; optional)
        - "pearson_r"
    Falls back to old "long_ratio" by mapping it to long_ratio_8_per_layer.
    """
    model.eval()
    if hasattr(model, "enable_attn_logging"):
        model.enable_attn_logging(True)

    # Optional: clear buffers if reusing same model object
    if hasattr(model, "_attn_logger") and hasattr(model._attn_logger, "buffers"):
        for buf in model._attn_logger.buffers:
            if hasattr(buf, "ent"):            buf.ent = []
            if hasattr(buf, "pearson"):        buf.pearson = []
            if hasattr(buf, "long_ratio"):     buf.long_ratio = []      # old
            if hasattr(buf, "long_ratio_8"):   buf.long_ratio_8 = []    # new
            if hasattr(buf, "long_ratio_16"):  buf.long_ratio_16 = []   # new

    # Populate logs
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        _ = model(batch.to(device))

    stats = model.get_attn_log_summary()  # dict: key -> list over layers of [H]-tensors

    def _mean_per_layer(key):
        """Mean over heads per layer; returns list[float] with NaNs if empty, or None if key missing."""
        if key not in stats:
            return None
        out = []
        for t in stats[key]:
            if hasattr(t, "numel") and t.numel() > 0:
                out.append(t.mean().item())
            else:
                out.append(float("nan"))
        return out

    entropy_layer     = _mean_per_layer("entropy")      or []
    pearson_layer     = _mean_per_layer("pearson_r")    or []
    lratio8_layer     = _mean_per_layer("long_ratio_8")     # may be None
    lratio16_layer    = _mean_per_layer("long_ratio_16")    # may be None

    # Back-compat: if only the old single ratio exists, map it to ratio@8
    if lratio8_layer is None and "long_ratio" in stats:
        lratio8_layer = _mean_per_layer("long_ratio")

    # Ensure lists (not None) so plotting doesn't break
    lratio8_layer  = lratio8_layer  or []
    lratio16_layer = lratio16_layer or []

    return {
        "entropy_per_layer":       entropy_layer,
        "long_ratio_8_per_layer":  lratio8_layer,
        "long_ratio_16_per_layer": lratio16_layer,
        "pearson_per_layer":       pearson_layer,
        "raw": stats,  # keep per-head tensors if you need deeper analysis later
    }

def aggregate_runs(stats_list):
    """
    Take a list of stats dicts (e.g., 5 runs of analyze_attention for same model)
    and return a single dict with mean and std per layer.
    """
    out = {}
    for key in ["entropy_per_layer", "long_ratio_8_per_layer",
                "long_ratio_16_per_layer", "pearson_per_layer"]:
        arr = np.array([s[key] for s in stats_list], dtype=float)  # shape: [n_runs, n_layers]
        out[key + "_mean"] = np.nanmean(arr, axis=0).tolist()
        out[key + "_std"]  = np.nanstd(arr, axis=0, ddof=1).tolist()  # unbiased std
    return out

# # Attention + SPD
# variance = [0, 1, 2, 3, 4]
# runs_on = []
# for var in variance:
#     cfg.spd_bias = True
#     cfg.variance = var
#     checkpoint_path = cfg.checkpoint_path
#     print(checkpoint_path)
#     checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
#     net = GPSPlusPlus(cf.NODE_DIM, cf.EDGE_DIM, cf.FACE_DIM, cf.GLOBAL_DIM, node_feature_map, edge_feature_map, cfg,
#                   max_degree=5, face_chem_dict=ring_feature_map)
#     optimizer = torch.optim.AdamW(
#         filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.peak_lr)
#     net.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     net.to(cf.DEVICE)
#     run_on = analyze_attention(net, test_loader)
#     runs_on.append(run_on)

# stats_on = aggregate_runs(runs_on)

# # Attention – SPD (same weights; just disable bias)
# variance = [1, 2, 3, 4, 5]
# runs_off = []
# for var in variance:
#     cfg.spd_bias = False
#     cfg.variance = var
#     checkpoint_path = cfg.checkpoint_path
#     print(checkpoint_path)
#     checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
#     net = GPSPlusPlus(cf.NODE_DIM, cf.EDGE_DIM, cf.FACE_DIM, cf.GLOBAL_DIM, node_feature_map, edge_feature_map, cfg,
#                   max_degree=5, face_chem_dict=ring_feature_map)
#     optimizer = torch.optim.AdamW(
#         filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.peak_lr)
#     net.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     net.to(cf.DEVICE)
#     run_off = analyze_attention(net, test_loader)
#     runs_off.append(run_off)

# stats_off = aggregate_runs(runs_off)

# payload_on = {
#     "label": "SPD bias ON",
#     "variance": [0, 1, 2, 3, 4],
#     "stats": stats_on,  # aggregated: *_mean / *_std
# }
# payload_off = {
#     "label": "SPD bias OFF",
#     "variance": [1, 2, 3, 4, 5],
#     "stats": stats_off,
# }


stats_on,  label_on  = load_stats_json(f"{cfg.utility_dir}/stats_on.json")
stats_off, label_off = load_stats_json(f"{cfg.utility_dir}/stats_off.json")

# Optional third line when you re-train a model with spd_bias=False:
# stats_trn_off = analyze_attention(model_trained_no_spd, loader)

# Plot side-by-side (2 or 3 lines per chart)
plot_all(
    [stats_on, stats_off], 
    ["SPD bias ON", "SPD bias OFF"],
    prefix="attn_compare_train"
)

# Delta curves (OFF − ON). If the model internalized the bias,
# these deltas will be ~0 (what you’re seeing).
plot_delta(stats_on, stats_off, base_label="SPD ON", cmp_label="SPD OFF", prefix="attn_delta_train")

# # write three files: individual + combined (timestamped)
# with open(f"{cfg.utility_dir}/stats_on.json", "w", encoding="utf-8") as f:
#     json.dump(_sanitize_for_json(payload_on), f, indent=2, allow_nan=False)

# with open(f"{cfg.utility_dir}/stats_off.json", "w", encoding="utf-8") as f:
#     json.dump(_sanitize_for_json(payload_off), f, indent=2, allow_nan=False)

# Later, add the model trained without SPD:
# plot_all([stats_on, stats_off, stats_trn_off],
#          ["SPD ON (infer)", "SPD OFF (infer)", "Trained w/o SPD"],
#          prefix="attn_compare_3way")
# plot_delta(stats_on, stats_trn_off, base_label="SPD ON", cmp_label="Trained w/o SPD", prefix="attn_delta_trainoff")


# import numpy as np
# from sklearn.linear_model import Ridge
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler

# def energy_distance_multivariate(X, Y, subsample=None, seed=0):
#     if subsample is not None:
#         rng = np.random.default_rng(seed)
#         if X.shape[0] > subsample: X = X[rng.choice(X.shape[0], subsample, replace=False)]
#         if Y.shape[0] > subsample: Y = Y[rng.choice(Y.shape[0], subsample, replace=False)]
#     try:
#         from scipy.stats import energy_distance
#         return float(energy_distance(X, Y))
#     except Exception:
#         from scipy.spatial.distance import cdist, pdist
#         XY = cdist(X, Y, metric="euclidean").mean()
#         XX = pdist(X, metric="euclidean").mean()
#         YY = pdist(Y, metric="euclidean").mean()
#         return float(2*XY - XX - YY)

# def energy_report(train_mat, eval_mat, subsample=50000):
#     rep = {}
#     for key in ("base","face","bary"):
#         if key in train_mat and key in eval_mat:
#             rep[f"ED_{key}"] = energy_distance_multivariate(train_mat[key], eval_mat[key], subsample=subsample)
#     return rep

# def r2_redundancy(X_base, Y_target, alpha=1e-3):
#     model = make_pipeline(StandardScaler(with_mean=True, with_std=True),
#                           Ridge(alpha=alpha, fit_intercept=True, random_state=0))
#     model.fit(X_base, Y_target)
#     mean_r2 = float(model.score(X_base, Y_target))
#     return mean_r2

# def r2_report(train_mat):
#     out = {}
#     if "base" in train_mat and "face" in train_mat:
#         out["R2_face_given_base"] = r2_redundancy(train_mat["base"], train_mat["face"])
#     if "base" in train_mat and "bary" in train_mat:
#         out["R2_bary_given_base"] = r2_redundancy(train_mat["base"], train_mat["bary"])
#     return out

# from utils.attn_logging import collect_encoded_summaries
# # load weights...
# train_enc = collect_encoded_summaries(net, train_loader, device="cuda")
# eval_enc  = collect_encoded_summaries(net, test_loader,  device="cuda")

# ed = energy_report(train_enc, eval_enc, subsample=50000)
# r2 = r2_report(train_enc)

# print("Energy distances (shift):", ed)         # e.g. {'ED_base':0.31, 'ED_face':0.92, 'ED_bary':1.05}
# print("R² redundancy (train):", r2)  