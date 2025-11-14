import json, math, argparse
from pathlib import Path
from collections import defaultdict, OrderedDict
import numpy as np
import configs.parameters as cf
from utils.plotting import load_summary, prepare_alpha_results, prepare_bucket_delta_inputs, prepare_cosine_curves, prepare_sigma_results, plot_alpha_sweep, plot_bucket_deltas, plot_cosine_curves, plot_sigma_sweep

cfg   = cf.get_config()
def _isclose(a, b, atol=1e-9):
    return abs(float(a) - float(b)) <= atol

def _mean_std(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return (math.nan, math.nan)
    if arr.size == 1:
        return (float(arr[0]), 0.0)
    return (float(arr.mean()), float(arr.std(ddof=1)))

def load_rows(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def per_variance_sigma_alpha(rows):
    """
    Returns:
      sigma_stats_by_var: [{variance, sigma, model_score_mean, model_score_std, model_scores_mean, model_scores_std}, ...]
      alpha_stats_by_var: [{variance, alpha, ...}, ...]
    Disjoint handling:
      - Sigma stats keep the FIRST occurrence per (variance, sigma) with alpha≈1.0 (picks sigma-sweep).
      - Alpha stats keep the LAST occurrence per (variance, alpha) with sigma≈0.0 (picks alpha-sweep).
    """
    # --- Sigma stats (alpha fixed ≈ 1.0); first occurrence wins
    sigma_seen = set()
    sigma_vals = defaultdict(list)
    sigma_splits = defaultdict(list)

    for r in rows:
        if r.get("model_score") is None:
            continue
        if not _isclose(r.get("alpha", 1.0), 1.0):
            continue
        key = (int(r["variance"]), float(r["sigma"]))
        if key in sigma_seen:
            # Skip duplicates (e.g., baseline repeated in alpha sweep)
            continue
        sigma_seen.add(key)
        sigma_vals[key].append(float(r["model_score"]))
        sigma_splits[key].append([float(x) for x in r.get("model_scores", [])])

    sigma_stats_by_var = []
    for key in sorted(sigma_vals):
        mu, sd = _mean_std(sigma_vals[key])
        splits = np.array(sigma_splits[key], dtype=float) if sigma_splits[key] else np.empty((0, 0))
        if splits.size:
            split_mean = splits.mean(0)
            split_std = splits.std(0, ddof=1) if splits.shape[0] > 1 else np.zeros_like(split_mean)
            split_mean = split_mean.tolist()
            split_std = split_std.tolist()
        else:
            split_mean, split_std = [], []
        sigma_stats_by_var.append({
            "variance": key[0], "sigma": key[1],
            "model_score_mean": mu, "model_score_std": sd,
            "model_scores_mean": split_mean, "model_scores_std": split_std,
        })

    # --- Alpha stats (sigma fixed ≈ 0.0); last occurrence wins
    alpha_vals = OrderedDict()
    alpha_splits = OrderedDict()

    for r in rows:
        if r.get("model_score") is None:
            continue
        if not _isclose(r.get("sigma", 0.0), 0.0):
            continue
        key = (int(r["variance"]), float(r["alpha"]))
        alpha_vals[key] = alpha_vals.get(key, []) + [float(r["model_score"])]
        alpha_splits[key] = alpha_splits.get(key, []) + [[float(x) for x in r.get("model_scores", [])]]

    alpha_stats_by_var = []
    for key in sorted(alpha_vals):
        mu, sd = _mean_std(alpha_vals[key])
        splits = np.array(alpha_splits[key], dtype=float) if alpha_splits[key] else np.empty((0, 0))
        if splits.size:
            split_mean = splits.mean(0)
            split_std = splits.std(0, ddof=1) if splits.shape[0] > 1 else np.zeros_like(split_mean)
            split_mean = split_mean.tolist()
            split_std = split_std.tolist()
        else:
            split_mean, split_std = [], []
        alpha_stats_by_var.append({
            "variance": key[0], "alpha": key[1],
            "model_score_mean": mu, "model_score_std": sd,
            "model_scores_mean": split_mean, "model_scores_std": split_std,
        })

    return sigma_stats_by_var, alpha_stats_by_var

def aggregate_across_variances(stats_list, key_name):
    """
    Pool across variances for each sigma (or alpha).
    stats_list: items with keys {"variance", key_name, "model_score_mean", "model_scores_mean"}
    Returns list of dicts keyed by key_name with aggregated mean/std over variances.
    """
    grouped = defaultdict(list)
    grouped_splits = defaultdict(list)
    for r in stats_list:
        grouped[r[key_name]].append(float(r["model_score_mean"]))
        grouped_splits[r[key_name]].append([float(x) for x in r.get("model_scores_mean", [])])

    agg = []
    for k in sorted(grouped):
        vals = np.array(grouped[k], dtype=float)
        mu = float(vals.mean())
        sd = float(vals.std(ddof=1)) if vals.size > 1 else 0.0

        splits = np.array(grouped_splits[k], dtype=float)
        if splits.size:
            split_mean = splits.mean(0).tolist()
            split_std = (splits.std(0, ddof=1) if splits.shape[0] > 1 else np.zeros_like(splits[0])).tolist()
        else:
            split_mean, split_std = [], []

        agg.append({
            key_name: float(k),
            "model_score_mean_over_variances": mu,
            "model_score_std_over_variances": sd,
            "model_scores_mean_over_variances": split_mean,
            "model_scores_std_over_variances": split_std,
        })
    return agg

def cosine_stats(rows):
    """
    Returns:
      curve_stats_by_var: [{variance, mean_curve, std_curve}]
      cosine_summary: {"mean_curve_over_variances", "std_curve_over_variances"}
    """
    curve_groups = defaultdict(list)
    for r in rows:
        mc = r.get("mean_curves", [])
        if mc:
            curve_groups[int(r["variance"])].append(np.asarray(mc, dtype=float))

    curve_stats_by_var = []
    all_vecs = []
    for var, vecs in sorted(curve_groups.items()):
        V = np.stack(vecs, axis=0)  # [N, L]
        mean_curve = V.mean(0)
        std_curve = V.std(0, ddof=1) if V.shape[0] > 1 else np.zeros_like(mean_curve)
        curve_stats_by_var.append({
            "variance": var,
            "mean_curve": [float(x) for x in mean_curve],
            "std_curve": [float(x) for x in std_curve],
        })
        all_vecs.append(mean_curve)

    if all_vecs:
        A = np.stack(all_vecs, axis=0)  # [V, L]
        overall_mean = A.mean(0)
        overall_std = A.std(0, ddof=1) if A.shape[0] > 1 else np.zeros_like(overall_mean)
        cosine_summary = {
            "mean_curve_over_variances": [float(x) for x in overall_mean],
            "std_curve_over_variances": [float(x) for x in overall_std],
        }
    else:
        cosine_summary = {
            "mean_curve_over_variances": [],
            "std_curve_over_variances": [],
        }

    return curve_stats_by_var, cosine_summary

for i in range(3):
    if i == 1:
        cfg.face = True
    if i == 2:
        cfg.face = True
        cfg.barycentric = True
    log_path = Path(f"/vol/bitbucket/{cf.USERNAME}/belka_dti/logs") / f"face_ablation_{int(cfg.barycentric)}_{int(cfg.face)}.json"
    print(log_path)
    rows_json = load_rows(log_path)
    sigma_by_var, alpha_by_var = per_variance_sigma_alpha(rows_json)

    # 2) Aggregation across variances for each sigma / alpha
    sigma_over_vars = aggregate_across_variances(sigma_by_var, "sigma")
    alpha_over_vars = aggregate_across_variances(alpha_by_var, "alpha")

    # 3) Cosine similarity stats
    curve_by_var, curve_stats_over_vars = cosine_stats(rows_json)

    summary = {
        "sigma_stats_by_variance": sigma_by_var,
        "alpha_stats_by_variance": alpha_by_var,
        "sigma_over_variances": sigma_over_vars,
        "alpha_over_variances": alpha_over_vars,
        "curve_stats_by_variance": curve_by_var,
        "cosine_summary": curve_stats_over_vars,
    }

    out_path = Path(f"/vol/bitbucket/{cf.USERNAME}/belka_dti/logs") / f"face_ablation_{int(cfg.barycentric)}_{int(cfg.face)}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] wrote summary to {out_path}")

    def _peek(name, arr, k=6):
        if isinstance(arr, list):
            print(f"\n== {name} (showing {min(k, len(arr))} of {len(arr)}) ==")
            for x in arr[:k]:
                print(x)
        elif isinstance(arr, dict):
            print(f"\n== {name} (dict with {len(arr)} keys) ==")
            for key, val in arr.items():
                if isinstance(val, list):
                    if len(val) > 16:
                        preview = ", ".join(str(v) for v in val[:16])
                        print(f"{key}: [{preview}, ...] (len={len(val)})")
                    else:
                        print(f"{key}: {val}")
                else:
                    print(f"{key}: {val}")
        else:
            print(f"\n== {name} (type {type(arr).__name__}) ==")
            print(arr)
    _peek("sigma_over_variances", sigma_over_vars)
    _peek("alpha_over_variances", alpha_over_vars)
    _peek("cosine_summary", curve_stats_over_vars)


# ---------- example usage ----------
# Adjust paths if needed; your writer used: face_ablation_{bary}_{face}_summary.json
logs_dir = Path(f"/vol/bitbucket/{cf.USERNAME}/belka_dti/logs")
base_sum      = load_summary(logs_dir / "face_ablation_0_0_summary.json")
face_sum      = load_summary(logs_dir / "face_ablation_0_1_summary.json")
face_bary_sum = load_summary(logs_dir / "face_ablation_1_1_summary.json")

# Figure X: MAP vs σ (with std)
sigma_results = prepare_sigma_results(base_sum, face_sum, face_bary_sum)
plot_sigma_sweep(sigma_results, title=r"MAP vs $\sigma$ (face-noise sweep)", savepath="sigma.png", ylim_max= 0.24)

# Figure Y: MAP vs α (with std)
alpha_results = prepare_alpha_results(base_sum, face_sum, face_bary_sum)
plot_alpha_sweep(alpha_results, title=r"MAP vs $\alpha$ ($\alpha$-gate sweep)", savepath="alpha.png")

# Figure Z: cosine similarity vs layer (with std)
cosine_curves = prepare_cosine_curves(base_sum, face_sum, face_bary_sum)
plot_cosine_curves(cosine_curves, title="Cosine similarity per layer", savepath="cosine.png")

# Figure W: bucket-wise ΔMAP vs Base (e.g., compare σ=0 and σ=1.0)
BUCKET_LABELS = [
    "BRD4 share", "HSA share", "sEH share",
    "BRD4 non-share", "HSA non-share", "sEH non-share",
    "BRD4 kin0", "HSA kin0", "sEH kin0"
]
# pick σ values you want to display for Face-only / Face+bary (here 0.0 and 1.0)
base_scores, var_scores = prepare_bucket_delta_inputs(
    base_sum, face_sum, face_bary_sum,
    mode="sigma", base_val=0.0, var_vals=(0.0, 1.0)  # Face-only @0.0, Face+bary @1.0 as example
)
plot_bucket_deltas(var_scores, base_scores, BUCKET_LABELS,
                   title=r"Bucket-wise $\Delta$MAP vs Base ($\sigma$ comparison)", savepath="bucketplot.png")