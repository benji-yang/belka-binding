import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json
from pathlib import Path
import configs.parameters as cf

# --- Global style (publication-ish but neutral) ---
plt.rcParams.update({
    "figure.figsize": (6.0, 3.8),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

def _as_np(x):
    if isinstance(x, list):
        return np.array([float(v) if v is not None else np.nan for v in x], dtype=float)
    return np.array(x, dtype=float)

def _clean_xy(y):
    """Return x=1..L and finite y mask applied."""
    y = _as_np(y)
    x = np.arange(1, len(y) + 1, dtype=int)
    m = np.isfinite(y)
    return x[m], y[m]

def _format_ax(ax, ylab, ylim=None):
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylab)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)
    # Add minor ticks for y
    ax.minorticks_on()
    ax.tick_params(which='both', direction='out', length=3, width=0.8)

def _save(prefix_or_path):
    if not prefix_or_path:
        return
    if prefix_or_path.lower().endswith((".png", ".pdf", ".svg")):
        plt.savefig(prefix_or_path, bbox_inches="tight")
    else:
        plt.savefig(f"{prefix_or_path}.png", bbox_inches="tight")
        plt.savefig(f"{prefix_or_path}.pdf", bbox_inches="tight")

def _pick_series(stats, key):
    """Return (y, e) where y is series and e is std or None (works for single or mean±std)."""
    mean_key, std_key = key + "_mean", key + "_std"
    if mean_key in stats:
        y = _as_np(stats[mean_key])
        e = _as_np(stats.get(std_key, None)) if std_key in stats else None
        return y, e
    return _as_np(stats.get(key, [])), None

def plot_metric_over_layers(stats_list, labels, key, ylabel, ylim=None,
                            savepath=None, show=False):
    """
    Supports either:
      - stats[key] (plain list per layer), or
      - stats[key+'_mean'] and stats[key+'_std'] (mean±std across runs).
    """
    fig, ax = plt.subplots()
    for stats, label in zip(stats_list, labels):
        if (key + "_mean") in stats:
            y = _as_np(stats[key + "_mean"])
            e = _as_np(stats.get(key + "_std", None)) if (key + "_std") in stats else None
        else:
            y, e = _as_np(stats.get(key, [])), None

        x = np.arange(1, len(y) + 1, dtype=int)
        m = np.isfinite(y)
        x, y = x[m], y[m]

        ax.plot(x, y, marker='o', linewidth=1.5, markersize=3.5, label=label)
        if e is not None and len(e) == len(y):
            e = e[m]
            # guard: length-1 arrays are ok with fill_between
            ax.fill_between(x, y - e, y + e, alpha=0.18, linewidth=0)

    _format_ax(ax, ylabel, ylim)
    # put legend below plot if many entries
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=min(3, len(labels)), frameon=False)

    if savepath:
        _save(savepath)
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_all(stats_list, labels, prefix=None, show=False,
             long_ylim=(0.0, 1.0), pearson_ylim=(-1.0, 1.0)):
    # 1) Entropy (normalized)
    plot_metric_over_layers(
        stats_list, labels,
        key="entropy_per_layer",
        ylabel="Attention entropy (normalised)",
        ylim=None,
        savepath=(f"{prefix}_entropy" if prefix else None),
        show=show,
    )

    # 2) Long-range ratios @8 and @16 (overlay), mean±std aware
    fig, ax = plt.subplots()
    for stats, label in zip(stats_list, labels):
        y8,  e8  = _pick_series(stats, "long_ratio_8_per_layer")
        y16, e16 = _pick_series(stats, "long_ratio_16_per_layer")

        if len(y8) > 0:
            x8 = np.arange(1, len(y8) + 1, dtype=int)
            m8 = np.isfinite(y8)
            ax.plot(x8[m8], y8[m8], marker='o', linewidth=1.5, markersize=3.5,
                    label=f"{label} (cutoff=8)")
            if e8 is not None and len(e8) == len(y8):
                ax.fill_between(x8[m8], (y8 - e8)[m8], (y8 + e8)[m8], alpha=0.18, linewidth=0)

        if len(y16) > 0:
            x16 = np.arange(1, len(y16) + 1, dtype=int)
            m16 = np.isfinite(y16)
            ax.plot(x16[m16], y16[m16], marker='s', linewidth=1.5, markersize=3.5,
                    label=f"{label} (cutoff=16)")
            if e16 is not None and len(e16) == len(y16):
                ax.fill_between(x16[m16], (y16 - e16)[m16], (y16 + e16)[m16], alpha=0.18, linewidth=0)

    _format_ax(ax, "Fraction of attention weight on >k-hop pairs", long_ylim)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=False)
    if prefix:
        _save(f"{prefix}_longratio_8_16")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # 3) Pearson r(attention, SPD distance)
    plot_metric_over_layers(
        stats_list, labels,
        key="pearson_per_layer",
        ylabel="Pearson r (Attention, SPD)",
        ylim=pearson_ylim,
        savepath=(f"{prefix}_pearson" if prefix else None),
        show=show,
    )

def plot_delta(base_stats, compare_stats, base_label="Model A", cmp_label="Model B",
               prefix=None, show=False):
    """
    Plots per-layer deltas (compare - base) for:
      - entropy_per_layer
      - long_ratio_8_per_layer
      - long_ratio_16_per_layer
      - pearson_per_layer

    Works with:
      • single-run dicts:            stats[key] -> list[float]
      • aggregated dicts (mean/std): stats[key+'_mean'], stats[key+'_std']
    """
    def pick_series(stats, key):
        mean_key, std_key = key + "_mean", key + "_std"
        if mean_key in stats:
            y = _as_np(stats[mean_key])
            e = _as_np(stats.get(std_key, None)) if std_key in stats else None
            return y, e
        return _as_np(stats.get(key, [])), None

    fig, ax = plt.subplots()
    for key, ylab in [
        ("entropy_per_layer", "Entropy"),
        ("long_ratio_8_per_layer", "Long-range ratio @8"),
        ("long_ratio_16_per_layer", "Long-range ratio @16"),
        ("pearson_per_layer", "Pearson r(attn, SPD)"),
    ]:
        a, sa = pick_series(base_stats, key)
        b, sb = pick_series(compare_stats, key)

        L = min(len(a), len(b))
        if L == 0:
            continue
        a, b = a[:L], b[:L]
        x = np.arange(1, L + 1, dtype=int)
        m = np.isfinite(a) & np.isfinite(b)
        if not np.any(m):
            continue
        d = (b - a)[m]
        xm = x[m]

        ax.plot(xm, d, marker='o', linewidth=1.5, markersize=3.5,
                label=f"Δ {ylab} ({cmp_label} − {base_label})")

        if sa is not None and sb is not None and len(sa) >= L and len(sb) >= L:
            sdiff = np.sqrt(sa[:L]**2 + sb[:L]**2)[m]  # assuming independence
            ax.fill_between(xm, d - sdiff, d + sdiff, alpha=0.18, linewidth=0)

    _format_ax(ax, "Change vs base", None)
    ax.axhline(0, linestyle="--", alpha=0.4, linewidth=1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=1, frameon=False)

    if prefix:
        _save(f"{prefix}_deltas")
    if show:
        plt.show()
    else:
        plt.close(fig)

# ---------- plotting helpers (with std) ----------
def plot_sigma_sweep(sigma_results, title="Face-noise sensitivity", savepath=None, show=False, ylim_max=None):
    """
    sigma_results: {variant: {sigma: (mean, std), ...}, ...}
    """
    plt.figure(figsize=(4.8,3.2))
    for variant, vals in sigma_results.items():
        sigmas = sorted(vals.keys())
        means  = [vals[s][0] for s in sigmas]
        stds   = [vals[s][1] for s in sigmas]
        plt.plot(sigmas, means, marker="o", label=variant)
        plt.fill_between(sigmas, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.2)
    plt.xlabel(r"$\sigma$ (noise scale)"); plt.ylabel("MAP") 
    # plt.title(title)
    if ylim_max is not None:
        plt.ylim(top=ylim_max)   # only sets the upper bound
    plt.legend(); plt.tight_layout()
    if savepath: 
        plt.savefig(savepath, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

def plot_alpha_sweep(alpha_results, title=r"Face→edge $\alpha$-gate sweep", savepath=None, show=False):
    """
    alpha_results: {variant: {alpha: (mean, std), ...}, ...}
    """
    plt.figure(figsize=(4.8,3.2))
    for variant, vals in alpha_results.items():
        alphas = sorted(vals.keys())
        means  = [vals[a][0] for a in alphas]
        stds   = [vals[a][1] for a in alphas]
        plt.plot(alphas, means, marker="s", label=variant)
        plt.fill_between(alphas, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.2)
    plt.xlabel(r"$\alpha$ (face→edge strength)"); plt.ylabel("MAP") 
    # plt.title(title)
    plt.legend(); plt.tight_layout()
    if savepath: 
        plt.savefig(savepath, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

def plot_cosine_curves(cosine_curves, title="Oversmoothing signature", savepath=None, show=False):
    """
    cosine_curves: {variant: {"mean":[L], "std":[L]}, ...}
    """
    plt.figure(figsize=(4.8,3.2))
    for variant, vals in cosine_curves.items():
        layers = np.arange(1, len(vals["mean"])+1)
        means, stds = np.array(vals["mean"]), np.array(vals["std"])
        plt.plot(layers, means, label=variant)
        plt.fill_between(layers, means-stds, means+stds, alpha=0.2)
    plt.xlabel("Layer index"); plt.ylabel("Mean pairwise cosine") 
    # plt.title(title)
    plt.legend(); plt.tight_layout()
    if savepath: 
        plt.savefig(savepath, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

def plot_bucket_deltas(bucket_scores, base_scores, bucket_labels, title=r"Bucket-wise $\Delta$MAP vs Base", savepath=None, show=False):
    """
    bucket_scores: {variant: {"mean":[9], "std":[9]}}
    base_scores:   {"mean":[9], "std":[9]}
    """
    x = np.arange(len(bucket_labels)); width = 0.38
    plt.figure(figsize=(8.5,3.6))
    i = 0
    for variant, vals in bucket_scores.items():
        if variant == "Base": continue
        deltas = np.array(vals["mean"]) - np.array(base_scores["mean"])
        # independent stds → add in quadrature
        errs = np.sqrt(np.array(vals["std"])**2 + np.array(base_scores["std"])**2)
        plt.bar(x + i*width, deltas, width, yerr=errs, capsize=3, label=variant, alpha=0.85)
        i += 1
    plt.axhline(0, color="k", lw=0.8)
    plt.xticks(x + (i-1)*width/2, bucket_labels, rotation=35, ha="right")
    plt.ylabel(r"$\Delta$MAP vs Base") 
    # plt.title(title)
    plt.legend(); plt.tight_layout()
    if savepath: 
        plt.savefig(savepath, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
# ---------- converters from your *_summary.json format ----------
BUCKET_LABELS = [
    "BRD4 share", "HSA share", "sEH share",
    "BRD4 non-share", "HSA non-share", "sEH non-share",
    "BRD4 kin0", "HSA kin0", "sEH kin0"
]

def load_summary(p: str | Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _to_sigma_dict(summary: dict) -> dict:
    """{sigma: (mean, std)} from summary['sigma_over_variances']"""
    out = {}
    for row in summary.get("sigma_over_variances", []):
        s = float(row["sigma"])
        mu = float(row["model_score_mean_over_variances"])
        sd = float(row["model_score_std_over_variances"])
        out[s] = (mu, sd)
    return out

def _to_alpha_dict(summary: dict) -> dict:
    """{alpha: (mean, std)} from summary['alpha_over_variances']"""
    out = {}
    for row in summary.get("alpha_over_variances", []):
        a = float(row["alpha"])
        mu = float(row["model_score_mean_over_variances"])
        sd = float(row["model_score_std_over_variances"])
        out[a] = (mu, sd)
    return out

def _to_cosine_dict(summary: dict) -> dict:
    """{"mean":[L], "std":[L]} from summary['cosine_summary']"""
    cs = summary.get("cosine_summary", {})
    return {"mean": cs.get("mean_curve_over_variances", []),
            "std":  cs.get("std_curve_over_variances", [])}

def _closest_entry(rows: list[dict], key: str, target: float) -> dict | None:
    """pick row in rows with key closest to target (for sigma/alpha bucket pulls)"""
    if not rows: return None
    arr = [(abs(float(r[key]) - target), r) for r in rows]
    return sorted(arr, key=lambda x: x[0])[0][1]

def bucket_scores_at(summary: dict, mode: str = "sigma", value: float = 0.0) -> dict:
    """
    Return {"mean":[9], "std":[9]} for a chosen sigma or alpha.
    - mode: "sigma" or "alpha"
    - value: desired sigma/alpha; chooses the closest available
    """
    key = "sigma_over_variances" if mode == "sigma" else "alpha_over_variances"
    rows = summary.get(key, [])
    row = _closest_entry(rows, mode, value)
    if row is None:
        return {"mean": [], "std": []}
    means = row.get("model_scores_mean_over_variances", [])
    stds  = row.get("model_scores_std_over_variances", [])
    return {"mean": means, "std": stds}

# ---------- assemble across variants ----------
def prepare_sigma_results(base_sum, face_sum, face_bary_sum):
    return {
        "Base":      _to_sigma_dict(base_sum),
        "Face-only": _to_sigma_dict(face_sum),
        "Face+bary": _to_sigma_dict(face_bary_sum),
    }

def prepare_alpha_results(base_sum, face_sum, face_bary_sum):
    return {
        "Base":      _to_alpha_dict(base_sum),
        "Face-only": _to_alpha_dict(face_sum),
        "Face+bary": _to_alpha_dict(face_bary_sum),
    }

def prepare_cosine_curves(base_sum, face_sum, face_bary_sum):
    return {
        "Base":      _to_cosine_dict(base_sum),
        "Face-only": _to_cosine_dict(face_sum),
        "Face+bary": _to_cosine_dict(face_bary_sum),
    }

def prepare_bucket_delta_inputs(base_sum, face_sum, face_bary_sum,
                                mode="sigma", base_val=0.0, var_vals=(0.0,)):
    """
    Return base bucket scores at base_val, and dict of variants at given values.
    Example: mode="sigma", base_val=0.0, var_vals=(0.0, 1.0) → use σ=0 for Base and
             compute deltas for Face-only and Face+bary at σ=0.0 and σ=1.0 separately.
    """
    base = bucket_scores_at(base_sum, mode=mode, value=base_val)
    out  = {}
    names = ["Face-only", "Face+bary"]
    sums  = [face_sum, face_bary_sum]
    for name, summ, v in zip(names, sums, var_vals):
        out[name] = bucket_scores_at(summ, mode=mode, value=v)
    return base, out

# # ---------- example usage ----------
# # Adjust paths if needed; your writer used: face_ablation_{bary}_{face}_summary.json
# logs_dir = Path(f"/vol/bitbucket/{cf.USERNAME}/belka_dti/logs")
# base_sum      = load_summary(logs_dir / "face_ablation_0_0_summary.json")
# face_sum      = load_summary(logs_dir / "face_ablation_0_1_summary.json")
# face_bary_sum = load_summary(logs_dir / "face_ablation_1_1_summary.json")

# # Figure X: MAP vs σ (with std)
# sigma_results = prepare_sigma_results(base_sum, face_sum, face_bary_sum)
# plot_sigma_sweep(sigma_results, title=r"MAP vs $\sigma$ (face-noise sweep)")

# # Figure Y: MAP vs α (with std)
# alpha_results = prepare_alpha_results(base_sum, face_sum, face_bary_sum)
# plot_alpha_sweep(alpha_results, title=r"MAP vs $\alpha$ ($\alpha$-gate sweep)")

# # Figure Z: cosine similarity vs layer (with std)
# cosine_curves = prepare_cosine_curves(base_sum, face_sum, face_bary_sum)
# plot_cosine_curves(cosine_curves, title="Cosine similarity per layer")

# # Figure W: bucket-wise ΔMAP vs Base (e.g., compare σ=0 and σ=1.0)
# BUCKET_LABELS = [
#     "BRD4 share", "HSA share", "sEH share",
#     "BRD4 non-share", "HSA non-share", "sEH non-share",
#     "BRD4 kin0", "HSA kin0", "sEH kin0"
# ]
# # pick σ values you want to display for Face-only / Face+bary (here 0.0 and 1.0)
# base_scores, var_scores = prepare_bucket_delta_inputs(
#     base_sum, face_sum, face_bary_sum,
#     mode="sigma", base_val=0.0, var_vals=(0.0, 1.0)  # Face-only @0.0, Face+bary @1.0 as example
# )
# plot_bucket_deltas(var_scores, base_scores, BUCKET_LABELS,
#                    title=r"Bucket-wise $\Delta$MAP vs Base ($\sigma$ comparison)")