import glob
import numpy as np
import configs.parameters as cf
from torch_geometric.data import Data
import polaris as po
from utils.helpers import load_from_zip
import json
import os
from utils.similarity import compare_three_sets_with_cache, pretty_print_comparison

cfg   = cf.get_config()
print(cfg.barycentric)
dataset = po.dataset.DatasetV2.from_json(cfg.load_path)
train_mask = (dataset[:, "split"] == "train")
test_mask = np.isin(dataset[:, "split"], ['test', ''])
smiles_array = dataset[:, "molecule_smiles"]
full_smiles = smiles_array[train_mask]
test_smiles = smiles_array[test_mask]

size_suffix = f"_{cfg.experiment_size}" if cfg.experiment_size is not None else ""
barycentric_suffix = "_barycentric" if cfg.barycentric else "_normal"
file_suffix = size_suffix + barycentric_suffix
zip_paths = glob.glob(f"{cfg.zip_dir}/features{file_suffix}*.zip")
json_path = f"{cfg.utility_dir}/subset_index_mask{file_suffix}.json"
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        subset_index_mask = json.load(f)
else:
    train_graph = [Data(**d) for d in load_from_zip(zip_paths[0])]
    subset_index_mask = [graph.idx for graph in train_graph]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(subset_index_mask, f, indent=2)

subset_smiles = full_smiles[subset_index_mask]

# report = compare_three_sets_with_cache(
#     train_full_smiles   = full_smiles,     # list/array
#     train_subset_smiles = subset_smiles,
#     test_smiles         = test_smiles,
#     out_cache_root      = "./chemcache",   # caches will be created here
#     fp_radius=2, fp_bits=2048, use_chirality=True,
#     n_procs=16, seed=42,
#     nn_cap_query=20000, mmd_cap_sub=5000, mmd_cap_full=5000
# )

report = compare_three_sets_with_cache(
    full_cache_dir      = "./chemcache/train_full_ECFP4_bits2048_chiral_dropDy_v1_sha256-85ec043f5e3eb3a2-98415610",     # list/array
    subset_cache_dir    = "./chemcache/train_subset_ECFP4_bits2048_chiral_dropDy_v1_sha256-cb4c1d14c72209e1-10000000",
    test_cache_dir      = "./chemcache/test_ECFP4_bits2048_chiral_dropDy_v1_sha256-c3283e193064c765-684407",
    out_cache_root      = "./chemcache",   # caches will be created here
    fp_radius=2, fp_bits=2048, use_chirality=True,
    n_procs=16, seed=42,
    nn_cap_query=20000, mmd_cap_sub=5000, mmd_cap_full=5000
)

print("Cache paths:", report["paths"])

pretty_print_comparison("Test vs Subset", report["test_vs_subset"], out_path=f"{cfg.utility_dir}/comparison_results.txt")
pretty_print_comparison("Test vs Full",   report["test_vs_full"], out_path=f"{cfg.utility_dir}/comparison_results.txt")
pretty_print_comparison("Subset vs Full", report["subset_vs_full"], out_path=f"{cfg.utility_dir}/comparison_results.txt")


# # sanity_check_cache.py
# import os, json, numpy as np
# from collections import Counter

# # ---------- helpers ----------
# _POP_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
# def popcount_bytes(U8): return _POP_LUT[U8]

# def load_cache(root: str):
#     with open(os.path.join(root, "meta.json"), encoding="utf-8") as f:
#         meta = json.load(f)
#     scaff = []
#     with open(os.path.join(root, "scaffolds.csv"), encoding="utf-8") as f:
#         next(f, None)  # skip header if present
#         for line in f:
#             scaff.append(line.rstrip("\n"))
#     fps = np.load(os.path.join(root, "fp_matrix.npz"))["fps"]       # [N, bytes] uint8
#     pop = np.load(os.path.join(root, "popcount.npy"))                # [N] int
#     valid = np.load(os.path.join(root, "valid_mask.npy"))            # [N] bool
#     return {"scaffolds": scaff, "fps_uint8": fps, "popcount": pop, "valid_mask": valid, "meta": meta}

# def summarize_cache(root: str, show_examples: int = 5):
#     print(f"\n=== CACHE: {root} ===")
#     C = load_cache(root)
#     fps = C["fps_uint8"]; pop = C["popcount"]; valid = C["valid_mask"]; scaff = C["scaffolds"]; M = C["meta"]

#     N, B = fps.shape
#     fp_bits = int(M.get("fp_bits", B*8))
#     assert B*8 == fp_bits, f"Bit-length mismatch: matrix has {B*8} bits, meta says {fp_bits}"

#     # Basic counts
#     n_valid = int(valid.sum())
#     n_scaff_valid = sum(1 for s in scaff if isinstance(s, str) and len(s) > 0)
#     print(f"mols: {N} | bytes/FP: {B} ({B*8} bits) | valid_fp: {n_valid} ({n_valid/N:.1%}) | valid_scaffolds: {n_scaff_valid}")

#     # Zero/nonzero rows
#     row_pop = popcount_bytes(fps).sum(axis=1, dtype=np.int32)
#     n_nonzero = int((row_pop > 0).sum())
#     print(f"nonzero_fp rows: {n_nonzero} ({n_nonzero/N:.1%}) | zero_fp rows: {N - n_nonzero} ({(N - n_nonzero)/N:.1%})")

#     # Check stored popcounts vs recomputed
#     mism = int((row_pop != pop).sum())
#     if mism:
#         print(f"WARNING: popcount mismatch on {mism} rows (stored vs recomputed)")
#     else:
#         print("popcount check: OK (stored matches recomputed)")

#     # Popcount statistics on valid/nonzero
#     nnz_mask = row_pop > 0
#     if nnz_mask.any():
#         rp = row_pop[nnz_mask]
#         q = np.quantile(rp, [0.05, 0.5, 0.95])
#         print(f"popcount (nonzero rows) mean={rp.mean():.1f} | p05={q[0]:.0f}, p50={q[1]:.0f}, p95={q[2]:.0f}")
#     else:
#         print("popcount: all rows are zero — fingerprints likely not generated")

#     # Scaffold stats
#     S = [s for s in scaff if isinstance(s, str) and s]
#     uniq = set(S)
#     print(f"unique scaffolds: {len(uniq)}")
#     if len(uniq) == 0:
#         print("scaffolds: NONE (extraction failed or empty)")
#     else:
#         top = Counter(S).most_common(5)
#         print("top scaffolds (count):", ", ".join(f"{k[:30]}…:{v}" if len(k)>30 else f"{k}:{v}" for k,v in top))

#     # Show a few example rows that look healthy
#     shown = 0
#     print("examples (row_idx, popcount, has_scaffold):")
#     for i in range(N):
#         if row_pop[i] > 0 or (isinstance(scaff[i], str) and scaff[i]):
#             print(f"  {i:>6d}  {int(row_pop[i]):>5d}  {bool(scaff[i]):>5}")
#             shown += 1
#             if shown >= show_examples:
#                 break

#     # Simple warnings
#     if n_nonzero == 0:
#         print("ERROR: all fingerprints are zero. Check your featurization/init code.")
#     if n_scaff_valid == 0:
#         print("WARNING: no Murcko scaffolds extracted.")

#     return C  # so you can reuse the loaded arrays

# # ---------- (Optional) tiny smoke tests for metrics ----------
# def tiny_avgnn(Q_u8, R_u8, cap_query=2000, seed=0):
#     if len(Q_u8)==0 or len(R_u8)==0: return None
#     rng = np.random.default_rng(seed)
#     if len(Q_u8) > cap_query:
#         Q = Q_u8[rng.choice(len(Q_u8), size=cap_query, replace=False)]
#     else:
#         Q = Q_u8
#     popR = popcount_bytes(R_u8).sum(axis=1, dtype=np.int32)
#     best = np.empty(len(Q), dtype=np.float32)
#     for i, q in enumerate(Q):
#         pc_q = int(popcount_bytes(q).sum())
#         inter = popcount_bytes(np.bitwise_and(R_u8, q)).sum(axis=1, dtype=np.int32)
#         union = popR + pc_q - inter
#         with np.errstate(divide='ignore', invalid='ignore'):
#             sims = np.divide(inter, union, out=np.zeros_like(union, dtype=np.float32), where=(union>0))
#         best[i] = float(sims.max(initial=0.0))
#     return float(best.mean())

# def tiny_mmd(A_u8, B_u8, nA=2000, nB=2000, seed=0, pair_cap=50_000):
#     if len(A_u8)<2 or len(B_u8)<2: return None
#     rng = np.random.default_rng(seed)
#     A = A_u8[rng.choice(len(A_u8), size=min(nA,len(A_u8)), replace=False)]
#     B = B_u8[rng.choice(len(B_u8), size=min(nB,len(B_u8)), replace=False)]
#     popA = popcount_bytes(A).sum(axis=1, dtype=np.int32)
#     popB = popcount_bytes(B).sum(axis=1, dtype=np.int32)
#     # self
#     def mean_self(F, popF):
#         N=len(F); P=min(pair_cap, N*(N-1)//2)
#         i=rng.integers(0,N,size=P); j=rng.integers(0,N,size=P)
#         m=(i!=j); i=i[m]; j=j[m]
#         Fi, Fj = F[i], F[j]; pi, pj = popF[i], popF[j]
#         inter = popcount_bytes(np.bitwise_and(Fi, Fj)).sum(axis=1, dtype=np.int32)
#         union = pi + pj - inter
#         with np.errstate(divide='ignore', invalid='ignore'):
#             sims = np.divide(inter, union, out=np.zeros_like(union, dtype=np.float32), where=(union>0))
#         return float(sims.mean())
#     # cross
#     def mean_cross(F,G,popF,popG):
#         NF,NG=len(F),len(G); P=min(pair_cap, NF*NG)
#         i=rng.integers(0,NF,size=P); j=rng.integers(0,NG,size=P)
#         Fi, Gj = F[i], G[j]; pi, pj = popF[i], popG[j]
#         inter = popcount_bytes(np.bitwise_and(Fi, Gj)).sum(axis=1, dtype=np.int32)
#         union = pi + pj - inter
#         with np.errstate(divide='ignore', invalid='ignore'):
#             sims = np.divide(inter, union, out=np.zeros_like(union, dtype=np.float32), where=(union>0))
#         return float(sims.mean())
#     Exx=mean_self(A,popA); Eyy=mean_self(B,popB); Exy=mean_cross(A,B,popA,popB)
#     return Exx + Eyy - 2*Exy

# if __name__ == "__main__":
#     # EDIT these paths to your three caches:
#     full_dir   = "./chemcache/train_full_ECFP4_bits2048_chiral_dropDy_v1_sha256-a2626b3dbc954faa-5000"    # path to cache folder
#     subset_dir = "./chemcache/train_subset_ECFP4_bits2048_chiral_dropDy_v1_sha256-746ea05640e19b69-1000"
#     test_dir   = "./chemcache/test_ECFP4_bits2048_chiral_dropDy_v1_sha256-7784753e1127ab5b-5000"

#     F_full  = summarize_cache(full_dir)
#     F_sub   = summarize_cache(subset_dir)
#     F_test  = summarize_cache(test_dir)

#     # Optional smoke tests (small caps) — should be >0 if fingerprints are sane:
#     print("\n--- quick smoke tests ---")
#     ann_ts = tiny_avgnn(F_test["fps_uint8"], F_sub["fps_uint8"], cap_query=1000)
#     ann_sf = tiny_avgnn(F_sub["fps_uint8"],  F_full["fps_uint8"], cap_query=1000)
#     mmd_sf = tiny_mmd(F_sub["fps_uint8"], F_full["fps_uint8"], nA=2000, nB=2000)
#     print(f"AvgNN(Test→Subset, 1k queries): {ann_ts}")
#     print(f"AvgNN(Subset→Full, 1k queries): {ann_sf}")
#     print(f"MMD²(Subset↔Full, 2k/2k): {mmd_sf}")