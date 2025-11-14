import os, re, json, itertools, hashlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import rdFingerprintGenerator as rFG
from tqdm import tqdm
from contextlib import contextmanager
from time import perf_counter

@contextmanager
def timer(label: str):
    t0 = perf_counter()
    try:
        yield
    finally:
        dt = perf_counter() - t0
        print(f"[time] {label}: {dt:.2f}s")

# ----------------------------
# Dy normalization + featurization
# ----------------------------
_DY_BARE = re.compile(r'(?<!\[)Dy')
_DY_SMARTS = Chem.MolFromSmarts('[Dy]')

def norm_drop_dy(smi: str) -> str | None:
    if smi is None:
        return None
    smi2 = _DY_BARE.sub('[Dy]', smi)      # normalize bare Dy
    m = Chem.MolFromSmiles(smi2)
    if m is None:
        return None
    if _DY_SMARTS is not None:
        m = Chem.DeleteSubstructs(m, _DY_SMARTS, onlyFrags=False)
    if m.GetNumAtoms() == 0:
        return None
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass  # keep partially sanitized; ECFP is robust
    return Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)

@lru_cache(maxsize=None)
def _get_morgan_generator(radius: int, fp_bits: int, use_chirality: bool):
    """Create once per process; works across RDKit versions."""
    try:
        return rFG.GetMorganGenerator(radius=radius, includeChirality=use_chirality, fpSize=fp_bits)
    except TypeError:
        # older signature (positional)
        return rFG.GetMorganGenerator(int(radius), False, bool(use_chirality),
                                      True, False, True, None, int(fp_bits),
                                      None, None, False)

def fp_bytes_from_norm(norm_smi: str, radius=2, fp_bits=2048, use_chirality=True) -> bytes | None:
    m = Chem.MolFromSmiles(norm_smi)
    if m is None:
        return None
    gen = _get_morgan_generator(int(radius), int(fp_bits), bool(use_chirality))
    bv = gen.GetFingerprint(m)
    return DataStructs.BitVectToBinaryText(bv)

def _featurize_one(idx, smi, radius, nBits, use_chirality):
    nrm = norm_drop_dy(smi)
    if nrm is None:
        return (idx, None, None, None)
    # scaffold may be None — that's OK
    try:
        sc_m = Chem.MolFromSmiles(nrm)
        scaf = Chem.MolToSmiles(
            Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(sc_m),
            isomericSmiles=True, canonical=True
        )
        scaff = scaf if sc_m and sc_m.GetNumAtoms() > 0 else None
    except Exception:
        scaff = None
    fpb = fp_bytes_from_norm(nrm, radius, nBits, use_chirality)
    return (idx, nrm, scaff, fpb)

def featurize_smiles_mp(smiles, fp_radius=2, fp_bits=2048, use_chirality=True,
                        n_procs=None, chunksize=1024, show_progress=True):
    smiles = list(smiles)
    N = len(smiles)
    byte_len = fp_bits // 8
    norm = [None]*N
    scaff = [None]*N
    fps_u8 = np.zeros((N, byte_len), dtype=np.uint8)
    valid = np.zeros(N, dtype=bool)

    n_procs = n_procs or max(1, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=n_procs) as ex:
        it = ex.map(
            _featurize_one,
            range(N), smiles,
            itertools.repeat(fp_radius),
            itertools.repeat(fp_bits),
            itertools.repeat(use_chirality),
            chunksize=chunksize
        )
        if show_progress:
            it = tqdm(it, total=N, desc="Featurizing", unit="mol", smoothing=0.1)

        for i, nrm, sc, fpb in it:
            norm[i] = nrm
            scaff[i] = sc
            if fpb is not None:
                fps_u8[i, :] = np.frombuffer(fpb, dtype=np.uint8, count=byte_len)
                valid[i] = True

    pop = (np.unpackbits(fps_u8, axis=1).sum(axis=1)).astype(np.int32)
    return {
        "normalized_smiles": norm,
        "scaffolds": scaff,
        "fps_uint8": fps_u8,
        "popcount": pop,
        "valid_mask": valid,
        "meta": {"fp_radius": int(fp_radius), "fp_bits": int(fp_bits),
                 "use_chirality": bool(use_chirality), "n": int(N)}
    }

# ----------------------------
# Cache I/O
# ----------------------------
def digest_smiles(smiles_iter, algorithm="sha256", limit: int | None = None):
    h = hashlib.new(algorithm)
    n = 0
    for s in smiles_iter:
        h.update((s or "").encode("utf-8"))
        h.update(b"\n")
        n += 1
        if limit and n >= limit:
            break
    return f"{algorithm[:6]}-{h.hexdigest()[:16]}-{n}"

def make_cache_id(name: str, fp_bits: int, fp_radius: int, use_chirality: bool,
                  preprocess_version: str, smiles_digest: str):
    return f"{name}_ECFP{2*fp_radius}_bits{fp_bits}_{'chiral' if use_chirality else 'achiral'}_{preprocess_version}_{smiles_digest}"

def write_cache(out_dir: str, payload: dict, name: str,
                preprocess_version: str = "dropDy_v1",
                smiles_for_digest = None):
    os.makedirs(out_dir, exist_ok=True)
    fp_bits = int(payload["meta"]["fp_bits"])
    fp_radius = int(payload["meta"]["fp_radius"])
    use_chirality = bool(payload["meta"]["use_chirality"])
    norm = payload["normalized_smiles"]
    scaff = payload["scaffolds"]
    fps = payload["fps_uint8"]
    pop = payload["popcount"]
    valid = payload["valid_mask"]

    digest = digest_smiles(smiles_for_digest or norm)
    cid = make_cache_id(name, fp_bits, fp_radius, use_chirality, preprocess_version, digest)
    root = os.path.join(out_dir, cid)
    os.makedirs(root, exist_ok=True)

    # write files
    with open(os.path.join(root, "scaffolds.csv"), "w", encoding="utf-8") as f:
        f.write("scaffold\n")
        for s in scaff:
            f.write((s or "") + "\n")

    np.savez_compressed(os.path.join(root, "fp_matrix.npz"), fps=fps)
    np.save(os.path.join(root, "popcount.npy"), pop)
    np.save(os.path.join(root, "valid_mask.npy"), valid)

    meta = dict(payload["meta"])
    meta.update({
        "preprocess_version": preprocess_version,
        "cache_id": cid,
        "smiles_digest": digest,
        "n_valid_fp": int(valid.sum()),
    })
    with open(os.path.join(root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return root  # cache root path

def load_cache(root: str):
    # minimal, robust loader
    with open(os.path.join(root, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    scaff = []
    with open(os.path.join(root, "scaffolds.csv"), encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            scaff.append(line.rstrip("\n"))
    fps = np.load(os.path.join(root, "fp_matrix.npz"))["fps"]
    pop = np.load(os.path.join(root, "popcount.npy"))
    valid = np.load(os.path.join(root, "valid_mask.npy"))
    return {"scaffolds": scaff, "fps_uint8": fps, "popcount": pop, "valid_mask": valid, "meta": meta}

# ----------------------------
# Tanimoto kernels (NumPy)
# ----------------------------
_POP_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
def popcount_bytes(x_uint8: np.ndarray) -> np.ndarray:
    return _POP_LUT[x_uint8]

def _tanimoto_block(q_row_u8: np.ndarray, R_block_u8: np.ndarray, pc_q: int, pc_R: np.ndarray) -> np.ndarray:
    inter = popcount_bytes(np.bitwise_and(R_block_u8, q_row_u8)).sum(axis=1, dtype=np.int32)
    union = pc_R + pc_q - inter
    sims  = np.where(union > 0, inter / union, 0.0)
    return sims.astype(np.float32)

def avg_nn_tanimoto_cached(Q_u8: np.ndarray, R_u8: np.ndarray, popR: np.ndarray,
                           cap_query: int = 20000,
                           ref_cap: int | None = None,      # ← NEW
                           seed: int = 0, ref_block: int = 32768,  # a bit bigger block is fine
                           desc: str = "AvgNN"):
    if len(Q_u8) == 0 or len(R_u8) == 0:
        return {"mean": np.nan, "median": np.nan, "p05": np.nan, "p95": np.nan, "n": 0}
    rng = np.random.default_rng(seed)

    # sample queries
    Q = Q_u8[rng.choice(len(Q_u8), size=cap_query, replace=False)] if cap_query and len(Q_u8) > cap_query else Q_u8

    # sample reference (NEW)
    if ref_cap and len(R_u8) > ref_cap:
        idxR = rng.choice(len(R_u8), size=ref_cap, replace=False)
        R = R_u8[idxR]
        popR_use = popR[idxR]
    else:
        R = R_u8
        popR_use = popR

    best = np.empty(len(Q), dtype=np.float32)
    with timer(f"{desc} {len(Q)} queries vs {len(R)} refs"), \
         tqdm(total=len(Q), unit="query", desc=desc, smoothing=0.05) as pbar:
        for i, q in enumerate(Q):
            pc_q = int(popcount_bytes(q).sum())
            mx = 0.0
            for s in range(0, len(R), ref_block):
                Rb   = R[s:s+ref_block]
                pcRb = popR_use[s:s+ref_block]
                sims = _tanimoto_block(q, Rb, pc_q, pcRb)
                if sims.size:
                    v = float(np.max(sims))
                    if v > mx: mx = v
            best[i] = mx
            pbar.update(1)

    return {
        "mean":  float(np.mean(best)),
        "median":float(np.median(best)),
        "p05":   float(np.percentile(best, 5)),
        "p95":   float(np.percentile(best, 95)),
        "n":     int(len(best)),
    }

def mmd_tanimoto_cached(A_u8: np.ndarray, B_u8: np.ndarray,
                        popA: np.ndarray | None, popB: np.ndarray | None,
                        nA: int = 5000, nB: int = 5000, seed: int = 0,
                        pair_cap: int = 1_500_000):
    rng = np.random.default_rng(seed)
    if len(A_u8) < 2 or len(B_u8) < 2:
        return {"mmd2": np.nan, "Exx": np.nan, "Eyy": np.nan, "Exy": np.nan, "nA": len(A_u8), "nB": len(B_u8)}

    idxA = rng.choice(len(A_u8), size=min(nA, len(A_u8)), replace=False)
    idxB = rng.choice(len(B_u8), size=min(nB, len(B_u8)), replace=False)
    A = A_u8[idxA]; B = B_u8[idxB]
    if popA is None: popA = popcount_bytes(A).sum(axis=1, dtype=np.int32)
    else: popA = popA[idxA]
    if popB is None: popB = popcount_bytes(B).sum(axis=1, dtype=np.int32)
    else: popB = popB[idxB]

    def mean_self(F, popF):
        N = len(F)
        if N < 2: return np.nan
        P = min(pair_cap, N*(N-1)//2)
        i = rng.integers(0, N, size=P, endpoint=False)
        j = rng.integers(0, N, size=P, endpoint=False)
        mask = i != j
        i, j = i[mask], j[mask]
        Fi, Fj = F[i], F[j]
        pi, pj = popF[i], popF[j]
        inter = popcount_bytes(np.bitwise_and(Fi, Fj)).sum(axis=1, dtype=np.int32)
        union = pi + pj - inter
        with np.errstate(divide='ignore', invalid='ignore'):
            sims = np.divide(inter, union, out=np.zeros_like(union, dtype=np.float32), where=(union > 0))
        return float(np.mean(sims))

    def mean_cross(F, G, popF, popG):
        NF, NG = len(F), len(G)
        P = min(pair_cap, NF*NG)
        i = rng.integers(0, NF, size=P, endpoint=False)
        j = rng.integers(0, NG, size=P, endpoint=False)
        Fi, Gj = F[i], G[j]
        pi, pj = popF[i], popG[j]
        inter = popcount_bytes(np.bitwise_and(Fi, Gj)).sum(axis=1, dtype=np.int32)
        union = pi + pj - inter
        with np.errstate(divide='ignore', invalid='ignore'):
            sims = np.divide(inter, union, out=np.zeros_like(union, dtype=np.float32), where=(union > 0))
        return float(np.mean(sims))

    Exx = mean_self(A, popA); Eyy = mean_self(B, popB); Exy = mean_cross(A, B, popA, popB)
    return {"mmd2": Exx + Eyy - 2.0*Exy, "Exx": Exx, "Eyy": Eyy, "Exy": Exy, "nA": int(len(A)), "nB": int(len(B))}

# ----------------------------
# Scaffold overlap
# ----------------------------
def scaffold_overlap_metrics(scaffolds_A: list[str | None], scaffolds_B: list[str | None]):
    SA = {s for s in scaffolds_A if isinstance(s, str) and s}
    SB = {s for s in scaffolds_B if isinstance(s, str) and s}
    inter = SA & SB
    uni   = SA | SB
    return {
        "n_scaff_A": len(SA),
        "n_scaff_B": len(SB),
        "n_overlap": len(inter),
        "coverage_A_in_B": len(inter) / float(len(SA)) if SA else 0.0,
        "coverage_B_in_A": len(inter) / float(len(SB)) if SB else 0.0,
        "jaccard_scaff": len(inter) / float(len(uni)) if uni else 0.0,
    }

# ----------------------------
# Pairwise comparison (cache → metrics)
# ----------------------------
def compare_cached_pair(cacheA: dict, cacheB: dict,
                        seed: int = 42,
                        nn_cap_query: int = 20000,
                        mmd_cap_A: int = 5000,
                        mmd_cap_B: int = 5000,
                        ref_cap_A_to_B: int | None = None, ref_cap_B_to_A: int | None = None):
    A = cacheA; B = cacheB
    scaff = scaffold_overlap_metrics(A["scaffolds"], B["scaffolds"])

    ann_A_to_B = avg_nn_tanimoto_cached(A["fps_uint8"], B["fps_uint8"], B["popcount"],
                                        cap_query=nn_cap_query, ref_cap=ref_cap_A_to_B, seed=seed,
                                        desc="AvgNN A→B")
    ann_B_to_A = avg_nn_tanimoto_cached(B["fps_uint8"], A["fps_uint8"], A["popcount"],
                                        cap_query=nn_cap_query, ref_cap=ref_cap_B_to_A, seed=seed,
                                        desc="AvgNN B→A")
    mmd = mmd_tanimoto_cached(A["fps_uint8"], B["fps_uint8"], A["popcount"], B["popcount"],
                              nA=mmd_cap_A, nB=mmd_cap_B, seed=seed)
    return {
        "scaffold_overlap": scaff,
        "avgNN_A→B": ann_A_to_B,
        "avgNN_B→A": ann_B_to_A,
        "mmd_tanimoto": mmd,
        "counts": {
            "n_mols_A": int(len(A["scaffolds"])),
            "n_mols_B": int(len(B["scaffolds"])),
        }
    }

# ----------------------------
# Top-level: build caches (if needed) and compare 3 sets
# ----------------------------
def build_cache_from_smiles(smiles, out_dir: str, name: str,
                            fp_radius=2, fp_bits=2048, use_chirality=True,
                            n_procs=None, show_progress=True, preprocess_version="dropDy_v1"):
    feats = featurize_smiles_mp(smiles, fp_radius, fp_bits, use_chirality, n_procs, show_progress=show_progress)
    return write_cache(out_dir, feats, name=name,
                       preprocess_version=preprocess_version,
                       smiles_for_digest=feats["normalized_smiles"])

def compare_three_sets_with_cache(train_full_smiles=None, train_subset_smiles=None, test_smiles=None,
                                  full_cache_dir=None, subset_cache_dir=None, test_cache_dir=None,
                                  out_cache_root="./chemcache",
                                  fp_radius=2, fp_bits=2048, use_chirality=True,
                                  n_procs=None, seed=42,
                                  nn_cap_query=20000, mmd_cap_sub=5000, mmd_cap_full=5000):
    # Build caches if paths not provided
    if full_cache_dir is None:
        assert train_full_smiles is not None, "Provide train_full_smiles or full_cache_dir"
        full_cache_dir = build_cache_from_smiles(train_full_smiles, out_cache_root, "train_full",
                                                 fp_radius, fp_bits, use_chirality, n_procs)
    if subset_cache_dir is None:
        assert train_subset_smiles is not None, "Provide train_subset_smiles or subset_cache_dir"
        subset_cache_dir = build_cache_from_smiles(train_subset_smiles, out_cache_root, "train_subset",
                                                   fp_radius, fp_bits, use_chirality, n_procs)
    if test_cache_dir is None:
        assert test_smiles is not None, "Provide test_smiles or test_cache_dir"
        test_cache_dir = build_cache_from_smiles(test_smiles, out_cache_root, "test",
                                                 fp_radius, fp_bits, use_chirality, n_procs)

    # Load caches
    F_full = load_cache(full_cache_dir)
    F_sub  = load_cache(subset_cache_dir)
    F_test = load_cache(test_cache_dir)

    ref_cap_full = 2_000_000    # or 2_000_000 if you want extra accuracy
    # Subset could be 1M or 10M; if 10M, sampling to 1M helps
    ref_cap_subset = 1_000_000  # set to None if subset≈1M
    ref_cap_inverse = 1_000_000

    return {
        "paths": {
            "full": full_cache_dir,
            "subset": subset_cache_dir,
            "test": test_cache_dir,
        },
        "test_vs_subset": compare_cached_pair(
            F_test, F_sub, seed=seed,
            nn_cap_query=20_000,
            mmd_cap_A=50_000, mmd_cap_B=50_000,
            ref_cap_A_to_B=ref_cap_subset,   # A=test, B=subset
            ref_cap_B_to_A=None              # subset→test small ref, no need to sample
        ),
        "test_vs_full": compare_cached_pair(
            F_test, F_full, seed=seed,
            nn_cap_query=20_000,
            mmd_cap_A=50_000, mmd_cap_B=50_000,
            ref_cap_A_to_B=ref_cap_full,     # A=test, B=full (HUGE ref → sample)
            ref_cap_B_to_A=None              # full→test uses small ref (test), no need
        ),
        "subset_vs_full": compare_cached_pair(
            F_sub, F_full, seed=seed,
            nn_cap_query=20_000,
            mmd_cap_A=50_000, mmd_cap_B=50_000,
            ref_cap_A_to_B=ref_cap_full,     # A=subset, B=full (HUGE ref → sample)
            ref_cap_B_to_A=ref_cap_inverse if F_sub["meta"]["n"] >= 1_000_000 else None  # full→subset uses smaller ref (subset), optional: ref_cap_subset
        ),
        "meta": {
            "fp_radius": fp_radius, "fp_bits": fp_bits, "use_chirality": use_chirality,
            "seed": seed, "nn_cap_query": nn_cap_query,
            "mmd_cap_sub": mmd_cap_sub, "mmd_cap_full": mmd_cap_full,
        }
    }

def _fmt(x, nd=3):
    """Format float or None nicely"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.{nd}f}"

def pretty_print_comparison(label: str, comp: dict, out_path: str | None = None):
    lines = []
    lines.append(f"\n=== {label} ===")

    scaff = comp.get("scaffold_overlap", {})
    if scaff:
        lines.append(
            "Scaffold overlap:"
            f" A={scaff.get('n_scaff_A', 'NA')},"
            f" B={scaff.get('n_scaff_B', 'NA')},"
            f" overlap={scaff.get('n_overlap', 'NA')}"
        )
        lines.append(
            "  coverage A∈B: " + _fmt(scaff.get('coverage_A_in_B')) +
            " | coverage B∈A: " + _fmt(scaff.get('coverage_B_in_A')) +
            " | Jaccard: " + _fmt(scaff.get('jaccard_scaff'))
        )

    # MMD results
    mmd = comp.get("mmd_tanimoto", {})
    if mmd:
        lines.append(f"MMD²: {_fmt(mmd.get('mmd2'))}")
        lines.append(f"  Exx: {_fmt(mmd.get('Exx'))}, "
                     f"Eyy: {_fmt(mmd.get('Eyy'))}, "
                     f"Exy: {_fmt(mmd.get('Exy'))}")
        lines.append(f"  nA={mmd.get('nA')}, nB={mmd.get('nB')}")

    # AvgNN results
    annAB = comp.get("avgNN_A→B", {})
    annBA = comp.get("avgNN_B→A", {})
    if annAB:
        lines.append("AvgNN A→B: mean=" + _fmt(annAB.get('mean')) +
                     ", median=" + _fmt(annAB.get('median')) +
                     ", p05=" + _fmt(annAB.get('p05')) +
                     ", p95=" + _fmt(annAB.get('p95')) +
                     f", n={annAB.get('n')}")
    if annBA:
        lines.append("AvgNN B→A: mean=" + _fmt(annBA.get('mean')) +
                     ", median=" + _fmt(annBA.get('median')) +
                     ", p05=" + _fmt(annBA.get('p05')) +
                     ", p95=" + _fmt(annBA.get('p95')) +
                     f", n={annBA.get('n')}")

    # Print to console
    for ln in lines:
        print(ln)

    # Optionally save to file
    if out_path is not None:
        with open(out_path, "a", encoding="utf-8") as f:
            for ln in lines:
                f.write(ln + "\n")
