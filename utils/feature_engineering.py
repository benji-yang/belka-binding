from torch_geometric.data import Data
import numpy as np
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter, deque
import torch
import configs.parameters as cf
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import shortest_path

# import duckdb
# import pandas as pd

# 2) Embed one or more 3D conformers

# def get_positional_weight(filepath, p_to_idx_map):
# 	con = duckdb.connect()
# 	df = con.query(f"""
# 	SELECT
# 		protein_name,
# 		/* binds is stored as 0/1, so SUM(binds) = # of positives        */
# 		SUM(binds)                       AS binds_count,
# 		COUNT(*) - SUM(binds)            AS non_binds_count,  -- # of zeros
# 		COUNT(*)                         AS total_molecules
# 	FROM parquet_scan('{filepath}')
# 	GROUP BY protein_name
# 	ORDER BY protein_name
# 	""").df()
# 	con.close()

# 	p_order = df.protein_name.values
# 	pos_counts = torch.tensor(df.binds_count.values, dtype=torch.float)
# 	neg_counts = torch.tensor(df.non_binds_count.values, dtype=torch.float)

# 	pos_weight = (neg_counts / pos_counts)  # shape (3,)
# 	pos_weight_ord = torch.zeros(cf.N_PROTEINS, dtype=torch.float)
# 	for ind, name in enumerate(p_order):
# 		protein_idx = p_to_idx_map[name]
# 		pos_weight_ord[protein_idx] = pos_weight[ind]
# 	return pos_weight_ord


#Get features of an atom (one-hot encoding:)
'''
	1.atom element: 44+1 dimensions
	2.the atom's hybridization: 5 dimensions
	3.degree of atom: 6 dimensions
	4.total number of H bound to atom: 6 dimensions
	5.number of implicit H bound to atom: 6 dimensions
	6.whether the atom is on ring: 1 dimension
	7.whether the atom is aromatic: 1 dimension
	Total: 70 dimensions
'''


def get_group_from_map(symbol, g_map):
    """
    A fast, lightweight function to get the group from a pre-computed map.
    """
    return g_map.get(symbol, "Symbol not found")


def parse_and_drop_belka_tag(smiles: str) -> Chem.Mol | None:
	_DY_BARE = re.compile(r'(?<!\[)Dy')  # normalize bare Dy -> [Dy]
	smi = _DY_BARE.sub('[Dy]', smiles)
	m = Chem.MolFromSmiles(smi)
	if m is None:
		return None
	m = Chem.DeleteSubstructs(m, Chem.MolFromSmarts('[Dy]'), onlyFrags=False)
	Chem.SanitizeMol(m)
	return m  # no Hs here (you'll AddHs later, as before)


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def get_atom_features(atom):
    """Returns a 1D numpy array of concatenated atom features."""
    a_type = atom.GetSymbol()
    Z = cf.ATOM_IDX.get(atom.GetAtomicNum(), cf.ATOM_IDX[cf.DEFAULT_IDX])
    period  = cf.PERIOD_IDX.get(cf.PT.GetRow(a_type), cf.PERIOD_IDX[cf.DEFAULT_IDX])
    group = cf.GROUP_IDX.get(get_group_from_map(a_type, cf.GROUP_MAP), cf.GROUP_IDX[cf.DEFAULT_IDX])
    degree = cf.DEGREE_IDX.get(atom.GetDegree(), cf.DEGREE_IDX[cf.DEFAULT_IDX])
    valence = cf.VALENCE_IDX.get(atom.GetValence(Chem.ValenceType.IMPLICIT), cf.VALENCE_IDX[cf.DEFAULT_IDX])
    num_h = cf.NUMH_IDX.get(atom.GetTotalNumHs(includeNeighbors=True), cf.NUMH_IDX[cf.DEFAULT_IDX])
    num_radical = cf.RAD_IDX.get(atom.GetNumRadicalElectrons(), cf.RAD_IDX[cf.DEFAULT_IDX])
    charge = cf.CHG_IDX.get(atom.GetFormalCharge(), cf.CHG_IDX[cf.DEFAULT_IDX])
    hybrid = cf.HYB_IDX.get(atom.GetHybridization().name, cf.HYB_IDX[cf.DEFAULT_IDX])
    aromatic = int(atom.GetIsAromatic())
    in_ring = int(atom.IsInRing())
    chiral_center = int(atom.HasProp('_ChiralityPossible'))
    return np.array([Z, period, group, degree, valence, num_h, num_radical, charge, hybrid, 
                        aromatic, in_ring, chiral_center], dtype=float)


#Get features of an edge (one-hot encoding)
'''
	1.single/double/triple/aromatic: 4 dimensions
	2.the atom's hybridization: 1 dimensions
	3.whether the bond is on ring: 1 dimension
	Total: 6 dimensions
'''

def get_bond_features(bond):
    bond_type = cf.BOND_TYPE_IDX.get(bond.GetBondType().name, cf.BOND_TYPE_IDX[cf.DEFAULT_IDX])
    bond_stereo = cf.BOND_STEREO_IDX.get(bond.GetStereo().name, cf.BOND_STEREO_IDX[cf.DEFAULT_IDX])
    is_conjugated = int(bond.GetIsConjugated())
    is_in_ring = int(bond.IsInRing())
    return np.array([bond_type, bond_stereo, is_conjugated, is_in_ring], dtype=float)


def get_face_features(mol, atoms, bonds, bond_counts):
	size = cf.RING_SIZE_IDX.get(len(atoms), cf.RING_SIZE_IDX[cf.DEFAULT_IDX])

	aromatic = int(all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in atoms))

	het_count = cf.HET_COUNT_IDX.get(sum(1 for a in atoms if mol.GetAtomWithIdx(a).GetAtomicNum() != 6), cf.HET_COUNT_IDX[cf.DEFAULT_IDX])

	saturated = int(all(
		(not mol.GetBondWithIdx(b).GetIsAromatic())
		and mol.GetBondWithIdx(b).GetBondType() == Chem.BondType.SINGLE
		for b in bonds
		))

	has_fusion = int(any(bond_counts[b] > 1 for b in bonds))

	Zs = [mol.GetAtomWithIdx(a).GetAtomicNum() for a in atoms]
	avg_en = np.mean([cf.PAULING_ARR[Z] for Z in Zs])
	if avg_en < (cf.MIN_EN - cf.TOLERANCE) or avg_en > (cf.MAX_EN + cf.TOLERANCE):
		en = cf.ELECTRONEGATIVITY_IDX[cf.DEFAULT_IDX]
	else:
	# snap to nearest bin index
		raw_idx = round((avg_en - cf.MIN_EN) / cf.INTERVAL)
		idx     = max(0, min(raw_idx, len(cf.ELECTRONEGATIVITY_LIST) - 1))
		bin_val = cf.ELECTRONEGATIVITY_LIST[idx]

		# lookup (fall back to misc just in case)
		en = cf.ELECTRONEGATIVITY_IDX.get(bin_val,
			cf.ELECTRONEGATIVITY_IDX[cf.DEFAULT_IDX])
	return np.array([size, aromatic, het_count, saturated, has_fusion, en])


def barycentric_1_skeleton(edge_index: np.ndarray,   # (2, E_dir) with both (u,v) and (v,u)
                           face_index: np.ndarray,   # (2, B_inc): [face_id, edge_id (directed, column into edge_index)]
                           num_nodes: int | None = None,
                           include_vertex_face: bool = True):
    """
    Build the 1-skeleton adjacency of the barycentric subdivision graph.

    Vertices:
      rank-0: original vertices (0..N0-1)
      rank-1: undirected edges (N0..N0+N1-1)
      rank-2: faces (N0+N1..N0+N1+N2-1)

    Edges (undirected):
      0–1, 1–2, and (optionally) 0–2.
    """
    u_dir, v_dir = edge_index
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    N0 = int(num_nodes)

    # -- collapse directed edges -> undirected, keep mapping --
    u_can = np.minimum(u_dir, v_dir)
    v_can = np.maximum(u_dir, v_dir)
    pairs = np.stack([u_can, v_can], axis=1)                         # (E_dir, 2)
    undirected_pairs, inv_dir2undir = np.unique(pairs, axis=0, return_inverse=True)
    e_u = undirected_pairs[:, 0].astype(np.int64)                    # endpoints of undirected edge i
    e_v = undirected_pairs[:, 1].astype(np.int64)
    N1 = int(undirected_pairs.shape[0])

    # -- faces --
    F = int(face_index[0].max()) + 1 if face_index.size > 0 else 0
    N2 = F

    off_e = N0
    off_f = N0 + N1
    N_total = N0 + N1 + N2

    # 0–1 (vertex–edge)
    edge_ids = np.arange(N1, dtype=np.int64)
    row_0_1 = np.concatenate([e_u, e_v])
    col_0_1 = np.concatenate([edge_ids + off_e, edge_ids + off_e])

    # 1–2 (edge–face): remap directed edge ids -> undirected, drop duplicates
    if face_index.size > 0 and N2 > 0:
        f_ids_dir = face_index[0].astype(np.int64)
        e_ids_dir = face_index[1].astype(np.int64)
        e_ids_undir = inv_dir2undir[e_ids_dir].astype(np.int64)

        fe_pairs = np.unique(np.stack([f_ids_dir, e_ids_undir], axis=1), axis=0)
        f_ids = fe_pairs[:, 0]
        e_ids = fe_pairs[:, 1]

        row_1_2 = e_ids + off_e
        col_1_2 = f_ids + off_f
    else:
        row_1_2 = col_1_2 = np.empty(0, dtype=np.int64)
        f_ids = e_ids = np.empty(0, dtype=np.int64)

    # 0–2 (vertex–face): compose (vertex–edge) with (edge–face), dedupe
    if include_vertex_face and face_index.size > 0 and N2 > 0:
        # vertices incident to each undirected edge in e_ids
        vf_left  = np.stack([e_u[e_ids], f_ids], axis=1)   # (vertex_u, face)
        vf_right = np.stack([e_v[e_ids], f_ids], axis=1)   # (vertex_v, face)
        vf_pairs = np.unique(np.concatenate([vf_left, vf_right], axis=0), axis=0)
        row_0_2 = vf_pairs[:, 0].astype(np.int64)
        col_0_2 = vf_pairs[:, 1].astype(np.int64) + off_f
    else:
        row_0_2 = col_0_2 = np.empty(0, dtype=np.int64)

    # symmetrize
    row = np.concatenate([row_0_1, col_0_1, row_1_2, col_1_2, row_0_2, col_0_2])
    col = np.concatenate([col_0_1, row_0_1, col_1_2, row_1_2, col_0_2, row_0_2])

    data = np.ones(row.shape[0], dtype=np.float64)
    A = sp.coo_matrix((data, (row, col)), shape=(N_total, N_total)).tocsr()
    A.data[:] = 1.0
    A.setdiag(0.0)
    A.eliminate_zeros()

    maps = {
        "undir_edge_index": undirected_pairs.T,   # (2, N1)
        "dir2undir": inv_dir2undir,               # len = E_dir
        "offsets": {"edge": off_e, "face": off_f},
        "sizes": {"N0": N0, "N1": N1, "N2": N2, "N_total": N_total},
    }
    return A, maps


def symmetric_laplacian(A: sp.csr_matrix) -> sp.csr_matrix:
    # L = I - D^{-1/2} A D^{-1/2}
    degs = np.asarray(A.sum(axis=1)).ravel()
    with np.errstate(divide='ignore'):
        invsqrt = np.where(degs > 0, 1.0 / np.sqrt(degs), 0.0)
    Dm12 = sp.diags(invsqrt, 0, format='csr')
    L = sp.eye(A.shape[0], format='csr') - (Dm12 @ A @ Dm12)
    # numeric symmetrization (guards tiny asymmetries)
    L = (L + L.T) * 0.5
    return L


def laplacian_encodings(edge_index, k=7, num_nodes=None, zero_tol=1e-10):
    """
    Laplacian positional encodings from the symmetric normalized Laplacian:
        L_sym = I - D^{-1/2} A_undirected D^{-1/2}

    edge_index: shape (2, E) directed edges (may contain both (u,v) and (v,u))
    k:          number of non-trivial eigenpairs to return
    num_nodes:  optional; inferred from edge_index if None
    zero_tol:   threshold to treat eigenvalues as zero (handles disconnected graphs)

    Returns:
        eigvecs: (N, k)  columns are eigenvectors
        eigvals: (k,)    selected non-trivial eigenvalues, rescaled to [0,1]
    """
    src, dst = edge_index
    src = np.asarray(src, dtype=int)
    dst = np.asarray(dst, dtype=int)

    if num_nodes is None:
        max_idx = -1
        if src.size: max_idx = max(max_idx, int(src.max()))
        if dst.size: max_idx = max(max_idx, int(dst.max()))
        N = max_idx + 1
    else:
        N = int(num_nodes)

    # Directed adjacency, then symmetrize once for Laplacian
    A = np.zeros((N, N), dtype=float)
    if src.size:
        A[src, dst] = 1.0
    A = np.maximum(A, A.T)  # undirected support (boolean-OR)

    # Degrees and D^{-1/2} (safe for zeros)
    deg = A.sum(axis=1)
    invsqrt = np.zeros_like(deg)
    nz = deg > 0
    invsqrt[nz] = 1.0 / np.sqrt(deg[nz])
    D_inv_sqrt = np.diag(invsqrt)

    # Symmetric normalized Laplacian
    L_sym = np.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt

    # Eigen-decomposition (L_sym is symmetric PSD)
    eigvals_all, eigvecs_all = np.linalg.eigh(L_sym)  # ascending order

    # Skip ALL near-zero eigenvalues (one per connected component)
    keep = eigvals_all > zero_tol
    eigvals_nt = eigvals_all[keep]
    eigvecs_nt = eigvecs_all[:, keep]

    # Take the first k non-trivial pairs
    eigvals = eigvals_nt[:k]
    eigvecs = eigvecs_nt[:, :k]

    # Rescale eigenvalues to [0,1] for convenience (optional)
    if eigvals.size:
        mx = eigvals.max()
        if mx > 0:
            eigvals = eigvals / mx

    # Pad if fewer than k were available
    m = eigvals.shape[0]
    if m < k:
        eigvals = np.pad(eigvals, (0, k - m), constant_values=0.0)
        if eigvecs.size == 0:
            eigvecs = np.zeros((N, 0), dtype=float)
        eigvecs = np.concatenate([eigvecs, np.zeros((N, k - m), dtype=float)], axis=1)

    # (Optional) stabilize sign: flip each eigenvector so its largest-magnitude entry is positive
    for j in range(eigvecs.shape[1]):
        col = eigvecs[:, j]
        if col.size:
            idx = np.argmax(np.abs(col))
            if col[idx] < 0:
                eigvecs[:, j] = -col

    return eigvecs, eigvals


def lap_pe_barycentric(edge_index, face_index, pos_enc_dim=7,
                       include_vertex_face=True, zero_tol=1e-9):
    # 1) Build barycentric 1-skeleton (with 0–2 links if you want)
    A, maps = barycentric_1_skeleton(edge_index, face_index,
                                     include_vertex_face=include_vertex_face)
    N0 = maps["sizes"]["N0"]
    N1 = maps["sizes"]["N1"]
    N2 = maps["sizes"]["N2"]
    N  = maps["sizes"]["N_total"]

    # 2) Symmetric normalized Laplacian
    L = symmetric_laplacian(A)

    # 3) Smallest-magnitude eigenpairs (oversample to drop all zeros)
    k_request = min(max(1, pos_enc_dim + 8), N - 1)  # guard
    evals_all, evecs_all = eigsh(L, k=k_request, which='SM')
    order = np.argsort(evals_all)
    evals_all = evals_all[order]
    evecs_all = evecs_all[:, order]

    # 4) Drop all ~zero eigenvalues (multiple comps ⇒ multiple zeros)
    nonzero_mask = evals_all > zero_tol
    evals_nz = evals_all[nonzero_mask]
    evecs_nz = evecs_all[:, nonzero_mask]

    # 5) Take up to pos_enc_dim, pad if needed
    take = min(pos_enc_dim, evecs_nz.shape[1])
    eigvals = evals_nz[:take]
    eigvecs = evecs_nz[:, :take]

    if take < pos_enc_dim:
        # pad columns with zeros
        pad_c = pos_enc_dim - take
        if eigvecs.size == 0:
            eigvecs = np.zeros((N, 0), dtype=np.float64)
        eigvecs = np.concatenate([eigvecs, np.zeros((N, pad_c), dtype=eigvecs.dtype)], axis=1)
        eigvals = np.pad(eigvals, (0, pad_c), mode='constant', constant_values=0.0)

    # 6) Deterministic sign fix + L2 normalize columns (safety)
    if eigvecs.shape[1] > 0:
        signs = np.sign(eigvecs[np.abs(eigvecs).argmax(axis=0), np.arange(eigvecs.shape[1])])
        signs[signs == 0] = 1.0
        eigvecs *= signs
        norms = np.linalg.norm(eigvecs, axis=0, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        eigvecs = eigvecs / norms

    # 7) (Optional) map eigenvalues to [0,1].
    # For symmetric L, spectrum ⊂ [0,2]; divide by 2 for [0,1].
    eigvals01 = np.clip(eigvals / 2.0, 0.0, 1.0)

    # 8) Slice back to 0/1/2 ranks
    node_pos_enc = eigvecs[:N0, :]                     # (N0, k)
    edge_pos_enc = eigvecs[N0:N0+N1, :]                # (N1, k)
    face_pos_enc = eigvecs[N0+N1:N0+N1+N2, :]          # (N2, k)

    # 9) Broadcast edge encodings to directed edge_index
    dir2undir = maps["dir2undir"]                      # len = E_dir
    edge_pos_enc_directed = edge_pos_enc[dir2undir, :] # (E_dir, k)

    # return {
    #     "eigvals": eigvals,          # raw (0..2)
    #     "eigvals01": eigvals01,      # scaled to [0,1]
    #     "eigvecs": eigvecs,          # (N_total, k)
    #     "node_pos_enc": node_pos_enc,
    #     "edge_pos_enc": edge_pos_enc,
    #     "edge_pos_enc_directed": edge_pos_enc_directed,
    #     "face_pos_enc": face_pos_enc,
    #     "maps": maps
    # }
    return eigvals01, node_pos_enc, edge_pos_enc_directed, face_pos_enc


def random_walk_landing_probs(edge_index, k_steps=list(range(1, 17)), space_dim=0):
    """
    Returns an (N, len(k_steps)) array whose (i, j) entry is
      diag(P^{k_j})[i] * (k_j ** (space_dim/2)),
    where P = D_out^{-1} A is the row-stochastic random-walk matrix
    on the **directed** graph defined by edge_index (shape (2, E)).
    """
    # Unpack and ensure integer arrays
    src, dst = edge_index
    src = np.asarray(src, dtype=int)
    dst = np.asarray(dst, dtype=int)

    # Infer number of nodes
    if src.size == 0 and dst.size == 0:
        return np.zeros((0, len(k_steps)))
    num_nodes = int(max(src.max() if src.size else -1,
                        dst.max() if dst.size else -1)) + 1

    # 1) Directed adjacency (count multiplicities if duplicates exist)
    A = np.zeros((num_nodes, num_nodes), dtype=float)
    for u, v in zip(src, dst):
        A[u, v] += 1.0

    # 2) Row-normalize to get random-walk matrix P
    deg = A.sum(axis=1)                      # out-degrees
    P = np.divide(A, deg[:, None],
                  out=np.zeros_like(A),
                  where=deg[:, None] > 0)

    # Dangling nodes: stay put
    dangling = (deg == 0)
    if np.any(dangling):
        idx = np.where(dangling)[0]
        P[idx, idx] = 1.0

    # 3) Compute diag(P^k) efficiently for the requested k's
    max_k = max(k_steps) if len(k_steps) else 0
    diags_seq = []
    if max_k > 0:
        Pk = P.copy()                # P^1
        diags_seq.append(np.diag(Pk))
        for _ in range(2, max_k + 1):
            Pk = Pk @ P
            diags_seq.append(np.diag(Pk))
    else:
        diags_seq = []

    # 4) Assemble in the original k order with scaling
    walks = [diags_seq[k - 1] * (k ** (space_dim / 2.0)) for k in k_steps]
    return np.stack(walks, axis=1) if walks else np.zeros((num_nodes, 0))


def random_walk_barycentric(edge_index,
                            face_index,
                            k_steps=list(range(1, 17)),
                            include_vertex_face=True,
                            space_dim=0,
                            return_directed_edges=True):
    """
    Random-walk return probabilities (diagonal of P^k) on the barycentric 1-skeleton.

    Args:
      edge_index: (2, E_dir) directed edges, contains both (u,v) and (v,u)
      face_index: (2, B_inc): rows = [face_id, edge_id (directed index into edge_index)]
      k_steps:    iterable of integers k >= 0
      include_vertex_face: if True, add 0–2 (vertex–face) links (standard barycentric 1-skeleton);
                           if False, only 0–1 and 1–2 (Hasse/cover-only)
      space_dim:  multiply diag(P^k) by k**(space_dim/2) (kept to match your scaling)
      return_directed_edges: if True, also return edge features broadcast to directed edges

    Returns:
      {
        "node_rw": (N0, len(k_steps)),
        "edge_rw": (N1, len(k_steps)),
        "face_rw": (N2, len(k_steps)),
        "edge_rw_directed": (E_dir, len(k_steps))  # only if return_directed_edges=True
        "maps": {...}  # includes sizes, offsets, dir2undir
      }
    """
    # 1) Build barycentric adjacency (handles dir→undir mapping, adds 0–2 if requested)
    A, maps = barycentric_1_skeleton(edge_index, face_index, include_vertex_face=include_vertex_face)
    N0, N1, N2 = maps["sizes"]["N0"], maps["sizes"]["N1"], maps["sizes"]["N2"]
    N_total    = maps["sizes"]["N_total"]

    # 2) Random-walk transition matrix P = D^{-1} A
    degs = np.asarray(A.sum(axis=1)).ravel()
    invdeg = np.divide(1.0, degs, out=np.zeros_like(degs, dtype=float), where=degs > 0)
    Dinv = sp.diags(invdeg, 0, format='csr')
    P = Dinv @ A  # sparse (CSR)

    # 3) Compute diag(P^k) for all k in k_steps (incremental powers, supports unsorted input)
    k_steps = np.asarray(k_steps, dtype=int)
    if (k_steps < 0).any():
        raise ValueError("k_steps must be nonnegative integers.")
    order = np.argsort(k_steps)
    k_sorted = k_steps[order]

    H = np.zeros((N_total, len(k_sorted)), dtype=float)
    P_t = sp.eye(N_total, format='csr')  # P^0
    last_k = 0
    for j, k in enumerate(k_sorted):
        # advance from last_k to k
        for _ in range(k - last_k):
            P_t = P_t @ P
        last_k = k

        diag = P_t.diagonal()
        # match your scaling: diag(P^k) * k^(d/2)
        if k == 0:
            # keep your original semantics; if you prefer scale=1 at k=0, set scale = 1.0
            scale = 0.0 if space_dim > 0 else 1.0
        else:
            scale = k ** (space_dim / 2.0)
        H[:, j] = diag * scale

    # put columns back in the original k_steps order
    inv = np.argsort(order)
    H = H[:, inv]

    # 4) Slice back to 0/1/2 ranks
    node_rw = H[:N0, :]
    edge_rw = H[N0:N0+N1, :]
    face_rw = H[N0+N1:N0+N1+N2, :]

    out = {
        "node_rw": node_rw,
        "edge_rw": edge_rw,
        "face_rw": face_rw,
        "maps": maps,
    }

    # 5) (Optional) broadcast edge features to directed edges so (u,v) and (v,u) share values
    if return_directed_edges:
        dir2undir = maps["dir2undir"]           # length = E_dir
        out["edge_rw_directed"] = edge_rw[dir2undir, :]

    return node_rw, out["edge_rw_directed"], face_rw


def all_pairs_shortest_paths_unweighted(edge_index, num_nodes=None):
    src, dst = edge_index
    if num_nodes is None:
        num_nodes = int(max(src.max(), dst.max())) + 1
    # build adjacency list
    nbrs = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        nbrs[u].append(v)
        # nbrs[v].append(u)   # omit if the graph is directed

    # distances[i] will be an array of length num_nodes
    D = np.full((num_nodes, num_nodes), np.inf)
    for start in range(num_nodes):
        dist = np.full(num_nodes, -1, dtype=int)
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in nbrs[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
        D[start] = dist
    return D


def all_pairs_shortest_paths_barycentric(edge_index: np.ndarray,
                                         face_index: np.ndarray,
                                         include_vertex_face: bool = True,
                                         return_full: bool = False):
    """
    Compute all-pairs unweighted shortest-path distances on the
    barycentric 1-skeleton (0–1, 1–2, and optionally 0–2).

    Args
    ----
    edge_index : (2, E_dir) np.ndarray
        Directed edges (contains both (u,v) and (v,u)). They will be
        collapsed to undirected for edge-nodes.
    face_index : (2, B_inc) np.ndarray
        [face_id, edge_id] where edge_id indexes *directed* edges;
        it will be remapped to undirected inside barycentric_1_skeleton.
    include_vertex_face : bool
        If True, add 0–2 links (standard barycentric 1-skeleton).
        If False, only 0–1 and 1–2 (Hasse / cover-only).
    return_full : bool
        If True, return the full N_total×N_total distance matrix too.

    Returns
    -------
    spd_nodes : (N0, N0) np.ndarray of int32
        Node–node hop distances; -1 for unreachable.
    (optionally) D_full : (N_total, N_total) np.ndarray of int32
        Full barycentric distances (nodes, edges, faces), -1 for unreachable.
    maps : dict
        Offsets and sizes (includes dir2undir).
    """
    # Build barycentric adjacency (collapses directed edges, remaps face→edge)
    A, maps = barycentric_1_skeleton(edge_index, face_index,
                                     include_vertex_face=include_vertex_face)
    N0 = maps["sizes"]["N0"]
    N1 = maps["sizes"]["N1"]
    N2 = maps["sizes"]["N2"]
    N_total = maps["sizes"]["N_total"]

    # All-pairs shortest paths on an unweighted, undirected graph
    # shortest_path returns float with inf for unreachable; we convert to int with -1.
    D_float = shortest_path(A, directed=False, unweighted=True, return_predecessors=False)
    D_int = np.full(D_float.shape, -1, dtype=np.int32)
    finite = np.isfinite(D_float)
    D_int[finite] = D_float[finite].astype(np.int32)

    spd_nodes = D_int[:N0, :N0]

    if return_full:
        return spd_nodes, D_int, maps
    return spd_nodes


def make_raw_inputs_from_spd(spd):
    """
    spd: (N,N) integer array of shortest‐path distances,
         with spd[i,j] >= 1 for reachable i≠j,
             spd[i,i] = 0,
             spd[i,j] = -1 for unreachable pairs.
    max_d: maximum distance you’ll embed (anything beyond is clamped).
    Returns raw_inputs of shape (N,N) with:
      -1          : no attention (unreachable)
       0          : padding / self‐attention (i==j)
       d+1 (1<=)  : distance d offset by 1, clamped at max_d+1
    """
    N = spd.shape[0]
    raw = np.full((N, N), -1, dtype=np.int8)

    # 1) self‐pairs → 0 (padding / untrainable)
    np.fill_diagonal(raw, 0)

    # 2) reachable off‐diagonals → distance+1 (clamped)
    # mask of entries with a valid distance ≥1
    reachable = (spd >= 1)
    # assign offset by 1
    raw[reachable] = spd[reachable] + 1

    return raw


# # A small Pauling-scale electronegativity map; extend as needed
# ELEC = {
#     1: 2.20,   # H
#     6: 2.55,   # C
#     7: 3.04,   # N
#     8: 3.44,   # O
#     9: 3.98,   # F
#    15: 2.19,   # P
#    16: 2.58,   # S
#    17: 3.16,   # Cl
#    35: 2.96,   # Br
#    53: 2.66,   # I
# }

# def ring_properties(smiles: str):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         raise ValueError(f"Invalid SMILES: {smiles}")

#     ri = mol.GetRingInfo()
#     atom_rings = ri.AtomRings()   # tuple of atom‐index tuples
#     bond_rings = ri.BondRings()   # tuple of bond‐index tuples

#     # Precompute how many rings each bond belongs to
#     bond_counts = Counter(b for ring in bond_rings for b in ring)

#     props = []
#     for ring_idx, (atoms, bonds) in enumerate(zip(atom_rings, bond_rings)):
#         size = len(atoms)

#         # aromatic if *all* atoms in the ring are flagged aromatic
#         aromatic = all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in atoms)

#         # heteroatom count = non-carbon (Z != 6)
#         het_count = sum(1 for a in atoms
#                         if mol.GetAtomWithIdx(a).GetAtomicNum() != 6)

#         # saturated if *all* ring bonds are SINGLE and non‐aromatic
#         saturated = all(
#             (not mol.GetBondWithIdx(b).GetIsAromatic())
#             and mol.GetBondWithIdx(b).GetBondType() == Chem.BondType.SINGLE
#             for b in bonds
#         )

#         # fused if any ring bond belongs to >1 ring
#         has_fusion = any(bond_counts[b] > 1 for b in bonds)

#         # average electronegativity over ring atoms (default to C=2.55 if unknown)
#         en_vals = [
#             ELEC.get(mol.GetAtomWithIdx(a).GetAtomicNum(), 2.55)
#             for a in atoms
#         ]
#         avg_en = sum(en_vals) / size

#         props.append({
#             'ring_index':       ring_idx,
#             'size':             size,
#             'aromatic':         aromatic,
#             'heteroatom_count': het_count,
#             'saturated':        saturated,
#             'has_fusion':       has_fusion,
#             'avg_electronegativity': avg_en,
#         })

#     return props


def smile_to_graph(smiles, faces=False, barycentrics=False, position=False, replace=True, check_symbol_alignment=False):
	if replace:
		mol = parse_and_drop_belka_tag(smiles)
	else:
		mol = Chem.MolFromSmiles(smiles)
	# if removeH:
	# 	mol = Chem.RemoveHs(mol) 

	mol_h = Chem.AddHs(mol)

	heavy_idx = [atom.GetIdx() for atom in mol_h.GetAtoms() if atom.GetSymbol() != 'H']
	heavy_symbols = [mol_h.GetAtomWithIdx(i).GetSymbol() for i in heavy_idx]
	if position:
		cid  = AllChem.EmbedMolecule(mol_h, cf.ETKDG_PARAMS)
		if cid == -1:
			raise ValueError("Embedding failed; try relaxing parameters.")
		AllChem.UFFOptimizeMolecule(mol_h, confId=cid)
		# results = AllChem.MMFFOptimizeMoleculeConfs(mol_h)
		# best_idx = min(range(len(results)), key=lambda i: results[i][1])
		# best_cid = cids[best_idx]
		conf = mol_h.GetConformer(cid)
		pos = np.vstack([conf.GetAtomPosition(i) for i in heavy_idx], dtype=float)

	else:
		pos = np.zeros((len(heavy_idx), 3), dtype=float)
	
	mol = Chem.RemoveHs(mol_h)
	node_feats = np.stack([get_atom_features(a) for a in mol.GetAtoms()], axis=0)  # shape [N, D_node]

	if check_symbol_alignment:
		new_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
		if heavy_symbols != new_symbols:
			raise ValueError(
				f"Atom‐symbol alignment failed:\n"
				f" before RemoveHs: {heavy_symbols}\n"
				f" after  RemoveHs: {new_symbols}"
			)
	# compute edges
	# edge_index = []
	# edge_feats = []
	# for b in mol.GetBonds():
	# 	i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
	# 	f = get_bond_features(b)
	# 	edge_index += [(i,j), (j,i)]
	# 	edge_feats  += [f, f]  # undirected
	# edge_index = np.array(edge_index).T  # shape [2, E]
	# edge_feats = np.stack(edge_feats, axis=0)  # shape [E, D_edge]

	bond_list = mol.GetBonds()
	bond2pos = {}   # maps RDKit bond ID → list of edge‐slot indices
	edge_slots = []
	for b in bond_list:
		bid = b.GetIdx()       # the raw RDKit bond ID
		u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()

		# forward direction
		ei = len(edge_slots)
		edge_slots.append((u, v))
		bond2pos.setdefault(bid, []).append(ei)

		# reverse direction
		ei = len(edge_slots)
		edge_slots.append((v, u))
		bond2pos.setdefault(bid, []).append(ei)
	# edge_index = np.array(
	# 	[(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in bond_list] +
	# 	[(b.GetEndAtomIdx(), b.GetBeginAtomIdx()) for b in bond_list]
	# ).T
	edge_index = np.array(edge_slots).T
	edge_feats = np.vstack(
		[get_bond_features(b) for b in bond_list for _ in (0,1)]
	)

	lap_eigenvectors, lap_eigenvalues = laplacian_encodings(edge_index)
	random_walk_encodings = random_walk_landing_probs(edge_index)
	spd = all_pairs_shortest_paths_unweighted(edge_index)
	raw_spd = make_raw_inputs_from_spd(spd)
	global_feats = np.zeros(1)

	if faces:
		# compute faces
		ri  = mol.GetRingInfo()
		atom_rings = ri.AtomRings()  
		bond_rings = ri.BondRings() 

		face_index = []
		face_feats = []
		bond_counts = Counter(b for ring in bond_rings for b in ring)
		# for ring_idx, (atoms, bonds) in enumerate(zip(atom_rings, bond_rings)):
		# # for f_idx, b_ids in enumerate(bond_rings):
		#     for bond in bonds:
		#         face_index.append((bond, ring_idx))
		#     # face_index.append(bonds)
		#     face_feats += [get_face_features(mol, atoms, bonds, bond_counts)]
		# face_index = np.array(face_index).T
		# face_feats = np.stack(face_feats, axis=0)

		for fidx, (atoms, bonds) in enumerate(zip(atom_rings, bond_rings)):
			face_feats.append(get_face_features(mol, atoms, bonds, bond_counts))
			# face_index.extend((fidx, bond) for bond in bonds)
			for raw_bond_id in bonds:
				# now look up its true edge‐slot(s):
				for slot in bond2pos[raw_bond_id]:
					face_index.append((fidx, slot))

		face_index = np.array(face_index).T
		face_feats = np.stack(face_feats, axis=0)

		if barycentrics:
			bary_eigvals, bary_nodevec, bary_edgevec, bary_ringvec = lap_pe_barycentric(edge_index, face_index)
			bary_noderw, bary_edgerw, bary_ringrw = random_walk_barycentric(edge_index, face_index)
			spd_barycentric = all_pairs_shortest_paths_barycentric(edge_index, face_index)
			raw_spd_barycentric = make_raw_inputs_from_spd(spd_barycentric)

	output = {
		'node_feats' : node_feats,       # (N, D_node)
		'edge_index' : edge_index,       # (2, E)
		'edge_feats' : edge_feats,       # (E, D_edge)
		'global_idx' : global_feats,
		'eigenvector': lap_eigenvectors,
		'eigenvalue' : lap_eigenvalues,
        'random_walk': random_walk_encodings,
		'sp_distance': raw_spd		
	}
    
	if position:
		output["pos_matrix"] = pos
    
	if faces:
		output["face_index"] = face_index
		output["face_feats"] = face_feats
		output["num_faces"]  = len(bond_rings)
	if barycentrics and faces:
		output["bary_eigvalues"] = bary_eigvals
		output["bary_node_vec"] = bary_nodevec
		output["bary_edge_vec"] = bary_edgevec
		output["bary_ring_vec"] = bary_ringvec
		output["bary_node_rw"] = bary_noderw
		output["bary_edge_rw"] = bary_edgerw
		output["bary_ring_rw"] = bary_ringrw
		output["bary_sp_distance"] = raw_spd_barycentric
	return output


def to_pyg_format(g_index, feature_map, labels, split_group, faces=False):
	if split_group is None:
		if faces:
			graph = Data(
					idx        = g_index,
					edge_index = torch.from_numpy(feature_map["edge_index"]).to(torch.uint8),
					x          = torch.from_numpy(feature_map["node_feats"]).to(torch.uint8),
					edge_attr  = torch.from_numpy(feature_map["edge_feats"]).to(torch.uint8),
					pos        = torch.from_numpy(feature_map["pos_matrix"]).half(),
					global_idx = torch.from_numpy(feature_map["global_idx"]).to(torch.uint8),
					face_index = torch.from_numpy(feature_map["face_index"]).to(torch.uint8),
					face_attr  = torch.from_numpy(feature_map["face_feats"]).to(torch.uint8),
					y          = torch.from_numpy(labels).unsqueeze(0).to(torch.uint8)
				)
		else:
			graph = Data(
					idx        = g_index,
					edge_index = torch.from_numpy(feature_map["edge_index"]).to(torch.uint8),
					x          = torch.from_numpy(feature_map["node_feats"]).to(torch.uint8),
					edge_attr  = torch.from_numpy(feature_map["edge_feats"]).to(torch.uint8),
					pos        = torch.from_numpy(feature_map["pos_matrix"]).half(),
					global_idx = torch.from_numpy(feature_map["global_idx"]).to(torch.uint8),
					y          = torch.from_numpy(labels).unsqueeze(0).to(torch.uint8)
				)
	else:
		if faces:
			graph = Data(
				idx         = g_index,
				edge_index  = torch.from_numpy(feature_map["edge_index"]).to(torch.uint8),
				x           = torch.from_numpy(feature_map["node_feats"]).to(torch.uint8),
				edge_attr   = torch.from_numpy(feature_map["edge_feats"]).to(torch.uint8),
				pos         = torch.from_numpy(feature_map["pos_matrix"]).half(),
				global_idx  = torch.from_numpy(feature_map["global_idx"]).to(torch.uint8),
				face_index  = torch.from_numpy(feature_map["face_index"]).to(torch.uint8),
				face_attr   = torch.from_numpy(feature_map["face_feats"]).to(torch.uint8),
				y           = torch.from_numpy(labels).unsqueeze(0).to(torch.uint8),
				split_group = split_group
			)
		else:
			graph = Data(
					idx         = g_index,
					edge_index  = torch.from_numpy(feature_map["edge_index"]).to(torch.uint8),
					x           = torch.from_numpy(feature_map["node_feats"]).to(torch.uint8),
					edge_attr   = torch.from_numpy(feature_map["edge_feats"]).to(torch.uint8),
					pos         = torch.from_numpy(feature_map["pos_matrix"]).half(),
					global_idx  = torch.from_numpy(feature_map["global_idx"]).to(torch.uint8),
					y           = torch.from_numpy(labels).unsqueeze(0).to(torch.uint8),
					split_group = split_group
				)

	return graph


# def to_pyg_plain_dict(g_index, feature_map, labels, split_group, faces=False):
# 	if split_group is None:
# 		if faces:
# 			return {'idx'        : int(g_index),
# 					'edge_index' : torch.from_numpy(feature_map["edge_index"]).to(torch.uint8),
# 					'x'          : torch.from_numpy(feature_map["node_feats"]).to(torch.uint8),
# 					'edge_attr'  : torch.from_numpy(feature_map["edge_feats"]).to(torch.uint8),
# 					'pos'        : torch.from_numpy(feature_map["pos_matrix"]).half(),
# 					'global_idx' : torch.from_numpy(feature_map["global_idx"]).to(torch.uint8),
# 					'face_index' : torch.from_numpy(feature_map["face_index"]).to(torch.uint8),
# 					'face_attr'  : torch.from_numpy(feature_map["face_feats"]).to(torch.uint8),
# 					'y'          : torch.from_numpy(labels).unsqueeze(0).to(torch.uint8)}
# 		else:
# 			return {'idx'        : int(g_index),
# 					'edge_index' : torch.from_numpy(feature_map["edge_index"]).to(torch.uint8),
# 					'x'          : torch.from_numpy(feature_map["node_feats"]).to(torch.uint8),
# 					'edge_attr'  : torch.from_numpy(feature_map["edge_feats"]).to(torch.uint8),
# 					'pos'        : torch.from_numpy(feature_map["pos_matrix"]).half(),
# 					'global_idx' : torch.from_numpy(feature_map["global_idx"]).to(torch.uint8),
# 					'y'          : torch.from_numpy(labels).unsqueeze(0).to(torch.uint8)}
# 	else:
# 		if faces:
# 			return {'idx'        : int(g_index),
# 					'edge_index' : torch.from_numpy(feature_map["edge_index"]).to(torch.uint8),
# 					'x'          : torch.from_numpy(feature_map["node_feats"]).to(torch.uint8),
# 					'edge_attr'  : torch.from_numpy(feature_map["edge_feats"]).to(torch.uint8),
# 					'pos'        : torch.from_numpy(feature_map["pos_matrix"]).half(),
# 					'global_idx' : torch.from_numpy(feature_map["global_idx"]).to(torch.uint8),
# 					'face_index' : torch.from_numpy(feature_map["face_index"]).to(torch.uint8),
# 					'face_attr'  : torch.from_numpy(feature_map["face_feats"]).to(torch.uint8),
# 					'y'          : torch.from_numpy(labels).unsqueeze(0).to(torch.uint8),
# 					'split_group': split_group}
# 		else:
# 			return {'idx'        : int(g_index),
# 					'edge_index' : torch.from_numpy(feature_map["edge_index"]).to(torch.uint8),
# 					'x'          : torch.from_numpy(feature_map["node_feats"]).to(torch.uint8),
# 					'edge_attr'  : torch.from_numpy(feature_map["edge_feats"]).to(torch.uint8),
# 					'pos'        : torch.from_numpy(feature_map["pos_matrix"]).half(),
# 					'global_idx' : torch.from_numpy(feature_map["global_idx"]).to(torch.uint8),
# 					'y'          : torch.from_numpy(labels).unsqueeze(0).to(torch.uint8),
# 					'split_group': split_group}	


def to_pyg_plain_dict(g_index, feature_map, labels, split_group, faces=False, barycentrics=False, position=False):
    """
    Build a dict of numpy arrays (and Python types) only.
    Downstream you can reconstruct torch.Tensors with torch.from_numpy(...) if needed.
    """
    out = {
        'idx': int(g_index),
        'edge_index':  np.asarray(feature_map["edge_index"], dtype=np.int64),
        'x':           np.asarray(feature_map["node_feats"],  dtype=np.uint8),
        'edge_attr':   np.asarray(feature_map["edge_feats"],  dtype=np.uint8),
        'global_idx':  np.asarray(feature_map["global_idx"], dtype=np.uint8),
        'y':           np.asarray(labels, dtype=np.uint8).reshape(1, -1),
        }

    if position:
        out['pos'] = np.asarray(feature_map["pos_matrix"], dtype=np.float16)

    if split_group is not None:
        out['split_group'] = split_group

    if faces:
        out['face_index'] = np.asarray(feature_map["face_index"], dtype=np.int64)
        out['face_attr']  = np.asarray(feature_map["face_feats"], dtype=np.uint8)
        out['faces_num'] = int(feature_map["num_faces"])
    if barycentrics:
        out["bary_eigvals"] = np.asarray(feature_map["bary_eigvalues"], dtype=np.float32).reshape(1, -1)
        out["bary_nodevec"] = np.asarray(feature_map["bary_node_vec"], dtype=np.float32)
        out["bary_edgevec"] = np.asarray(feature_map["bary_edge_vec"], dtype=np.float32)
        out["bary_ringvec"] = np.asarray(feature_map["bary_ring_vec"], dtype=np.float32)
        out["bary_noderw"] = np.asarray(feature_map["bary_node_rw"], dtype=np.float32)
        out["bary_edgerw"] = np.asarray(feature_map["bary_edge_rw"], dtype=np.float32)
        out["bary_ringrw"] = np.asarray(feature_map["bary_ring_rw"], dtype=np.float32)
        out["bary_spd"] = np.asarray(feature_map["bary_sp_distance"], dtype=np.int16)
    else:
        out["eigenvec"] = np.asarray(feature_map["eigenvector"], dtype=np.float32)
        out["eigenval"] = np.asarray(feature_map["eigenvalue"], dtype=np.float32).reshape(1, -1)
        out["random_walk"] = np.asarray(feature_map["random_walk"], dtype=np.float32)
        out["spd"] = np.asarray(feature_map["sp_distance"], dtype=np.int8)

    return out	