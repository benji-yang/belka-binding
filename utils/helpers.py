import torch
import math
from sklearn.metrics import average_precision_score
import numpy as np
import configs.parameters as cf
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.loader.dataloader import Collater
from torch.utils.data import Sampler
from typing import Iterable, List, Optional
from torch.utils.data import IterableDataset, get_worker_info
import random
import zipfile


class dotdict(dict):
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError:
			raise AttributeError(name)

def tensor_dim_slice(tensor, dim, dim_slice):
	return tensor[(dim if dim >= 0 else dim + tensor.dim()) * (slice(None),) + (dim_slice,)]

# @torch.jit.script
def packshape(shape, dim: int = -1, mask: int = 0b00000001, dtype=torch.uint8, pack=True):
	dim = dim if dim >= 0 else dim + len(shape)
	bits, nibble = (
		8 if dtype is torch.uint8 else 16 if dtype is torch.int16 else 32 if dtype is torch.int32 else 64 if dtype is torch.int64 else 0), (
		1 if mask == 0b00000001 else 2 if mask == 0b00000011 else 4 if mask == 0b00001111 else 8 if mask == 0b11111111 else 0)
	# bits = torch.iinfo(dtype).bits # does not JIT compile
	assert nibble <= bits and bits % nibble == 0
	nibbles = bits // nibble
	shape = (shape[:dim] + (int(math.ceil(shape[dim] / nibbles)),) + shape[1 + dim:]) if pack else (
				shape[:dim] + (shape[dim] * nibbles,) + shape[1 + dim:])
	return shape, nibbles, nibble

# @torch.jit.script
def F_unpackbits(tensor, dim: int = -1, mask: int = 0b00000001, shape=None, out=None, dtype=torch.uint8):
	dim = dim if dim >= 0 else dim + tensor.dim()
	shape_, nibbles, nibble = packshape(tensor.shape, dim=dim, mask=mask, dtype=tensor.dtype, pack=False)
	shape = shape if shape is not None else shape_
	out = out if out is not None else torch.empty(shape, device=tensor.device, dtype=dtype)
	assert out.shape == shape

	if shape[dim] % nibbles == 0:
		shift = torch.arange((nibbles - 1) * nibble, -1, -nibble, dtype=torch.uint8, device=tensor.device)
		shift = shift.view(nibbles, *((1,) * (tensor.dim() - dim - 1)))
		return torch.bitwise_and((tensor.unsqueeze(1 + dim) >> shift).view_as(out), mask, out=out)

	else:
		for i in range(nibbles):
			shift = nibble * i
			sliced_output = tensor_dim_slice(out, dim, slice(i, None, nibbles))
			sliced_input = tensor.narrow(dim, 0, sliced_output.shape[dim])
			torch.bitwise_and(sliced_input >> shift, mask, out=sliced_output)
	return out

def convert_p_to_idx(proteins_list):
	all_unique_proteins = proteins_list
	protein_vocabulary = sorted(list(all_unique_proteins))
	protein_to_idx = {name: i for i, name in enumerate(protein_vocabulary)}
	return protein_to_idx


def calculate_score(predicted_labels, true_labels, split_group_array):
	"""
	Args:
		predicted_labels : shape [N x 3]
		true_labels : shape [N x 3]
		split_group_array : shape [N]
	"""

	share_mask = split_group_array == "share"
	non_share_mask = split_group_array == "non_share"
	kin0_mask = split_group_array == "kin0"
	masks = [share_mask, non_share_mask, kin0_mask]

	scores = []
	sentinel = cf.SENTINEL

	for mask in masks:
		y_true = true_labels[mask, :]
		y_predicted = predicted_labels[mask, :]
		for j in range(y_predicted.shape[1]):
			non_nan = (y_true[:, j] != sentinel)
			if non_nan.sum() > 0:
				yj = y_true[non_nan, j]
				scores_j = y_predicted[non_nan, j]
				ap_j = average_precision_score(yj, scores_j)
				scores.append(ap_j)
			else:
				print(f"Class {j} has no labels; skipping.")
		# score = average_precision_score(true_labels[mask, :], predicted_labels[mask, :])
		# scores.append(score)
	return np.mean(scores), scores


def convert_numpy_to_torch_inplace(graphs):
    """
    Walks through a list of dicts and replaces any np.ndarray values
    with torch.from_numpy(...) tensors, in place. Returns the same list.
    """
    for g in graphs:
        for k, v in list(g.items()):
            if isinstance(v, np.ndarray):
                g[k] = torch.from_numpy(v)
    return graphs

def build_dense_batch(edge_index, batch, num_nodes):
	A = to_dense_adj(edge_index, batch)                        # [B, Nmax, Nmax]

	# 2) build a [B, Nmax] mask of real vs. padded nodes
	# we can use to_dense_batch on a dummy feature to get the mask
	dummy = torch.ones(num_nodes, 1, device=edge_index.device)
	_, mask = to_dense_batch(dummy, batch)                          # mask: BoolTensor [B, Nmax]
	mask = mask.squeeze(-1)

	return A, mask


def save_as_zip(zip_path, torch_dict):
	# 2) (Optional) Ensure the parent directory exists
	zip_path.parent.mkdir(parents=True, exist_ok=True)

	# 3) Write straight into that ZIP
	with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
		# store the file inside a folder called "checkpoints/" within the zip
		with zf.open("training_features.pt", "w", force_zip64=True) as dest:
			torch.save(torch_dict, dest, _use_new_zipfile_serialization=False)


def load_from_zip(zip_path):
	with zipfile.ZipFile(zip_path, "r") as zf:
    # match whatever arcname you used when saving:
		with zf.open("training_features.pt", "r") as src:
			data = torch.load(src, map_location='cpu', weights_only=True)
	return data


def pad_to_max_nodes(mats):
	B = len(mats)
	sizes = [m.size(0) for m in mats]
	N_max = max(sizes)

	device = mats[0].device
	dtype  = mats[0].dtype
	batch_mat = torch.full((B, N_max, N_max), -1, device=device, dtype=dtype)

	# 3) Fill each slice
	for i, (m, n) in enumerate(zip(mats, sizes)):
		batch_mat[i, :n, :n] = m
	return batch_mat


def add_offset_to_face_index(face_mats, no_faces, no_edges):
	# compute cumulative offsets
	face_offsets = torch.tensor([0] + no_faces[:-1]).cumsum(0)
	edge_offsets = torch.tensor([0] + no_edges[:-1]).cumsum(0)

	batched_fei = []
	for fei, fo, eo in zip(face_mats, face_offsets, edge_offsets):
		# shift row by #faces, col by #nodes
		fei_copy = fei.clone()
		fei_copy[0] += fo
		fei_copy[1] += eo

		batched_fei.append(fei_copy)

	face_index = torch.cat(batched_fei, dim=1)      # [2, total_incidence]
	return face_index


class CustomCollater(Collater):
	def __init__(self,
		dataset,
		bary_config,
		follow_batch=None,
		exclude_keys=None):
		super().__init__(dataset, follow_batch, exclude_keys)
		self.bary_config = bary_config

	def __call__(self, batch):
		# 1) Extract your custom matrices
		if self.bary_config:
			spd_matrices = [data.bary_spd.clone() for data in batch]
		else:
			spd_matrices = [data.spd.clone() for data in batch]
		face_matrices = [data.face_index.clone() for data in batch]
		no_faces = [data.faces_num for data in batch]
		no_edges = [data.num_edges for data in batch]

		# 2) Remove them so the base Collater won't try to torch.cat them
		if self.bary_config:
			for data in batch:
				del data.face_index
				del data.eigenvec
				del data.eigenval
				del data.random_walk
				del data.bary_spd
				del data.spd
		else:
			for data in batch:
				del data.bary_eigvals
				del data.bary_nodevec
				del data.bary_edgevec
				del data.bary_ringvec
				del data.bary_noderw
				del data.bary_edgerw
				del data.bary_ringrw
				del data.spd
				del data.face_index
				del data.bary_spd

		# 3) Do the normal batching for everything else
		batched = super().__call__(batch)

		for data, spd, f_mats in zip(batch, spd_matrices, face_matrices):
			if self.bary_config:
				data.bary_spd = spd
			else:
				data.spd = spd
			data.face_index = f_mats

		# 4) Attach your list-of-[n_i,n_i] matrices back onto the batch
		if self.bary_config:
			batched.bary_spd = pad_to_max_nodes(spd_matrices)
		else:
			batched.spd = pad_to_max_nodes(spd_matrices)		
		batched.face_index = add_offset_to_face_index(face_matrices, no_faces, no_edges)
		return batched


class CustomDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        bary_config: bool = False,
        batch_size: int = 1,                  # ignored when use_token_bucket=True
        shuffle: bool = False,                # ignored when use_token_bucket=True
        follow_batch=None,
        exclude_keys=None,
        use_token_bucket: bool = True,        # ← turn on
        max_nodes_per_batch: int = 8_000,     # tune for your GPU
        max_edges_per_batch: int | None = None,
        max_graphs_per_batch: int | None = None,
        sortish_window: int = 1024,
        drop_last: bool = False,
        seed: int = 0,
        **kwargs,
    ):
        kwargs.pop('collate_fn', None)
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        collater = CustomCollater(dataset, bary_config, follow_batch, exclude_keys)

        if use_token_bucket:
            batch_sampler = TokenBucketBatchSampler(
                dataset=dataset,
                max_nodes_per_batch=max_nodes_per_batch,
                max_edges_per_batch=max_edges_per_batch,
                max_graphs_per_batch=max_graphs_per_batch,
                shuffle=shuffle,
                sortish_window=sortish_window,
                drop_last=drop_last,
                seed=seed,
            )
            super().__init__(dataset, batch_sampler=batch_sampler, collate_fn=collater, **kwargs)
            self._tb_sampler = batch_sampler
        else:
            super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                             drop_last=drop_last, collate_fn=collater, **kwargs)

    def set_epoch(self, epoch: int):
        if hasattr(self, "_tb_sampler"):
            self._tb_sampler.set_epoch(epoch)


def _graph_size(dataset, i):
    d = dataset[i]
    # Prefer num_nodes/num_edges if available; otherwise infer
    n = int(getattr(d, "num_nodes", getattr(d, "x").size(0)))
    e = int(getattr(d, "num_edges", getattr(d, "edge_index").size(1)))
    return n, e

class TokenBucketBatchSampler(Sampler[List[int]]):
    """
    Greedily packs dataset indices into variable-size batches such that:
        sum(nodes)  <= max_nodes_per_batch,
        sum(edges)  <= max_edges_per_batch (if given),
        len(batch)  <= max_graphs_per_batch (if given).
    It reshuffles every epoch and optionally 'sortish' within windows to improve packing.

    Call .set_epoch(epoch) each epoch to reshuffle.
    """

    def __init__(
        self,
        dataset,
        max_nodes_per_batch: int,
        max_edges_per_batch: Optional[int] = None,
        max_graphs_per_batch: Optional[int] = None,
        shuffle: bool = True,
        sortish_window: int = 1024,        # set 0/None to disable local sorting
        drop_last: bool = False,
        seed: int = 0,
        allow_oversize_single: bool = True # if one graph exceeds budget, yield it alone
    ):
        self.dataset = dataset
        self.maxN = int(max_nodes_per_batch)
        self.maxE = None if max_edges_per_batch is None else int(max_edges_per_batch)
        self.maxG = None if max_graphs_per_batch is None else int(max_graphs_per_batch)
        self.shuffle = bool(shuffle)
        self.sortish_window = int(sortish_window) if sortish_window else 0
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self.allow_oversize_single = bool(allow_oversize_single)

        # Precompute sizes once (cheap compared to training)
        Ns, Es = [], []
        for i in range(len(dataset)):
            n, e = _graph_size(dataset, i)
            Ns.append(n); Es.append(e)
        self.Ns = torch.tensor(Ns, dtype=torch.long)
        self.Es = torch.tensor(Es, dtype=torch.long)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self) -> int:
        # Rough estimate (not exact when edges/graphs limits are active)
        total_nodes = int(self.Ns.sum().item())
        est = math.ceil(total_nodes / max(self.maxN, 1))
        return max(est, 1)

    def _ordered_indices(self) -> List[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx = torch.arange(len(self.Ns))

        if not self.shuffle:
            return idx.tolist()

        # Global shuffle
        idx = idx[torch.randperm(idx.numel(), generator=g)]

        # Sortish within windows to pack large graphs first
        if self.sortish_window > 0:
            chunks = idx.split(self.sortish_window)
            sorted_chunks = []
            for c in chunks:
                # sort by node count (desc) to pack big ones early
                n = self.Ns[c]
                order = torch.argsort(n, descending=True)
                sorted_chunks.append(c[order])
            idx = torch.cat(sorted_chunks, dim=0)

        return idx.tolist()

    def __iter__(self) -> Iterable[List[int]]:
        indices = self._ordered_indices()

        batch, n_acc, e_acc = [], 0, 0
        for i in indices:
            n_i = int(self.Ns[i])
            e_i = int(self.Es[i])

            # Would this sample overflow any cap?
            would_overflow = (
                (n_acc + n_i > self.maxN) or
                (self.maxE is not None and e_acc + e_i > self.maxE) or
                (self.maxG is not None and len(batch) + 1 > self.maxG)
            )

            if would_overflow:
                if batch:
                    yield batch
                    batch, n_acc, e_acc = [], 0, 0
                # Handle single oversized sample
                if (n_i > self.maxN) or (self.maxE is not None and e_i > self.maxE):
                    if self.allow_oversize_single:
                        yield [i]
                        continue
                    else:
                        # skip or raise; here we yield alone anyway
                        yield [i]
                        continue

            # Add to current batch
            batch.append(i)
            n_acc += n_i
            e_acc += e_i

        # leftover
        if batch and not self.drop_last:
            yield batch

def count_parameters(module, trainable_only=True):
    """Count the number of parameters in a module."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())

# import os, io
# from typing import List, Dict, Any
# import torch
# from safetensors.torch import save_file, load_file
# import zstandard as zstd

# _INT_TAG = ".__int"  # suffix to mark wrapped Python ints

# def _approx_nbytes(t: torch.Tensor) -> int:
#     return int(t.numel()) * max(t.element_size(), 1)

# def save_graphs_safetensors_zstd(
#     out_dir: str,
#     graphs: List[Dict[str, Any]],
#     target_shard_bytes: int = 1_200_000_000,  # ~1.2 GiB (pre-compress) per shard
#     zstd_level: int = 12
# ) -> None:
#     """
#     graphs: list of dict[str, (torch.Tensor | int)]
#     - Tensors are written as-is (contiguous).
#     - Python ints are stored as 0-D int64 tensors with key '<key>.__int'.
#     Other types are ignored.
#     """
#     os.makedirs(out_dir, exist_ok=True)
#     shard_idx = 0
#     current: Dict[str, torch.Tensor] = {}
#     approx_bytes = 0

#     def flush():
#         nonlocal shard_idx, current, approx_bytes
#         if not current:
#             return
#         buf = io.BytesIO()
#         save_file(current, buf)  # write safetensors into memory
#         raw = buf.getvalue()
#         comp = zstd.ZstdCompressor(level=zstd_level).compress(raw)
#         path = os.path.join(out_dir, f"dataset-shard-{shard_idx:05d}.safetensors.zst")
#         with open(path, "wb") as f:
#             f.write(comp)
#         shard_idx += 1
#         current.clear()
#         approx_bytes = 0

#     for i, g in enumerate(graphs):
#         for k, v in g.items():
#             namespaced = None
#             tensor = None

#             if isinstance(v, torch.Tensor):
#                 namespaced = f"{i}/{k}"
#                 tensor = v.contiguous()

#             elif isinstance(v, int):
#                 # store as 0-D int64 tensor under a tagged key so we can unwrap
#                 namespaced = f"{i}/{k}{_INT_TAG}"
#                 tensor = torch.tensor(v, dtype=torch.int64)

#             else:
#                 # skip unsupported types; add a serializer if needed later
#                 continue

#             current[namespaced] = tensor
#             approx_bytes += _approx_nbytes(tensor)

#         if approx_bytes >= target_shard_bytes:
#             flush()

#     flush()  # final shard


# def load_graphs_safetensors_zstd(in_dir: str) -> List[Dict[str, Any]]:
#     """
#     Reconstruct list[dict[str, Any]] with tensors and ints restored.
#     """
#     files = sorted(p for p in os.listdir(in_dir) if p.endswith(".safetensors.zst"))
#     if not files:
#         return []

#     # Pass 1: determine number of graphs
#     max_gid = -1
#     for fname in files:
#         with open(os.path.join(in_dir, fname), "rb") as f:
#             dec = zstd.ZstdDecompressor().decompress(f.read())
#         md = load_file(io.BytesIO(dec))
#         for k in md.keys():
#             gid = int(k.split("/", 1)[0])
#             if gid > max_gid:
#                 max_gid = gid
#     N = max_gid + 1
#     out: List[Dict[str, Any]] = [dict() for _ in range(N)]

#     # Pass 2: assemble and unwrap ints
#     for fname in files:
#         with open(os.path.join(in_dir, fname), "rb") as f:
#             dec = zstd.ZstdDecompressor().decompress(f.read())
#         blob = load_file(io.BytesIO(dec))  # dict[str, Tensor]

#         for full_key, t in blob.items():
#             gid_str, subk = full_key.split("/", 1)
#             gid = int(gid_str)

#             if subk.endswith(_INT_TAG):
#                 base_key = subk[: -len(_INT_TAG)]
#                 out[gid][base_key] = int(t.item())
#             else:
#                 out[gid][subk] = t  # keep dtype exactly as saved

#     return out
# class PackDataset(IterableDataset):
# 	"""
# 	Streams through dataset[i] in a sharded, per-worker range, and
# 	yields *lists* of graphs (i.e. raw packs) subject to the
# 	(max_nodes, max_edges, max_graphs) caps.
# 	"""
# 	def __init__(self,
# 		dataset,
# 		max_nodes: int = 60,
# 		max_edges: int = 120,
# 		max_graphs: int = 8,
# 		shuffle: bool = False):
# 		self.dataset = dataset
# 		self.max_nodes = max_nodes
# 		self.max_edges = max_edges
# 		self.max_graphs = max_graphs
# 		self.shuffle = shuffle

# 	def __iter__(self):
# 		# --- 1) Determine this worker's exclusive slice of dataset ---
# 		worker_info = get_worker_info()
# 		if worker_info is None:
# 			# single‐worker: use the whole dataset
# 			iter_start, iter_end = 0, len(self.dataset)
# 		else:
# 			# N workers: split indices [0..len) into (almost) equal chunks
# 			per_worker = int(math.ceil(len(self.dataset) / float(worker_info.num_workers)))
# 			worker_id  = worker_info.id
# 			iter_start = worker_id * per_worker
# 			iter_end   = min(iter_start + per_worker, len(self.dataset))

# 		# Prepare indices, optionally shuffled
# 		if self.shuffle:
# 			indices = list(range(iter_start, iter_end))
# 			random.shuffle(indices)
# 		else:
# 			indices = range(iter_start, iter_end)

# 		# --- 2) Stream & pack ---
# 		pack       = []
# 		nodes_acc  = 0
# 		edges_acc  = 0

# 		for idx in range(iter_start, iter_end):
# 			data = self.dataset[idx]
# 			n, e = data.num_nodes, data.num_edges

# 			# if adding this graph would overflow any cap, yield current pack
# 			if pack and (
# 				nodes_acc + n > self.max_nodes
# 				or edges_acc + e > self.max_edges
# 				or len(pack) >= self.max_graphs
# 				):
# 				yield pack
# 				pack      = []
# 				nodes_acc = 0
# 				edges_acc = 0

# 			# add graph to current pack
# 			pack.append(data)
# 			nodes_acc += n
# 			edges_acc += e

# 		# yield leftover
# 		if pack:
# 			yield pack


# class HierarchicalDataLoader(torch.utils.data.DataLoader):
# 	"""
# 	A DataLoader that internally packs graphs into fixed-size packs,
# 	sharded across workers, then forms micro-batches of N packs,
# 	collating each pack via a CustomCollater.
# 	"""
# 	def __init__(
# 		self,
# 		dataset,
# 		pack_size=(60, 120, 8),         # (max_nodes, max_edges, max_graphs)
# 		packs_per_microbatch: int = 8,
# 		shuffle: bool = False,
# 		follow_batch=None,
# 		exclude_keys=None,
# 		**kwargs,
# 		):
# 		# drop external collate_fn if given
# 		kwargs.pop('collate_fn', None)
# 		kwargs.pop('shuffle', None)

# 		# save for Lightning compatibility
# 		self.follow_batch = follow_batch
# 		self.exclude_keys = exclude_keys
# 		self.packs_per_microbatch = packs_per_microbatch

# 		# 1) Build the sharded pack-streaming dataset
# 		pack_ds = PackDataset(
# 			dataset,
# 			max_nodes=pack_size[0],
# 			max_edges=pack_size[1],
# 			max_graphs=pack_size[2],
# 			shuffle=shuffle
# 			)

# 		# 2) Instantiate your per-pack collater
# 		self._collater = CustomCollater(dataset, follow_batch, exclude_keys)

# 		# 3) Initialize DataLoader: batch_size = # packs per microbatch
# 		super().__init__(
# 			pack_ds,
# 			batch_size=packs_per_microbatch,
# 			shuffle=False,
# 			collate_fn=self._micro_collate,
# 			**kwargs,
# 			)

# 	def _micro_collate(self, list_of_packs):
# 		# list_of_packs: List[List[Data]]
# 		# Apply your CustomCollater to each pack
# 		microbatch = [self._collater(pack) for pack in list_of_packs]
# 		return microbatch

# def profile_layer_memory(model: torch.nn.Module,
#                          example_input: torch.Tensor,
#                          device: torch.device = None):
#     """
#     Runs a single forward pass through `model` with `example_input`
#     and prints a module-by-module breakdown of GPU memory usage.
#     """

#     # 1. Setup device
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     example_input = example_input.to(device)

#     # 2. Prepare a dict to record memory after each module
#     mem_stats = {}

#     # 3. Hook factory that snapshots allocated memory
#     def make_hook(name):
#         def hook(module, inp, out):
#             # ensure all CUDA ops are done
#             torch.cuda.synchronize(device)
#             mem_stats[name] = torch.cuda.memory_allocated(device)
#         return hook

#     # 4. Register hooks on all modules
#     hooks = []
#     for name, module in model.named_modules():
#         hooks.append(module.register_forward_hook(make_hook(name)))

#     # 5. Reset and clear
#     torch.cuda.reset_peak_memory_stats(device)
#     torch.cuda.empty_cache()

#     # 6. Forward pass (no grads needed here)
#     model.eval()
#     with torch.no_grad():
#         _ = model(example_input)

#     # 7. Remove hooks
#     for h in hooks:
#         h.remove()

#     # 8. Print a sorted breakdown
#     prev = 0
#     print(f"{'Module':<40s} {'Δ Memory (MB)':>12s} {'Total (MB)':>12s}")
#     print("-" * 68)
#     for name, usage in mem_stats.items():
#         delta_mb = (usage - prev) / (1024**2)
#         total_mb = usage / (1024**2)
#         print(f"{name:<40s} {delta_mb:12.1f} {total_mb:12.1f}")
#         prev = usage