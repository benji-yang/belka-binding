from multiprocessing import Pool
from multiprocessing import get_context
import polaris as po
import numpy as np
import configs.parameters as cf
import torch
import math
from utils.feature_engineering import to_pyg_format, smile_to_graph, to_pyg_plain_dict
import time
from tqdm import tqdm
import os

def _worker_wrapper(args_tuple):
    """
    Simplified wrapper. It just unpacks the arguments for _make_graph.
    No local index is needed.
    """
    # Unpack the arguments needed by the graph creation function
    index, smile, label, face, barycentric, position, split_group = args_tuple
    
    # Call the function that does the real work
    return _make_graph(index, smile, label, face, barycentric, position, split_group)

def _make_graph(index, smile, label, face, barycentric, position, split_group):
    fmap = smile_to_graph(smile, face, barycentric, position)
    return to_pyg_plain_dict(index, fmap, label, split_group, face, barycentric, position)


def load_features(task_type="train", m=None, sample_idx=None, cfg = cf.get_config()):
    pbs_idx = os.environ.get("PBS_ARRAY_INDEX")
    slurm_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    if pbs_idx is not None:
        chunk_id   = int(pbs_idx) - 1
        num_chunks = cfg.chunk_size
    elif slurm_idx is not None:
        chunk_id = int(slurm_idx) - 1
        num_chunks = cfg.chunk_size
    else:
        chunk_id   = None
        num_chunks = None
    
    dataset = po.dataset.DatasetV2.from_json(cfg.load_path)
    smiles_arrays = dataset[:, "molecule_smiles"]
    
    if task_type == 'test':
        mask = np.isin(dataset[:, "split"], ['test', ''])
    else:
        mask = (dataset[:, "split"] == task_type)
    smiles_arrays = smiles_arrays[mask]
    labels = np.column_stack((dataset[:, "binds_BRD4"], dataset[:, "binds_HSA"], dataset[:, "binds_sEH"]))
    labels = labels[mask, :]
    split_group = dataset[:, 'split_group']
    split_group = split_group[mask]
    indexes = np.array(list(range(len(labels))))
    
    if chunk_id is not None and num_chunks is not None and num_chunks > 1:
        total       = len(indexes)
        chunk_size  = total // num_chunks
        start       = chunk_id * chunk_size
        end         = start + chunk_size if chunk_id < num_chunks - 1 else total

        smiles_arrays = smiles_arrays[start:end]
        labels        = labels[start:end]
        indexes       = indexes[start:end]
        split_group   = split_group[start:end]
    
    ones_per_col = (labels == 1).sum(axis=0)
    zeros_per_col = (labels == 0).sum(axis=0)
    
    default = np.ones_like(zeros_per_col, dtype=float)
    
    # do the elementwise divide, but only where ones_per_col != 0
    pos_weight = np.divide(
        zeros_per_col,
        ones_per_col,
        out=default,               # fills result with this array first
        where=ones_per_col != 0    # only actually divides where denom!=0
    )
    pos_weight = torch.from_numpy(pos_weight).float()

    # -------- apply provided sample_idx if given --------
    if sample_idx is not None:
        sample_idx = np.asarray(sample_idx, dtype=int)
        if sample_idx.ndim != 1:
            raise ValueError("sample_idx must be a 1D integer array (indices relative to the masked/chunked arrays).")
        N = len(labels)
        if sample_idx.size == 0:
            raise ValueError("sample_idx is empty.")
        if sample_idx.min() < 0 or sample_idx.max() >= N:
            raise ValueError(f"sample_idx contains out-of-bounds indices (valid range: 0..{N-1}).")
        # If m was provided, ensure consistency (optional)
        if m is not None and len(sample_idx) != m:
            raise ValueError(f"len(sample_idx) ({len(sample_idx)}) != m ({m}). "
                             "Either pass matching values or set m=None.")
        # Apply indexing
        smiles_arrays = smiles_arrays[sample_idx]
        indexes = indexes[sample_idx]
        labels = labels[sample_idx]
        split_group = split_group[sample_idx]
        ones_per_col = (labels == 1).sum(axis=0)
        zeros_per_col = (labels == 0).sum(axis=0)

        # sanity‐check:
        col_counts = np.nansum(labels, axis=0)
        print("Ones per column in your sample:", col_counts)
        
        default = np.ones_like(zeros_per_col, dtype=float)
        
        # do the elementwise divide, but only where ones_per_col != 0
        pos_weight = np.divide(
            zeros_per_col,
            ones_per_col,
            out=default,               # fills result with this array first
            where=ones_per_col != 0    # only actually divides where denom!=0
        )
        # print(zeros_per_col, ones_per_col)
        pos_weight = torch.from_numpy(pos_weight).float()

    
    elif m:
        N = len(labels)

        # 1) for each column, grab min_ones random positive‐indices
        pos_idx = [np.where(labels[:, j] == 1)[0] for j in range(3)]
        if any(len(idx) < cfg.min_ones for idx in pos_idx):
            raise ValueError("Not enough 1s in one of the columns to meet the 100‐ones requirement.")

        selected = set()
        for idx in pos_idx:
            chosen = np.random.choice(idx, size=cfg.min_ones, replace=False)
            selected.update(chosen)

        # 2) fill out to num_samples with totally random draws from the *remaining* rows
        remaining = np.setdiff1d(np.arange(N), list(selected), assume_unique=True)
        n_remaining = m - len(selected)
        if n_remaining < 0:
            raise ValueError(f"Quota selection ({len(selected)}) exceeds total sample size ({m}).")
        if len(remaining) < n_remaining:
            raise ValueError("Not enough leftover rows to fill the sample.")

        extra = np.random.choice(remaining, size=n_remaining, replace=False)
        sample_idx = np.array(list(selected) + list(extra))
        np.random.shuffle(sample_idx)
        
        smiles_arrays = smiles_arrays[sample_idx]
        indexes = indexes[sample_idx]
        labels = labels[sample_idx]
        split_group = split_group[sample_idx]
        ones_per_col = (labels == 1).sum(axis=0)
        zeros_per_col = (labels == 0).sum(axis=0)

        # sanity‐check:
        col_counts = np.nansum(labels, axis=0)
        print("Ones per column in your sample:", col_counts)
        
        default = np.ones_like(zeros_per_col, dtype=float)
        
        # do the elementwise divide, but only where ones_per_col != 0
        pos_weight = np.divide(
            zeros_per_col,
            ones_per_col,
            out=default,               # fills result with this array first
            where=ones_per_col != 0    # only actually divides where denom!=0
        )
        # print(zeros_per_col, ones_per_col)
        pos_weight = torch.from_numpy(pos_weight).float()
    
    labels = np.where(np.isnan(labels), cf.SENTINEL, labels)
    faces = [cfg.face for i in range(len(labels))]
    barycentrics = [cfg.barycentric for i in range(len(labels))]
    positions = [cfg.position for i in range(len(labels))]
    if task_type != 'test':
        split_group[:] = None
    args = list(zip(indexes, smiles_arrays, labels, faces, barycentrics, positions, split_group))  
    train_graphs = []

    start = time.perf_counter()
    with Pool() as pool:
        # Use tqdm to get a progress bar.
        with tqdm(total=len(args), desc="Generating Graphs (Unordered)") as pbar:
            #
            # --- STEP 3: Use imap_unordered and simply append results ---
            #
            for graph_dict in pool.imap_unordered(_worker_wrapper, args):
                
                # If graph generation was successful, add it to our list
                if graph_dict is not None:
                    train_graphs.append(graph_dict)
    
                pbar.update()
    end = time.perf_counter()
    print(f"load_features took {end - start:.3f} seconds")
    return train_graphs, pos_weight