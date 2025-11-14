# import cudf
# import pyarrow.parquet as pq
# import pyarrow as pa
# import torch
# from torch_geometric.data import Batch
# from utils.feature_engineering import smile_to_graph, to_pyg_format
# import duckdb
# from pathlib import Path
from rdkit import Chem
from multiprocessing import Pool, cpu_count
import json
import configs.parameters as cf
import numpy as np
from utils.feature_engineering import all_pairs_shortest_paths_unweighted, all_pairs_shortest_paths_barycentric

# def streaming_pyg_generator(file_path, selected_columns, model_batch_size, p_to_idx_map, num_total_proteins, seed: int = 42, shuffle_row_groups: bool = True, shuffle_within_group: bool = True):
#     parquet_file = pq.ParquetFile(file_path)
#     num_row_groups = parquet_file.num_row_groups

#     row_group_order = list(range(num_row_groups))
#     if shuffle_row_groups:
#         rng = np.random.default_rng(seed)
#         rng.shuffle(row_group_order)      # in-place

#     # This list will hold Data objects until a full model batch is ready
#     pyg_data_objects_for_current_batch = []

#     for i, rg_idx in enumerate(row_group_order):
#         print(f"Streaming: Reading Parquet row group "
#               f"{i+1}/{num_row_groups} (physical RG {rg_idx})")
#         gdf_chunk = cudf.read_parquet(file_path,
#                                       columns=selected_columns,
#                                       row_groups=[rg_idx])

#         if shuffle_within_group:
#             gdf_chunk = gdf_chunk.sample(frac=1, random_state=seed) \
#                                  .reset_index(drop=True)
#         # Bring the string data and corresponding targets to CPU (as pandas Series)
#         # for iteration and use by your Python string processing function.
#         graph_id = gdf_chunk['id'].to_pandas()
#         smile_strings_pd = gdf_chunk['molecule_smiles'].to_pandas()
#         protein_names_pd = gdf_chunk['protein_name'].to_pandas()
#         binds_values_pd = gdf_chunk['binds'].to_pandas()

#         print(f"  Processing {len(smile_strings_pd)} string entries from current Parquet chunk...")
#         for j, data_str in enumerate(smile_strings_pd):
#             if data_str is None: # Handle potential missing data
#                 print(f"  Skipping None string at index {j} in chunk.")
#                 continue

#             # 1. Convert the string to a PyG Data object using your function
#             try:
#                 # This function call happens on the CPU with a Python string
#                 pyg_data = to_pyg_format(*smile_to_graph(smiles=data_str), g_index=graph_id)
#                 # Ensure pyg_data's tensors are on GPU if not already handled inside string_to_data_func
#                 pyg_data = pyg_data.to('cuda') # If needed
#             except Exception as e:
#                 print(f"  Error processing string (idx {j}): '{data_str[:50]}...' with error: {e}")
#                 continue # Skip this data point

#             # --- Apply your one-hot encoding and masking logic for the current row ---
#             current_protein_name = protein_names_pd.iloc[j]
#             current_binds_value = binds_values_pd.iloc[j] # This is now a float (0.0 or 1.0)

#             # 1. Initialize one-hot vector (all zeros)
#             one_hot_protein_tensor = torch.zeros(num_total_proteins, dtype=torch.float)

#             # 2. Set the appropriate index to 1 if protein is known
#             if current_protein_name in p_to_idx_map:
#                 protein_idx = p_to_idx_map[current_protein_name]
#                 one_hot_protein_tensor[protein_idx] = 1.0
#             else:
#                 # Optional: Handle unknown proteins encountered during streaming
#                 # (should ideally not happen if vocabulary is complete from preprocessing)
#                 print(f"  Warning: Protein '{current_protein_name}' not in pre-defined vocabulary. Target will be all zeros.")

#             # 3. Mask by 'binds' value (element-wise multiplication)
#             #    If current_binds_value is 0.0, vector becomes all zeros.
#             #    If current_binds_value is 1.0, vector remains the one-hot encoding.
#             final_train_label_tensor = one_hot_protein_tensor * current_binds_value

#             # 4. Attach the processed label tensor to the PyG Data object
#             #    It should be on the GPU for efficient batching and training.
#             pyg_data.y = final_train_label_tensor.unsqueeze(0).to('cuda')
#             # --- End of target processing for the current row ---

#             pyg_data_objects_for_current_batch.append(pyg_data)

#             # 3. If a full model batch is ready, collate and yield
#             if len(pyg_data_objects_for_current_batch) == model_batch_size:
#                 model_ready_batch = Batch.from_data_list(pyg_data_objects_for_current_batch)
#                 yield model_ready_batch
#                 pyg_data_objects_for_current_batch = [] # Reset list

#         # Cleanup VRAM for the processed cuDF chunk
#         del gdf_chunk
#         cudf.utils.gc.collect()
#         torch.cuda.empty_cache()

#     # Yield any remaining Data objects that didn't form a full batch
#     if pyg_data_objects_for_current_batch:
#         model_ready_batch = Batch.from_data_list(pyg_data_objects_for_current_batch)
#         yield model_ready_batch


# def build_experiment_parquet(
#         src_path: str | Path,
#         out_path: str | Path,
#         n_per_class: int = 30_000,
#         seed: int = 42,
#         row_group_size: int = 50_000,   # good size for cuDF streaming
# ) -> Path:
#     """
#     Creates a new Parquet file `out_path` that contains
#     `n_per_class` random rows for each class (binds==0 and binds==1).

#     The schema, dtypes and column order are preserved so it is
#     100 % compatible with `streaming_pyg_generator`.
#     """
#     con = duckdb.connect()

#     query = f"""
#         WITH sampled AS (
#             (SELECT *
#                FROM parquet_scan('{src_path}')
#               WHERE binds = 0
#               ORDER BY random()
#               LIMIT {n_per_class}
#             )
#             UNION ALL
#             (SELECT *
#                FROM parquet_scan('{src_path}')
#               WHERE binds = 1
#               ORDER BY random()
#               LIMIT {n_per_class}
#             )
#         )
#         SELECT * FROM sampled
#         ORDER BY random()          -- same seed, so reproducible
#     """

#     # Pull the result as a PyArrow table (fast & zero-copy from DuckDB)
#     arrow_tbl: pa.Table = con.execute(query).arrow()
#     con.close()

#     # OPTIONAL: shuffle again with NumPy for extra randomness
#     rng = np.random.default_rng(seed)
#     arrow_tbl = arrow_tbl.take(rng.permutation(arrow_tbl.num_rows))

#     # Write as a single Parquet file with reasonably sized row groups.
#     # cuDF can then read one group at a time exactly like in your current code.
#     pq.write_table(
#         arrow_tbl,
#         out_path,
#         row_group_size=row_group_size,
#         use_dictionary=True,
#         compression="zstd",
#     )
#     return Path(out_path)


def get_group_from_map(symbol, g_map):
    """
    A fast, lightweight function to get the group from a pre-computed map.
    """
    return g_map.get(symbol, "Symbol not found")


def _unique_from_smiles_chunk(smiles_chunk):
    """
    Worker function: processes a sub-list of SMILES and returns
    a dict of sets for all atom/bond features in that chunk.
    """

    # Prepare one set for each feature
    unique = {
        'number_of_atoms': [],
        'atomic_num': set(),
        'element': set(),
        'period': set(),
        'group': set(),
        'degree': set(),
        'valence': set(),
        'num_h': set(),
        'num_radical': set(),
        'charge': set(),
        'hybrid': set(),
        'number_of_bonds' : [],
        'bond_type': set(),
        'bond_stereo': set(),
        'number_of_rings' : [],
        'ring_sizes' : set(),
        'het_counts' : set(),
        'avg_en': [],
        'spd_max': [],
        'barycentric_spd_max': []
    }

    for smi in smiles_chunk:
        mol = Chem.MolFromSmiles(smi.replace("[Dy]", "[C]"))
        unique['number_of_atoms'].append(len(mol.GetAtoms()))
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            Z = atom.GetAtomicNum()
            unique['atomic_num'].add(Z)

            # RDKit’s PT.GetPeriod/GetGroup want the actual atomic number
            unique['element'].add(symbol)
            period = cf.PT.GetRow(Z)
            unique['period'].add(period)
            unique['group'].add(get_group_from_map(symbol, cf.GROUP_MAP))

            unique['degree'].add(atom.GetDegree())
            unique['valence'].add(atom.GetValence(Chem.ValenceType.IMPLICIT))
            unique['num_h'].add(atom.GetTotalNumHs(includeNeighbors=True))
            unique['num_radical'].add(atom.GetNumRadicalElectrons())
            unique['charge'].add(atom.GetFormalCharge())
            unique['hybrid'].add(atom.GetHybridization().name)

        unique["number_of_bonds"].append(len(mol.GetBonds()))           
        for bond in mol.GetBonds():
            unique['bond_type'].add(bond.GetBondType().name)
            unique['bond_stereo'].add(bond.GetStereo().name)
        
        ri  = mol.GetRingInfo()
        atom_rings = ri.AtomRings()  
        bond_rings = ri.BondRings() 
        unique["number_of_rings"].append(len(bond_rings))
        for atoms in atom_rings:
            unique["ring_sizes"].add(len(atoms))
            unique['het_counts'].add(sum(1 for a in atoms if mol.GetAtomWithIdx(a).GetAtomicNum() != 6))
            Zs = [mol.GetAtomWithIdx(a).GetAtomicNum() for a in atoms]
            avg_en = np.mean([cf.PAULING_ARR[Z] for Z in Zs])
            unique["avg_en"].append(avg_en)
        
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
        edge_index = np.array(edge_slots).T
        spd = all_pairs_shortest_paths_unweighted(edge_index)
        unique['spd_max'].append(spd.max())
        face_index = []

        for fidx, (atoms, bonds) in enumerate(zip(atom_rings, bond_rings)):
            # face_index.extend((fidx, bond) for bond in bonds)
            for raw_bond_id in bonds:
                # now look up its true edge‐slot(s):
                for slot in bond2pos[raw_bond_id]:
                    face_index.append((fidx, slot))

        face_index = np.array(face_index).T
        spd_barycentric = all_pairs_shortest_paths_barycentric(edge_index, face_index)
        unique['barycentric_spd_max'].append(spd_barycentric.max())
    return unique

def collect_unique_atom_values_parallel(smiles, n_workers=None):
    """
    Parallel version: split `smiles` into chunks, run `_unique_from_smiles_chunk`
    in parallel, then merge all the partial sets into one.
    Finally, convert each set → sorted list.
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    # 1. Split the SMILES list into roughly equal chunks
    chunk_size = (len(smiles) + n_workers - 1) // n_workers
    chunks = [smiles[i : i + chunk_size] for i in range(0, len(smiles), chunk_size)]

    # 2. Use a Pool to process each chunk in parallel
    with Pool(n_workers) as pool:
        results = pool.map(_unique_from_smiles_chunk, chunks)

    # 3. Merge all partial results into one “unique” dict of sets
    merged = {
        'number_of_atoms' : [],
        'atomic_num': set(),
        'element': set(),
        'period': set(),
        'group': set(),
        'degree': set(),
        'valence': set(),
        'num_h': set(),
        'num_radical': set(),
        'charge': set(),
        'hybrid': set(),
        'number_of_bonds' : [],
        'bond_type': set(),
        'bond_stereo': set(),
        'number_of_rings' : [],
        'ring_sizes' : set(),
        'het_counts' : set(),
        'avg_en': [],
        'spd_max': [],
        'barycentric_spd_max': []
    }

    for part in results:
        for key, s in part.items():
            val = merged[key]
            if isinstance(val, set):
                val.update(s)
            else:
                val += s

    # 4. Convert each set → a sorted list
    for key, v in merged.items():
        if isinstance(v, list):
            merged[key] = [ float(np.mean(v)),
                            float(np.max(v)),
                            float(np.min(v)) ]
        else:
            merged[key] = sorted(v)
    return merged


def convert_dict_to_json(filepath, unique_dict):
    serializable = {}
    for key, val_list in unique_dict.items():
        # # Detect if the list contains RDKit‐enum objects by checking type of first element
        # if len(val_list) > 0 and hasattr(val_list[0], "__class__") and \
        # "rdchem" in val_list[0].__class__.__module__:
        #     # Convert each item to string
        #     serializable[key] = [str(v) for v in val_list]
        # else:
        #     # Already JSON‐native (ints, strings)
        serializable[key] = val_list

    # 3) Write out as JSON
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def read_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    return loaded