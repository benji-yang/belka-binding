from utils.preprocessing import convert_dict_to_json, collect_unique_atom_values_parallel
# from polaris.hub.client import PolarisHubClient
# from polaris.hub.settings import PolarisHubSettings
import polaris as po
import torch.multiprocessing as mp

# 1) Must be before any torch.* or multiprocessing pools are created:
mp.set_start_method('fork', force=True)        # or 'forkserver' if you like
mp.set_sharing_strategy('file_system')
import configs.parameters as cf
from utils.loader import load_features
from utils.helpers import convert_numpy_to_torch_inplace, save_as_zip, load_from_zip
import torch
from torch_geometric.data import Data
import json

# settings = PolarisHubSettings(
#     username="",
#     password="",
# )

# with PolarisHubClient(settings=settings, cache_auth_token=True) as client:
#     token = client.fetch_token()
#     # print(client.list_datasets())
#     dataset = po.load_dataset("leash-bio/belka-v1", verify_checksum="verify")

# path = dataset.to_json(destination=cf.DATA_PATH)

# print("loading done")

# cfg   = cf.get_config()
# dataset = po.dataset.DatasetV2.from_json(cfg.load_path)

# smiles_array = dataset[:, "molecule_smiles"]

# unique_vals = collect_unique_atom_values_parallel(smiles_array)

# convert_dict_to_json(cf.UNIQUE_JSON_PATH, unique_vals)

# print("Check done.")

def main():
    cfg   = cf.get_config()
    # only inside main (not at import time) do we build the Pool
    size_suffix = f"_{cfg.experiment_size}" if cfg.experiment_size is not None else ""
    barycentric_suffix = "_normal"
    file_suffix = size_suffix + barycentric_suffix
    json_path = f"{cfg.utility_dir}/subset_index_mask{file_suffix}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        subset_index_mask = json.load(f)
    train_graph, pos_weight = load_features(task_type="train", m=cfg.experiment_size, sample_idx=subset_index_mask, cfg=cfg)
    train_graph = convert_numpy_to_torch_inplace(train_graph)
    # torch.save(train_graph, cf.TRAIN_GRAPH_PATH, _use_new_zipfile_serialization=False)
    torch.save(pos_weight, cfg.train_weight_path)
    save_as_zip(cfg.zip_train_path, train_graph)
    # train_data = load_from_zip(cf.ZIP_PATH)

if __name__ == '__main__':
    # # safe‐guard for Windows; no harm on Linux
    # from multiprocessing import freeze_support
    # freeze_support()

    main()
