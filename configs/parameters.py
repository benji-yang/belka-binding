from mendeleev import get_all_elements, element
from rdkit.Chem import AllChem
from rdkit import Chem
import numpy as np
import os
import torch
from pathlib import Path
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional

# CLUSTER = 'GPU'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SENTINEL = 255
USERNAME = ""
# pbs_idx = os.environ.get("PBS_ARRAY_INDEX")

# # build a suffix like "_3" or "" if not present
# _suffix = f"_{pbs_idx}" if pbs_idx is not None else ""

def s_to_g_map():
    symbol_to_group_map = {elem.symbol: elem.group_id for elem in get_all_elements()}
    # Handle elements without a group (like Lanthanides/Actinides) if necessary
    for symbol in symbol_to_group_map:
        if symbol_to_group_map[symbol] is None:
            symbol_to_group_map[symbol] = -1 # Use a placeholder like -1
    return symbol_to_group_map

def get_electroneg(max_z=118):
    # 1) As a NumPy array (fast vector indexing + .mean()):
    pauling_arr = np.zeros(max_z+1, dtype=float)
    for z in range(1, max_z+1):
        pauling_arr[z] = element(z).en_pauling
    return pauling_arr

def get_electronegativity_list(interval, min_en, max_en):
    interval       = 0.5                    # your desired bin width
    min_en, max_en = 0.0, 5.0               # the “allowed” range endpoints

    # build your numeric bins [0.0, 0.5, 1.0, …, 5.0] for interval=0.5
    num_bins = int((max_en - min_en) / interval) + 1
    eneg_list = [min_en + i * interval for i in range(num_bins)] + ['misc']
    return eneg_list

@dataclass
class Config:
    # --- raw inputs / tunables ---
    cluster: str                   = 'GPU'
    attention: bool                = True
    face: bool                     = True
    barycentric : bool             = True
    batch_size: int                = 512
    max_nodes : int                = 15000
    use_bucket : bool              = False
    experiment_size: Optional[int] = 1000000
    loss: str                      = 'ASYM'
    stopping_epochs: int           = 450
    lambda_: float                 = 0.01
    chunk_size: int                = 8
    peak_lr: float                 = 3e-4
    min_lr: float                  = 0.0
    total_epochs: int              = 450
    warmup_epochs: int             = 10
    grad_clip: float               = 5.0
    patience: int                  = 15
    num_heads: int                 = 32
    position: bool                 = False
    variance: int                  = 0
    spd_bias : bool                = True
    eval_face_noise_sigma : float  = 0.0
    log_cosine_per_layer : bool    = False
    cos_sample_cap : int           = 4096
    num_layers : int               = 16

    # --- derived after parsing ---
    size_suffix: str               = field(init=False)
    pbs_suffix: str                = field(init=False)
    file_suffix: str               = field(init=False)
    min_ones: int                  = field(init=False)

    # --- cluster-specific paths ---
    data_path: str                 = field(init=False)
    train_graph_path: Optional[Path] = field(init=False)
    train_weight_path: str         = field(init=False)
    val_graph_path: Optional[Path] = field(init=False)
    test_graph_path: Optional[Path] = field(init=False)
    # checkpoint_path: str           = field(init=False)
    zip_dir: str                   = field(init=False)
    zip_train_path: Path           = field(init=False)
    zip_val_path: Path             = field(init=False)
    zip_test_path: Path            = field(init=False)
    load_path: str                 = field(init=False)
    utility_dir: str               = field(init=False)

    def __post_init__(self):

        # suffix for experiment_size
        c = self.cluster.upper()
        self.size_suffix = f"_{self.experiment_size}" if self.experiment_size is not None else ""
        self.barycentric_suffix = "_barycentric" if self.barycentric else "_normal"
        self.face_suffix = f"_{int(self.face)}"
        # suffix for PBS_ARRAY_INDEX
        pbs_idx = os.environ.get("PBS_ARRAY_INDEX")
        slurm_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
        self.pbs_suffix = f"_{pbs_idx}" if pbs_idx is not None else ""
        self.slurm_suffix = f"_{slurm_idx}" if slurm_idx is not None else ""
        self.checkpoint_suffix = f"_{self.experiment_size}_{self.loss}{self.lambda_}_{int(self.attention)}_{int(self.face)}_{int(self.barycentric)}"
        self.train_file_suffix = self.size_suffix + self.barycentric_suffix
        self.test_file_suffix = self.barycentric_suffix

        # self.ablation = f"_{self.num_layers}_{int(self.spd_bias)}" if not self.spd_bias or self.num_layers != 16 else ""

        # combined file suffix
        if c == 'HPC':
            self.train_file_suffix += self.pbs_suffix
        elif c == 'GPU':
            self.train_file_suffix += self.slurm_suffix

        # compute min_ones based on experiment_size
        self.min_ones = int(self.experiment_size / 1000) if self.experiment_size is not None else 10000

        # cluster-dependent paths
        fs = self.train_file_suffix
        ns = self.test_file_suffix
        cs = self.checkpoint_suffix
        vs = self.variance if self.variance != 0 else ""

        if c == 'HPC':
            self.data_path = 'data'
            self.train_weight_path = f"compressed/training_weights{fs}.pt"
            self.zip_dir           = "compressed"
            self.zip_train_path    = Path(f"compressed/features{fs}.zip")
            self.zip_val_path      = Path(f"compressed/validation_features{ns}.zip")
            self.zip_test_path     = Path(f"compressed/testing_features{ns}.zip")
            self.utility_dir       = "utils_out"
        elif c == 'DOC':
            self.data_path = f'/vol/bitbucket/{USERNAME}/belka_dti/data'
            self.train_weight_path = f"/vol/bitbucket/{USERNAME}/belka_dti/compressed/training_weights{fs}.pt"
            self.zip_dir           = f"/vol/bitbucket/{USERNAME}/belka_dti/compressed"
            self.zip_train_path    = Path(f"/vol/bitbucket/{USERNAME}/belka_dti/compressed/features{fs}.zip")
            self.zip_val_path      = Path(f"/vol/bitbucket/{USERNAME}/belka_dti/compressed/validation_features{fs}.zip")
            self.zip_test_path     = Path(f"/vol/bitbucket/{USERNAME}/belka_dti/compressed/testing_features{fs}.zip")
            self.utility_dir       = f"/vol/bitbucket/{USERNAME}/belka_dti/utils_out"
        elif c == 'GPU':
            self.data_path = f'/vol/bitbucket/{USERNAME}/belka_dti/data'
            self.train_weight_path = f"/vol/bitbucket/{USERNAME}/belka_dti/compressed_gpu/training_weights{fs}.pt"
            self.zip_dir           = f"/vol/bitbucket/{USERNAME}/belka_dti/compressed_gpu"
            self.zip_train_path    = Path(f"/vol/bitbucket/{USERNAME}/belka_dti/compressed_gpu/features{fs}.zip")
            self.zip_val_path      = Path(f"/vol/bitbucket/{USERNAME}/belka_dti/compressed_gpu/validation_features{ns}.zip")
            self.zip_test_path     = Path(f"/vol/bitbucket/{USERNAME}/belka_dti/compressed_gpu/testing_features{ns}.zip")
            self.utility_dir       = f"/vol/bitbucket/{USERNAME}/belka_dti/utils_out"
        else:
            raise ValueError(f"Unknown cluster: {self.cluster}")

        # JSON load path
        self.load_path = f"{self.data_path}/belka-v1.json"
    
    # --- NEW: dynamic vs suffix (only variance) ---
    @property
    def vs(self) -> str:
        """Dynamic variance suffix string used in checkpoint path."""
        return f"{self.variance}" if self.variance != 0 else ""
    
    @property
    def ablation_suffix(self) -> str:
        # add ablation only if spd_bias is off OR layers != 16
        return f"_{self.num_layers}_{int(self.spd_bias)}" if (not self.spd_bias or self.num_layers != 16) else ""

    # --- NEW: checkpoint_path computed on access, using static cs and dynamic vs ---
    @property
    def checkpoint_path(self) -> str:
        c = self.cluster.upper()
        vs = self.vs  # dynamic
        cs = self.checkpoint_suffix + self.ablation_suffix  # static (from __post_init__)
        if c == 'HPC':
            return f"checkpoint{vs}/best_model{cs}.pt"
        elif c == 'DOC':
            return f"/vol/bitbucket/{USERNAME}/belka_dti/checkpoint/best_model{cs}.pt"
        elif c == "GPU":
            return f"/vol/bitbucket/{USERNAME}/belka_dti/checkpoint{vs}/best_model_gpu{cs}.pt"
        else:
            raise ValueError(f"Unknown cluster: {self.cluster}")

    def to_dict(self):
        # merge everything for logging/tracking
        base = asdict(self)
        return base


def get_config() -> Config:
    # build a default CFG just to read its defaults
    default_cfg = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster", choices=["HPC","DOC","GPU"], default=default_cfg.cluster,
        help="Target cluster: HPC, DOC, or GPU"
    )
    parser.add_argument("--attention", action="store_true",
                        default=default_cfg.attention)
    parser.add_argument("--no-attention", dest="attention", action="store_false")
    parser.add_argument("--position", action="store_true",
                        default=default_cfg.position)
    parser.add_argument("--no-position", dest="position", action="store_false")
    parser.add_argument("--face", action="store_true",
                        default=default_cfg.face)
    parser.add_argument("--no-face", dest="face", action="store_false")
    parser.add_argument("--barycentric", action="store_true",
                        default=default_cfg.barycentric)
    parser.add_argument("--no-barycentric", dest="barycentric", action="store_false")
    parser.add_argument("--num_heads",    type=int,
                        default=default_cfg.num_heads)
    parser.add_argument("--batch_size",   type=int,
                        default=default_cfg.batch_size)
    parser.add_argument("--experiment_size", type=int,
                        default=default_cfg.experiment_size,
                        help="Optional experiment size for suffix and min_ones calculation"
    )
    parser.add_argument("--chunk_size", type=int,
                        default=default_cfg.chunk_size)
    parser.add_argument("--peak_lr",    type=float,
                        default=default_cfg.peak_lr)
    parser.add_argument("--min_lr",     type=float,
                        default=default_cfg.min_lr)
    parser.add_argument("--total_epochs",  type=int,
                        default=default_cfg.total_epochs)
    parser.add_argument("--stopping_epochs",  type=int,
                        default=default_cfg.stopping_epochs)
    parser.add_argument("--warmup_epochs", type=int,
                        default=default_cfg.warmup_epochs)
    parser.add_argument("--grad_clip",   type=float,
                        default=default_cfg.grad_clip)
    parser.add_argument("--patience",    type=int,
                        default=default_cfg.patience)
    parser.add_argument("--loss",        type=str,
                        default=default_cfg.loss)
    parser.add_argument("--lambda_",     type=float,
                        default=default_cfg.lambda_)
    parser.add_argument("--max_nodes",     type=int,
                        default=default_cfg.max_nodes)
    parser.add_argument("--use_bucket",  action="store_true",
                        default=default_cfg.use_bucket)
    parser.add_argument("--no-use_bucket", dest="use_bucket", action="store_false")
    parser.add_argument("--variance", type=int,
                        default=default_cfg.variance)
    parser.add_argument("--spd_bias", action="store_true",
                        default=default_cfg.spd_bias)
    parser.add_argument("--no-spd_bias", dest="spd_bias", action="store_false")
    parser.add_argument("--cos_sample_cap", type=int,
                        default=default_cfg.cos_sample_cap)
    parser.add_argument("--eval_face_noise_sigma", type=float,
                        default=default_cfg.eval_face_noise_sigma)
    parser.add_argument("--log_cosine_per_layer", action="store_true",
                        default=default_cfg.log_cosine_per_layer)
    parser.add_argument("--no-log_cosine_per_layer", dest="log_cosine_per_layer", action="store_false")
    parser.add_argument("--num_layers", type=int, default=default_cfg.num_layers)

    args = parser.parse_args()
    cfg  = Config(**vars(args))
    return cfg

# ATTENTION = True
# POSITION = False
# FACE = False
# NUM_HEADS = 32
# BATCH_SIZE = 512
# EXPERIMENT_SIZE = None
# _size = f"_{EXPERIMENT_SIZE}" if EXPERIMENT_SIZE is not None else ""
# MIN_ONES = int(EXPERIMENT_SIZE / 1000) if EXPERIMENT_SIZE is not None else 10000
# CHUNK_SIZE = 8
# PEAK_LR       = 4e-4
# MIN_LR = 0
# TOTAL_EPOCHS  = 450
# WARMUP_EPOCHS = 10
# GRAD_CLIP     = 5.0
# PATIENCE      = 15
# LOSS = 'FOCAL'
# LAMBDA = 0.1


# Path type
# CHECKPOINT_PATH = "/vol/bitbucket/{USERNAME}/belka_dti/checkpoint/best_model.pt"
UNIQUE_JSON_PATH = 'unique_values.json'

# if CLUSTER == 'HPC':
#     DATA_PATH = 'data'
#     TRAIN_GRAPH_PATH   = f"preprocessed/training_features{_suffix}.pt"
#     TRAIN_WEIGHT_PATH  = f"preprocessed/training_weights{_suffix}.pt"
#     VAL_GRAPH_PATH   = f"preprocessed/validation_features{_suffix}.pt"
#     TEST_GRAPH_PATH = f"preprocessed/testing_features{_suffix}.pt"
#     CHECKPOINT_PATH = "checkpoint/best_model.pt"
#     ZIP_DIR = "compressed"
#     ZIP_TRAIN_PATH = Path(f"compressed/features{_suffix}.zip")
#     ZIP_VAL_PATH   = Path(f"compressed/validation_features{_suffix}.zip")
#     ZIP_TEST_PATH = Path(f"compressed/testing_features{_suffix}.zip")

# elif CLUSTER == 'DOC':
#     DATA_PATH = '/vol/bitbucket/{USERNAME}/belka_dti/data'
#     TRAIN_GRAPH_PATH   = f"/vol/bitbucket/{USERNAME}/belka_dti/preprocessed/training_features{_suffix}.pt"
#     TRAIN_WEIGHT_PATH  = f"/vol/bitbucket/{USERNAME}/belka_dti/preprocessed/training_weights{_suffix}.pt"
#     VAL_GRAPH_PATH   = f"/vol/bitbucket/{USERNAME}/belka_dti/preprocessed/validation_features{_suffix}.pt"
#     TEST_GRAPH_PATH = f"/vol/bitbucket/{USERNAME}/belka_dti/preprocessed/testing_features{_suffix}.pt"
#     CHECKPOINT_PATH = "/vol/bitbucket/{USERNAME}/belka_dti/checkpoint/best_model.pt"
#     ZIP_DIR = "/vol/bitbucket/{USERNAME}/belka_dti/compressed"
#     ZIP_TRAIN_PATH = Path(f"/vol/bitbucket/{USERNAME}/belka_dti/compressed/features{_suffix}{_size}.zip")
#     ZIP_VAL_PATH   = Path(f"/vol/bitbucket/{USERNAME}/belka_dti/preprocessed/validation_features{_suffix}{_size}.zip")
#     ZIP_TEST_PATH = Path(f"/vol/bitbucket/{USERNAME}/belka_dti/preprocessed/testing_features{_suffix}{_size}.zip")

# elif CLUSTER == 'GPU':
#     DATA_PATH = '/vol/bitbucket/{USERNAME}/belka_dti/data'
#     TRAIN_WEIGHT_PATH  = f"/vol/bitbucket/{USERNAME}/belka_dti/compressed_gpu/training_weights{_suffix}.pt"
#     CHECKPOINT_PATH = "/vol/bitbucket/{USERNAME}/belka_dti/checkpoint/best_model_gpu.pt"
#     ZIP_DIR = "/vol/bitbucket/{USERNAME}/belka_dti/compressed_gpu"
#     ZIP_TRAIN_PATH = Path(f"/vol/bitbucket/{USERNAME}/belka_dti/compressed_gpu/features{_suffix}{_size}.zip")
#     ZIP_VAL_PATH   = Path(f"/vol/bitbucket/{USERNAME}/belka_dti/compressed_gpu/validation_features{_suffix}{_size}.zip")
#     ZIP_TEST_PATH = Path(f"/vol/bitbucket/{USERNAME}/belka_dti/compressed_gpu/testing_features{_suffix}{_size}.zip")

# else:
#     raise ValueError("Please specify a valid cluster destination.")

# LOAD_PATH = DATA_PATH + '/belka-v1.json'

# Essential model initialisation parameters
N_PROTEINS = 3
NODE_DIM=256
EDGE_DIM=128
FACE_DIM=128
GLOBAL_DIM=64
ALL_UNIQUE_PROTEIN = ['BRD4', 'HSA', 'sEH']


# Chemical parameters
DEFAULT_IDX = 'misc'
MAX_Z = 118
ETKDG_PARAMS = AllChem.ETKDGv3()  # modern distance-geometry parameters
ETKDG_PARAMS.pruneRmsThresh = 0.5
ETKDG_PARAMS.maxIterations = 1000
# NUM_CONF = 10
GROUP_MAP = s_to_g_map()
PAULING_ARR = get_electroneg(MAX_Z)
PT = Chem.GetPeriodicTable()
INTERVAL = 0.05
MIN_EN = 2.5
MAX_EN = 2.95
TOLERANCE = INTERVAL / 2


# Atom encoding
ATOM_NUM_LIST = [5, 6, 7, 8, 9, 14, 16, 17, 35, 53] + ['misc']
PERIOD_LIST = [2, 3, 4, 5, 'misc']
GROUP_LIST = list(range(13, 18)) + ['misc']
DEGREE_LIST = [0, 1, 2, 3, 4, 'misc']
IMPLICIT_VALENCE_LIST = [0, 1, 2, 3, 'misc']
POSSIBLE_NUMH_LIST = [0, 1, 2, 3, 'misc']
POSSIBLE_NUMBER_RADICAL_E_LIST = [0, 1, 2, 3, 'misc']
POSSIBLE_FORMAL_CHARGE_LIST = [-1, 0, 1, 'misc']
POSSIBLE_HYBRIDISATION_LIST = ['SP', 'SP2', 'SP3', 'misc']
ATOM_IDX    = {z:i for i,z in enumerate(ATOM_NUM_LIST)}
PERIOD_IDX  = {p:i for i,p in enumerate(PERIOD_LIST)}
GROUP_IDX   = {g:i for i,g in enumerate(GROUP_LIST)}
DEGREE_IDX  = {d:i for i,d in enumerate(DEGREE_LIST)}
VALENCE_IDX = {v:i for i,v in enumerate(IMPLICIT_VALENCE_LIST)}
NUMH_IDX    = {h:i for i,h in enumerate(POSSIBLE_NUMH_LIST)}
RAD_IDX     = {r:i for i,r in enumerate(POSSIBLE_NUMBER_RADICAL_E_LIST)}
CHG_IDX     = {c:i for i,c in enumerate(POSSIBLE_FORMAL_CHARGE_LIST)}
HYB_IDX     = {h:i for i,h in enumerate(POSSIBLE_HYBRIDISATION_LIST)}

# Bond encoding
POSSIBLE_BOND_TYPE_LIST = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'misc']
POSSIBLE_BOND_STEREO_LIST = ['STEREONONE', 'STEREOE', 'misc',]
BOND_TYPE_IDX   = {b:i for i,b in enumerate(POSSIBLE_BOND_TYPE_LIST)}
BOND_STEREO_IDX = {s:i for i,s in enumerate(POSSIBLE_BOND_STEREO_LIST)}

# Face encoding
RING_SIZE_LIST = [3, 4, 5, 6, 7, 'misc']
HET_COUNT_LIST = [0, 1, 2, 3, 4, 'misc']
ELECTRONEGATIVITY_LIST = get_electronegativity_list(INTERVAL, MIN_EN, MAX_EN)
RING_SIZE_IDX = {r:i for i,r in enumerate(RING_SIZE_LIST)}
HET_COUNT_IDX = {h:i for i,h in enumerate(HET_COUNT_LIST)}
ELECTRONEGATIVITY_IDX = {e:i for i,e in enumerate(ELECTRONEGATIVITY_LIST)}
