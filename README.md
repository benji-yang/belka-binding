# Binding Prediction with Topological Deep Learning  

This repository contains code for our experiments on **predicting ligand–protein binding** using **topological deep learning architectures**. We systematically evaluate message-passing and attention-based models with positional, structural, and higher-order encodings on the BELKA dataset.  

## Abstract  

Ligand–protein binding prediction remains challenging due to extreme class imbalance, scaffold-level generalisation, and long-range dependencies. We develop and benchmark **GPS-CC** variants equipped with distance encodings, Laplacian eigenvectors, random walk, barycentric subdivisions, and higher-order cell representations. Our results highlight the trade-offs between message passing and attention, the stabilising role of structural encodings, and the challenges of out-of-distribution scaffold generalisation.  

## Installation  

Clone the repository and install dependencies:  

```bash
git clone repo_link
cd your-repo
pip install -r requirements.txt
```

## Important Note on Reproducibility  

This repository is **not designed to run out-of-the-box**. The codebase has been optimised for our internal computing environments, including:  

- **GPU cluster** (multi-GPU training, job arrays) with Slurm schedulers  
- **HPC (High Performance Computing) systems** with PBS schedulers  
- **Local lab workstations** for debugging and small-scale experiments  

As a result:  
- Certain scripts assume **cluster-specific paths** (e.g. `/vol/bitbucket/` or mounted storage).  
- Job submission relies on **custom `.sh` files** tailored to our cluster’s scheduler.  
- Configuration files (`configs/parameters.py`) may need **manual modification** depending on environment.  
- Precomputed features and datasets are stored externally and are **not included in this repository**.  

If you wish to adapt this repository:  
1. Adjust the paths and configs in `configs/parameters.py`.  
2. Modify job submission scripts `.sh` for your own cluster environment.  
3. Ensure access to the BELKA dataset and precomputed features.  

We provide this repository primarily to **document our methodology and experiments**, not as a ready-to-run package.  

Below are some guidance on what may work if setup properly.

### Training

Run the main training script:

```bash
python training.py 
```

For smaller-scale evaluation runs:

```bash
python small_training.py
```

### Preprocessing & Feature Generation

```bash
python pre_run.py
```

This computes and saves processed graph features (atoms, bonds, rings, Laplacian encodings, etc.).

### Analysis

* **Attention diagnostics**

  ```bash
  python attention_metrics.py
  ```
* **Higher-order feature ablations**

  ```bash
  python cells_analysis.py
  python cells_metrics.py
  ```
* **Dataset similarity analysis**

  ```bash
  python dataset_eda.py
  ```

(Additional `.sh` scripts are provided for cluster/HPC job submission.)

```
.
├── README.md                # Project overview, installation, usage, citation
├── requirements.txt         # Python dependencies
│
├── configs/                 
│   ├── parameters.py        # Default config files
│   └── __init__.py
│
├── models/                  
│   ├── __init__.py
│   ├── distance.py          # Distance encodings
│   ├── encoder.py           # Node, edge, Laplacian, and feature encoders
│   ├── gps.py               # Main model architecture (GPS-CC variants)
│   ├── lossfunc.py          # Loss functions (BCE, ASL, focal, etc.)
│   └── mpnn.py              # Baseline MPNN model
│
├── utils/                   
│   ├── attn_logging.py      # Functions for logging attention diagnostics
│   ├── feature_engineering.py # Convert SMILES to graph/cell features
│   ├── helpers.py           # Utility helpers
│   ├── loader.py            # Multiprocessing feature loader
│   ├── plotting.py          # Plotting utilities
│   ├── preprocessing.py     # Dataset exploration and statistics
│   └── similarity.py        # Similarity metrics (Tanimoto, MMD, Jaccard, etc.)
│
├── figures/                 # Figures used in the report
│   └── ...       
│
├── attention_metrics.py     # Attention logging and visualisation
├── cells_analysis.py        # Analysis of higher-order (cell-based) features
├── cells_metrics.py         # Metrics for higher-order ablations
├── dataset_eda.py           # Subset similarity analysis
├── debugging_test.py        # Debugging playground
├── pre_run.py               # Feature preprocessing runner
├── small_training.py        # Quick evaluation runs
└── training.py              # Main training script
```

## Results

We report performance across multiple BELKA dataset subsets. Key findings:

* Plain MPNNs with rich structural encodings outperform attention-augmented models.
* Adding higher-order features stabilises training but reduces generalisation.
* Scaling from 1M to 10M improves MAP via scaffold coverage but does not solve OOD collapse.

(Figures are provided in `figures/`.)
