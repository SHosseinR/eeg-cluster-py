"""
Configuration file for EEG connectivity analysis
"""

import os
import tomllib
from pathlib import Path

import numpy as np

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# Change only this filename to switch datasets. It can also be overridden for a
# single run with the EEG_DATASET_CONFIG environment variable.
DATASET_CONFIG_FILE = os.environ.get("EEG_DATASET_CONFIG", "tdbrain_coherence.toml")
DATASET_CONFIG_PATH = Path(__file__).parent / "dataset_configs" / DATASET_CONFIG_FILE

try:
    with DATASET_CONFIG_PATH.open("rb") as config_file:
        DATASET_CONFIG = tomllib.load(config_file)
except FileNotFoundError as exc:
    raise FileNotFoundError(
        f"Dataset config not found: {DATASET_CONFIG_PATH}. "
        "Choose a TOML file from the dataset_configs directory."
    ) from exc

_REQUIRED_DATASET_KEYS = {
    "dataset_root",
    "healthy_subdirectory",
    "patient_subdirectory",
    "output_directory",
    "optimization_output_subdirectory",
    "optimization_debug_subject",
    "optimization_measures_by_band",
}
_missing_dataset_keys = _REQUIRED_DATASET_KEYS - DATASET_CONFIG.keys()
if _missing_dataset_keys:
    raise ValueError(
        f"Dataset config {DATASET_CONFIG_PATH} is missing keys: "
        f"{', '.join(sorted(_missing_dataset_keys))}"
    )

DATASET_ROOT = os.path.normpath(DATASET_CONFIG["dataset_root"])
HC_DATA_PATH = os.path.join(DATASET_ROOT, DATASET_CONFIG["healthy_subdirectory"])
PATIENT_DATA_PATH = os.path.join(DATASET_ROOT, DATASET_CONFIG["patient_subdirectory"])
OUTPUT_DIR = DATASET_CONFIG["output_directory"]
STEP_TO_START = 1

# Channel display metadata. Exact labels remain authoritative for indexing.
CHANNEL_LABEL_STYLE = "exact"
CHANNEL_ALIAS_MONTAGE = "standard_1005"
CHANNEL_ALIAS_MAX_DISTANCE_M = 0.02

# TD-BRAIN restEC preprocessing already keeps the 26 standard scalp channels.
CHANNEL_SELECTION_MODE = "all"  # "standard_32" or "all"
CHANNEL_SOURCE_MONTAGE = "GSN-HydroCel-128"
CHANNEL_SELECTION_MONTAGE = "standard_1005"
CHANNEL_SELECTION_TARGETS = [
    "Fp1", "Fp2",
    "AF3", "AF4",
    "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "Pz", "P4", "P8",
    "PO3", "PO4",
    "O1", "Oz", "O2",
]

# ============================================================================
# SIGNAL PROCESSING PARAMETERS
# ============================================================================
EPOCH_DURATION = 10.0  # seconds
OVERLAP = 0.0  # No overlap between epochs

# Frequency bands (Hz)
# Theta and gamma are intentionally excluded from analysis/optimization based on
# prior knowledge for this project.
FREQUENCY_BANDS = {
    'delta': (1, 4),
    'alpha': (8, 13),
    'beta': (13, 30),
}

# ============================================================================
# CONNECTIVITY PARAMETERS
# ============================================================================
# Dataset profiles may opt into a scientifically distinct connectivity run.
# Existing profiles retain their historical settings; the *_connectivity_v2
# profiles use broadband spectral input, natural estimator scales, and explicit
# failure handling without overwriting legacy result trees.
CONNECTIVITY_SETTINGS = DATASET_CONFIG.get("connectivity", {})
CONNECTIVITY_METHODS = list(CONNECTIVITY_SETTINGS.get("methods", ["gc"]))
CONNECTIVITY_N_JOBS = None  # None: use all available CPU cores, 1: disable multiprocessing

# Selected method for network analysis (change after visualization 2)
SELECTED_METHOD = CONNECTIVITY_SETTINGS.get("selected_method", "gc")
GC_N_LAGS = int(CONNECTIVITY_SETTINGS.get("gc_n_lags", 40))
CONNECTIVITY_NORMALIZATION = CONNECTIVITY_SETTINGS.get("normalization", "minmax")
CONNECTIVITY_ERROR_POLICY = CONNECTIVITY_SETTINGS.get("error_policy", "zeros")
SPECTRAL_CONNECTIVITY_INPUT = CONNECTIVITY_SETTINGS.get(
    "spectral_input", "band_filtered"
)
if CONNECTIVITY_NORMALIZATION not in {"none", "minmax", "maxabs"}:
    raise ValueError(f"Unknown connectivity normalization: {CONNECTIVITY_NORMALIZATION}")
if CONNECTIVITY_ERROR_POLICY not in {"raise", "zeros"}:
    raise ValueError(f"Unknown connectivity error policy: {CONNECTIVITY_ERROR_POLICY}")
if SPECTRAL_CONNECTIVITY_INPUT not in {"broadband", "band_filtered"}:
    raise ValueError(f"Unknown spectral connectivity input: {SPECTRAL_CONNECTIVITY_INPUT}")

# Frequency resolution for spectral connectivity
FMIN = 1.0
FMAX = 45.0
N_FREQS = 100

# ============================================================================
# NETWORK MEASURES
# ============================================================================
NETWORK_MEASURES = [
    'global_efficiency',
    'local_efficiency',
    'clustering_coefficient',
    'transitivity',
    'modularity',
    'degree',
    'betweenness_centrality',
    'rich_club',
    'assortativity',
    'spectral_radius',
    'small_worldness',
    'diameter',
    'density',
    'mean_weight',
    'std_weight',
    'median_weight',
    'max_weight',
    'min_weight',
    'cv_weight',
    'char_path_length'
]

# ============================================================================
# STATISTICAL PARAMETERS
# ============================================================================
ALPHA_LEVEL = 0.05  # Significance level
N_PERMUTATIONS = 1000  # For permutation tests

# ============================================================================
# CLASSIFICATION PARAMETERS
# ============================================================================
N_FEATURES_COMBINATION = 3  # Number of features in each combination
N_FOLDS = 5  # Cross-validation folds
N_TOP_FEATURES = 10  # Number of top feature sets to report
RANDOM_STATE = 42  # For reproducibility
CLASSIFICATION_MODE = 'all_metrics'  # 'triplet' or 'all_metrics'
CLASSIFICATION_MODEL = 'linear_svm'  # 'linear_svm' or 'logistic' (used in all_metrics mode)
CLASSIFICATION_C = 0.1  # Regularization strength for linear models
CLASSIFICATION_FEATURE_IMPORTANCE_TOP_N = 10  # Top features to show in summaries/plots

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================
FIGURE_DPI = 300
FIGURE_SIZE = (12, 10)
CMAP_CONNECTIVITY = 'viridis'
CMAP_PVALUE = 'RdYlGn_r'  # Red for low p-values, green for high

# P-value thresholds for visualization
PVALUE_THRESHOLDS = [0.001, 0.01, 0.05]

# ============================================================================
# CHANNELS TO DROP
# ============================================================================
# CHANNELS_TO_DROP = ['23A-23R', '24A-24R', 'A2-A1']
CHANNELS_TO_DROP = []
