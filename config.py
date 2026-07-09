"""
Configuration file for EEG connectivity analysis
"""

import os

import numpy as np

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# HC_DATA_PATH = "D:\\university\\projects\\graph-opt\\adhd-dataset\\preprocessed\\set2\\Control"
# PATIENT_DATA_PATH = "D:\\university\\projects\\graph-opt\\adhd-dataset\\preprocessed\\set2\\ADHD"
# OUTPUT_DIR = "./results-ADHD"
MODMA_PREPROCESSED_ROOT = (
    "D:\\university\\projects\\graph-opt\\mdd-dataset-2\\MODMA_EEG_BIDS_format\\"
    "MODMA_EEG_BIDS_format\\output\\preprocessed_resting_state_full"
)
HC_DATA_PATH = os.path.join(MODMA_PREPROCESSED_ROOT, "Control")
PATIENT_DATA_PATH = os.path.join(MODMA_PREPROCESSED_ROOT, "Patient")
OUTPUT_DIR = "./results-MODMA-resting"
STEP_TO_START = 1

# Channel display metadata. Exact labels remain authoritative for indexing.
CHANNEL_LABEL_STYLE = "e_alias"
CHANNEL_ALIAS_MONTAGE = "standard_1005"
CHANNEL_ALIAS_MAX_DISTANCE_M = 0.02

# Reduce dense HydroCel-128 data to a conventional near-30 scalp layout before
# connectivity. Exact selected labels remain E#; display labels use these names.
CHANNEL_SELECTION_MODE = "standard_32"  # "standard_32" or "all"
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
# Connectivity methods to compute
# CONNECTIVITY_METHODS = ['pdc', 'gc', 'psi', 'plv']
# CONNECTIVITY_METHODS = ['gc_tr', 'gc', 'psi', 'plv']
CONNECTIVITY_METHODS = ['gc']
CONNECTIVITY_N_JOBS = None  # None: use all available CPU cores, 1: disable multiprocessing

# Selected method for network analysis (change after visualization 2)
SELECTED_METHOD = 'gc'  # Change this based on visualization results

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
CHANNELS_TO_DROP = ['23A-23R', '24A-24R', 'A2-A1']
# CHANNELS_TO_DROP = []
