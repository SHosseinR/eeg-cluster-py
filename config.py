"""
Configuration file for EEG connectivity analysis
"""

import numpy as np

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# HC_DATA_PATH = "D:\\university\\projects\\graph-opt\\adhd-dataset\\preprocessed\\set2\\Control"  # UPDATE THIS
# PATIENT_DATA_PATH = "D:\\university\\projects\\graph-opt\\adhd-dataset\\preprocessed\\set2\\ADHD"      # UPDATE THIS
# OUTPUT_DIR = "./results-ADHD" 
HC_DATA_PATH = "D:\\university\\projects\\graph-opt\\paper-data\\EC\\set2\\HC"  # UPDATE THIS
PATIENT_DATA_PATH = "D:\\university\\projects\\graph-opt\\paper-data\\EC\\set2\\MDD"      # UPDATE THIS
OUTPUT_DIR = "./results-MDD" 
STEP_TO_START = 7

# ============================================================================
# SIGNAL PROCESSING PARAMETERS
# ============================================================================
EPOCH_DURATION = 10.0  # seconds
OVERLAP = 0.0  # No overlap between epochs

# Frequency bands (Hz)
FREQUENCY_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
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
    'diameter'
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
