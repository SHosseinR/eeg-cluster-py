"""
Configuration for NSGA-II optimization of EEG connectivity
"""

import os

from config import OUTPUT_DIR

# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================

# Network measures to optimize (select as many as needed from available measures)
# OPTIMIZATION_MEASURES = [
#     # 'global_efficiency',
#     'betweenness_centrality', 
#     # 'small_worldness'
#     # 'modularity',
#     'clustering_coefficient',
#     'degree',
# ]

OPTIMIZATION_MEASURES = [
    'global_efficiency',
    'betweenness_centrality', 
    'small_worldness'
]

# Run a separate optimization per band (band is fixed, not a decision variable).
OPTIMIZATION_PER_BAND = True

# Optional per-band measure lists. If a band is missing, OPTIMIZATION_MEASURES is used.
# Example:
OPTIMIZATION_MEASURES_BY_BAND = {
    #results-MDD-5
    # 'delta': ['cv_weight', 'min_weight', 'small_worldness'],
    # 'alpha': ['min_weight', 'transitivity', 'clustering_coefficient'],
    # 'beta': ['cv_weight', 'diameter', 'modularity'],
    # don't know wth is this!
    # 'delta': ['cv_weight', 'diameter', 'mean_weight'],
    # 'alpha': ['mean_weight', 'small_worldness', 'betweenness_centrality'],
    # 'beta': ['cv_weight', 'betweenness_centrality', 'diameter'],
    #tdbrainset
    'delta': ['char_path_length', 'diameter', 'transitivity'],
    'alpha': ['local_efficiency', 'global_efficiency', 'char_path_length'],
    'beta': ['median_weight', 'std_weight', 'diameter'],
    #small set
    # 'delta': ['std_weight', 'median_weight', 'betweenness_centrality'],
    # 'alpha': ['small_worldness', 'spectral_radius', 'std_weight'],
    # 'beta': ['betweenness_centrality', 'diameter', 'cv_weight'],
}
# OPTIMIZATION_MEASURES_BY_BAND = {}

# NSGA-II Algorithm parameters (using pymoo)
NSGA_CONFIG = {
    'population_size': 100,           # Population size for NSGA-II
    'n_generations': 50,              # Number of generations
    'crossover_prob': 0.9,            # Crossover probability
    'crossover_eta': 15.0,            # Distribution index for SBX crossover
    'mutation_prob': None,            # Mutation probability (None = 1/n_var)
    'mutation_eta': 20.0,             # Distribution index for polynomial mutation
    'seed': None,                     # Random seed for reproducibility (None = random)
}
OPTIMIZATION_N_JOBS = None  # None: use all available CPU cores, 1: disable multiprocessing

# Number of top-ranked Pareto solutions to keep per subject (used for weighted summaries)
OPTIMIZATION_TOP_K = 5

# Optional patient filtering before optimization.
# Generate this ranking with plot_svm_margin_rejection_3d.py.
PATIENT_REJECTION_PERCENT = 50
PATIENT_REJECTION_PERCENT_BY_BAND = {
    'delta': 50,
    'alpha': 50,
    'beta': 50,
}
PATIENT_REJECTION_RANKING_FILE = os.path.join(
    OUTPUT_DIR,
    'data',
    'svm_margin_subject_ranking.csv'
)
PATIENT_REJECTION_RANKING_FILE_BY_BAND = os.path.join(
    OUTPUT_DIR,
    'data',
    'svm_margin_subject_ranking_{band}.csv'
)

# Ranking pool for distance-based selection (grid + NSGA)
# - True: rank and select from Pareto front only
# - False: rank and select from all solutions (including dominated)
GRID_USE_PARETO_ONLY = True

# Optimization mode
# - 'nsga': NSGA-II with continuous stimulation duration/amplitude
# - 'grid': exhaustive node x band evaluation using fixed SIMULATION_CONFIG values
OPTIMIZATION_MODE = 'nsga'

# Objective mode (how objectives are computed)
# - 'directional': maximize/minimize based on Patient vs Healthy direction
# - 'distance_to_gt': minimize distance to Healthy mean (ground truth)
OPTIMIZATION_OBJECTIVE_MODE = 'distance_to_gt'

# State-space simulation parameters
# SIMULATION_CONFIG = {
#     'stimulation_duration': 1,      # Stimulation duration in seconds
#     'stimulation_amplitude': 1,     # Stimulation amplitude
#     'dt': 0.001,                      # Time step for simulation (seconds)
#     'stability_constant': 0.01,       # Constant for A matrix normalization (c in A/(c+lambda))
#     'leak': 0,                      # Identity damping for A' = A - leak * I
# }
SIMULATION_CONFIG = {
    'stimulation_duration': 10,      # Stimulation duration in seconds
    'stimulation_amplitude': 1,     # Stimulation amplitude
    'dt': 0.01,                      # Time step for simulation (seconds)
    'stability_constant': 0.01,       # Constant for A matrix normalization (c in A/(c+lambda))
    'leak': 1,                      # Identity damping for A' = A - leak * I
}

# Optimization bounds for stimulation parameters
# STIMULATION_DURATION_BOUNDS = (1, 20)
# STIMULATION_AMPLITUDE_BOUNDS = (0.03, 0.3)
# STIMULATION_LEAK_BOUNDS = (0.0, 2.0)
STIMULATION_DURATION_BOUNDS = (1, 30)
# Signed and fully configurable:
#   (0.1, 3.0)   enhancement only
#   (-3.0, -0.1) suppression only
#   (-3.0, 3.0)  let the optimizer choose polarity (current default)
STIMULATION_AMPLITUDE_BOUNDS = (0.1, 3.0)
STIMULATION_LEAK_BOUNDS = (0.0, 2.0)

# Hard feasibility bounds on raw final/baseline activation ratios. Candidates
# outside this range are rejected by NSGA-II before the clipped ratios can make
# extreme stimulation solutions look artificially equivalent.
ACTIVATION_RATIO_FEASIBILITY_BOUNDS = (0.1, 10.0)

# Plasticity parameters
PLASTICITY_CONFIG = {
    'plasticity_enabled': True,       # Enable plasticity-based connectivity updates
    'plasticity_scaling': 1.0,        # Scaling factor for plasticity updates
}

# Debug/plotting defaults
# OPTIMIZATION_DEBUG_SUBJECT = 'sub-88064657'
OPTIMIZATION_DEBUG_SUBJECT = 'MDD S2  EC'

# Output paths for optimization
OPTIMIZATION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'optimization-nsga-distance_to_gt')
OPTIMIZATION_RESULTS_FILE = 'optimization_results.npy'
OPTIMIZATION_FIGURES_DIR = os.path.join(OPTIMIZATION_OUTPUT_DIR, 'optimization', 'figures')
