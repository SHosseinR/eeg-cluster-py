"""
Configuration for NSGA-II optimization of EEG connectivity
"""

# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================

# Network measures to optimize (select as many as needed from available measures)
OPTIMIZATION_MEASURES = [
    # 'global_efficiency',
    'betweenness_centrality', 
    # 'small_worldness'
    # 'modularity',
    'clustering_coefficient',
    'degree',
]

# NSGA-II Algorithm parameters (using pymoo)
NSGA_CONFIG = {
    'population_size': 100,           # Population size for NSGA-II
    'n_generations': 50,              # Number of generations
    'crossover_prob': 0.9,            # Crossover probability
    'crossover_eta': 15.0,            # Distribution index for SBX crossover
    'mutation_prob': None,            # Mutation probability (None = 1/n_var = 0.5 for 2 variables)
    'mutation_eta': 20.0,             # Distribution index for polynomial mutation
    'seed': None,                     # Random seed for reproducibility (None = random)
}
OPTIMIZATION_N_JOBS = None  # None: use all available CPU cores, 1: disable multiprocessing

# Number of top-ranked Pareto solutions to keep per subject (used for weighted summaries)
OPTIMIZATION_TOP_K = 5

# Optimization mode
# - 'nsga': NSGA-II with continuous stimulation duration/amplitude
# - 'grid': exhaustive node x band evaluation using fixed SIMULATION_CONFIG values
OPTIMIZATION_MODE = 'grid'

# State-space simulation parameters
SIMULATION_CONFIG = {
    'stimulation_duration': 1.0,      # Stimulation duration in seconds
    'stimulation_amplitude': 1.0,     # Stimulation amplitude
    'dt': 0.001,                      # Time step for simulation (seconds)
    'stability_constant': 0.01,       # Constant for A matrix normalization (c in A/(c+lambda))
}

# Optimization bounds for stimulation parameters
STIMULATION_DURATION_BOUNDS = (0.0, 2.0)
STIMULATION_AMPLITUDE_BOUNDS = (0.0, 2.0)

# Plasticity parameters
PLASTICITY_CONFIG = {
    'plasticity_enabled': True,       # Enable plasticity-based connectivity updates
    'plasticity_scaling': 1.0,        # Scaling factor for plasticity updates
}

# Output paths for optimization
OPTIMIZATION_OUTPUT_DIR = 'results-ADHD-kaggle/optimization'
OPTIMIZATION_RESULTS_FILE = 'optimization_results.npy'
OPTIMIZATION_FIGURES_DIR = 'results-ADHD-kaggle/optimization/figures'
