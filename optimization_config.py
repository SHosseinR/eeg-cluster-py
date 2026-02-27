"""
Configuration for NSGA-II optimization of EEG connectivity
"""

# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================

# Network measures to optimize (select 3 from the available measures)
OPTIMIZATION_MEASURES = [
    'global_efficiency',
    'clustering_coefficient', 
    'modularity'
]

# NSGA-II Algorithm parameters
NSGA_CONFIG = {
    'population_size': 100,           # Population size for NSGA-II
    'n_generations': 50,              # Number of generations
    'crossover_prob': 0.9,            # Crossover probability
    'mutation_prob': 0.1,             # Mutation probability
    'tournament_size': 3,             # Tournament selection size
}

# State-space simulation parameters
SIMULATION_CONFIG = {
    'stimulation_duration': 1.0,      # Stimulation duration in seconds
    'stimulation_amplitude': 1.0,     # Stimulation amplitude
    'dt': 0.001,                      # Time step for simulation (seconds)
    'stability_constant': 0.01,       # Constant for A matrix normalization (c in A/(c+lambda))
}

# Plasticity parameters
PLASTICITY_CONFIG = {
    'plasticity_enabled': True,       # Enable plasticity-based connectivity updates
    'plasticity_scaling': 1.0,        # Scaling factor for plasticity updates
}

# Output paths for optimization
OPTIMIZATION_OUTPUT_DIR = 'results/optimization'
OPTIMIZATION_RESULTS_FILE = 'optimization_results.npy'
OPTIMIZATION_FIGURES_DIR = 'results/optimization/figures'
