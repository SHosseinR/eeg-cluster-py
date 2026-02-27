# NSGA-II Optimization Module - File Summary

## Overview
Complete implementation of multi-objective optimization for EEG connectivity analysis using NSGA-II algorithm.

## Core Implementation Files

### 1. `optimization_config.py`
**Purpose:** Configuration parameters for optimization
**Contains:**
- Optimization measures selection (3 network measures)
- NSGA-II algorithm parameters (population size, generations, etc.)
- State-space simulation parameters (duration, amplitude, time step)
- Plasticity update parameters
- Output directory paths

**Key Variables:**
```python
OPTIMIZATION_MEASURES = ['global_efficiency', 'clustering_coefficient', 'modularity']
NSGA_CONFIG = {...}
SIMULATION_CONFIG = {...}
PLASTICITY_CONFIG = {...}
```

---

### 2. `state_space_simulation.py`
**Purpose:** State-space dynamics simulation with brain stimulation
**Key Functions:**
- `normalize_adjacency_matrix()` - Ensure stable dynamics
- `create_control_matrix()` - Define stimulation input
- `create_stimulation_signal()` - Generate stimulation signal
- `simulate_eeg_dynamics()` - Main simulation loop
- `run_full_simulation()` - Complete pipeline
- `compute_activation_changes()` - Compute node activation ratios

**Implements:** 
```
dx/dt = A*x - A*xbar + B*u
```

**Dependencies:** numpy, scipy

---

### 3. `plasticity.py`
**Purpose:** Plasticity-based connectivity updates
**Key Functions:**
- `apply_plasticity_updates()` - Update connectivity based on activation
- `normalize_connectivity_matrix()` - Normalize to [0,1] range
- `compute_plasticity_effect()` - Complete plasticity pipeline
- `analyze_plasticity_changes()` - Statistics on connectivity changes

**Implements:**
```
E_new(i,j) = E(i,j) * (R_i * R_j)
```

**Dependencies:** numpy

---

### 4. `nsga_optimizer.py`
**Purpose:** NSGA-II multi-objective genetic algorithm
**Key Classes:**
- `Individual` - Represents a solution (node, band, objectives)
- `NSGAIIOptimizer` - Main optimizer class

**Key Methods:**
- `initialize_population()` - Create initial random population
- `evaluate_population()` - Compute objectives
- `fast_non_dominated_sort()` - Pareto ranking
- `calculate_crowding_distance()` - Diversity preservation
- `tournament_selection()` - Parent selection
- `crossover()` - Combine parent solutions
- `mutate()` - Random variation
- `optimize()` - Main optimization loop
- `get_best_solution()` - Select single best from Pareto front

**Dependencies:** numpy, dataclasses

---

### 5. `eeg_optimization.py`
**Purpose:** EEG-specific optimization pipeline
**Key Class:**
- `EEGOptimizer` - Main optimization orchestrator

**Key Methods:**
- `_determine_optimization_directions()` - Min/max based on Patient vs Healthy
- `_compute_baseline_activation()` - Extract baseline from raw EEG
- `_create_evaluation_function()` - Create objective function for NSGA-II
- `optimize_subject()` - Optimize single patient
- `optimize_all_patients()` - Optimize all patients
- `save_results()` / `load_results()` - Persistence

**Integrates:** All above modules + network measures

**Dependencies:** numpy, all optimization modules

---

### 6. `optimization_visualization.py`
**Purpose:** Visualization of optimization results
**Key Functions:**
- `plot_node_histogram()` - Distribution of optimal nodes
- `plot_band_histogram()` - Distribution of optimal bands
- `plot_node_band_heatmap()` - 2D node × band distribution
- `plot_pareto_fronts()` - Pareto fronts for sample subjects
- `plot_optimization_summary()` - Generate all plots
- `create_optimization_report()` - Text summary report

**Outputs:** PNG figures and TXT report

**Dependencies:** numpy, matplotlib, seaborn

---

### 7. `run_optimization.py`
**Purpose:** Main execution script
**Workflow:**
1. Create output directories
2. Load data (connectivity, measures, raw EEG)
3. Verify requirements
4. Create optimizer
5. Run optimization for all patients
6. Save results
7. Generate visualizations and report

**Usage:** `python run_optimization.py`

**Dependencies:** All optimization modules + data_loader, config

---

## Testing and Documentation

### 8. `test_optimization.py`
**Purpose:** Comprehensive test suite
**Tests:**
- State-space simulation components
- Plasticity updates
- NSGA-II optimizer
- Full integration with synthetic data

**Usage:** `python test_optimization.py`

**Expected Output:** All tests should PASS

---

### 9. `OPTIMIZATION_README.md`
**Purpose:** Complete documentation
**Sections:**
- Overview and features
- Workflow description
- Installation instructions
- Configuration guide
- Usage examples
- Theory and methods
- Troubleshooting
- Performance notes

**Audience:** Users and developers

---

### 10. `QUICKSTART_OPTIMIZATION.md`
**Purpose:** Quick integration guide
**Sections:**
- Prerequisites checklist
- Step-by-step setup
- Integration with main pipeline
- Customization examples
- Troubleshooting tips
- Expected outputs

**Audience:** Users wanting quick setup

---

### 11. `requirements_optimization.txt`
**Purpose:** Python dependencies
**Contents:**
- Core: numpy, scipy, matplotlib, seaborn
- Optional: tqdm, plotly, networkx

**Note:** Most dependencies already in main requirements.txt

---

## Data Flow

```
Input Data (from main pipeline):
├── connectivity_matrices.npy
│   └── [group][subject][method][band] -> adjacency matrix
├── network_measures.npy
│   └── [group][subject][method][band][measure] -> scalar value
└── Raw EEG data (.set files)
    └── subject['data'] -> (n_channels, n_samples)

↓

Optimization Process:
├── For each Patient subject:
│   ├── Compute baseline activation
│   ├── Run NSGA-II:
│   │   ├── Test (node, band) combinations
│   │   ├── Simulate dynamics
│   │   ├── Apply plasticity
│   │   └── Evaluate objectives
│   └── Return Pareto-optimal solutions
└── Aggregate results across subjects

↓

Output:
├── optimization_results.npy
│   └── [subject_id] -> {best_solution, best_front, history, ...}
├── figures/
│   ├── optimal_nodes_histogram.png
│   ├── optimal_bands_histogram.png
│   ├── node_band_heatmap.png
│   └── pareto_fronts_sample.png
└── optimization_report.txt
```

## File Dependencies

```
run_optimization.py
├── config.py (main pipeline config)
├── optimization_config.py
├── data_loader.py (from main pipeline)
├── eeg_optimization.py
│   ├── state_space_simulation.py
│   ├── plasticity.py
│   ├── nsga_optimizer.py
│   └── network_measures.py (from main pipeline)
└── optimization_visualization.py
```

## Directory Structure

```
project/
├── config.py                          # Main config (update with optimization params)
├── optimization_config.py             # Optimization-specific config
├── state_space_simulation.py          # State-space dynamics
├── plasticity.py                      # Connectivity updates
├── nsga_optimizer.py                  # NSGA-II algorithm
├── eeg_optimization.py                # EEG optimization pipeline
├── optimization_visualization.py      # Visualization functions
├── run_optimization.py                # Main execution script
├── test_optimization.py               # Test suite
├── OPTIMIZATION_README.md             # Full documentation
├── QUICKSTART_OPTIMIZATION.md         # Quick start guide
├── OPTIMIZATION_FILES_SUMMARY.md      # This file
└── requirements_optimization.txt      # Python dependencies
```

## Integration with Main Pipeline

### Required from Main Pipeline:
- `config.py` - Configuration variables
- `data_loader.py` - Data loading functions
- `network_measures.py` - Network measure computation
- `connectivity_matrices.npy` - Pre-computed connectivity
- `network_measures.npy` - Pre-computed measures

### Adds to Main Pipeline:
- Step 6: NSGA-II Optimization
- New output directory: `results/optimization/`

## Key Concepts

### NSGA-II Algorithm
- **Multi-objective:** Optimizes multiple measures simultaneously
- **Pareto optimization:** Finds trade-off solutions
- **Genetic algorithm:** Uses evolution (selection, crossover, mutation)
- **Elitism:** Preserves best solutions

### State-Space Simulation
- **Linear dynamics:** dx/dt = A*x - A*xbar + B*u
- **Stimulation:** Input applied to specific node
- **Stability:** Matrix normalization ensures convergence

### Plasticity
- **Hebbian:** "Neurons that fire together, wire together"
- **Bidirectional:** Increases and decreases connections
- **Activity-dependent:** Based on node activation changes

### Optimization Variables
- **Node:** Which electrode to stimulate (discrete)
- **Band:** Which frequency band to use (discrete)

### Objectives
- **3 network measures:** Selected from 12 available
- **Direction:** Minimize or maximize based on Patient vs Healthy
- **Goal:** Move Patient connectivity toward Healthy

## Usage Patterns

### Pattern 1: Standard Run
```bash
# After running main pipeline Steps 1-5
python run_optimization.py
```

### Pattern 2: Custom Configuration
```python
# Modify optimization_config.py
OPTIMIZATION_MEASURES = ['local_efficiency', 'modularity', 'rich_club']
NSGA_CONFIG['population_size'] = 50
NSGA_CONFIG['n_generations'] = 25

# Then run
python run_optimization.py
```

### Pattern 3: Programmatic Use
```python
from eeg_optimization import create_optimizer_from_config

optimizer = create_optimizer_from_config(...)
results = optimizer.optimize_all_patients()
optimizer.save_results('my_results.npy')
```

### Pattern 4: Analysis Only
```python
import numpy as np
results = np.load('optimization_results.npy', allow_pickle=True).item()

# Analyze results
for subject_id, res in results.items():
    print(f"{subject_id}: Node {res['best_solution'].node}")
```

## Performance Characteristics

### Time Complexity
- **Per evaluation:** O(n_nodes × n_timesteps) for simulation
- **Per generation:** O(population_size × n_objectives) for sorting
- **Total:** O(n_generations × population_size × evaluation_time)

### Space Complexity
- **Population:** O(population_size × n_objectives)
- **Simulation:** O(n_nodes × n_timesteps)
- **History:** O(n_generations × Pareto_front_size)

### Typical Runtime
- Small (10 nodes, 3 bands): ~5 min per subject
- Medium (20 nodes, 5 bands): ~15 min per subject
- Large (64 nodes, 5 bands): ~30 min per subject

## Version History

- **v1.0** (2024): Initial implementation
  - NSGA-II optimization
  - State-space simulation
  - Plasticity updates
  - Comprehensive visualization

## Future Enhancements

Potential additions:
- [ ] Parallel processing for multiple subjects
- [ ] GPU acceleration for simulation
- [ ] Alternative genetic operators
- [ ] Constraint handling (e.g., avoid certain nodes)
- [ ] Multi-node stimulation
- [ ] Time-varying stimulation patterns
- [ ] Validation against experimental data

## License and Citation

See main project LICENSE and README for details.

## Contact

For questions about specific files:
- Algorithm: `nsga_optimizer.py`
- Simulation: `state_space_simulation.py`
- Integration: `eeg_optimization.py`
- Usage: `QUICKSTART_OPTIMIZATION.md`

---

**Last Updated:** 2024
**Total Lines of Code:** ~3000
**Test Coverage:** All major components
