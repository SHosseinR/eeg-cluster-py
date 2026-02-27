# Quick Start Guide: NSGA-II Optimization

## Prerequisites

1. **Complete main pipeline first** (Steps 1-5)
   - This generates the required `connectivity_matrices.npy` and `network_measures.npy` files

2. **Ensure data files exist:**
   ```
   results/data/connectivity_matrices.npy
   results/data/network_measures.npy
   ```

## Step-by-Step Guide

### 1. Add Configuration to `config.py`

Add these lines to your existing `config.py`:

```python
# Optimization parameters
OPTIMIZATION_MEASURES = [
    'global_efficiency',
    'clustering_coefficient', 
    'modularity'
]

# NSGA-II Configuration
NSGA_CONFIG = {
    'population_size': 100,
    'n_generations': 50,
    'crossover_prob': 0.9,
    'mutation_prob': 0.1,
    'tournament_size': 3,
}

# Simulation parameters
SIMULATION_CONFIG = {
    'stimulation_duration': 1.0,
    'stimulation_amplitude': 1.0,
    'dt': 0.001,
    'stability_constant': 0.01,
}

# Plasticity parameters
PLASTICITY_CONFIG = {
    'plasticity_enabled': True,
    'plasticity_scaling': 1.0,
}

# Output directories
OPTIMIZATION_OUTPUT_DIR = 'results/optimization'
OPTIMIZATION_RESULTS_FILE = 'optimization_results.npy'
OPTIMIZATION_FIGURES_DIR = 'results/optimization/figures'
```

### 2. Copy Optimization Files

Copy these files to your project directory:
```
optimization_config.py
state_space_simulation.py
plasticity.py
nsga_optimizer.py
eeg_optimization.py
optimization_visualization.py
run_optimization.py
```

### 3. Run Tests (Optional but Recommended)

```bash
python test_optimization.py
```

This verifies all components work correctly.

### 4. Run Optimization

```bash
python run_optimization.py
```

**Expected runtime:** 2-10 hours for ~20 patient subjects

**To test with shorter runtime:**
Edit `optimization_config.py`:
```python
NSGA_CONFIG = {
    'population_size': 20,    # Smaller population
    'n_generations': 10,      # Fewer generations
    ...
}
```

### 5. View Results

Results will be saved to:
- `results/optimization/optimization_results.npy` - Raw results
- `results/optimization/figures/` - Visualization plots
- `results/optimization/optimization_report.txt` - Text summary

## Integration with Existing Pipeline

### Option A: Add to `main.py`

Add this section after Step 5 (Network Measures):

```python
# ========================================================================
# STEP 6: NSGA-II OPTIMIZATION
# ========================================================================
print("\n" + "="*80)
print("STEP 6: NSGA-II OPTIMIZATION")
print("="*80)

if STEP_TO_START <= 6:
    from eeg_optimization import create_optimizer_from_config
    from optimization_visualization import (
        plot_optimization_summary, create_optimization_report
    )
    
    # Prepare subject data
    subject_data = {}
    for group_data, group_name in [(healthy_data, "Healthy"), (patient_data, "Patient")]:
        for subject in group_data:
            subject_data[subject['subject_id']] = {
                'data': subject['data'],
                'fs': subject['fs'],
                'channels': subject['channels'],
                'group': group_name
            }
    
    # Create optimizer
    optimizer = create_optimizer_from_config(
        connectivity_matrices=connectivity_matrices,
        network_measures=network_measures,
        subject_data=subject_data,
        frequency_bands=FREQUENCY_BANDS,
        channel_names=healthy_data[0]['channels'],
        selected_method=SELECTED_METHOD
    )
    
    # Run optimization
    optimization_results = optimizer.optimize_all_patients(verbose=True)
    
    # Save results
    os.makedirs(OPTIMIZATION_OUTPUT_DIR, exist_ok=True)
    optimizer.save_results(
        os.path.join(OPTIMIZATION_OUTPUT_DIR, OPTIMIZATION_RESULTS_FILE)
    )
    
    # Generate visualizations
    plot_optimization_summary(
        optimization_results=optimization_results,
        channel_names=healthy_data[0]['channels'],
        band_names=list(FREQUENCY_BANDS.keys()),
        optimization_measures=OPTIMIZATION_MEASURES,
        output_dir=OPTIMIZATION_FIGURES_DIR
    )
    
    # Create report
    create_optimization_report(
        optimization_results=optimization_results,
        channel_names=healthy_data[0]['channels'],
        band_names=list(FREQUENCY_BANDS.keys()),
        optimization_measures=OPTIMIZATION_MEASURES,
        optimization_directions=optimizer.optimization_directions,
        output_path=os.path.join(OPTIMIZATION_OUTPUT_DIR, 'optimization_report.txt')
    )
```

### Option B: Run Separately

Keep optimization separate and run after main pipeline:

```bash
# Run main pipeline first
python main.py

# Then run optimization
python run_optimization.py
```

## Customization Examples

### Change Optimization Measures

Edit `optimization_config.py`:
```python
OPTIMIZATION_MEASURES = [
    'local_efficiency',      # Replace with different measures
    'betweenness_centrality',
    'small_worldness'
]
```

Available measures:
- `global_efficiency`
- `local_efficiency`
- `clustering_coefficient`
- `transitivity`
- `modularity`
- `degree`
- `betweenness_centrality`
- `rich_club`
- `assortativity`
- `spectral_radius`
- `small_worldness`
- `diameter`

### Adjust Algorithm Parameters

Edit `optimization_config.py`:

```python
# For faster but less thorough optimization
NSGA_CONFIG = {
    'population_size': 50,     # Smaller population
    'n_generations': 25,       # Fewer generations
    ...
}

# For more thorough but slower optimization
NSGA_CONFIG = {
    'population_size': 200,    # Larger population
    'n_generations': 100,      # More generations
    ...
}
```

### Modify Simulation Parameters

Edit `optimization_config.py`:

```python
SIMULATION_CONFIG = {
    'stimulation_duration': 2.0,    # Longer stimulation
    'stimulation_amplitude': 0.5,   # Weaker stimulation
    'dt': 0.001,                    # Finer time steps
    'stability_constant': 0.05,     # More stable dynamics
}
```

### Disable Plasticity

Edit `optimization_config.py`:

```python
PLASTICITY_CONFIG = {
    'plasticity_enabled': False,    # Turn off plasticity
}
```

## Troubleshooting

### Error: "Connectivity matrices not found"

**Solution:** Run main pipeline first (Steps 1-5) to generate required files.

### Error: "Memory error"

**Solutions:**
- Reduce `population_size` to 50 or less
- Process fewer subjects at a time
- Increase system swap space

### Error: "Unstable dynamics"

**Solutions:**
- Increase `stability_constant` to 0.05 or 0.1
- Reduce `stimulation_amplitude`
- Check for NaN/Inf in connectivity matrices

### Optimization is too slow

**Solutions:**
- Reduce `population_size` (try 20-50 for testing)
- Reduce `n_generations` (try 10-20 for testing)
- Use fewer patient subjects for initial testing

### Results don't make sense

**Solutions:**
- Check that Patient and Healthy groups show clear differences in Step 4
- Verify selected network measures are appropriate
- Try different optimization measures
- Increase `plasticity_scaling` for stronger effects

## Expected Outputs

After successful optimization, you should see:

1. **Console output** showing:
   - Optimization directions for each measure
   - Progress for each patient subject
   - Pareto front sizes
   - Best solutions

2. **Files generated:**
   ```
   results/optimization/
   ├── optimization_results.npy       # Raw results
   ├── optimization_report.txt        # Text summary
   └── figures/
       ├── optimal_nodes_histogram.png
       ├── optimal_bands_histogram.png
       ├── node_band_heatmap.png
       └── pareto_fronts_sample.png
   ```

3. **Key insights from results:**
   - Which brain regions (nodes) are most frequently optimal
   - Which frequency bands are most important
   - Trade-offs between objectives (Pareto fronts)

## Next Steps

After running optimization:

1. **Analyze node distribution** - Which brain regions are optimal targets?
2. **Examine band preferences** - Which frequencies are most important?
3. **Study Pareto fronts** - What are the trade-offs?
4. **Compare with literature** - Do results match known targets?
5. **Clinical validation** - Test predicted targets experimentally

## Advanced Usage

### Load and Analyze Previous Results

```python
import numpy as np
from optimization_visualization import plot_optimization_summary

# Load results
results = np.load('results/optimization/optimization_results.npy', 
                  allow_pickle=True).item()

# Analyze specific subject
subject_id = 'P1'
best_sol = results[subject_id]['best_solution']
print(f"Node: {best_sol.node}, Band: {best_sol.band}")
print(f"Objectives: {best_sol.objectives}")

# Plot Pareto front
front = results[subject_id]['best_front']
objectives = np.array([ind.objectives for ind in front])
import matplotlib.pyplot as plt
plt.scatter(objectives[:, 0], objectives[:, 1])
plt.show()
```

### Run Optimization Programmatically

```python
from eeg_optimization import EEGOptimizer

# Create optimizer
optimizer = EEGOptimizer(
    connectivity_matrices=conn_matrices,
    network_measures=measures,
    subject_data=subj_data,
    frequency_bands=bands,
    channel_names=channels,
    selected_method='plv',
    optimization_measures=['global_efficiency', 'modularity', 'clustering_coefficient']
)

# Optimize specific subject
results = optimizer.optimize_subject('P1', verbose=True)

# Or optimize all patients
all_results = optimizer.optimize_all_patients(verbose=True)

# Save
optimizer.save_results('my_results.npy')
```

## Support

For issues or questions:
1. Check the main `OPTIMIZATION_README.md`
2. Run `test_optimization.py` to verify installation
3. Review error messages carefully
4. Check configuration parameters

## Performance Tips

1. **Start small:** Test with 10 generations first
2. **Profile code:** Identify bottlenecks if too slow
3. **Use good hardware:** Multi-core CPU helps
4. **Monitor memory:** Watch for memory leaks
5. **Save intermediate results:** Don't lose progress

## References

- NSGA-II paper: Deb et al. (2002)
- State-space models: Strogatz (2015)
- Brain connectivity: Bullmore & Sporns (2009)
