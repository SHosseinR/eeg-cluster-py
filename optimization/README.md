# NSGA-II EEG Optimization Module

## What's Included

This directory contains a complete, production-ready implementation of multi-objective optimization for EEG connectivity analysis using the NSGA-II algorithm.

## Files

### Core Implementation
1. **optimization_config.py** - Configuration parameters
2. **state_space_simulation.py** - State-space dynamics with stimulation
3. **plasticity.py** - Plasticity-based connectivity updates
4. **nsga_optimizer.py** - NSGA-II algorithm implementation
5. **eeg_optimization.py** - EEG-specific optimization pipeline
6. **optimization_visualization.py** - Visualization functions
7. **run_optimization.py** - Main execution script
8. **test_optimization.py** - Comprehensive test suite

### Documentation
9. **IMPLEMENTATION_SUMMARY.md** - ⭐ **START HERE** - Complete overview
10. **QUICKSTART_OPTIMIZATION.md** - Quick start guide
11. **OPTIMIZATION_README.md** - Detailed documentation
12. **OPTIMIZATION_FILES_SUMMARY.md** - File descriptions
13. **requirements_optimization.txt** - Python dependencies

## Quick Start

### 1. Prerequisites
- Complete main pipeline Steps 1-5
- Ensure these files exist:
  - `results/data/connectivity_matrices.npy`
  - `results/data/network_measures.npy`

### 2. Installation
```bash
# Copy all files to your project directory
cp optimization_module/* /path/to/your/project/

# No additional packages needed (uses main pipeline dependencies)
```

### 3. Test (Optional)
```bash
python test_optimization.py
# Should see: "ALL TESTS PASSED ✓"
```

### 4. Run
```bash
python run_optimization.py
# Runtime: 2-10 hours for ~20 patients
```

## What It Does

1. **Optimizes** brain stimulation parameters (node and frequency) for each patient
2. **Uses** NSGA-II multi-objective genetic algorithm
3. **Simulates** brain dynamics with state-space models
4. **Updates** connectivity using plasticity rules
5. **Evaluates** network measures on updated connectivity
6. **Produces** optimal parameters, visualizations, and reports

## Key Features

✅ Multi-objective optimization (3 network measures)
✅ Biologically plausible (state-space + Hebbian plasticity)
✅ Fully modular and testable
✅ Comprehensive documentation
✅ Production-ready

## Output

After running, you'll get:
```
results/optimization/
├── optimization_results.npy       # Raw results
├── optimization_report.txt        # Summary report
└── figures/
    ├── optimal_nodes_histogram.png
    ├── optimal_bands_histogram.png
    ├── node_band_heatmap.png
    └── pareto_fronts_sample.png
```

## Documentation

- **New to this?** Read `IMPLEMENTATION_SUMMARY.md` first
- **Want quick setup?** See `QUICKSTART_OPTIMIZATION.md`
- **Need details?** Check `OPTIMIZATION_README.md`
- **Looking for specific file?** See `OPTIMIZATION_FILES_SUMMARY.md`

## Support

- Run tests: `python test_optimization.py`
- Check documentation files
- Review error messages carefully

## Statistics

- **Files:** 13 (8 code + 5 docs)
- **Code:** ~3000 lines
- **Documentation:** ~3300 lines
- **Test Coverage:** All major components
- **Status:** Production ready ✅

## License

See main project LICENSE.

---

**Ready to use. Just run `python run_optimization.py`**
