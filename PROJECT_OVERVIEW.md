# EEG Connectivity Analysis - Project Overview

## 📋 Project Summary

This is a complete, production-ready pipeline for analyzing functional connectivity in EEG data with the following capabilities:

- **Multi-modal connectivity analysis** (4 methods: PDC, GC, PSI, PLV)
- **Comprehensive network analysis** (12 graph theory measures)
- **Statistical group comparisons** (healthy vs. patient)
- **Machine learning classification** (feature selection with cross-validation)
- **Publication-ready visualizations** (6 types of plots)

## 📁 File Structure

### Core Pipeline Modules
```
├── config.py                 # All configuration parameters
├── data_loader.py           # Load EEGLAB .set files
├── signal_processing.py     # Epoching and frequency filtering
├── connectivity.py          # 4 connectivity methods
├── network_measures.py      # 12 network metrics
├── statistics.py            # Statistical tests and comparisons
├── visualization.py         # All 6 visualizations
├── classification.py        # ML feature selection
└── main.py                  # Complete pipeline orchestration
```

### Supporting Files
```
├── requirements.txt         # Python dependencies
├── README.md               # Complete usage guide
├── TROUBLESHOOTING.md      # Problem-solving guide
├── setup.py                # Installation verification
├── examples.py             # Usage examples
└── utils.py                # Helper functions
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python setup.py  # Verify installation
```

### 2. Configure Data Paths
Edit `config.py`:
```python
HC_DATA_PATH = "/path/to/healthy/controls"
PATIENT_DATA_PATH = "/path/to/patients"
```

### 3. Run Complete Pipeline
```bash
python main.py
```

This will:
- ✅ Load all subject data
- ✅ Create 10-second epochs
- ✅ Filter into 5 frequency bands
- ✅ Compute connectivity (4 methods)
- ✅ Calculate network measures
- ✅ Perform statistical tests
- ✅ Run feature selection
- ✅ Generate all visualizations
- ✅ Save results to `./results/`

## 📊 Pipeline Workflow

```
RAW EEG DATA (.set files)
    ↓
EPOCHING (10-second chunks)
    ↓
FILTERING (5 frequency bands: delta, theta, alpha, beta, gamma)
    ↓
CONNECTIVITY ANALYSIS (4 methods: PDC, GC, PSI, PLV)
    ↓
NORMALIZATION (per subject, per band, per method)
    ↓
NETWORK MEASURES (12 metrics per graph)
    ↓
STATISTICAL TESTS (group comparisons)
    ↓
FEATURE SELECTION (combinations of 3 features)
    ↓
CLASSIFICATION (5-fold CV, logistic regression)
    ↓
VISUALIZATIONS & REPORTS
```

## 📈 Output Files

### Visualizations (in `results/figures/`)
1. **viz1_connectivity_matrices.png** - Average connectivity per method
2. **viz2_pvalue_matrices.png** - Statistical significance per method
3. **viz3_pvalue_per_band.png** - Significance per frequency band
4. **viz4_network_pvalues.png** - Group comparison p-values
5. **viz5_top_triplets.png** - Best feature combinations
6. **viz6_feature_importance.png** - Feature weights

### Data Files (in `results/data/`)
- **connectivity_matrices.npy** - All connectivity matrices (normalized)
- **network_measures.npy** - All network metrics
- **network_measures_pvalues.csv** - Statistical test results
- **top_feature_triplets.csv** - Classification results

### Reports (in `results/reports/`)
- **classification_report.txt** - ML analysis summary
- **summary_report.png** - Overall analysis overview

## 🔧 Customization Options

### Change Selected Method for Network Analysis
After reviewing visualization 2 (p-value matrices):
```python
# In config.py
SELECTED_METHOD = 'gc'  # Options: 'pdc', 'gc', 'psi', 'plv'
```

### Adjust Epoch Duration
```python
# In config.py
EPOCH_DURATION = 10.0  # seconds
```

### Modify Frequency Bands
```python
# In config.py
FREQUENCY_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}
```

### Add/Remove Network Measures
```python
# In config.py
NETWORK_MEASURES = [
    'global_efficiency',
    'local_efficiency',
    # Add your measures here
]
```

### Classification Parameters
```python
# In config.py
N_FEATURES_COMBINATION = 3  # Features per combination
N_FOLDS = 5                 # CV folds
N_TOP_FEATURES = 10         # Top combinations to report
```

## 🔬 Technical Details

### Connectivity Methods

| Method | Type | Description | Use Case |
|--------|------|-------------|----------|
| **PLV** | Undirected | Phase synchronization | General connectivity |
| **PSI** | Directed | Time delay without amplitude | Causal relationships |
| **GC** | Directed | Predictive information flow | Effective connectivity |
| **PDC** | Directed | Direct vs indirect connections | Network connectivity |

### Network Measures

| Measure | Type | Description |
|---------|------|-------------|
| Global Efficiency | Global | Information integration |
| Local Efficiency | Local | Fault tolerance |
| Clustering Coefficient | Local | Local clustering |
| Transitivity | Global | Overall clustering |
| Modularity | Global | Community structure |
| Degree | Nodal | Connectivity strength |
| Betweenness Centrality | Nodal | Information flow |
| Rich Club | Global | Hub connectivity |
| Assortativity | Global | Homophily |
| Spectral Radius | Global | Network stability |
| Small-worldness | Global | Integration/segregation |
| Diameter | Global | Communication distance |

### Statistical Tests
- **Group comparison**: Mann-Whitney U (non-parametric)
- **Connectivity significance**: One-sample t-test vs. zero
- **Multiple comparison correction**: Available (Bonferroni, FDR)

## 💡 Usage Tips

### For Small Datasets (< 10 subjects per group)
- Reduce cross-validation folds: `N_FOLDS = 3`
- Use permutation tests for more robust statistics
- Consider leave-one-out cross-validation

### For Large Datasets (> 50 subjects)
- Process in batches to manage memory
- Use parallel processing (see examples.py)
- Consider subsampling for initial exploration

### For High-Density EEG (> 64 channels)
- May need more memory
- Consider spatial downsampling
- Network measures will take longer

### For Clinical Studies
- Always check data quality first (use `utils.check_data_quality()`)
- Verify preprocessing before running pipeline
- Document all parameter choices
- Consider multiple comparison corrections

## 🐛 Troubleshooting

See `TROUBLESHOOTING.md` for detailed solutions to common issues:
- Installation problems
- Memory errors
- Connectivity computation issues
- Classification problems
- Visualization issues

## 📚 Key References

### Methods
- **Granger Causality**: Granger (1969), Econometrica
- **Phase Slope Index**: Nolte et al. (2008), Phys Rev Lett
- **PLV**: Lachaux et al. (1999), Hum Brain Mapp

### Software
- **MNE-Python**: Gramfort et al. (2013), Front Neurosci
- **Brain Connectivity Toolbox**: Rubinov & Sporns (2010), Neuroimage

## 📝 Citation

If you use this pipeline in your research, please cite:
```
[Your citation information]
```

## 🤝 Contributing

To extend the pipeline:
1. Add new connectivity method in `connectivity.py`
2. Add new network measure in `network_measures.py`
3. Update `config.py` with new parameters
4. Add tests in `examples.py`

## 📄 License

[Your license information]

## 👥 Authors & Contact

[Your information]

## ✅ Validation Status

- [x] Installation tested on Python 3.7+
- [x] All dependencies verified
- [x] Example data processing works
- [x] Statistical tests validated
- [x] Visualizations render correctly
- [x] Classification pipeline functional

## 🎯 Next Steps

1. **Setup**: Run `python setup.py` to verify installation
2. **Configure**: Update paths in `config.py`
3. **Explore**: Try `python examples.py` with sample data
4. **Analyze**: Run `python main.py` for complete pipeline
5. **Customize**: Modify parameters based on your needs
6. **Extend**: Add custom methods or measures as needed

---

**Version**: 1.0  
**Last Updated**: February 2026  
**Status**: Production Ready ✅
