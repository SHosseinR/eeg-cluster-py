# EEG Connectivity Analysis Pipeline

A comprehensive pipeline for analyzing functional connectivity in EEG data, comparing healthy controls with patients using network analysis and machine learning.

## Overview

This pipeline performs:
1. **Data loading** from EEGLAB .set files
2. **Signal processing** (epoching and frequency band filtering)
3. **Connectivity analysis** using multiple methods (PDC, Granger Causality, PSI, PLV)
4. **Network measures** computation using graph theory
5. **Statistical analysis** comparing groups
6. **Machine learning** for feature selection and classification
7. **Comprehensive visualizations** of results

## Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv eeg_env
source eeg_env/bin/activate  # On Windows: eeg_env\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify installation

```python
python -c "import mne; import bct; print('Installation successful!')"
```

## Directory Structure

```
.
├── config.py                 # Configuration and parameters
├── data_loader.py           # Data loading utilities
├── signal_processing.py     # Epoching and filtering
├── connectivity.py          # Connectivity analysis
├── network_measures.py      # Graph theory metrics
├── statistics.py            # Statistical tests
├── visualization.py         # Plotting functions
├── classification.py        # Machine learning
├── main.py                  # Main pipeline
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Configuration

Before running, update `config.py` with your data paths:

```python
HC_DATA_PATH = "/path/to/healthy/controls"
PATIENT_DATA_PATH = "/path/to/patients"
OUTPUT_DIR = "./results"
```

### Key Parameters

- **Epoch Duration**: 10 seconds (configurable in `config.py`)
- **Frequency Bands**: 
  - Delta: 1-4 Hz
  - Theta: 4-8 Hz
  - Alpha: 8-13 Hz
  - Beta: 13-30 Hz
  - Gamma: 30-45 Hz
- **Connectivity Methods**: PDC, Granger Causality, PSI, PLV
- **Network Measures**: 12 different graph metrics
- **Classification**: 5-fold cross-validation with logistic regression

## Usage

### Basic Usage

Run the complete pipeline:

```bash
python main.py
```

This will:
1. Load data from both groups
2. Process and filter EEG signals
3. Compute connectivity matrices
4. Calculate network measures
5. Perform statistical comparisons
6. Run feature selection
7. Generate all visualizations
8. Save results to the output directory

### Output

The pipeline generates:

#### Figures (in `results/figures/`)
- `viz1_connectivity_matrices.png` - Average connectivity per method
- `viz2_pvalue_matrices.png` - Statistical significance per method
- `viz3_pvalue_per_band.png` - Statistical significance per frequency band
- `viz4_network_pvalues.png` - Group comparison p-values
- `viz5_top_triplets.png` - Best feature combinations
- `viz6_feature_importance.png` - Feature importance for classification

#### Data (in `results/data/`)
- `connectivity_matrices.npy` - All connectivity matrices
- `network_measures.npy` - All network metrics
- `network_measures_pvalues.csv` - Statistical test results
- `top_feature_triplets.csv` - Best feature combinations

#### Reports (in `results/reports/`)
- `classification_report.txt` - Classification analysis summary
- `summary_report.png` - Overall analysis summary

## Customization

### Changing Connectivity Method for Network Analysis

After reviewing Visualization 2 (p-value matrices), you can change the selected method:

```python
# In config.py
SELECTED_METHOD = 'gc'  # Options: 'pdc', 'gc', 'psi', 'plv'
```

### Adding Network Measures

Edit `NETWORK_MEASURES` in `config.py`:

```python
NETWORK_MEASURES = [
    'global_efficiency',
    'local_efficiency',
    # Add your custom measure here
]
```

Then implement the function in `network_measures.py`.

### Adjusting Classification Parameters

```python
# In config.py
N_FEATURES_COMBINATION = 3  # Number of features per combination
N_FOLDS = 5                 # Cross-validation folds
N_TOP_FEATURES = 10         # Top combinations to report
```

## Data Format

### Input Data Structure

```
data_directory/
├── subject_01/
│   ├── file1.set
│   ├── file2.set
│   └── ...
├── subject_02/
│   └── ...
```

### Requirements
- Data must be in EEGLAB .set format
- Each subject has a separate directory
- Multiple .set files per subject are automatically concatenated
- Preprocessing should be done beforehand (no preprocessing is performed)

### Channels
The following channels are automatically dropped:
- `23A-23R`
- `24A-24R`
- `A2-A1`

Modify `CHANNELS_TO_DROP` in `config.py` to change this.

## Connectivity Methods

### 1. Phase Locking Value (PLV)
- **Type**: Undirected
- **Measures**: Phase synchronization
- **Good for**: General connectivity patterns

### 2. Phase Slope Index (PSI)
- **Type**: Directed
- **Measures**: Time delays between signals
- **Good for**: Causal relationships without amplitude effects

### 3. Granger Causality (GC)
- **Type**: Directed
- **Measures**: Predictive information flow
- **Good for**: Effective connectivity

### 4. Partial Directed Coherence (PDC)
- **Type**: Directed
- **Measures**: Direct vs indirect connections
- **Good for**: Network-level connectivity

## Network Measures

1. **Global Efficiency**: Information integration across the network
2. **Local Efficiency**: Fault tolerance and local information processing
3. **Clustering Coefficient**: Tendency to form local clusters
4. **Transitivity**: Overall clustering tendency
5. **Modularity**: Community structure strength
6. **Degree**: Average connectivity strength
7. **Betweenness Centrality**: Importance of nodes in information flow
8. **Rich Club**: Tendency for hubs to connect to each other
9. **Assortativity**: Tendency for similar nodes to connect
10. **Spectral Radius**: Network stability indicator
11. **Small-worldness**: Balance of integration and segregation
12. **Diameter**: Maximum communication distance

## Statistical Analysis

### Group Comparisons
- **Test**: Mann-Whitney U (non-parametric)
- **Hypothesis**: Two groups have different distributions
- **Correction**: Available through `correct_multiple_comparisons()`

### Connectivity Significance
- **Test**: One-sample t-test against zero
- **Purpose**: Identify significant connections
- **Visualization**: P-value heatmaps

## Troubleshooting

### Memory Issues
If you encounter memory errors:
1. Process subjects in batches
2. Reduce epoch duration
3. Use fewer frequency bands

### PDC Not Available
If PDC computation fails:
- The pipeline uses GC time-reversed as approximation
- Install specialized PDC packages if needed
- Or use other connectivity methods

### Convergence Warnings in Classification
If logistic regression doesn't converge:
```python
# In classification.py, increase max_iter
clf = LogisticRegression(random_state=random_state, max_iter=2000)
```

## Citation

If you use this pipeline in your research, please cite:

```
[Your citation information here]
```

## References

Key papers for the methods:
- **MNE-Python**: Gramfort et al., 2013
- **Brain Connectivity Toolbox**: Rubinov & Sporns, 2010
- **Granger Causality**: Granger, 1969
- **Phase Slope Index**: Nolte et al., 2008
- **PLV**: Lachaux et al., 1999

## License

[Your license information]

## Contact

[Your contact information]

## Acknowledgments

This pipeline uses several excellent open-source packages:
- MNE-Python for EEG processing
- Brain Connectivity Toolbox Python (bctpy) for network analysis
- scikit-learn for machine learning
- matplotlib and seaborn for visualization
