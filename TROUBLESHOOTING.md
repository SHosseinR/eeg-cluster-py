# TROUBLESHOOTING GUIDE

## Table of Contents
1. [Installation Issues](#installation-issues)
2. [Data Loading Problems](#data-loading-problems)
3. [Memory Errors](#memory-errors)
4. [Connectivity Computation Issues](#connectivity-computation-issues)
5. [Network Measures Problems](#network-measures-problems)
6. [Classification Issues](#classification-issues)
7. [Visualization Problems](#visualization-problems)
8. [Performance Optimization](#performance-optimization)

---

## Installation Issues

### Problem: Package Installation Fails

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement mne-connectivity
```

**Solutions:**
1. Update pip:
   ```bash
   pip install --upgrade pip
   ```

2. Install packages individually:
   ```bash
   pip install numpy scipy pandas
   pip install matplotlib seaborn
   pip install mne
   pip install mne-connectivity
   pip install bctpy
   pip install scikit-learn statsmodels
   ```

3. Use conda (alternative):
   ```bash
   conda install -c conda-forge mne mne-connectivity
   conda install scikit-learn matplotlib seaborn pandas
   pip install bctpy
   ```

### Problem: BCT Installation Fails

**Symptoms:**
```
ERROR: Could not build wheels for bctpy
```

**Solutions:**
1. Install from GitHub:
   ```bash
   pip install git+https://github.com/aestrivex/bctpy.git
   ```

2. Or install dependencies first:
   ```bash
   pip install numpy networkx
   pip install bctpy
   ```

---

## Data Loading Problems

### Problem: "No .set files found"

**Symptoms:**
```
ValueError: No .set files found in /path/to/data
```

**Solutions:**
1. Check data path in `config.py`:
   ```python
   HC_DATA_PATH = "/correct/path/to/healthy/controls"
   PATIENT_DATA_PATH = "/correct/path/to/patients"
   ```

2. Verify directory structure:
   ```
   data/
   ├── subject_01/
   │   ├── file1.set
   │   └── file1.fdt
   ```

3. Check file permissions:
   ```bash
   ls -la /path/to/data
   ```

### Problem: Channel Mismatch Between Files

**Symptoms:**
```
ValueError: Channel names mismatch in subject_01/file2.set
```

**Solutions:**
1. Ensure all files have same channels
2. Add problematic channels to `CHANNELS_TO_DROP` in `config.py`
3. Manually standardize channels before loading

### Problem: Sampling Frequency Mismatch

**Symptoms:**
```
ValueError: Sampling frequency mismatch in file2.set
```

**Solutions:**
1. Resample files to same frequency before loading
2. In Python:
   ```python
   raw = mne.io.read_raw_eeglab('file.set', preload=True)
   raw_resampled = raw.resample(sfreq=250)
   raw_resampled.save('file_resampled.set')
   ```

---

## Memory Errors

### Problem: Out of Memory During Processing

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Process fewer subjects at once:**
   ```python
   # In main.py, process in batches
   for i in range(0, len(healthy_data), 5):
       batch = healthy_data[i:i+5]
       # Process batch
   ```

2. **Reduce epoch duration:**
   ```python
   # In config.py
   EPOCH_DURATION = 5.0  # Instead of 10.0
   ```

3. **Process fewer frequency bands:**
   ```python
   # In config.py
   FREQUENCY_BANDS = {
       'alpha': (8, 13),
       'beta': (13, 30)
   }
   ```

4. **Use float32 instead of float64:**
   ```python
   data = data.astype(np.float32)
   ```

5. **Clear memory between subjects:**
   ```python
   import gc
   gc.collect()
   ```

---

## Connectivity Computation Issues

### Problem: PDC Computation Fails

**Symptoms:**
```
Warning: PDC computation failed. Using placeholder.
```

**Solutions:**
1. Use alternative method temporarily:
   ```python
   CONNECTIVITY_METHODS = ['gc', 'psi', 'plv']  # Remove 'pdc'
   ```

2. Install specialized PDC package (if available)

3. Use GC time-reversed as approximation (already implemented as fallback)

### Problem: "Connectivity matrix all zeros"

**Symptoms:**
```
connectivity_matrix contains only zeros
```

**Solutions:**
1. Check if data has enough signal:
   ```python
   print(f"Data range: {np.min(data)} to {np.max(data)}")
   print(f"Data std: {np.std(data)}")
   ```

2. Verify frequency band matches data:
   ```python
   # Don't filter for gamma (30-45 Hz) if sampling rate is only 100 Hz
   ```

3. Check epoch duration is sufficient:
   ```python
   # Use at least 2 seconds for delta band
   EPOCH_DURATION = 10.0
   ```

### Problem: NaN values in connectivity matrices

**Symptoms:**
```
RuntimeWarning: invalid value encountered in connectivity computation
```

**Solutions:**
1. Remove NaN from input data:
   ```python
   data = np.nan_to_num(data, nan=0.0)
   ```

2. Check for flat channels:
   ```python
   from utils import check_data_quality
   quality_report = check_data_quality(data, fs, subject_id)
   ```

---

## Network Measures Problems

### Problem: Network Measure Returns NaN

**Symptoms:**
```
modularity: nan
rich_club: nan
```

**Solutions:**
1. **Check if matrix is valid:**
   ```python
   print(f"Matrix range: {np.min(matrix)} to {np.max(matrix)}")
   print(f"Matrix has NaN: {np.any(np.isnan(matrix))}")
   print(f"Matrix is symmetric: {np.allclose(matrix, matrix.T)}")
   ```

2. **Threshold matrix to remove weak connections:**
   ```python
   from utils import threshold_matrix
   thresholded = threshold_matrix(matrix, method='percentile', value=90)
   ```

3. **For modularity issues:**
   - Network may be too sparse or too dense
   - Try different thresholds

4. **For rich-club issues:**
   - Degree threshold may be too high
   - Function automatically uses median degree

### Problem: "Matrix not positive definite"

**Symptoms:**
```
LinAlgError: Matrix is not positive definite
```

**Solutions:**
1. Add small constant to diagonal:
   ```python
   matrix = matrix + 1e-6 * np.eye(matrix.shape[0])
   ```

2. Check for negative values:
   ```python
   matrix = np.abs(matrix)
   ```

---

## Classification Issues

### Problem: "Convergence warning in Logistic Regression"

**Symptoms:**
```
ConvergenceWarning: lbfgs failed to converge
```

**Solutions:**
1. Increase max iterations:
   ```python
   # In classification.py
   clf = LogisticRegression(max_iter=2000)
   ```

2. Scale features:
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

### Problem: Very Low Classification Accuracy

**Symptoms:**
```
Best accuracy: 0.52
```

**Solutions:**
1. **Check for data leakage:**
   - Ensure train/test split is correct
   - No data from same subject in both sets

2. **Remove NaN features:**
   ```python
   from utils import remove_nan_features
   X_clean, features_clean = remove_nan_features(X, y, feature_names)
   ```

3. **Check class balance:**
   ```python
   print(f"Group 0: {np.sum(y==0)} subjects")
   print(f"Group 1: {np.sum(y==1)} subjects")
   ```

4. **Try different feature combinations:**
   ```python
   N_FEATURES_COMBINATION = 4  # Instead of 3
   ```

### Problem: Too Few Subjects for Classification

**Symptoms:**
```
Not enough subjects for 5-fold cross-validation
```

**Solutions:**
1. Reduce number of folds:
   ```python
   N_FOLDS = 3
   ```

2. Use Leave-One-Out CV for very small samples:
   ```python
   from sklearn.model_selection import LeaveOneOut
   loo = LeaveOneOut()
   ```

---

## Visualization Problems

### Problem: Figures Not Showing

**Symptoms:**
- No plot windows appear
- Script completes but no visualization

**Solutions:**
1. Check if running in non-interactive environment:
   ```python
   import matplotlib
   matplotlib.use('Agg')  # For saving without display
   ```

2. Use explicit show:
   ```python
   import matplotlib.pyplot as plt
   plt.show()
   ```

3. For Jupyter notebooks:
   ```python
   %matplotlib inline
   ```

### Problem: Plots Look Distorted

**Symptoms:**
- Overlapping labels
- Cut-off text

**Solutions:**
1. Increase figure size:
   ```python
   fig, ax = plt.subplots(figsize=(14, 10))
   ```

2. Use tight layout:
   ```python
   plt.tight_layout()
   ```

3. Adjust margins:
   ```python
   plt.subplots_adjust(bottom=0.15, left=0.15)
   ```

---

## Performance Optimization

### Speed Up Processing

1. **Use fewer epochs:**
   ```python
   EPOCH_DURATION = 20.0  # Longer epochs = fewer to process
   ```

2. **Parallel processing (advanced):**
   ```python
   from joblib import Parallel, delayed
   
   results = Parallel(n_jobs=-1)(
       delayed(process_subject)(subject) 
       for subject in subjects
   )
   ```

3. **Process subset of data first:**
   ```python
   # Test on 2-3 subjects first
   healthy_data = healthy_data[:3]
   patient_data = patient_data[:3]
   ```

4. **Skip time-consuming measures:**
   ```python
   NETWORK_MEASURES = [
       'global_efficiency',
       'clustering_coefficient',
       'degree'
       # Skip: small_worldness (slow)
   ]
   ```

### Reduce Disk Space Usage

1. **Save compressed:**
   ```python
   np.savez_compressed('results.npz', data=data)
   ```

2. **Delete intermediate results:**
   ```python
   # After processing, delete raw epochs
   del filtered_epochs
   import gc
   gc.collect()
   ```

---

## Common Error Messages

### "module 'mne' has no attribute 'create_info'"
- **Cause:** Old MNE version
- **Solution:** `pip install --upgrade mne`

### "ImportError: cannot import name 'spectral_connectivity_epochs'"
- **Cause:** Missing mne-connectivity
- **Solution:** `pip install mne-connectivity`

### "AttributeError: module 'bct' has no attribute 'efficiency_wei'"
- **Cause:** Wrong BCT version or installation
- **Solution:** `pip install --upgrade bctpy`

### "ValueError: setting an array element with a sequence"
- **Cause:** Inconsistent array shapes
- **Solution:** Check that all matrices have same size

---

## Getting Help

If you're still having issues:

1. **Check versions:**
   ```python
   import mne, bct, sklearn
   print(f"MNE: {mne.__version__}")
   print(f"BCT: {bct.__version__}")
   print(f"sklearn: {sklearn.__version__}")
   ```

2. **Run setup verification:**
   ```bash
   python setup.py
   ```

3. **Enable debug mode:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

4. **Check issue tracker** for known problems

5. **Contact:** [Your contact information]

---

## Useful Debug Commands

```python
# Check data shape
print(f"Data shape: {data.shape}")

# Check for NaN/Inf
print(f"Has NaN: {np.any(np.isnan(data))}")
print(f"Has Inf: {np.any(np.isinf(data))}")

# Memory usage
import sys
print(f"Matrix size: {sys.getsizeof(matrix) / 1e6:.2f} MB")

# Timing
import time
start = time.time()
# ... code ...
print(f"Time elapsed: {time.time() - start:.2f} seconds")
```
