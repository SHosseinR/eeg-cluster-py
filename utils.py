"""
Utility functions for EEG connectivity analysis
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# =============================================================================
# DATA UTILITIES
# =============================================================================

def check_data_quality(data, fs, subject_id="Unknown"):
    """
    Perform basic data quality checks.
    
    Parameters
    ----------
    data : ndarray
        EEG data
    fs : float
        Sampling frequency
    subject_id : str
        Subject identifier
        
    Returns
    -------
    quality_report : dict
        Dictionary with quality metrics
    """
    n_channels, n_samples = data.shape
    duration = n_samples / fs
    
    quality_report = {
        'subject_id': subject_id,
        'n_channels': n_channels,
        'n_samples': n_samples,
        'duration_seconds': duration,
        'sampling_frequency': fs,
        'has_nan': np.any(np.isnan(data)),
        'has_inf': np.any(np.isinf(data)),
        'mean_amplitude': np.mean(np.abs(data)),
        'max_amplitude': np.max(np.abs(data)),
        'min_amplitude': np.min(np.abs(data))
    }
    
    # Check for flat channels
    flat_channels = []
    for i in range(n_channels):
        if np.std(data[i, :]) < 1e-6:
            flat_channels.append(i)
    quality_report['flat_channels'] = flat_channels
    
    # Check for noisy channels (high variance)
    channel_std = np.std(data, axis=1)
    threshold = np.mean(channel_std) + 3 * np.std(channel_std)
    noisy_channels = np.where(channel_std > threshold)[0].tolist()
    quality_report['noisy_channels'] = noisy_channels
    
    return quality_report


def print_quality_report(quality_report):
    """Print a formatted quality report."""
    print("\n" + "="*60)
    print(f"DATA QUALITY REPORT: {quality_report['subject_id']}")
    print("="*60)
    print(f"Channels:          {quality_report['n_channels']}")
    print(f"Samples:           {quality_report['n_samples']}")
    print(f"Duration:          {quality_report['duration_seconds']:.2f} seconds")
    print(f"Sampling Freq:     {quality_report['sampling_frequency']} Hz")
    print(f"Mean Amplitude:    {quality_report['mean_amplitude']:.4f}")
    print(f"Max Amplitude:     {quality_report['max_amplitude']:.4f}")
    print("-"*60)
    
    # Warnings
    warnings = []
    if quality_report['has_nan']:
        warnings.append("⚠️  Contains NaN values")
    if quality_report['has_inf']:
        warnings.append("⚠️  Contains Inf values")
    if quality_report['flat_channels']:
        warnings.append(f"⚠️  Flat channels: {quality_report['flat_channels']}")
    if quality_report['noisy_channels']:
        warnings.append(f"⚠️  Noisy channels: {quality_report['noisy_channels']}")
    
    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("✓ No quality issues detected")
    
    print("="*60 + "\n")


# =============================================================================
# MATRIX UTILITIES
# =============================================================================

def threshold_matrix(matrix, method='percentile', value=90):
    """
    Threshold connectivity matrix to keep only strong connections.
    
    Parameters
    ----------
    matrix : ndarray
        Connectivity matrix
    method : str
        'percentile', 'absolute', or 'std'
    value : float
        Threshold value (percentile, absolute value, or std multiplier)
        
    Returns
    -------
    thresholded : ndarray
        Thresholded matrix
    """
    if method == 'percentile':
        threshold = np.percentile(matrix, value)
    elif method == 'absolute':
        threshold = value
    elif method == 'std':
        threshold = np.mean(matrix) + value * np.std(matrix)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    thresholded = np.where(matrix > threshold, matrix, 0)
    
    print(f"Thresholding: {method} = {value}")
    print(f"  Threshold value: {threshold:.4f}")
    print(f"  Edges kept: {np.sum(thresholded > 0)} / {matrix.size}")
    
    return thresholded


def binarize_matrix(matrix, threshold=0):
    """
    Convert weighted matrix to binary.
    
    Parameters
    ----------
    matrix : ndarray
        Connectivity matrix
    threshold : float
        Threshold for binarization
        
    Returns
    -------
    binary : ndarray
        Binary matrix
    """
    return (matrix > threshold).astype(int)


def symmetrize_matrix(matrix, method='average'):
    """
    Make a directed matrix symmetric.
    
    Parameters
    ----------
    matrix : ndarray
        Asymmetric matrix
    method : str
        'average', 'max', or 'min'
        
    Returns
    -------
    symmetric : ndarray
        Symmetric matrix
    """
    if method == 'average':
        symmetric = (matrix + matrix.T) / 2
    elif method == 'max':
        symmetric = np.maximum(matrix, matrix.T)
    elif method == 'min':
        symmetric = np.minimum(matrix, matrix.T)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return symmetric


# =============================================================================
# STATISTICAL UTILITIES
# =============================================================================

def compute_effect_size(group1, group2):
    """
    Compute Cohen's d effect size.
    
    Parameters
    ----------
    group1, group2 : array-like
        Data from two groups
        
    Returns
    -------
    d : float
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d


def bootstrap_confidence_interval(data, statistic=np.mean, n_bootstrap=1000, confidence=95):
    """
    Compute bootstrap confidence interval.
    
    Parameters
    ----------
    data : array-like
        Data to bootstrap
    statistic : callable
        Statistic to compute
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level (0-100)
        
    Returns
    -------
    ci_lower, ci_upper : float
        Confidence interval bounds
    """
    bootstrap_stats = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))
    
    alpha = (100 - confidence) / 2
    ci_lower = np.percentile(bootstrap_stats, alpha)
    ci_upper = np.percentile(bootstrap_stats, 100 - alpha)
    
    return ci_lower, ci_upper


# =============================================================================
# FEATURE UTILITIES
# =============================================================================

def normalize_features(X, method='zscore'):
    """
    Normalize feature matrix.
    
    Parameters
    ----------
    X : ndarray
        Feature matrix
    method : str
        'zscore', 'minmax', or 'robust'
        
    Returns
    -------
    X_normalized : ndarray
        Normalized features
    """
    if method == 'zscore':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    X_normalized = scaler.fit_transform(X)
    return X_normalized, scaler


def remove_nan_features(X, y, feature_names):
    """
    Remove features containing NaN values.
    
    Parameters
    ----------
    X : ndarray
        Feature matrix
    y : ndarray
        Labels
    feature_names : list
        Feature names
        
    Returns
    -------
    X_clean, feature_names_clean : ndarray, list
        Cleaned features and names
    """
    # Find columns without NaN
    valid_cols = ~np.any(np.isnan(X), axis=0)
    
    X_clean = X[:, valid_cols]
    feature_names_clean = [name for name, valid in zip(feature_names, valid_cols) if valid]
    
    n_removed = np.sum(~valid_cols)
    if n_removed > 0:
        print(f"Removed {n_removed} features with NaN values")
    
    return X_clean, feature_names_clean


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def plot_distribution_comparison(group1, group2, labels=['Group 1', 'Group 2'], 
                                 title='Distribution Comparison', xlabel='Value'):
    """
    Plot distributions of two groups.
    
    Parameters
    ----------
    group1, group2 : array-like
        Data from two groups
    labels : list
        Labels for groups
    title : str
        Plot title
    xlabel : str
        X-axis label
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(group1, bins=20, alpha=0.5, label=labels[0], density=True)
    axes[0].hist(group2, bins=20, alpha=0.5, label=labels[1], density=True)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel('Density')
    axes[0].set_title('Histogram')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    axes[1].boxplot([group1, group2], labels=labels)
    axes[1].set_ylabel(xlabel)
    axes[1].set_title('Box Plot')
    axes[1].grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(X, feature_names, title='Feature Correlations'):
    """
    Plot correlation matrix of features.
    
    Parameters
    ----------
    X : ndarray
        Feature matrix
    feature_names : list
        Feature names
    title : str
        Plot title
    """
    corr_matrix = np.corrcoef(X.T)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                xticklabels=feature_names, yticklabels=feature_names,
                square=True, ax=ax)
    ax.set_title(title, fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()


# =============================================================================
# FILE UTILITIES
# =============================================================================

def save_results_summary(results_dict, output_path):
    """
    Save a summary of results to text file.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with results
    output_path : str
        Path to save file
    """
    with open(output_path, 'w') as f:
        f.write("EEG CONNECTIVITY ANALYSIS - RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        for key, value in results_dict.items():
            f.write(f"{key}:\n")
            if isinstance(value, dict):
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"  {value}\n")
            f.write("\n")
    
    print(f"Saved results summary to: {output_path}")


def load_saved_results(results_dir):
    """
    Load all saved results from a directory.
    
    Parameters
    ----------
    results_dir : str
        Path to results directory
        
    Returns
    -------
    results : dict
        Dictionary with loaded results
    """
    results = {}
    
    # Load .npy files
    npy_files = ['connectivity_matrices.npy', 'network_measures.npy']
    for filename in npy_files:
        filepath = os.path.join(results_dir, 'data', filename)
        if os.path.exists(filepath):
            results[filename.replace('.npy', '')] = np.load(filepath, allow_pickle=True).item()
    
    # Load .csv files
    csv_files = ['network_measures_pvalues.csv', 'top_feature_triplets.csv']
    for filename in csv_files:
        filepath = os.path.join(results_dir, 'data', filename)
        if os.path.exists(filepath):
            results[filename.replace('.csv', '')] = pd.read_csv(filepath)
    
    print(f"Loaded {len(results)} result files from {results_dir}")
    return results

def pretty_print_matrix(A, max_rows=10, max_columns=10):
    A = np.asarray(A)
    sub = A[:max_rows, :max_columns]

    with np.printoptions(
        precision=4,       # decimals
        suppress=True,     # no scientific for small numbers
        linewidth=120      # fit in terminal width
    ):
        print(sub)

if __name__ == "__main__":
    print("EEG Connectivity Analysis - Utility Functions")
    print("Import this module to use utility functions")
    print("Example: from utils import check_data_quality, threshold_matrix")
