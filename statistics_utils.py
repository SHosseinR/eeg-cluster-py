"""
Statistical analysis utilities
"""

import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
import pandas as pd

def compute_pvalue_matrix(matrices_list, alternative='two-sided'):
    """
    Compute p-value for each element of connectivity matrices.
    Tests if mean is different from zero.
    
    Parameters
    ----------
    matrices_list : list of ndarray
        List of connectivity matrices, one per subject
        Each matrix has shape (n_channels, n_channels)
    alternative : str
        'two-sided', 'greater', or 'less'
        
    Returns
    -------
    pvalue_matrix : ndarray, shape (n_channels, n_channels)
        P-values for each connection
    mean_matrix : ndarray, shape (n_channels, n_channels)
        Mean connectivity across subjects
    """
    # Stack matrices
    matrices_array = np.stack(matrices_list, axis=0)  # (n_subjects, n_channels, n_channels)
    
    # Compute mean
    mean_matrix = np.mean(matrices_array, axis=0)
    
    # Compute p-values element-wise (test against zero)
    n_channels = matrices_array.shape[1]
    pvalue_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(n_channels):
            values = matrices_array[:, i, j]
            # One-sample t-test against zero
            _, p = stats.ttest_1samp(values, 0, alternative=alternative)
            pvalue_matrix[i, j] = p
    
    return pvalue_matrix, mean_matrix


def compute_group_comparison_pvalues(group1_measures, group2_measures, measure_names, band_names):
    """
    Compare network measures between two groups.
    
    Parameters
    ----------
    group1_measures : dict
        Network measures for group 1
        Structure: {subject_id: {method: {band: {measure: value}}}}
    group2_measures : dict
        Network measures for group 2 (same structure)
    measure_names : list
        List of measure names to compare
    band_names : list
        List of frequency band names
        
    Returns
    -------
    pvalue_df : pd.DataFrame
        DataFrame with p-values (rows=measures, cols=bands)
    """
    pvalue_dict = {}
    
    for measure in measure_names:
        pvalue_dict[measure] = {}
        
        for band in band_names:
            # Collect values from both groups
            group1_values = []
            group2_values = []
            
            # Extract values (assuming single method is selected)
            for subject_data in group1_measures.values():
                for method_data in subject_data.values():
                    if band in method_data and measure in method_data[band]:
                        val = method_data[band][measure]
                        if not np.isnan(val):
                            group1_values.append(val)
                    break  # Only first method
                
            for subject_data in group2_measures.values():
                for method_data in subject_data.values():
                    if band in method_data and measure in method_data[band]:
                        val = method_data[band][measure]
                        if not np.isnan(val):
                            group2_values.append(val)
                    break  # Only first method
            
            # Perform statistical test
            if len(group1_values) > 0 and len(group2_values) > 0:
                # Use Mann-Whitney U test (non-parametric)
                _, p = mannwhitneyu(group1_values, group2_values, alternative='two-sided')
                pvalue_dict[measure][band] = p
            else:
                pvalue_dict[measure][band] = np.nan
    
    # Convert to DataFrame
    pvalue_df = pd.DataFrame(pvalue_dict).T
    
    return pvalue_df


def extract_features_for_classification(network_measures_dict, measure_names, band_names, selected_method):
    """
    Extract features for classification from network measures.
    
    Parameters
    ----------
    network_measures_dict : dict
        Network measures with structure:
        {group: {subject_id: {method: {band: {measure: value}}}}}
    measure_names : list
        List of measure names to extract
    band_names : list
        List of frequency band names
    selected_method : str
        Selected connectivity method
        
    Returns
    -------
    X : ndarray, shape (n_subjects, n_features)
        Feature matrix where features are (measure, band) combinations
    y : ndarray, shape (n_subjects,)
        Labels (0 for first group, 1 for second group)
    feature_names : list
        List of feature names
    subject_ids : list
        List of subject IDs
    """
    X_list = []
    y_list = []
    subject_ids = []
    
    # Create feature names
    feature_names = []
    for measure in measure_names:
        for band in band_names:
            feature_names.append(f"{measure}_{band}")
    
    # Extract features for each group
    group_labels = {}
    for group_idx, (group_name, group_data) in enumerate(network_measures_dict.items()):
        group_labels[group_name] = group_idx
        
        for subject_id, subject_data in group_data.items():
            features = []
            
            # Extract features in consistent order
            for measure in measure_names:
                for band in band_names:
                    if selected_method in subject_data:
                        if band in subject_data[selected_method]:
                            if measure in subject_data[selected_method][band]:
                                val = subject_data[selected_method][band][measure]
                                features.append(val if not np.isnan(val) else 0.0)
                            else:
                                features.append(0.0)
                        else:
                            features.append(0.0)
                    else:
                        features.append(0.0)
            
            X_list.append(features)
            y_list.append(group_idx)
            subject_ids.append(subject_id)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y, feature_names, subject_ids


def perform_feature_selection_stats(X, y, feature_names):
    """
    Perform statistical tests for each feature between groups.
    
    Parameters
    ----------
    X : ndarray, shape (n_subjects, n_features)
        Feature matrix
    y : ndarray, shape (n_subjects,)
        Group labels
    feature_names : list
        Feature names
        
    Returns
    -------
    stats_df : pd.DataFrame
        DataFrame with statistics for each feature
    """
    stats_list = []
    
    for i, feature_name in enumerate(feature_names):
        feature_values = X[:, i]
        
        group0_values = feature_values[y == 0]
        group1_values = feature_values[y == 1]
        
        # Compute statistics
        mean0 = np.mean(group0_values)
        mean1 = np.mean(group1_values)
        std0 = np.std(group0_values)
        std1 = np.std(group1_values)
        
        # Statistical test
        if len(group0_values) > 0 and len(group1_values) > 0:
            stat, p = mannwhitneyu(group0_values, group1_values, alternative='two-sided')
        else:
            stat, p = np.nan, np.nan
        
        stats_list.append({
            'feature': feature_name,
            'mean_group0': mean0,
            'mean_group1': mean1,
            'std_group0': std0,
            'std_group1': std1,
            'statistic': stat,
            'pvalue': p
        })
    
    stats_df = pd.DataFrame(stats_list)
    
    return stats_df


def correct_multiple_comparisons(pvalues, method='fdr_bh'):
    """
    Correct p-values for multiple comparisons.
    
    Parameters
    ----------
    pvalues : array-like
        P-values to correct
    method : str
        Correction method ('bonferroni' or 'fdr_bh')
        
    Returns
    -------
    corrected_pvalues : ndarray
        Corrected p-values
    """
    from statsmodels.stats.multitest import multipletests
    
    pvalues_flat = np.array(pvalues).flatten()
    valid_mask = ~np.isnan(pvalues_flat)
    
    corrected_pvalues = np.full_like(pvalues_flat, np.nan)
    
    if np.sum(valid_mask) > 0:
        _, corrected_pvalues[valid_mask], _, _ = multipletests(
            pvalues_flat[valid_mask], method=method
        )
    
    return corrected_pvalues.reshape(np.array(pvalues).shape)
