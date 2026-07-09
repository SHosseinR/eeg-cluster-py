"""
Statistical analysis utilities
"""

import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
import pandas as pd
import re

try:
    from scipy.stats import binomtest
except ImportError:  # pragma: no cover - for older scipy versions
    binomtest = None

def _exact_binomial_pvalue(successes, trials, alternative='greater'):
    """Compatibility wrapper for scipy's exact binomial test."""
    if trials <= 0:
        return np.nan

    if binomtest is not None:
        return float(binomtest(successes, trials, p=0.5, alternative=alternative).pvalue)

    return float(stats.binom_test(successes, trials, p=0.5, alternative=alternative))


def compute_pvalue_matrix(
    matrices_list,
    alternative='two-sided',
    popmean=0.0,
    min_samples=2,
    diagonal_value=1.0
):
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
    popmean : float
        Null-hypothesis mean for the one-sample test
    min_samples : int
        Minimum finite values required per edge
    diagonal_value : float or None
        Optional p-value assigned to diagonal entries
        
    Returns
    -------
    pvalue_matrix : ndarray, shape (n_channels, n_channels)
        P-values for each connection
    mean_matrix : ndarray, shape (n_channels, n_channels)
        Mean connectivity across subjects
    """
    if not matrices_list:
        raise ValueError("matrices_list must contain at least one matrix")

    # Stack matrices
    matrices_array = np.stack(matrices_list, axis=0).astype(float)  # (n_subjects, n_channels, n_channels)
    
    # Compute mean
    mean_matrix = np.nanmean(matrices_array, axis=0)
    
    # Compute p-values element-wise (test against zero)
    n_channels = matrices_array.shape[1]
    pvalue_matrix = np.full((n_channels, n_channels), np.nan, dtype=float)
    
    for i in range(n_channels):
        for j in range(n_channels):
            values = matrices_array[:, i, j]
            values = values[np.isfinite(values)]
            if len(values) < min_samples:
                continue

            if np.allclose(values, values[0]):
                value = float(values[0])
                if np.isclose(value, popmean):
                    p = 1.0
                elif alternative == 'greater':
                    p = 0.0 if value > popmean else 1.0
                elif alternative == 'less':
                    p = 0.0 if value < popmean else 1.0
                else:
                    p = 0.0
            else:
                _, p = stats.ttest_1samp(
                    values,
                    popmean,
                    alternative=alternative,
                    nan_policy='omit'
                )
                if not np.isfinite(p):
                    p = np.nan

            # One-sample t-test against zero
            pvalue_matrix[i, j] = p

    if diagonal_value is not None:
        np.fill_diagonal(pvalue_matrix, diagonal_value)
    
    return pvalue_matrix, mean_matrix


def summarize_connectivity_stability(connectivity_dict, methods, band_names, thresholds=(0.001, 0.01, 0.05)):
    """
    Summarize edge-wise functional-connectivity stability p-values.

    Stability is estimated as a one-sample test for each edge across subjects
    against zero connectivity, using the one-sided alternative mean > 0.
    """
    rows = []

    def add_summary(scope, label, matrices):
        if not matrices:
            return
        pvalue_matrix, mean_matrix = compute_pvalue_matrix(matrices, alternative='greater')
        finite = np.isfinite(pvalue_matrix)
        row = {
            'scope': scope,
            'label': label,
            'n_matrices': len(matrices),
            'n_edges': int(pvalue_matrix.size),
            'finite_edges': int(np.sum(finite)),
            'mean_connectivity': float(np.nanmean(mean_matrix)),
            'median_pvalue': float(np.nanmedian(pvalue_matrix[finite])) if np.any(finite) else np.nan,
            'min_pvalue': float(np.nanmin(pvalue_matrix[finite])) if np.any(finite) else np.nan,
        }
        for threshold in thresholds:
            row[f'n_edges_p_lt_{threshold}'] = int(np.sum((pvalue_matrix < threshold) & finite))
        rows.append(row)

    for method in methods:
        matrices = []
        for group_data in connectivity_dict.values():
            for subject_data in group_data.values():
                if method in subject_data:
                    for band in band_names:
                        if band in subject_data[method]:
                            matrices.append(subject_data[method][band])
        add_summary('method_all_bands', method, matrices)

    for band in band_names:
        matrices = []
        for group_data in connectivity_dict.values():
            for subject_data in group_data.values():
                for method_data in subject_data.values():
                    if band in method_data:
                        matrices.append(method_data[band])
        add_summary('band_all_methods', band, matrices)

    return pd.DataFrame(rows)


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


def _base_electrode_name(channel_name):
    """Return the EEG electrode token before references such as '-LE'."""
    label = str(channel_name).strip()
    if "/" in label:
        label = label.split("/", 1)[1]
    token = re.split(r'[\s_\-]+', label)[0]
    return token.upper()


def infer_symmetric_electrode_pairs(channel_names):
    """
    Infer left/right homologous electrode pairs from standard EEG names.

    Odd-numbered electrodes are treated as left, even-numbered electrodes as
    right. Midline electrodes ending in Z map to themselves.
    """
    base_to_index = {
        _base_electrode_name(name): idx
        for idx, name in enumerate(channel_names)
    }
    legacy_pairs = {
        'T3': 'T4',
        'T4': 'T3',
        'T5': 'T6',
        'T6': 'T5',
    }
    pairs = {}

    for idx, channel_name in enumerate(channel_names):
        base = _base_electrode_name(channel_name)
        mirror = None

        if base.endswith('Z'):
            mirror = base
        elif base in legacy_pairs:
            mirror = legacy_pairs[base]
        else:
            match = re.match(r'^([A-Z]+)(\d+)$', base)
            if match:
                prefix, number_text = match.groups()
                number = int(number_text)
                mirror_number = number + 1 if number % 2 == 1 else number - 1
                mirror = f"{prefix}{mirror_number}"

        if mirror in base_to_index:
            pairs[idx] = base_to_index[mirror]

    return pairs


def compute_candidate_region_selection_stats(optimization_results, channel_names, alpha=0.05):
    """
    Test whether the most-selected stimulation electrode is selected more often
    than its contralateral homolog and every other electrode.

    Each optimization result contributes one categorical best-solution node.
    The pairwise p-value is an exact one-sided binomial test over runs where
    either the top node or the comparison node was selected.
    """
    selected_nodes = []
    for result in optimization_results.values():
        best_solution = result.get('best_solution') if isinstance(result, dict) else None
        if best_solution is None or best_solution.get('node') is None:
            continue
        selected_nodes.append(int(best_solution['node']))

    n_units = len(selected_nodes)
    n_channels = len(channel_names)
    if n_units == 0 or n_channels == 0:
        return pd.DataFrame()

    counts = np.bincount(selected_nodes, minlength=n_channels)
    top_node = int(np.argmax(counts))
    top_count = int(counts[top_node])
    mirror_pairs = infer_symmetric_electrode_pairs(channel_names)
    symmetric_node = mirror_pairs.get(top_node)

    rows = []
    for node_idx, node_name in enumerate(channel_names):
        if node_idx == top_node:
            continue

        comparison_count = int(counts[node_idx])
        conditional_n = top_count + comparison_count
        p_uncorrected = _exact_binomial_pvalue(
            successes=top_count,
            trials=conditional_n,
            alternative='greater'
        )

        rows.append({
            'top_node': top_node,
            'top_region': channel_names[top_node],
            'comparison_node': node_idx,
            'comparison_region': node_name,
            'comparison_relation': 'symmetric' if node_idx == symmetric_node else 'other',
            'n_optimization_units': n_units,
            'top_count': top_count,
            'comparison_count': comparison_count,
            'top_selection_rate': top_count / n_units,
            'comparison_selection_rate': comparison_count / n_units,
            'conditional_top_rate': top_count / conditional_n if conditional_n > 0 else np.nan,
            'p_uncorrected': p_uncorrected,
        })

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return stats_df

    corrected = correct_multiple_comparisons(stats_df['p_uncorrected'].to_numpy(), method='fdr_bh')
    stats_df['p_fdr_bh'] = corrected
    stats_df[f'significant_fdr_{alpha}'] = stats_df['p_fdr_bh'] < alpha

    sort_relation = stats_df['comparison_relation'].map({'symmetric': 0, 'other': 1}).fillna(2)
    stats_df = stats_df.assign(_sort_relation=sort_relation).sort_values(
        by=['_sort_relation', 'p_uncorrected', 'comparison_region']
    ).drop(columns=['_sort_relation']).reset_index(drop=True)

    return stats_df


def _weighted_rank_vectors_from_results(optimization_results, channel_names, top_k=None):
    """Build one per-optimization-unit electrode weight vector from ranked solutions."""
    n_channels = len(channel_names)
    unit_vectors = []

    for result in optimization_results.values():
        if not isinstance(result, dict):
            continue

        ranked = result.get('top_solutions') or []
        if top_k is not None:
            ranked = ranked[:int(top_k)]

        weights = np.zeros(n_channels, dtype=float)
        if ranked:
            for position, sol in enumerate(ranked, start=1):
                if sol is None or sol.get('node') is None:
                    continue
                node = int(sol['node'])
                if node < 0 or node >= n_channels:
                    continue
                rank = sol.get('rank', position)
                try:
                    rank = float(rank)
                except (TypeError, ValueError):
                    rank = float(position)
                weight = sol.get('strength')
                if weight is None:
                    weight = 1.0 / max(rank, 1.0)
                weights[node] += float(weight)
        else:
            best_solution = result.get('best_solution')
            if best_solution is None or best_solution.get('node') is None:
                continue
            node = int(best_solution['node'])
            if 0 <= node < n_channels:
                weights[node] = 1.0

        if np.any(weights > 0):
            unit_vectors.append(weights)

    if not unit_vectors:
        return np.empty((0, n_channels), dtype=float)
    return np.vstack(unit_vectors)


def _paired_sign_flip_pvalue(differences, n_permutations=10000, random_state=42):
    """
    One-sided paired sign-flip permutation p-value for mean(differences) > 0.
    """
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0:
        return np.nan

    observed = float(np.mean(differences))
    if observed <= 0:
        return 1.0

    nonzero = differences[np.abs(differences) > 1e-12]
    if len(nonzero) == 0:
        return 1.0

    max_exact = 2 ** len(nonzero)
    if max_exact <= n_permutations and len(nonzero) <= 20:
        stats_perm = []
        for mask in range(max_exact):
            signs = np.ones(len(nonzero), dtype=float)
            for bit_idx in range(len(nonzero)):
                if (mask >> bit_idx) & 1:
                    signs[bit_idx] = -1.0
            stats_perm.append(float(np.sum(signs * nonzero) / len(differences)))
        stats_perm = np.asarray(stats_perm, dtype=float)
        return float(np.mean(stats_perm >= observed - 1e-12))

    rng = np.random.default_rng(random_state)
    signs = rng.choice([-1.0, 1.0], size=(int(n_permutations), len(nonzero)))
    stats_perm = np.sum(signs * nonzero[None, :], axis=1) / len(differences)
    return float((np.sum(stats_perm >= observed - 1e-12) + 1.0) / (len(stats_perm) + 1.0))


def compute_candidate_region_weighted_rank_stats(
    optimization_results,
    channel_names,
    top_k=None,
    n_permutations=10000,
    random_state=42,
    alpha=0.05
):
    """
    Test whether the top rank-weighted electrode is superior to other electrodes.

    Each optimization unit contributes a paired weight vector over electrodes,
    using top_solutions strength values (default strength = 1/rank). Pairwise
    p-values use a one-sided paired sign-flip permutation test on
    weight(top) - weight(comparison).
    """
    unit_weights = _weighted_rank_vectors_from_results(
        optimization_results,
        channel_names,
        top_k=top_k
    )
    n_units = unit_weights.shape[0]
    n_channels = len(channel_names)
    if n_units == 0 or n_channels == 0:
        return pd.DataFrame()

    weighted_counts = np.sum(unit_weights, axis=0)
    top_node = int(np.argmax(weighted_counts))
    top_weight = float(weighted_counts[top_node])
    mirror_pairs = infer_symmetric_electrode_pairs(channel_names)
    symmetric_node = mirror_pairs.get(top_node)

    rows = []
    for node_idx, node_name in enumerate(channel_names):
        if node_idx == top_node:
            continue

        differences = unit_weights[:, top_node] - unit_weights[:, node_idx]
        p_uncorrected = _paired_sign_flip_pvalue(
            differences,
            n_permutations=n_permutations,
            random_state=random_state + node_idx
        )
        comparison_weight = float(weighted_counts[node_idx])
        rows.append({
            'top_node': top_node,
            'top_region': channel_names[top_node],
            'comparison_node': node_idx,
            'comparison_region': node_name,
            'comparison_relation': 'symmetric' if node_idx == symmetric_node else 'other',
            'n_optimization_units': n_units,
            'top_weighted_count': top_weight,
            'comparison_weighted_count': comparison_weight,
            'top_weighted_rate': top_weight / n_units,
            'comparison_weighted_rate': comparison_weight / n_units,
            'mean_paired_difference': float(np.mean(differences)),
            'median_paired_difference': float(np.median(differences)),
            'p_uncorrected': p_uncorrected,
            'n_permutations': int(n_permutations),
            'top_k': top_k if top_k is not None else np.nan,
        })

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return stats_df

    corrected = correct_multiple_comparisons(stats_df['p_uncorrected'].to_numpy(), method='fdr_bh')
    stats_df['p_fdr_bh'] = corrected
    stats_df[f'significant_fdr_{alpha}'] = stats_df['p_fdr_bh'] < alpha

    sort_relation = stats_df['comparison_relation'].map({'symmetric': 0, 'other': 1}).fillna(2)
    stats_df = stats_df.assign(_sort_relation=sort_relation).sort_values(
        by=['_sort_relation', 'p_uncorrected', 'comparison_region']
    ).drop(columns=['_sort_relation']).reset_index(drop=True)

    return stats_df
