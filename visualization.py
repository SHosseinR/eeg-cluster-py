"""
Visualization utilities for connectivity and network analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import FIGURE_DPI, FIGURE_SIZE, CMAP_CONNECTIVITY, CMAP_PVALUE

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = FIGURE_DPI

def plot_connectivity_matrices(connectivity_dict, methods, output_path=None):
    """
    Visualization 1: Plot average connectivity matrices for each method.
    Average across all frequency bands and subjects.
    
    Parameters
    ----------
    connectivity_dict : dict
        Structure: {group: {subject: {method: {band: matrix}}}}
    methods : list
        List of connectivity methods
    output_path : str, optional
        Path to save figure
    """
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
    
    if n_methods == 1:
        axes = [axes]
    
    for idx, method in enumerate(methods):
        # Collect all matrices for this method (across subjects and bands)
        all_matrices = []
        
        for group_data in connectivity_dict.values():
            for subject_data in group_data.values():
                if method in subject_data:
                    for band, matrix in subject_data[method].items():
                        all_matrices.append(matrix)
        
        if len(all_matrices) > 0:
            # Average across all matrices
            avg_matrix = np.mean(np.stack(all_matrices), axis=0)
            
            # Plot
            im = axes[idx].imshow(avg_matrix, cmap=CMAP_CONNECTIVITY, aspect='auto')
            axes[idx].set_title(f'{method.upper()}\n(Avg across bands & subjects)', 
                              fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Target Node')
            axes[idx].set_ylabel('Source Node')
            
            # Colorbar
            plt.colorbar(im, ax=axes[idx], label='Connectivity Strength')
        else:
            axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[idx].set_title(f'{method.upper()}')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()


def plot_pvalue_matrices(connectivity_dict, methods, output_path=None):
    """
    Visualization 2: Plot p-value matrices for each method.
    P-values test if average connectivity is different from zero.
    
    Parameters
    ----------
    connectivity_dict : dict
        Structure: {group: {subject: {method: {band: matrix}}}}
    methods : list
        List of connectivity methods
    output_path : str, optional
        Path to save figure
    """
    from statistics_utils import compute_pvalue_matrix
    
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
    
    if n_methods == 1:
        axes = [axes]
    
    for idx, method in enumerate(methods):
        # Collect all matrices for this method (across subjects and bands)
        all_matrices = []
        
        for group_data in connectivity_dict.values():
            for subject_data in group_data.values():
                if method in subject_data:
                    for band, matrix in subject_data[method].items():
                        all_matrices.append(matrix)
        
        if len(all_matrices) > 0:
            # Compute p-values
            pvalue_matrix, mean_matrix = compute_pvalue_matrix(all_matrices)
            
            # Plot p-values
            im = axes[idx].imshow(pvalue_matrix, cmap=CMAP_PVALUE, 
                                 aspect='auto', vmin=0, vmax=0.1)
            axes[idx].set_title(f'{method.upper()}\nP-values (H0: mean=0)', 
                              fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Target Node')
            axes[idx].set_ylabel('Source Node')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=axes[idx], label='P-value')
            
            # Add significance threshold lines
            axes[idx].text(0.02, 0.98, f'p<0.001: {np.sum(pvalue_matrix < 0.001)} edges\n'
                                       f'p<0.01: {np.sum(pvalue_matrix < 0.01)} edges\n'
                                       f'p<0.05: {np.sum(pvalue_matrix < 0.05)} edges',
                          transform=axes[idx].transAxes, 
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                          fontsize=8)
        else:
            axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[idx].set_title(f'{method.upper()}')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()


def plot_pvalue_matrices_per_band(connectivity_dict, band_names, output_path=None):
    """
    Visualization 3: Plot p-value matrices per frequency band.
    Averaged over methods.
    
    Parameters
    ----------
    connectivity_dict : dict
        Structure: {group: {subject: {method: {band: matrix}}}}
    band_names : list
        List of frequency band names
    output_path : str, optional
        Path to save figure
    """
    from statistics_utils import compute_pvalue_matrix
    
    n_bands = len(band_names)
    fig, axes = plt.subplots(1, n_bands, figsize=(4*n_bands, 3))
    
    if n_bands == 1:
        axes = [axes]
    
    for idx, band in enumerate(band_names):
        # Collect all matrices for this band (across subjects and methods)
        all_matrices = []
        
        for group_data in connectivity_dict.values():
            for subject_data in group_data.values():
                for method, method_data in subject_data.items():
                    if band in method_data:
                        all_matrices.append(method_data[band])
        
        if len(all_matrices) > 0:
            # Compute p-values
            pvalue_matrix, mean_matrix = compute_pvalue_matrix(all_matrices)
            
            # Plot p-values
            im = axes[idx].imshow(pvalue_matrix, cmap=CMAP_PVALUE, 
                                 aspect='auto', vmin=0, vmax=0.1)
            axes[idx].set_title(f'{band.upper()}\n(Avg over methods)', 
                              fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Target')
            axes[idx].set_ylabel('Source')
            
            # Colorbar
            if idx == n_bands - 1:
                cbar = plt.colorbar(im, ax=axes[idx], label='P-value')
            
            # Add significance counts
            axes[idx].text(0.02, 0.98, 
                          f'p<0.05:\n{np.sum(pvalue_matrix < 0.05)}',
                          transform=axes[idx].transAxes, 
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                          fontsize=8)
        else:
            axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[idx].set_title(f'{band.upper()}')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()


def plot_network_measures_pvalues(pvalue_df, output_path=None):
    """
    Visualization 4: Heatmap of network measures p-values.
    Rows: measures, Columns: bands
    
    Parameters
    ----------
    pvalue_df : pd.DataFrame
        DataFrame with p-values (rows=measures, cols=bands)
    output_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(pvalue_df, annot=True, fmt='.4f', cmap=CMAP_PVALUE, 
                vmin=0, vmax=0.1, ax=ax, cbar_kws={'label': 'P-value'})
    
    ax.set_title('Network Measures: Group Comparison P-values\n(Healthy vs Patient)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Frequency Band', fontsize=12, fontweight='bold')
    ax.set_ylabel('Network Measure', fontsize=12, fontweight='bold')
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()


def plot_top_feature_sets(top_features_df, output_path=None):
    """
    Visualization 5: Table of top 10 feature triplets with accuracy.
    
    Parameters
    ----------
    top_features_df : pd.DataFrame
        DataFrame with columns: ['Rank', 'Features', 'Accuracy']
    output_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=top_features_df.values,
                    colLabels=top_features_df.columns,
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.08, 0.72, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(top_features_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code by accuracy
    for i in range(1, len(top_features_df) + 1):
        accuracy = top_features_df.iloc[i-1]['Accuracy']
        
        # Color gradient based on accuracy
        if accuracy >= 0.9:
            color = '#E8F5E9'
        elif accuracy >= 0.8:
            color = '#F1F8E9'
        elif accuracy >= 0.7:
            color = '#FFF9C4'
        else:
            color = '#FFECB3'
        
        for j in range(len(top_features_df.columns)):
            table[(i, j)].set_facecolor(color)
    
    plt.title('Top 10 Feature Triplets for Classification\n(Sorted by Accuracy)', 
             fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()


def plot_feature_importance(feature_names, importances, output_path=None):
    """
    Visualization 6: Bar plot of feature importance for best triplet.
    
    Parameters
    ----------
    feature_names : list
        List of 3 feature names
    importances : list
        List of 3 importance values (logistic regression weights)
    output_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by absolute importance
    sorted_indices = np.argsort(np.abs(importances))[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importances = [importances[i] for i in sorted_indices]
    
    # Create bar plot
    colors = ['#4CAF50' if imp > 0 else '#F44336' for imp in sorted_importances]
    bars = ax.barh(sorted_features, sorted_importances, color=colors, alpha=0.7)
    
    # Add value labels
    for i, (feat, imp) in enumerate(zip(sorted_features, sorted_importances)):
        ax.text(imp, i, f' {imp:.4f}', 
               va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Feature Importance (Logistic Regression Weight)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance for Best Classification Triplet', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()


def create_summary_report(results_dict, output_path):
    """
    Create a comprehensive summary report with all key findings.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary containing all analysis results
    output_path : str
        Path to save the report
    """
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    # Add text summary
    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis('off')
    
    summary_text = f"""
    EEG CONNECTIVITY ANALYSIS - SUMMARY REPORT
    ==========================================
    
    Dataset:
    - Healthy Controls: {results_dict.get('n_healthy', 'N/A')} subjects
    - Patients: {results_dict.get('n_patients', 'N/A')} subjects
    - Channels: {results_dict.get('n_channels', 'N/A')}
    - Frequency Bands: {', '.join(results_dict.get('bands', []))}
    
    Connectivity Methods: {', '.join(results_dict.get('methods', []))}
    Selected Method for Network Analysis: {results_dict.get('selected_method', 'N/A')}
    
    Best Classification Results:
    - Best Accuracy: {results_dict.get('best_accuracy', 'N/A'):.2%}
    - Best Features: {results_dict.get('best_features', 'N/A')}
    
    Significant Network Measures (p < 0.05):
    {results_dict.get('significant_measures', 'N/A')}
    """
    
    ax_text.text(0.05, 0.95, summary_text, transform=ax_text.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Saved summary report: {output_path}")
    plt.close()
