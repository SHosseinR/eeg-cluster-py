"""
Visualization functions for NSGA-II optimization results
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os


def _rank_best_front(best_front: List[Dict], top_k: int, objective_mode: str = None) -> List[Dict]:
    """Rank Pareto solutions by distance to ideal point and keep top-k."""
    if not best_front:
        return []

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = len(best_front)

    top_k = max(1, top_k)
    top_k = min(top_k, len(best_front))

    objectives = np.array([sol['objectives'] for sol in best_front])
    if objective_mode == "distance_to_gt":
        ideal_point = np.zeros(objectives.shape[1], dtype=float)
    else:
        ideal_point = objectives.min(axis=0)
    distances = np.linalg.norm(objectives - ideal_point, axis=1)
    order = np.argsort(distances)

    ranked = []
    for rank, idx in enumerate(order[:top_k], start=1):
        sol = best_front[idx]
        ranked.append({
            'node': sol['node'],
            'band': sol['band'],
            'band_name': sol.get('band_name'),
            'stimulation_duration': sol.get('stimulation_duration'),
            'stimulation_amplitude': sol.get('stimulation_amplitude'),
            'objectives': sol['objectives'],
            'distance': float(distances[idx]),
            'rank': rank,
            'strength': 1.0 / float(rank)
        })

    return ranked


def _collect_ranked_solutions(optimization_results: Dict, top_k: int = None) -> List[Dict]:
    """Collect ranked solutions across subjects, computing ranks if missing."""
    collected = []

    for _, results in optimization_results.items():
        ranked = []
        if results.get('top_solutions'):
            ranked = results['top_solutions']
        elif results.get('best_front'):
            ranked = _rank_best_front(
                results['best_front'],
                top_k or len(results['best_front']),
                objective_mode=results.get('objective_mode')
            )

        if top_k is not None:
            ranked = ranked[:top_k]

        for sol in ranked:
            collected.append(sol)

    return collected


def plot_node_histogram(optimization_results: Dict, 
                        channel_names: List[str],
                        save_path: str = None,
                        figsize: Tuple = (12, 6)):
    """
    Plot histogram of optimal stimulation nodes across subjects.
    
    Parameters
    ----------
    optimization_results : dict
        Results from optimization (subject_id -> results dict)
    channel_names : list of str
        Names of EEG channels
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size (default: (12, 6))
    """
    # Extract optimal nodes
    optimal_nodes = []
    for subject_id, results in optimization_results.items():
        if 'best_solution' in results and results['best_solution'] is not None:
            optimal_nodes.append(results['best_solution']['node'])
    
    if not optimal_nodes:
        print("No optimization results to plot")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create histogram
    node_counts = np.bincount(optimal_nodes, minlength=len(channel_names))
    x_pos = np.arange(len(channel_names))
    
    bars = ax.bar(x_pos, node_counts, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Highlight most common nodes
    max_count = np.max(node_counts)
    for i, (bar, count) in enumerate(zip(bars, node_counts)):
        if count == max_count and count > 0:
            bar.set_color('crimson')
            bar.set_alpha(0.8)
    
    # Labels and formatting
    ax.set_xlabel('Channel/Node', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Optimal Stimulation Nodes', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(channel_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Node histogram saved to: {save_path}")
    
    # plt.show()
    
    return fig


def _candidate_region_counts(candidate_stats_df, channel_names):
    """Rebuild node selection counts from candidate-region pairwise stats."""
    counts = np.zeros(len(channel_names), dtype=int)
    if candidate_stats_df is None or candidate_stats_df.empty:
        return counts, None, None

    first_row = candidate_stats_df.iloc[0]
    top_node = int(first_row['top_node'])
    counts[top_node] = int(first_row['top_count'])
    symmetric_node = None

    for _, row in candidate_stats_df.iterrows():
        comparison_node = int(row['comparison_node'])
        counts[comparison_node] = int(row['comparison_count'])
        if row.get('comparison_relation') == 'symmetric':
            symmetric_node = comparison_node

    return counts, top_node, symmetric_node


def _format_pvalue(pvalue):
    if pvalue is None or not np.isfinite(float(pvalue)):
        return "N/A"
    pvalue = float(pvalue)
    if pvalue < 0.001:
        return f"{pvalue:.2e}"
    return f"{pvalue:.3f}"


def plot_candidate_region_selection_counts(candidate_stats_df,
                                           channel_names: List[str],
                                           save_path: str = None,
                                           figsize: Tuple = (13, 6)):
    """
    Plot final target selection counts across electrodes.

    Highlights the most-selected target and, if available, its symmetric
    contralateral homolog.
    """
    counts, top_node, symmetric_node = _candidate_region_counts(candidate_stats_df, channel_names)
    if top_node is None:
        print("No candidate-region statistics to plot")
        return None

    n_units = int(candidate_stats_df.iloc[0]['n_optimization_units'])
    x_pos = np.arange(len(channel_names))
    colors = ['#5B8DB8'] * len(channel_names)
    colors[top_node] = '#C73E3A'
    if symmetric_node is not None:
        colors[symmetric_node] = '#F0A202'

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(x_pos, counts, color=colors, edgecolor='black', linewidth=0.6, alpha=0.92)

    ax.set_title('Final Candidate Stimulation Targets', fontsize=15, fontweight='bold')
    ax.set_xlabel('Electrode')
    ax.set_ylabel('Selection count')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(channel_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.25)

    for idx, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            pct = height / n_units * 100.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}\n{pct:.0f}%",
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold' if idx == top_node else 'normal'
            )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color='#C73E3A', label='Most selected target'),
        plt.Rectangle((0, 0), 1, 1, color='#F0A202', label='Symmetric homolog'),
        plt.Rectangle((0, 0), 1, 1, color='#5B8DB8', label='Other electrodes')
    ]
    ax.legend(handles=handles, frameon=False, loc='upper right')
    ax.text(
        0.01,
        0.97,
        f"Optimization units: {n_units}",
        transform=ax.transAxes,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
        fontsize=9
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Candidate-region selection count figure saved to: {save_path}")
    plt.close(fig)
    return fig


def plot_candidate_region_symmetric_comparison(candidate_stats_df,
                                               save_path: str = None,
                                               figsize: Tuple = (7, 5)):
    """Plot the most-selected target against the symmetric homolog."""
    if candidate_stats_df is None or candidate_stats_df.empty:
        print("No candidate-region statistics to plot")
        return None

    symmetric_rows = candidate_stats_df[
        candidate_stats_df['comparison_relation'] == 'symmetric'
    ]

    fig, ax = plt.subplots(figsize=figsize)
    if symmetric_rows.empty:
        first_row = candidate_stats_df.iloc[0]
        ax.axis('off')
        ax.set_title('Top Target vs Symmetric Homolog', fontsize=14, fontweight='bold')
        ax.text(
            0.5,
            0.58,
            f"Most-selected target: {first_row['top_region']}",
            ha='center',
            va='center',
            fontsize=14,
            fontweight='bold'
        )
        ax.text(
            0.5,
            0.42,
            "No contralateral homolog is defined\nfor this electrode.",
            ha='center',
            va='center',
            fontsize=11
        )
    else:
        row = symmetric_rows.iloc[0]
        labels = [row['top_region'], row['comparison_region']]
        counts = [int(row['top_count']), int(row['comparison_count'])]
        p_uncorrected = float(row['p_uncorrected'])
        p_fdr = float(row['p_fdr_bh'])

        bars = ax.bar(labels, counts, color=['#C73E3A', '#F0A202'],
                      edgecolor='black', alpha=0.93)
        ax.set_title('Top Target vs Symmetric Homolog', fontsize=14, fontweight='bold')
        ax.set_ylabel('Selection count')
        ax.grid(axis='y', alpha=0.25)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                str(int(height)),
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )

        ax.text(
            0.5,
            0.96,
            f"one-sided exact p = {_format_pvalue(p_uncorrected)}\n"
            f"FDR p = {_format_pvalue(p_fdr)}",
            transform=ax.transAxes,
            ha='center',
            va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.88),
            fontsize=10
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Candidate-region symmetric comparison figure saved to: {save_path}")
    plt.close(fig)
    return fig


def plot_candidate_region_pairwise_superiority(candidate_stats_df,
                                               save_path: str = None,
                                               figsize: Tuple = (11, 8)):
    """Plot FDR-corrected pairwise tests: top target versus each other electrode."""
    if candidate_stats_df is None or candidate_stats_df.empty:
        print("No candidate-region statistics to plot")
        return None

    plot_df = candidate_stats_df.copy()
    plot_df['neg_log10_fdr_p'] = -np.log10(
        np.clip(plot_df['p_fdr_bh'].astype(float), 1e-300, 1.0)
    )
    plot_df['is_significant'] = plot_df['p_fdr_bh'].astype(float) < 0.05
    plot_df = plot_df.sort_values(
        by=['is_significant', 'neg_log10_fdr_p', 'comparison_region'],
        ascending=[True, True, True]
    )

    colors = []
    for _, row in plot_df.iterrows():
        if row['comparison_relation'] == 'symmetric':
            colors.append('#F0A202')
        elif bool(row['is_significant']):
            colors.append('#3A7D44')
        else:
            colors.append('#7A8FA6')

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        plot_df['comparison_region'],
        plot_df['neg_log10_fdr_p'],
        color=colors,
        edgecolor='black',
        linewidth=0.5,
        alpha=0.93
    )

    threshold = -np.log10(0.05)
    ax.axvline(threshold, color='#C73E3A', linestyle='--',
               linewidth=1.5, label='FDR p = 0.05')
    top_region = str(plot_df['top_region'].iloc[0])
    ax.set_title(f'Pairwise Superiority: {top_region} vs Other Electrodes',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('-log10(FDR-corrected p-value)')
    ax.set_ylabel('Comparison electrode')
    ax.grid(axis='x', alpha=0.25)

    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        label = f"{int(row['top_count'])} vs {int(row['comparison_count'])}"
        ax.text(
            bar.get_width() + 0.03,
            bar.get_y() + bar.get_height() / 2,
            label,
            va='center',
            fontsize=8
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color='#3A7D44', label='FDR significant'),
        plt.Rectangle((0, 0), 1, 1, color='#F0A202', label='Symmetric homolog'),
        plt.Rectangle((0, 0), 1, 1, color='#7A8FA6', label='Not significant')
    ]
    ax.legend(handles=handles, frameon=False, loc='lower right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Candidate-region pairwise superiority figure saved to: {save_path}")
    plt.close(fig)
    return fig


def plot_candidate_region_statistics_dashboard(candidate_stats_df,
                                               channel_names: List[str],
                                               save_path: str = None,
                                               figsize: Tuple = (14, 9)):
    """Create a compact dashboard for final-target selection statistics."""
    counts, top_node, symmetric_node = _candidate_region_counts(candidate_stats_df, channel_names)
    if top_node is None:
        print("No candidate-region statistics to plot")
        return None

    top_region = channel_names[top_node]
    n_units = int(candidate_stats_df.iloc[0]['n_optimization_units'])
    top_count = int(candidate_stats_df.iloc[0]['top_count'])
    significant_count = int(np.sum(candidate_stats_df['p_fdr_bh'].astype(float) < 0.05))

    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], width_ratios=[1.0, 1.1])
    ax_summary = fig.add_subplot(grid[0, 0])
    ax_counts = fig.add_subplot(grid[0, 1])
    ax_pvalues = fig.add_subplot(grid[1, :])

    ax_summary.axis('off')
    symmetric_text = "N/A"
    if symmetric_node is not None:
        symmetric_rows = candidate_stats_df[candidate_stats_df['comparison_relation'] == 'symmetric']
        if not symmetric_rows.empty:
            row = symmetric_rows.iloc[0]
            symmetric_text = (
                f"{row['comparison_region']} "
                f"({int(row['top_count'])} vs {int(row['comparison_count'])}, "
                f"FDR p={_format_pvalue(row['p_fdr_bh'])})"
            )

    summary_text = (
        f"Most-selected target: {top_region}\n"
        f"Selection rate: {top_count}/{n_units} ({top_count / n_units * 100:.1f}%)\n"
        f"Symmetric comparison: {symmetric_text}\n"
        f"FDR-significant pairwise wins: {significant_count}/{len(candidate_stats_df)}"
    )
    ax_summary.text(
        0.03,
        0.94,
        summary_text,
        ha='left',
        va='top',
        fontsize=12,
        linespacing=1.5,
        bbox=dict(boxstyle='round', facecolor='#F7F7F7', edgecolor='#BDBDBD')
    )
    ax_summary.set_title('Final Target Statistics Summary', fontsize=13, fontweight='bold')

    top_order = np.argsort(counts)[::-1][:min(8, len(channel_names))]
    top_order = top_order[::-1]
    count_colors = ['#C73E3A' if idx == top_node else '#5B8DB8' for idx in top_order]
    ax_counts.barh([channel_names[idx] for idx in top_order], counts[top_order],
                   color=count_colors, edgecolor='black', linewidth=0.5)
    ax_counts.set_title('Top Selected Electrodes', fontsize=13, fontweight='bold')
    ax_counts.set_xlabel('Selection count')
    ax_counts.grid(axis='x', alpha=0.25)

    plot_df = candidate_stats_df.copy()
    plot_df['neg_log10_fdr_p'] = -np.log10(
        np.clip(plot_df['p_fdr_bh'].astype(float), 1e-300, 1.0)
    )
    plot_df = plot_df.sort_values('neg_log10_fdr_p', ascending=False).head(12).iloc[::-1]
    p_colors = np.where(plot_df['p_fdr_bh'].astype(float) < 0.05, '#3A7D44', '#7A8FA6')
    ax_pvalues.barh(plot_df['comparison_region'], plot_df['neg_log10_fdr_p'],
                    color=p_colors, edgecolor='black', linewidth=0.5)
    ax_pvalues.axvline(-np.log10(0.05), color='#C73E3A', linestyle='--', linewidth=1.4)
    ax_pvalues.set_title('Strongest Pairwise Superiority Tests', fontsize=13, fontweight='bold')
    ax_pvalues.set_xlabel('-log10(FDR-corrected p-value)')
    ax_pvalues.grid(axis='x', alpha=0.25)

    fig.suptitle('Final Candidate Stimulation Target Statistics',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Candidate-region statistics dashboard saved to: {save_path}")
    plt.close(fig)
    return fig


def plot_candidate_region_statistics(candidate_stats_df,
                                     channel_names: List[str],
                                     output_dir: str,
                                     prefix: str = 'final_target_statistics'):
    """Create all final-target statistical figures."""
    os.makedirs(output_dir, exist_ok=True)
    figure_specs = [
        (
            plot_candidate_region_statistics_dashboard,
            os.path.join(output_dir, f'{prefix}_dashboard.png'),
            (candidate_stats_df, channel_names)
        ),
        (
            plot_candidate_region_selection_counts,
            os.path.join(output_dir, f'{prefix}_selection_counts.png'),
            (candidate_stats_df, channel_names)
        ),
        (
            plot_candidate_region_symmetric_comparison,
            os.path.join(output_dir, f'{prefix}_symmetric_comparison.png'),
            (candidate_stats_df,)
        ),
        (
            plot_candidate_region_pairwise_superiority,
            os.path.join(output_dir, f'{prefix}_pairwise_superiority.png'),
            (candidate_stats_df,)
        )
    ]

    created = []
    for plot_func, save_path, args in figure_specs:
        fig = plot_func(*args, save_path=save_path)
        if fig is not None:
            created.append(save_path)

    return created


def plot_band_histogram(optimization_results: Dict,
                       band_names: List[str],
                       save_path: str = None,
                       figsize: Tuple = (10, 6)):
    """
    Plot histogram of optimal frequency bands across subjects.
    
    Parameters
    ----------
    optimization_results : dict
        Results from optimization
    band_names : list of str
        Names of frequency bands
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size (default: (10, 6))
    """
    # Extract optimal bands
    optimal_bands = []
    for subject_id, results in optimization_results.items():
        if 'best_solution' in results and results['best_solution'] is not None:
            optimal_bands.append(results['best_solution']['band'])
    
    if not optimal_bands:
        print("No optimization results to plot")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create histogram
    band_counts = np.bincount(optimal_bands, minlength=len(band_names))
    x_pos = np.arange(len(band_names))
    
    # Color scheme for bands
    colors = ['#9467bd', '#17becf', '#2ca02c', '#ff7f0e', '#d62728']
    
    bars = ax.bar(x_pos, band_counts, alpha=0.8, edgecolor='black',
                 color=colors[:len(band_names)])
    
    # Labels and formatting
    ax.set_xlabel('Frequency Band', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Optimal Frequency Bands',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(band_names, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add count and percentage labels
    total = len(optimal_bands)
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            percentage = (height / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({percentage:.1f}%)',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Band histogram saved to: {save_path}")
    
    # plt.show()
    
    return fig


def plot_node_band_heatmap(optimization_results: Dict,
                          channel_names: List[str],
                          band_names: List[str],
                          save_path: str = None,
                          figsize: Tuple = (14, 10)):
    """
    Plot 2D heatmap of node x band combinations.
    
    Parameters
    ----------
    optimization_results : dict
        Results from optimization
    channel_names : list of str
        Names of EEG channels
    band_names : list of str
        Names of frequency bands
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size (default: (14, 10))
    """
    # Extract optimal node-band pairs
    node_band_counts = np.zeros((len(channel_names), len(band_names)))
    
    for subject_id, results in optimization_results.items():
        if 'best_solution' in results and results['best_solution'] is not None:
            node = results['best_solution']['node']
            band = results['best_solution']['band']
            node_band_counts[node, band] += 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(node_band_counts, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Count', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(band_names)))
    ax.set_yticks(np.arange(len(channel_names)))
    ax.set_xticklabels(band_names, fontsize=11)
    ax.set_yticklabels(channel_names, fontsize=9)
    
    # Labels
    ax.set_xlabel('Frequency Band', fontsize=12, fontweight='bold')
    ax.set_ylabel('Channel/Node', fontsize=12, fontweight='bold')
    ax.set_title('2D Distribution: Optimal Node × Frequency Band',
                fontsize=14, fontweight='bold', pad=20)
    
    # Add count annotations
    for i in range(len(channel_names)):
        for j in range(len(band_names)):
            count = int(node_band_counts[i, j])
            if count > 0:
                text_color = 'white' if count > node_band_counts.max() / 2 else 'black'
                ax.text(j, i, str(count), ha='center', va='center',
                       color=text_color, fontsize=9, fontweight='bold')
    
    # Grid
    ax.set_xticks(np.arange(len(band_names)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(channel_names)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Node-band heatmap saved to: {save_path}")
    
    # plt.show()
    
    return fig


def plot_weighted_node_histogram(optimization_results: Dict,
                                channel_names: List[str],
                                top_k: int = None,
                                save_path: str = None,
                                figsize: Tuple = (12, 6)):
    """
    Plot weighted histogram of top-k stimulation nodes across subjects.
    """
    ranked_solutions = _collect_ranked_solutions(optimization_results, top_k=top_k)
    if not ranked_solutions:
        print("No ranked solutions to plot")
        return

    weights = np.zeros(len(channel_names), dtype=float)
    for sol in ranked_solutions:
        weight = float(sol.get('strength', 1.0))
        weights[int(sol['node'])] += weight

    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(len(channel_names))
    bars = ax.bar(x_pos, weights, alpha=0.7, color='teal', edgecolor='black')

    ax.set_xlabel('Channel/Node', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weighted Strength', fontsize=12, fontweight='bold')
    ax.set_title('Weighted Distribution of Top-K Stimulation Nodes',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(channel_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Weighted node histogram saved to: {save_path}")

    return fig


def plot_weighted_band_histogram(optimization_results: Dict,
                                band_names: List[str],
                                top_k: int = None,
                                save_path: str = None,
                                figsize: Tuple = (10, 6)):
    """
    Plot weighted histogram of top-k frequency bands across subjects.
    """
    ranked_solutions = _collect_ranked_solutions(optimization_results, top_k=top_k)
    if not ranked_solutions:
        print("No ranked solutions to plot")
        return

    weights = np.zeros(len(band_names), dtype=float)
    for sol in ranked_solutions:
        weight = float(sol.get('strength', 1.0))
        weights[int(sol['band'])] += weight

    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(len(band_names))

    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#7f7f7f']
    bars = ax.bar(x_pos, weights, alpha=0.8, edgecolor='black',
                 color=colors[:len(band_names)])

    ax.set_xlabel('Frequency Band', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weighted Strength', fontsize=12, fontweight='bold')
    ax.set_title('Weighted Distribution of Top-K Frequency Bands',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(band_names, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Weighted band histogram saved to: {save_path}")

    return fig


def plot_weighted_node_band_heatmap(optimization_results: Dict,
                                   channel_names: List[str],
                                   band_names: List[str],
                                   top_k: int = None,
                                   save_path: str = None,
                                   figsize: Tuple = (14, 10)):
    """
    Plot weighted heatmap of node x band combinations using top-k ranks.
    """
    ranked_solutions = _collect_ranked_solutions(optimization_results, top_k=top_k)
    if not ranked_solutions:
        print("No ranked solutions to plot")
        return

    node_band_weights = np.zeros((len(channel_names), len(band_names)), dtype=float)
    for sol in ranked_solutions:
        weight = float(sol.get('strength', 1.0))
        node_band_weights[int(sol['node']), int(sol['band'])] += weight

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(node_band_weights, cmap='YlGnBu', aspect='auto', interpolation='nearest')

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Weighted Strength', fontsize=12, fontweight='bold', rotation=270, labelpad=20)

    ax.set_xticks(np.arange(len(band_names)))
    ax.set_yticks(np.arange(len(channel_names)))
    ax.set_xticklabels(band_names, fontsize=11)
    ax.set_yticklabels(channel_names, fontsize=9)

    ax.set_xlabel('Frequency Band', fontsize=12, fontweight='bold')
    ax.set_ylabel('Channel/Node', fontsize=12, fontweight='bold')
    ax.set_title('Weighted Node x Frequency Band (Top-K)',
                fontsize=14, fontweight='bold', pad=20)

    for i in range(len(channel_names)):
        for j in range(len(band_names)):
            value = node_band_weights[i, j]
            if value > 0:
                text_color = 'white' if value > node_band_weights.max() / 2 else 'black'
                ax.text(j, i, f"{value:.2f}", ha='center', va='center',
                       color=text_color, fontsize=8, fontweight='bold')

    ax.set_xticks(np.arange(len(band_names)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(channel_names)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Weighted node-band heatmap saved to: {save_path}")

    return fig


def plot_pareto_fronts(optimization_results: Dict,
                      optimization_measures: List[str],
                      save_path: str = None,
                      figsize: Tuple = (15, 5)):
    """
    Plot Pareto fronts for a subset of subjects (3D if 3 objectives).
    
    Parameters
    ----------
    optimization_results : dict
        Results from optimization
    optimization_measures : list of str
        Names of optimized measures
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    n_objectives = len(optimization_measures)
    
    # Select first 3 subjects for visualization
    subjects_to_plot = list(optimization_results.keys())[:3]
    
    if n_objectives == 3:
        # 3D Pareto front
        fig = plt.figure(figsize=figsize)
        
        for idx, subject_id in enumerate(subjects_to_plot):
            ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
            
            results = optimization_results[subject_id]
            if 'best_front' not in results or not results['best_front']:
                continue
            
            # Extract objectives from Pareto front
            objectives = np.array([ind['objectives'] for ind in results['best_front']])
            
            # Plot Pareto front
            ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                      c='steelblue', s=50, alpha=0.6, edgecolors='black')
            
            # Highlight best solution
            if results['best_solution'] is not None:
                best_obj = results['best_solution']['objectives']
                ax.scatter([best_obj[0]], [best_obj[1]], [best_obj[2]],
                          c='crimson', s=200, marker='*', 
                          edgecolors='black', linewidths=2, label='Best')
            
            # Labels
            ax.set_xlabel(optimization_measures[0], fontsize=10)
            ax.set_ylabel(optimization_measures[1], fontsize=10)
            ax.set_zlabel(optimization_measures[2], fontsize=10)
            ax.set_title(f'{subject_id}', fontsize=11, fontweight='bold')
            ax.legend()
        
    else:
        # 2D plots for pairs of objectives
        n_plots = min(3, len(subjects_to_plot))
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        
        for idx, (subject_id, ax) in enumerate(zip(subjects_to_plot, axes)):
            results = optimization_results[subject_id]
            if 'best_front' not in results or not results['best_front']:
                continue
            
            objectives = np.array([ind['objectives'] for ind in results['best_front']])
            
            # Plot first two objectives
            ax.scatter(objectives[:, 0], objectives[:, 1],
                      c='steelblue', s=50, alpha=0.6, edgecolors='black')
            
            if results['best_solution'] is not None:
                best_obj = results['best_solution']['objectives']
                ax.scatter([best_obj[0]], [best_obj[1]],
                          c='crimson', s=200, marker='*',
                          edgecolors='black', linewidths=2, label='Best')
            
            ax.set_xlabel(optimization_measures[0], fontsize=10)
            ax.set_ylabel(optimization_measures[1], fontsize=10)
            ax.set_title(f'{subject_id}', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
    
    plt.suptitle('Pareto Fronts (Sample Subjects)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pareto fronts plot saved to: {save_path}")
    
    # plt.show()
    
    return fig


def plot_optimization_summary(optimization_results: Dict,
                             channel_names: List[str],
                             band_names: List[str],
                             optimization_measures: List[str],
                             output_dir: str,
                             top_k: int = None):
    """
    Create comprehensive summary plots for optimization results.
    
    Parameters
    ----------
    optimization_results : dict
        Results from optimization
    channel_names : list of str
        Names of EEG channels
    band_names : list of str
        Names of frequency bands
    optimization_measures : list of str
        Names of optimized measures
    output_dir : str
        Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating optimization visualization plots...")
    
    # 1. Node histogram
    print("  - Node distribution histogram")
    plot_node_histogram(
        optimization_results, 
        channel_names,
        save_path=os.path.join(output_dir, 'optimal_nodes_histogram.png')
    )

    # 1b. Weighted node histogram
    print("  - Weighted node distribution histogram")
    plot_weighted_node_histogram(
        optimization_results,
        channel_names,
        top_k=top_k,
        save_path=os.path.join(output_dir, 'weighted_nodes_histogram.png')
    )
    
    # 2. Band histogram
    print("  - Band distribution histogram")
    plot_band_histogram(
        optimization_results,
        band_names,
        save_path=os.path.join(output_dir, 'optimal_bands_histogram.png')
    )

    # 2b. Weighted band histogram
    print("  - Weighted band distribution histogram")
    plot_weighted_band_histogram(
        optimization_results,
        band_names,
        top_k=top_k,
        save_path=os.path.join(output_dir, 'weighted_bands_histogram.png')
    )
    
    # 3. Node-Band heatmap
    print("  - Node × Band heatmap")
    plot_node_band_heatmap(
        optimization_results,
        channel_names,
        band_names,
        save_path=os.path.join(output_dir, 'node_band_heatmap.png')
    )

    # 3b. Weighted node-band heatmap
    print("  - Weighted Node x Band heatmap")
    plot_weighted_node_band_heatmap(
        optimization_results,
        channel_names,
        band_names,
        top_k=top_k,
        save_path=os.path.join(output_dir, 'weighted_node_band_heatmap.png')
    )
    
    # 4. Pareto fronts
    print("  - Pareto fronts (sample subjects)")
    plot_pareto_fronts(
        optimization_results,
        optimization_measures,
        save_path=os.path.join(output_dir, 'pareto_fronts_sample.png')
    )
    
    print(f"\nAll plots saved to: {output_dir}")


def create_optimization_report(optimization_results: Dict,
                              channel_names: List[str],
                              band_names: List[str],
                              optimization_measures: List[str],
                              optimization_directions: Dict[str, str],
                              output_path: str,
                              top_k: int = None):
    """
    Create text report summarizing optimization results.
    
    Parameters
    ----------
    optimization_results : dict
        Results from optimization
    channel_names : list of str
        Names of EEG channels
    band_names : list of str
        Names of frequency bands
    optimization_measures : list of str
        Names of optimized measures
    optimization_directions : dict
        Optimization direction for each measure
    output_path : str
        Path to save report
    """
    objective_mode = None
    healthy_baselines = None
    stored_measures = None
    stored_directions = None
    for _, results in optimization_results.items():
        if isinstance(results, dict):
            if objective_mode is None:
                objective_mode = results.get('objective_mode')
            if healthy_baselines is None:
                healthy_baselines = results.get('healthy_measure_baselines')
            if stored_measures is None:
                stored_measures = results.get('optimization_measures')
            if stored_directions is None:
                stored_directions = results.get('optimization_directions')
        if objective_mode or healthy_baselines or stored_measures:
            break

    if stored_measures:
        optimization_measures = list(stored_measures)

    if not optimization_directions and stored_directions:
        optimization_directions = dict(stored_directions)

    unit_label = (
        "subject-band units"
        if any("::" in str(result_key) for result_key in optimization_results.keys())
        else "subjects"
    )

    def _distance_to_gt(values):
        if healthy_baselines is None:
            return None
        diffs = []
        for measure_name, value in zip(optimization_measures, values):
            baseline = healthy_baselines.get(measure_name)
            if baseline is None:
                return None
            denom = baseline if abs(baseline) > 1e-10 else 1.0
            diffs.append(abs(float(value) - float(baseline)) / abs(denom))
        return float(np.linalg.norm(diffs))

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("NSGA-II OPTIMIZATION RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write(f"Total optimization units: {len(optimization_results)} ({unit_label})\n")
        f.write(f"Number of nodes: {len(channel_names)}\n")
        f.write(f"Number of frequency bands: {len(band_names)}\n")
        f.write(f"Optimization measures: {', '.join(optimization_measures)}\n")
        f.write(f"Objective mode: {objective_mode or 'unknown'}\n\n")

        if healthy_baselines:
            f.write("Healthy baselines (GT):\n")
            for measure_name in optimization_measures:
                if measure_name in healthy_baselines:
                    f.write(f"  - {measure_name}: {healthy_baselines[measure_name]:.6f}\n")
            f.write("\n")
        
        # Optimization directions
        f.write("Optimization directions:\n")
        for measure, direction in optimization_directions.items():
            f.write(f"  - {measure}: {direction.upper()}\n")
        f.write("\n")

        f.write("Top-K ranking:\n")
        if objective_mode == 'distance_to_gt':
            f.write("  - Distance to GT (L2 norm of objective vector; 0 = perfect match)\n")
        else:
            f.write("  - Distance to ideal point (L2 norm of objectives)\n")
        f.write("  - Strength = 1 / rank (rank 1 is strongest)\n\n")
        
        # Node distribution
        optimal_nodes = [r['best_solution']['node'] for r in optimization_results.values() 
                        if r['best_solution'] is not None]
        node_counts = np.bincount(optimal_nodes, minlength=len(channel_names))
        
        f.write("="*80 + "\n")
        f.write("OPTIMAL STIMULATION NODES\n")
        f.write("="*80 + "\n")
        top_nodes = np.argsort(node_counts)[::-1][:10]
        for rank, node_idx in enumerate(top_nodes, 1):
            if node_counts[node_idx] > 0:
                f.write(f"  {rank}. {channel_names[node_idx]}: "
                       f"{node_counts[node_idx]} {unit_label} ({node_counts[node_idx]/len(optimal_nodes)*100:.1f}%)\n")
        f.write("\n")

        try:
            from statistics_utils import compute_candidate_region_selection_stats
            candidate_stats_df = compute_candidate_region_selection_stats(
                optimization_results,
                channel_names
            )
        except Exception as exc:
            candidate_stats_df = None
            f.write("Candidate-region statistics could not be computed: "
                    f"{exc}\n\n")

        if candidate_stats_df is not None and not candidate_stats_df.empty:
            top_region = candidate_stats_df.iloc[0]['top_region']
            top_count = int(candidate_stats_df.iloc[0]['top_count'])
            n_units = int(candidate_stats_df.iloc[0]['n_optimization_units'])
            f.write("="*80 + "\n")
            f.write("CANDIDATE REGION SELECTION STATISTICS\n")
            f.write("="*80 + "\n")
            f.write(
                f"Most-selected region: {top_region} "
                f"({top_count}/{n_units}, {top_count/n_units*100:.1f}%)\n"
            )
            f.write(
                "Test: exact one-sided binomial/McNemar-style count test "
                "for top region > comparison region; p_fdr_bh corrects all "
                "pairwise electrode comparisons.\n\n"
            )

            symmetric_rows = candidate_stats_df[
                candidate_stats_df['comparison_relation'] == 'symmetric'
            ]
            if not symmetric_rows.empty:
                row = symmetric_rows.iloc[0]
                f.write("Contralateral homolog comparison:\n")
                f.write(
                    f"  {row['top_region']} vs {row['comparison_region']}: "
                    f"{int(row['top_count'])} vs {int(row['comparison_count'])}, "
                    f"p={row['p_uncorrected']:.6g}, "
                    f"p_fdr_bh={row['p_fdr_bh']:.6g}\n\n"
                )
            else:
                f.write("Contralateral homolog comparison: not available for this top region.\n\n")

            f.write("Strongest pairwise comparisons against other regions:\n")
            for _, row in candidate_stats_df.head(10).iterrows():
                f.write(
                    f"  {row['top_region']} vs {row['comparison_region']} "
                    f"({row['comparison_relation']}): "
                    f"{int(row['top_count'])} vs {int(row['comparison_count'])}, "
                    f"p={row['p_uncorrected']:.6g}, "
                    f"p_fdr_bh={row['p_fdr_bh']:.6g}\n"
                )
            f.write("\n")
        
        # Band distribution
        optimal_bands = [r['best_solution']['band'] for r in optimization_results.values()
                        if r['best_solution'] is not None]
        band_counts = np.bincount(optimal_bands, minlength=len(band_names))
        
        f.write("="*80 + "\n")
        f.write("OPTIMAL FREQUENCY BANDS\n")
        f.write("="*80 + "\n")
        for band_idx, count in enumerate(band_counts):
            if count > 0:
                f.write(f"  {band_names[band_idx]}: "
                       f"{count} {unit_label} ({count/len(optimal_bands)*100:.1f}%)\n")
        f.write("\n")
        
        # Per-subject results
        f.write("="*80 + "\n")
        f.write("PER-SUBJECT RESULTS\n")
        f.write("="*80 + "\n")
        for subject_id, results in optimization_results.items():
            if results['best_solution'] is not None:
                sol = results['best_solution']
                f.write(f"\n{subject_id}:\n")
                f.write(f"  Optimal node: {channel_names[sol['node']]}\n")
                f.write(f"  Optimal band: {band_names[sol['band']]}\n")
                if sol.get('stimulation_duration') is not None:
                    f.write(f"  Stimulation duration: {sol['stimulation_duration']:.4f}\n")
                if sol.get('stimulation_amplitude') is not None:
                    f.write(f"  Stimulation amplitude: {sol['stimulation_amplitude']:.4f}\n")
                f.write(f"  Objectives: {sol['objectives']}\n")

                initial_metrics = results.get('initial_metrics')
                final_metrics = results.get('final_metrics')
                if initial_metrics is not None:
                    f.write(f"  Initial metrics: {initial_metrics}\n")
                if final_metrics is not None:
                    f.write(f"  Final metrics: {final_metrics}\n")

                if initial_metrics is not None:
                    dist_initial = _distance_to_gt(initial_metrics)
                    if dist_initial is not None:
                        f.write(f"  Distance to GT (initial): {dist_initial:.6f}\n")
                if final_metrics is not None:
                    dist_final = _distance_to_gt(final_metrics)
                    if dist_final is not None:
                        f.write(f"  Distance to GT (final): {dist_final:.6f}\n")
                f.write(f"  Pareto front size: {len(results['best_front'])}\n")

                ranked = []
                if results.get('top_solutions'):
                    ranked = results['top_solutions']
                elif results.get('best_front'):
                    ranked = _rank_best_front(
                        results['best_front'],
                        top_k or len(results['best_front']),
                        objective_mode=objective_mode
                    )

                if top_k is not None:
                    ranked = ranked[:top_k]

                if ranked:
                    f.write("  Top-K ranked solutions (distance to ideal):\n")
                    for ranked_sol in ranked:
                        node_name = channel_names[ranked_sol['node']]
                        band_name = band_names[ranked_sol['band']]
                        strength = float(ranked_sol.get('strength', 0.0))
                        distance = float(ranked_sol.get('distance', 0.0))
                        duration = ranked_sol.get('stimulation_duration')
                        amplitude = ranked_sol.get('stimulation_amplitude')
                        duration_text = f"{duration:.4f}" if duration is not None else "N/A"
                        amplitude_text = f"{amplitude:.4f}" if amplitude is not None else "N/A"
                        gt_distance = None
                        if objective_mode == 'distance_to_gt':
                            obj_vals = ranked_sol.get('objectives')
                            if obj_vals is not None:
                                try:
                                    gt_distance = float(np.linalg.norm(np.array(obj_vals, dtype=float)))
                                except (TypeError, ValueError):
                                    gt_distance = None

                        if gt_distance is not None:
                            distance_text = f"{gt_distance:.6f}"
                        else:
                            distance_text = f"{distance:.6f}"

                        f.write(
                            f"    {ranked_sol.get('rank', '?')}. "
                            f"Node: {node_name}, Band: {band_name}, "
                            f"Duration: {duration_text}, Amplitude: {amplitude_text}, "
                            f"Strength: {strength:.3f}, Distance: {distance_text}, "
                            f"Objectives: {ranked_sol.get('objectives')}\n"
                        )
    
    print(f"\nOptimization report saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # This would be called from the main optimization script
    print("Optimization visualization module loaded.")
