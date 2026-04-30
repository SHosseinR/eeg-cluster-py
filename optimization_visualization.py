"""
Visualization functions for NSGA-II optimization results
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os


def _rank_best_front(best_front: List[Dict], top_k: int) -> List[Dict]:
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
            ranked = _rank_best_front(results['best_front'], top_k or len(results['best_front']))

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
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("NSGA-II OPTIMIZATION RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write(f"Total subjects optimized: {len(optimization_results)}\n")
        f.write(f"Number of nodes: {len(channel_names)}\n")
        f.write(f"Number of frequency bands: {len(band_names)}\n")
        f.write(f"Optimization measures: {', '.join(optimization_measures)}\n\n")
        
        # Optimization directions
        f.write("Optimization directions:\n")
        for measure, direction in optimization_directions.items():
            f.write(f"  - {measure}: {direction.upper()}\n")
        f.write("\n")

        f.write("Top-K ranking:\n")
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
                       f"{node_counts[node_idx]} subjects ({node_counts[node_idx]/len(optimal_nodes)*100:.1f}%)\n")
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
                       f"{count} subjects ({count/len(optimal_bands)*100:.1f}%)\n")
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
                f.write(f"  Pareto front size: {len(results['best_front'])}\n")

                ranked = []
                if results.get('top_solutions'):
                    ranked = results['top_solutions']
                elif results.get('best_front'):
                    ranked = _rank_best_front(results['best_front'], top_k or len(results['best_front']))

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
                        f.write(
                            f"    {ranked_sol.get('rank', '?')}. "
                            f"Node: {node_name}, Band: {band_name}, "
                            f"Duration: {duration_text}, Amplitude: {amplitude_text}, "
                            f"Strength: {strength:.3f}, Distance: {distance:.6f}, "
                            f"Objectives: {ranked_sol.get('objectives')}\n"
                        )
    
    print(f"\nOptimization report saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # This would be called from the main optimization script
    print("Optimization visualization module loaded.")
