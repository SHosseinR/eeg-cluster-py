"""
Example usage of individual modules
This script demonstrates how to use the pipeline components independently
"""

import numpy as np
from config import HC_DATA_PATH, PATIENT_DATA_PATH, FREQUENCY_BANDS
from data_loader import load_group_data
from signal_processing import process_subject_epochs
from connectivity import compute_all_connectivity
from network_measures import compute_all_network_measures
from visualization import plot_connectivity_matrices
from utils import pretty_print_matrix

# =============================================================================
# EXAMPLE 1: Load and process a single subject
# =============================================================================
def example_single_subject():
    """Load and process data for a single subject."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Subject Processing")
    print("="*80)
    
    # Load one group
    healthy_data = load_group_data(HC_DATA_PATH, group_name="Healthy")
    
    # Get first subject
    subject = healthy_data[0]
    print(f"\nProcessing subject: {subject['subject_id']}")
    
    # Process epochs
    filtered_epochs = process_subject_epochs(subject['data'], subject['fs'])
    
    print(f"\nFiltered epochs created:")
    for band, epochs in filtered_epochs.items():
        print(f"  {band}: {epochs.shape}")
    
    return filtered_epochs, subject['fs']


# =============================================================================
# EXAMPLE 2: Compute connectivity for specific band and method
# =============================================================================
def example_connectivity_single_band():
    """Compute connectivity for a specific frequency band."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Connectivity for Single Band")
    print("="*80)
    
    from connectivity import compute_connectivity_for_band
    
    # Get filtered epochs from example 1
    filtered_epochs, fs = example_single_subject()
    
    # Compute PLV connectivity for alpha band
    band_name = 'alpha'
    # methods = ['plv', 'psi', 'gc', 'gc_tr']
    methods = ['plv']
    
    for method in methods:
        print(f"\nComputing {method.upper()} connectivity for {band_name} band...")
        connectivity_matrix = compute_connectivity_for_band(
            filtered_epochs, band_name, fs, method
        )
        pretty_print_matrix(connectivity_matrix, max_rows=5, max_columns=5)
        print(f"Connectivity matrix shape: {connectivity_matrix.shape}")
        print(f"Mean connectivity: {np.mean(connectivity_matrix):.4f}")
        print(f"Max connectivity: {np.max(connectivity_matrix):.4f}")
        print("="*50)
    
    return connectivity_matrix


# =============================================================================
# EXAMPLE 3: Compute all network measures for a single matrix
# =============================================================================
def example_network_measures():
    """Compute network measures for a connectivity matrix."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Network Measures")
    print("="*80)
    
    # Get connectivity matrix from example 2
    connectivity_matrix = example_connectivity_single_band()
    
    # Compute all network measures
    print("\nComputing network measures...")
    measures = compute_all_network_measures(connectivity_matrix)
    
    print("\nNetwork Measures:")
    print("-" * 40)
    for measure_name, value in measures.items():
        print(f"  {measure_name:30s}: {value:.6f}")
    
    return measures


# =============================================================================
# EXAMPLE 4: Compare two groups on a specific measure
# =============================================================================
def example_group_comparison():
    """Compare network measures between two groups."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Group Comparison")
    print("="*80)
    
    from scipy.stats import mannwhitneyu
    
    # Simulate some measures for demonstration
    # In practice, these would come from actual computations
    healthy_measures = np.random.randn(20) + 0.5  # 20 healthy subjects
    patient_measures = np.random.randn(15) + 0.3  # 15 patient subjects
    
    # Statistical test
    statistic, pvalue = mannwhitneyu(healthy_measures, patient_measures)
    
    print(f"\nGroup Comparison:")
    print(f"  Healthy (n={len(healthy_measures)}): {np.mean(healthy_measures):.4f} ± {np.std(healthy_measures):.4f}")
    print(f"  Patient (n={len(patient_measures)}): {np.mean(patient_measures):.4f} ± {np.std(patient_measures):.4f}")
    print(f"  Mann-Whitney U statistic: {statistic:.4f}")
    print(f"  P-value: {pvalue:.6f}")
    
    if pvalue < 0.05:
        print("  Result: Significant difference (p < 0.05)")
    else:
        print("  Result: No significant difference (p >= 0.05)")


# =============================================================================
# EXAMPLE 5: Custom visualization
# =============================================================================
def example_custom_visualization():
    """Create a custom visualization of connectivity matrix."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Visualization")
    print("="*80)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get connectivity matrix
    connectivity_matrix = example_connectivity_single_band()
    
    # Create custom plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Full matrix
    im1 = axes[0].imshow(connectivity_matrix, cmap='viridis', aspect='auto')
    axes[0].set_title('Full Connectivity Matrix', fontweight='bold')
    axes[0].set_xlabel('Target Node')
    axes[0].set_ylabel('Source Node')
    plt.colorbar(im1, ax=axes[0], label='Connectivity')
    
    # Plot 2: Thresholded matrix
    threshold = np.percentile(connectivity_matrix, 75)
    thresholded = np.where(connectivity_matrix > threshold, connectivity_matrix, 0)
    im2 = axes[1].imshow(thresholded, cmap='viridis', aspect='auto')
    axes[1].set_title(f'Thresholded (Top 25%)', fontweight='bold')
    axes[1].set_xlabel('Target Node')
    axes[1].set_ylabel('Source Node')
    plt.colorbar(im2, ax=axes[1], label='Connectivity')
    
    plt.tight_layout()
    plt.savefig('custom_connectivity_plot.png', dpi=300, bbox_inches='tight')
    print("\nSaved: custom_connectivity_plot.png")
    plt.show()


# =============================================================================
# EXAMPLE 6: Batch processing multiple subjects
# =============================================================================
def example_batch_processing():
    """Process multiple subjects efficiently."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Batch Processing")
    print("="*80)
    
    # Load all healthy subjects
    healthy_data = load_group_data(HC_DATA_PATH, group_name="Healthy")
    
    # Process all subjects
    all_connectivity = {}
    
    for subject in healthy_data[:3]:  # Process first 3 subjects as example
        subject_id = subject['subject_id']
        print(f"\nProcessing {subject_id}...")
        
        # Process
        filtered_epochs = process_subject_epochs(subject['data'], subject['fs'])
        connectivity = compute_all_connectivity(filtered_epochs, subject['fs'])
        
        all_connectivity[subject_id] = connectivity
    
    print(f"\nProcessed {len(all_connectivity)} subjects")
    
    return all_connectivity


# =============================================================================
# MAIN: Run all examples
# =============================================================================
def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("EEG CONNECTIVITY ANALYSIS - EXAMPLES")
    print("="*80)
    
    # Uncomment the examples you want to run:
    
    # Example 1: Single subject
    # example_single_subject()
    
    # Example 2: Connectivity for single band
    # example_connectivity_single_band()
    
    # Example 3: Network measures
    example_network_measures()
    
    # Example 4: Group comparison
    # example_group_comparison()
    
    # Example 5: Custom visualization
    # example_custom_visualization()
    
    # Example 6: Batch processing
    # example_batch_processing()
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
