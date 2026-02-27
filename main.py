"""
Main pipeline for EEG connectivity analysis
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

# Import all modules
from config import (
    HC_DATA_PATH, PATIENT_DATA_PATH, OUTPUT_DIR,
    FREQUENCY_BANDS, CONNECTIVITY_METHODS, SELECTED_METHOD,
    NETWORK_MEASURES
)
from data_loader import load_group_data, verify_data_consistency
from signal_processing import process_subject_epochs
from connectivity import compute_all_connectivity
from network_measures import compute_network_measures_for_subjects, compute_all_network_measures
from statistics_utils import (
    compute_pvalue_matrix, compute_group_comparison_pvalues,
    extract_features_for_classification
)
from visualization import (
    plot_connectivity_matrices, plot_pvalue_matrices,
    plot_pvalue_matrices_per_band, plot_network_measures_pvalues,
    plot_top_feature_sets, plot_feature_importance,
    create_summary_report
)
from classification import (
    find_best_feature_triplets, get_best_triplet_details,
    create_classification_report
)

def create_output_directories():
    """Create output directories if they don't exist."""
    dirs = [
        OUTPUT_DIR,
        os.path.join(OUTPUT_DIR, 'figures'),
        os.path.join(OUTPUT_DIR, 'data'),
        os.path.join(OUTPUT_DIR, 'reports')
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print(f"Output directories created in: {OUTPUT_DIR}")


def main():
    """Main analysis pipeline."""
    
    print("\n" + "="*80)
    print("EEG CONNECTIVITY ANALYSIS PIPELINE")
    print("="*80)
    
    # Create output directories
    create_output_directories()
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    healthy_data = load_group_data(HC_DATA_PATH, group_name="Healthy")
    patient_data = load_group_data(PATIENT_DATA_PATH, group_name="Patient")
    
    print(f"\nLoaded:")
    print(f"  Healthy: {len(healthy_data)} subjects")
    print(f"  Patient: {len(patient_data)} subjects")
    
    # Verify consistency
    all_data = healthy_data + patient_data
    if not verify_data_consistency(all_data):
        raise ValueError("Data consistency check failed!")
    
    # ========================================================================
    # STEP 2: SIGNAL PROCESSING (EPOCHING & FILTERING)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: SIGNAL PROCESSING")
    print("="*80)
    
    all_subjects_filtered = {}
    
    for group_data, group_name in [(healthy_data, "Healthy"), (patient_data, "Patient")]:
        all_subjects_filtered[group_name] = {}
        
        for subject in group_data:
            subject_id = subject['subject_id']
            print(f"\nProcessing {subject_id} ({group_name})...")
            
            filtered_epochs = process_subject_epochs(subject['data'], subject['fs'])
            all_subjects_filtered[group_name][subject_id] = {
                'filtered_epochs': filtered_epochs,
                'fs': subject['fs'],
                'channels': subject['channels']
            }
    
    # ========================================================================
    # STEP 3: CONNECTIVITY ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: CONNECTIVITY ANALYSIS")
    print("="*80)
    
    connectivity_matrices = {}
    
    for group_name, subjects_dict in all_subjects_filtered.items():
        connectivity_matrices[group_name] = {}
        
        for subject_id, subject_data in subjects_dict.items():
            print(f"\nComputing connectivity for {subject_id} ({group_name})...")
            
            conn_results = compute_all_connectivity(
                subject_data['filtered_epochs'],
                subject_data['fs'],
                methods=CONNECTIVITY_METHODS
            )
            
            connectivity_matrices[group_name][subject_id] = conn_results
    
    # Save connectivity matrices
    np.save(os.path.join(OUTPUT_DIR, 'data', 'connectivity_matrices.npy'), 
            connectivity_matrices, allow_pickle=True)
    print(f"\nSaved connectivity matrices")
    
    # ========================================================================
    # STEP 4: VISUALIZATIONS 1-3
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: CONNECTIVITY VISUALIZATIONS")
    print("="*80)
    
    # Visualization 1: Average connectivity matrices per method
    print("\nCreating Visualization 1: Connectivity matrices per method...")
    plot_connectivity_matrices(
        connectivity_matrices,
        CONNECTIVITY_METHODS,
        output_path=os.path.join(OUTPUT_DIR, 'figures', 'viz1_connectivity_matrices.png')
    )
    
    # Visualization 2: P-value matrices per method
    print("\nCreating Visualization 2: P-value matrices per method...")
    plot_pvalue_matrices(
        connectivity_matrices,
        CONNECTIVITY_METHODS,
        output_path=os.path.join(OUTPUT_DIR, 'figures', 'viz2_pvalue_matrices.png')
    )
    
    # Visualization 3: P-value matrices per band
    print("\nCreating Visualization 3: P-value matrices per band...")
    plot_pvalue_matrices_per_band(
        connectivity_matrices,
        list(FREQUENCY_BANDS.keys()),
        output_path=os.path.join(OUTPUT_DIR, 'figures', 'viz3_pvalue_per_band.png')
    )
    
    # ========================================================================
    # STEP 5: NETWORK MEASURES
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: COMPUTING NETWORK MEASURES")
    print("="*80)
    print(f"Using connectivity method: {SELECTED_METHOD.upper()}")
    
    # Filter connectivity matrices to selected method only
    selected_connectivity = {}
    for group_name in connectivity_matrices.keys():
        selected_connectivity[group_name] = {}
        for subject_id, methods_dict in connectivity_matrices[group_name].items():
            if SELECTED_METHOD in methods_dict:
                selected_connectivity[group_name][subject_id] = {
                    SELECTED_METHOD: methods_dict[SELECTED_METHOD]
                }
    
    # Compute network measures
    network_measures = compute_network_measures_for_subjects(
        selected_connectivity,
        list(FREQUENCY_BANDS.keys())
    )
    
    # Save network measures
    np.save(os.path.join(OUTPUT_DIR, 'data', 'network_measures.npy'),
            network_measures, allow_pickle=True)
    print(f"\nSaved network measures")
    
    # ========================================================================
    # STEP 6: STATISTICAL ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: STATISTICAL ANALYSIS")
    print("="*80)
    
    # Compute p-values for group comparison
    pvalue_df = compute_group_comparison_pvalues(
        network_measures['Healthy'],
        network_measures['Patient'],
        NETWORK_MEASURES,
        list(FREQUENCY_BANDS.keys())
    )
    
    # Save p-values
    pvalue_df.to_csv(os.path.join(OUTPUT_DIR, 'data', 'network_measures_pvalues.csv'))
    print(f"\nSaved p-values to CSV")
    
    # Visualization 4: P-value heatmap
    print("\nCreating Visualization 4: Network measures p-values...")
    plot_network_measures_pvalues(
        pvalue_df,
        output_path=os.path.join(OUTPUT_DIR, 'figures', 'viz4_network_pvalues.png')
    )
    
    # ========================================================================
    # STEP 7: FEATURE EXTRACTION & CLASSIFICATION
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 7: CLASSIFICATION ANALYSIS")
    print("="*80)
    
    # Extract features for classification
    X, y, feature_names, subject_ids = extract_features_for_classification(
        network_measures,
        NETWORK_MEASURES,
        list(FREQUENCY_BANDS.keys()),
        SELECTED_METHOD
    )
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Number of subjects: {len(y)}")
    print(f"Group 0 (Healthy): {np.sum(y == 0)} subjects")
    print(f"Group 1 (Patient): {np.sum(y == 1)} subjects")
    
    # Find best feature triplets
    top_triplets_df, all_results = find_best_feature_triplets(
        X, y, feature_names, verbose=True
    )
    
    # Save results
    top_triplets_df.to_csv(
        os.path.join(OUTPUT_DIR, 'data', 'top_feature_triplets.csv'),
        index=False
    )
    
    # Visualization 5: Top feature triplets
    print("\nCreating Visualization 5: Top feature triplets...")
    plot_top_feature_sets(
        top_triplets_df,
        output_path=os.path.join(OUTPUT_DIR, 'figures', 'viz5_top_triplets.png')
    )
    
    # Get best triplet details
    best_triplet = get_best_triplet_details(all_results, rank=1)
    
    # Visualization 6: Feature importance
    print("\nCreating Visualization 6: Feature importance...")
    plot_feature_importance(
        best_triplet['feature_names'],
        best_triplet['coefficients'],
        output_path=os.path.join(OUTPUT_DIR, 'figures', 'viz6_feature_importance.png')
    )
    
    # Create classification report
    classification_report = create_classification_report(
        X, y, feature_names, all_results,
        output_path=os.path.join(OUTPUT_DIR, 'reports', 'classification_report.txt')
    )
    
    # ========================================================================
    # STEP 8: FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 8: CREATING SUMMARY REPORT")
    print("="*80)
    
    # Compile summary information
    summary_info = {
        'n_healthy': len(healthy_data),
        'n_patients': len(patient_data),
        'n_channels': len(all_data[0]['channels']),
        'bands': list(FREQUENCY_BANDS.keys()),
        'methods': CONNECTIVITY_METHODS,
        'selected_method': SELECTED_METHOD,
        'best_accuracy': best_triplet['accuracy'],
        'best_features': ', '.join(best_triplet['feature_names']),
        'significant_measures': '\n    '.join([
            f"{measure}: {band}"
            for measure in pvalue_df.index
            for band in pvalue_df.columns
            if pvalue_df.loc[measure, band] < 0.05
        ])
    }
    
    create_summary_report(
        summary_info,
        output_path=os.path.join(OUTPUT_DIR, 'reports', 'summary_report.png')
    )
    
    # ========================================================================
    # FINAL OUTPUT
    # ========================================================================
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  Figures:")
    print("    - viz1_connectivity_matrices.png")
    print("    - viz2_pvalue_matrices.png")
    print("    - viz3_pvalue_per_band.png")
    print("    - viz4_network_pvalues.png")
    print("    - viz5_top_triplets.png")
    print("    - viz6_feature_importance.png")
    print("  Data:")
    print("    - connectivity_matrices.npy")
    print("    - network_measures.npy")
    print("    - network_measures_pvalues.csv")
    print("    - top_feature_triplets.csv")
    print("  Reports:")
    print("    - classification_report.txt")
    print("    - summary_report.png")
    
    print(f"\n{'='*80}\n")
    
    return {
        'connectivity_matrices': connectivity_matrices,
        'network_measures': network_measures,
        'pvalue_df': pvalue_df,
        'classification_results': classification_report,
        'summary': summary_info
    }


if __name__ == "__main__":
    results = main()
