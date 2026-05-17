there is a project which is a pipeline for eeg processing and optimization for determining best stimulation node. 
I give you full code and then ask you for modifications. 

there is two major files which currently i run. first main.py and second run_optimization.py
main.py: 
"""
Main pipeline for EEG connectivity analysis
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import all modules
from config import (
    HC_DATA_PATH, PATIENT_DATA_PATH, OUTPUT_DIR,
    FREQUENCY_BANDS, CONNECTIVITY_METHODS, SELECTED_METHOD,
    NETWORK_MEASURES, STEP_TO_START, CONNECTIVITY_N_JOBS,
    CLASSIFICATION_MODE, CLASSIFICATION_MODEL, CLASSIFICATION_C,
    CLASSIFICATION_FEATURE_IMPORTANCE_TOP_N
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
    plot_top_feature_sets_per_band, plot_feature_importance_per_band,
    create_summary_report
)
from classification import (
    find_best_feature_triplets, get_best_triplet_details,
    create_classification_report, evaluate_all_features,
    create_full_feature_report, analyze_feature_importance
)


def _compute_subject_connectivity_task(task):
    """Worker task for per-subject connectivity computation."""
    group_name, subject_id, filtered_epochs, fs, methods = task
    conn_results = compute_all_connectivity(
        filtered_epochs,
        fs,
        methods=methods
    )
    return group_name, subject_id, conn_results


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

    if STEP_TO_START <= 1:
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

    if STEP_TO_START <= 2:
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

    if STEP_TO_START <= 3:
        connectivity_matrices = {group_name: {} for group_name in all_subjects_filtered.keys()}
        connectivity_tasks = []

        for group_name, subjects_dict in all_subjects_filtered.items():
            for subject_id, subject_data in subjects_dict.items():
                connectivity_tasks.append(
                    (
                        group_name,
                        subject_id,
                        subject_data['filtered_epochs'],
                        subject_data['fs'],
                        CONNECTIVITY_METHODS
                    )
                )

        total_tasks = len(connectivity_tasks)
        requested_workers = CONNECTIVITY_N_JOBS
        max_workers = (os.cpu_count() or 1) if requested_workers is None else max(1, int(requested_workers))
        max_workers = min(max_workers, total_tasks) if total_tasks > 0 else 1

        if max_workers <= 1 or total_tasks <= 1:
            print(f"\nRunning connectivity sequentially (workers={max_workers})")
            for i, task in enumerate(connectivity_tasks, start=1):
                group_name, subject_id, conn_results = _compute_subject_connectivity_task(task)
                connectivity_matrices[group_name][subject_id] = conn_results
                print(f"[{i}/{total_tasks}] Completed {subject_id} ({group_name})")
        else:
            print(f"\nRunning connectivity in parallel with {max_workers} processes...")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_subject = {
                    executor.submit(_compute_subject_connectivity_task, task): (task[0], task[1])
                    for task in connectivity_tasks
                }

                for i, future in enumerate(as_completed(future_to_subject), start=1):
                    group_name, subject_id = future_to_subject[future]
                    try:
                        _, _, conn_results = future.result()
                        connectivity_matrices[group_name][subject_id] = conn_results
                        print(f"[{i}/{total_tasks}] Completed {subject_id} ({group_name})")
                    except Exception as e:
                        print(f"[{i}/{total_tasks}] ERROR for {subject_id} ({group_name}): {e}")
        
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

    if STEP_TO_START <= 4:
        if STEP_TO_START == 4:
            connectivity_matrices = np.load(
                os.path.join(OUTPUT_DIR, 'data', 'connectivity_matrices.npy'),
                allow_pickle=True
            ).item()

            print("Loaded connectivity matrices")

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

    if STEP_TO_START <= 5:
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

    if STEP_TO_START <= 6:
        if STEP_TO_START == 6:
            network_measures = np.load(
                os.path.join(OUTPUT_DIR, 'data', 'network_measures.npy'),
                allow_pickle=True
            ).item()

            print("Loaded network measures")

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

    if STEP_TO_START <= 7:
        if STEP_TO_START == 7:
            network_measures = np.load(
                os.path.join(OUTPUT_DIR, 'data', 'network_measures.npy'),
                allow_pickle=True
            ).item()
            print("Loaded network measures")

            pvalue_csv = os.path.join(OUTPUT_DIR, 'data', 'network_measures_pvalues.csv')
            if os.path.exists(pvalue_csv):
                pvalue_df = pd.read_csv(pvalue_csv, index_col=0)
                print("Loaded network measure p-values")

        band_names = list(FREQUENCY_BANDS.keys())
        classification_reports_by_band = {}
        summary_rows = []

        if CLASSIFICATION_MODE == 'triplet':
            top_triplets_by_band = {}
            best_triplets_by_band = {}

            for band in band_names:
                print("\n" + "-"*80)
                print(f"Band-wise classification: {band.upper()}")
                print("-"*80)

                X, y, feature_names, subject_ids = extract_features_for_classification(
                    network_measures,
                    NETWORK_MEASURES,
                    [band],
                    SELECTED_METHOD
                )

                print(f"Feature matrix shape ({band}): {X.shape}")
                print(f"Number of subjects: {len(y)}")
                print(f"Group 0 (Healthy): {np.sum(y == 0)} subjects")
                print(f"Group 1 (Patient): {np.sum(y == 1)} subjects")

                top_triplets_df, all_results = find_best_feature_triplets(
                    X, y, feature_names, verbose=True
                )

                top_triplets_by_band[band] = top_triplets_df
                best_triplet = get_best_triplet_details(all_results, rank=1)
                best_triplets_by_band[band] = best_triplet

                top_triplets_df.to_csv(
                    os.path.join(OUTPUT_DIR, 'data', f'top_feature_triplets_{band}.csv'),
                    index=False
                )

                classification_report = create_classification_report(
                    X, y, feature_names, all_results,
                    output_path=os.path.join(OUTPUT_DIR, 'reports', f'classification_report_{band}.txt')
                )
                classification_reports_by_band[band] = classification_report

                summary_rows.append({
                    'band': band,
                    'best_accuracy': best_triplet['accuracy'],
                    'best_accuracy_std': best_triplet['accuracy_std'],
                    'best_features': ', '.join(best_triplet['feature_names'])
                })

            classification_summary_df = pd.DataFrame(summary_rows).sort_values(
                by='best_accuracy', ascending=False
            )
            classification_summary_df.to_csv(
                os.path.join(OUTPUT_DIR, 'data', 'classification_summary_by_band.csv'),
                index=False
            )

            print("\nCreating Visualization 5: Top feature triplets per band (4 panels per figure)...")
            plot_top_feature_sets_per_band(
                top_triplets_by_band,
                output_path=os.path.join(OUTPUT_DIR, 'figures', 'viz5_top_triplets_per_band.png')
            )

            print("\nCreating Visualization 6: Feature importance per band (4 panels per figure)...")
            plot_feature_importance_per_band(
                best_triplets_by_band,
                output_path=os.path.join(OUTPUT_DIR, 'figures', 'viz6_feature_importance_per_band.png'),
                top_n=CLASSIFICATION_FEATURE_IMPORTANCE_TOP_N
            )

        elif CLASSIFICATION_MODE == 'all_metrics':
            best_models_by_band = {}

            for band in band_names:
                print("\n" + "-"*80)
                print(f"Band-wise classification (all metrics): {band.upper()}")
                print("-"*80)

                X, y, feature_names, subject_ids = extract_features_for_classification(
                    network_measures,
                    NETWORK_MEASURES,
                    [band],
                    SELECTED_METHOD
                )

                print(f"Feature matrix shape ({band}): {X.shape}")
                print(f"Number of subjects: {len(y)}")
                print(f"Group 0 (Healthy): {np.sum(y == 0)} subjects")
                print(f"Group 1 (Patient): {np.sum(y == 1)} subjects")

                mean_acc, std_acc, coeffs = evaluate_all_features(
                    X,
                    y,
                    model_type=CLASSIFICATION_MODEL,
                    c_value=CLASSIFICATION_C
                )

                importance_df = analyze_feature_importance(coeffs, feature_names)
                top_features = ', '.join(
                    importance_df['Feature'].head(CLASSIFICATION_FEATURE_IMPORTANCE_TOP_N)
                )

                best_models_by_band[band] = {
                    'feature_names': feature_names,
                    'coefficients': coeffs,
                    'accuracy': mean_acc,
                    'accuracy_std': std_acc
                }

                importance_df.to_csv(
                    os.path.join(OUTPUT_DIR, 'data', f'feature_importance_all_metrics_{band}.csv'),
                    index=False
                )

                classification_report = create_full_feature_report(
                    X,
                    y,
                    feature_names,
                    model_type=CLASSIFICATION_MODEL,
                    c_value=CLASSIFICATION_C,
                    cv_accuracy=mean_acc,
                    cv_accuracy_std=std_acc,
                    cv_coefficients=coeffs,
                    output_path=os.path.join(
                        OUTPUT_DIR,
                        'reports',
                        f'classification_report_all_metrics_{band}.txt'
                    )
                )
                classification_reports_by_band[band] = classification_report

                summary_rows.append({
                    'band': band,
                    'best_accuracy': mean_acc,
                    'best_accuracy_std': std_acc,
                    'best_features': top_features
                })

            classification_summary_df = pd.DataFrame(summary_rows).sort_values(
                by='best_accuracy', ascending=False
            )
            classification_summary_df.to_csv(
                os.path.join(OUTPUT_DIR, 'data', 'classification_summary_by_band_all_metrics.csv'),
                index=False
            )

            print("\nCreating Visualization 6: Feature importance per band (4 panels per figure)...")
            plot_feature_importance_per_band(
                best_models_by_band,
                output_path=os.path.join(OUTPUT_DIR, 'figures', 'viz6_feature_importance_per_band.png'),
                top_n=CLASSIFICATION_FEATURE_IMPORTANCE_TOP_N
            )
        else:
            raise ValueError(
                f"Unsupported CLASSIFICATION_MODE '{CLASSIFICATION_MODE}'. "
                "Use 'triplet' or 'all_metrics'."
            )
    
    # ========================================================================
    # STEP 8: FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 8: CREATING SUMMARY REPORT")
    print("="*80)
    
    # Compile summary information
    n_healthy = len(network_measures.get('Healthy', {})) if 'network_measures' in locals() else 0
    n_patients = len(network_measures.get('Patient', {})) if 'network_measures' in locals() else 0

    n_channels = 'N/A'
    if 'connectivity_matrices' in locals():
        for group_data in connectivity_matrices.values():
            for subject_data in group_data.values():
                if SELECTED_METHOD in subject_data:
                    first_band_matrix = next(iter(subject_data[SELECTED_METHOD].values()))
                    n_channels = first_band_matrix.shape[0]
                    break
            if n_channels != 'N/A':
                break

    if 'classification_summary_df' in locals() and not classification_summary_df.empty:
        best_band_row = classification_summary_df.iloc[0]
        best_band_name = best_band_row['band']
        best_accuracy = best_band_row['best_accuracy']
        best_features = f"{best_band_name}: {best_band_row['best_features']}"
    else:
        best_accuracy = 0.0
        best_features = 'N/A'

    summary_info = {
        'n_healthy': n_healthy,
        'n_patients': n_patients,
        'n_channels': n_channels,
        'bands': list(FREQUENCY_BANDS.keys()),
        'methods': CONNECTIVITY_METHODS,
        'selected_method': SELECTED_METHOD,
        'best_accuracy': best_accuracy,
        'best_features': best_features,
        'significant_measures': '\n    '.join([
            f"{measure}: {band}"
            for measure in pvalue_df.index
            for band in pvalue_df.columns
            if pvalue_df.loc[measure, band] < 0.05
        ]) if 'pvalue_df' in locals() else 'N/A'
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
    if CLASSIFICATION_MODE == 'triplet':
        print("    - viz5_top_triplets_per_band.png (and additional parts if needed)")
    print("    - viz6_feature_importance_per_band.png (and additional parts if needed)")
    print("  Data:")
    print("    - connectivity_matrices.npy")
    print("    - network_measures.npy")
    print("    - network_measures_pvalues.csv")
    if CLASSIFICATION_MODE == 'triplet':
        print("    - top_feature_triplets_<band>.csv")
        print("    - classification_summary_by_band.csv")
    else:
        print("    - feature_importance_all_metrics_<band>.csv")
        print("    - classification_summary_by_band_all_metrics.csv")
    print("  Reports:")
    if CLASSIFICATION_MODE == 'triplet':
        print("    - classification_report_<band>.txt")
    else:
        print("    - classification_report_all_metrics_<band>.txt")
    print("    - summary_report.png")
    
    print(f"\n{'='*80}\n")
    
    return {
        'connectivity_matrices': connectivity_matrices if 'connectivity_matrices' in locals() else {},
        'network_measures': network_measures if 'network_measures' in locals() else {},
        'pvalue_df': pvalue_df if 'pvalue_df' in locals() else pd.DataFrame(),
        'classification_results': classification_reports_by_band if 'classification_reports_by_band' in locals() else {},
        'summary': summary_info
    }


if __name__ == "__main__":
    results = main()

###
run_optimization.py: 
