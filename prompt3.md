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
side files (functions):
config.py: 
"""
Configuration file for EEG connectivity analysis
"""

import numpy as np

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# HC_DATA_PATH = "D:\\university\\projects\\graph-opt\\adhd-dataset\\preprocessed\\set2\\Control"  # UPDATE THIS
# PATIENT_DATA_PATH = "D:\\university\\projects\\graph-opt\\adhd-dataset\\preprocessed\\set2\\ADHD"      # UPDATE THIS
# OUTPUT_DIR = "./results-ADHD" 
HC_DATA_PATH = "D:\\university\\projects\\graph-opt\\paper-data\\EC\\set2\\HC"  # UPDATE THIS
PATIENT_DATA_PATH = "D:\\university\\projects\\graph-opt\\paper-data\\EC\\set2\\MDD"      # UPDATE THIS
OUTPUT_DIR = "./results-MDD" 
STEP_TO_START = 7

# ============================================================================
# SIGNAL PROCESSING PARAMETERS
# ============================================================================
EPOCH_DURATION = 10.0  # seconds
OVERLAP = 0.0  # No overlap between epochs

# Frequency bands (Hz)
FREQUENCY_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# ============================================================================
# CONNECTIVITY PARAMETERS
# ============================================================================
# Connectivity methods to compute
# CONNECTIVITY_METHODS = ['pdc', 'gc', 'psi', 'plv']
# CONNECTIVITY_METHODS = ['gc_tr', 'gc', 'psi', 'plv']
CONNECTIVITY_METHODS = ['gc']
CONNECTIVITY_N_JOBS = None  # None: use all available CPU cores, 1: disable multiprocessing

# Selected method for network analysis (change after visualization 2)
SELECTED_METHOD = 'gc'  # Change this based on visualization results

# Frequency resolution for spectral connectivity
FMIN = 1.0
FMAX = 45.0
N_FREQS = 100

# ============================================================================
# NETWORK MEASURES
# ============================================================================
NETWORK_MEASURES = [
    'global_efficiency',
    'local_efficiency',
    'clustering_coefficient',
    'transitivity',
    'modularity',
    'degree',
    'betweenness_centrality',
    'rich_club',
    'assortativity',
    'spectral_radius',
    'small_worldness',
    'diameter'
]

# ============================================================================
# STATISTICAL PARAMETERS
# ============================================================================
ALPHA_LEVEL = 0.05  # Significance level
N_PERMUTATIONS = 1000  # For permutation tests

# ============================================================================
# CLASSIFICATION PARAMETERS
# ============================================================================
N_FEATURES_COMBINATION = 3  # Number of features in each combination
N_FOLDS = 5  # Cross-validation folds
N_TOP_FEATURES = 10  # Number of top feature sets to report
RANDOM_STATE = 42  # For reproducibility
CLASSIFICATION_MODE = 'all_metrics'  # 'triplet' or 'all_metrics'
CLASSIFICATION_MODEL = 'linear_svm'  # 'linear_svm' or 'logistic' (used in all_metrics mode)
CLASSIFICATION_C = 0.1  # Regularization strength for linear models
CLASSIFICATION_FEATURE_IMPORTANCE_TOP_N = 10  # Top features to show in summaries/plots

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================
FIGURE_DPI = 300
FIGURE_SIZE = (12, 10)
CMAP_CONNECTIVITY = 'viridis'
CMAP_PVALUE = 'RdYlGn_r'  # Red for low p-values, green for high

# P-value thresholds for visualization
PVALUE_THRESHOLDS = [0.001, 0.01, 0.05]

# ============================================================================
# CHANNELS TO DROP
# ============================================================================
CHANNELS_TO_DROP = ['23A-23R', '24A-24R', 'A2-A1']
# CHANNELS_TO_DROP = []
###

data_loader.py:
"""
Data loading utilities for EEG data
"""

import os
import glob
import numpy as np
import mne
from config import CHANNELS_TO_DROP

def load_subject_epochs(subject_folder):
    """
    Load all .set files from a subject folder and combine them.
    
    Parameters
    ----------
    subject_folder : str
        Path to subject folder containing .set files
        
    Returns
    -------
    data : ndarray, shape (n_channels, n_samples)
        Combined EEG data from all files
    fs : float
        Sampling frequency
    channel_names : list
        Channel names
    """
    set_files = sorted(glob.glob(os.path.join(subject_folder, '*.set')))
    
    if not set_files:
        raise ValueError(f"No .set files found in {subject_folder}")
    
    print(f"Loading {len(set_files)} file(s) from {subject_folder}")
    
    all_data = []
    fs = None
    channel_names = None
    
    for set_file in set_files:
        print(f"  Loading: {os.path.basename(set_file)}")
        raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
        
        # Drop specified channels
        existing_to_drop = [ch for ch in CHANNELS_TO_DROP if ch in raw.ch_names]
        if existing_to_drop:
            raw.drop_channels(existing_to_drop)
            print(f"    Dropped channels: {existing_to_drop}")
        
        # Get data
        data = raw.get_data()
        
        # Store sampling frequency and channel names from first file
        if fs is None:
            fs = raw.info['sfreq']
            channel_names = raw.ch_names
        else:
            # Verify consistency across files
            if fs != raw.info['sfreq']:
                raise ValueError(f"Sampling frequency mismatch in {set_file}")
            if channel_names != raw.ch_names:
                raise ValueError(f"Channel names mismatch in {set_file}")
        
        all_data.append(data)
    
    # Concatenate all data along time axis
    combined_data = np.concatenate(all_data, axis=1)
    
    print(f"  Total data shape: {combined_data.shape}")
    print(f"  Sampling frequency: {fs} Hz")
    print(f"  Number of channels: {len(channel_names)}")
    
    return combined_data, fs, channel_names


def load_group_data(data_path, group_name="Group"):
    """
    Load data from all subjects in a group.
    
    Parameters
    ----------
    data_path : str
        Path to directory containing subject folders
    group_name : str
        Name of the group (for logging)
        
    Returns
    -------
    subjects_data : list of dict
        List of dictionaries containing data for each subject
        Each dict has keys: 'data', 'fs', 'channels', 'subject_id'
    """
    subject_folders = [f.path for f in os.scandir(data_path) if f.is_dir()]
    subject_folders = sorted(subject_folders)
    
    if not subject_folders:
        raise ValueError(f"No subject folders found in {data_path}")
    
    print(f"\n{'='*80}")
    print(f"Loading {group_name} data from: {data_path}")
    print(f"Found {len(subject_folders)} subjects")
    print(f"{'='*80}\n")
    
    subjects_data = []
    
    for i, subject_folder in enumerate(subject_folders):
        subject_id = os.path.basename(subject_folder)
        print(f"\n[{i+1}/{len(subject_folders)}] Processing {subject_id}")
        
        try:
            data, fs, channels = load_subject_epochs(subject_folder)
            
            subjects_data.append({
                'data': data,
                'fs': fs,
                'channels': channels,
                'subject_id': subject_id,
                'group': group_name
            })
            
        except Exception as e:
            print(f"  ERROR loading {subject_id}: {str(e)}")
            continue
    
    print(f"\n{'='*80}")
    print(f"Successfully loaded {len(subjects_data)}/{len(subject_folders)} subjects")
    print(f"{'='*80}\n")
    
    return subjects_data


def verify_data_consistency(subjects_data):
    """
    Verify that all subjects have consistent sampling frequency and channels.
    
    Parameters
    ----------
    subjects_data : list of dict
        Subject data from load_group_data
        
    Returns
    -------
    bool
        True if all subjects are consistent
    """
    if not subjects_data:
        return False
    
    reference_fs = subjects_data[0]['fs']
    reference_channels = subjects_data[0]['channels']
    
    for subject in subjects_data:
        if subject['fs'] != reference_fs:
            print(f"WARNING: Subject {subject['subject_id']} has different sampling frequency")
            return False
        if subject['channels'] != reference_channels:
            print(f"WARNING: Subject {subject['subject_id']} has different channels")
            return False
    
    print(f"✓ All subjects consistent:")
    print(f"  Sampling frequency: {reference_fs} Hz")
    print(f"  Number of channels: {len(reference_channels)}")
    print(f"  Channels: {reference_channels}")
    
    return True

##
signal_processing.py:
"""
Signal processing utilities for EEG data
"""

import numpy as np
from scipy import signal
from config import EPOCH_DURATION, OVERLAP, FREQUENCY_BANDS

def create_epochs(data, fs, epoch_duration=EPOCH_DURATION, overlap=OVERLAP):
    """
    Chunk continuous data into fixed-duration epochs.
    
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
        Continuous EEG data
    fs : float
        Sampling frequency
    epoch_duration : float
        Duration of each epoch in seconds
    overlap : float
        Overlap between epochs in seconds
        
    Returns
    -------
    epochs : ndarray, shape (n_epochs, n_channels, n_samples_per_epoch)
        Epoched data
    """
    n_channels, n_samples = data.shape
    
    # Calculate epoch parameters
    samples_per_epoch = int(epoch_duration * fs)
    step_size = int((epoch_duration - overlap) * fs)
    
    # Calculate number of epochs
    n_epochs = int((n_samples - samples_per_epoch) / step_size) + 1
    
    epochs = np.zeros((n_epochs, n_channels, samples_per_epoch))
    
    for i in range(n_epochs):
        start_idx = i * step_size
        end_idx = start_idx + samples_per_epoch
        
        if end_idx <= n_samples:
            epochs[i] = data[:, start_idx:end_idx]
        else:
            # Pad last epoch if necessary
            available_samples = n_samples - start_idx
            epochs[i, :, :available_samples] = data[:, start_idx:]
            # Zero-pad the rest
            epochs[i, :, available_samples:] = 0
    
    print(f"Created {n_epochs} epochs of {epoch_duration}s duration")
    print(f"Epoch shape: {epochs[0].shape}")
    
    return epochs


def bandpass_filter(data, fs, low_freq, high_freq, order=4):
    """
    Apply bandpass filter to data.
    
    Parameters
    ----------
    data : ndarray
        Input data (can be 2D or 3D)
    fs : float
        Sampling frequency
    low_freq : float
        Low cutoff frequency
    high_freq : float
        High cutoff frequency
    order : int
        Filter order
        
    Returns
    -------
    filtered_data : ndarray
        Filtered data with same shape as input
    """
    nyquist = fs / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Design Butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter along the last axis (time)
    filtered_data = signal.filtfilt(b, a, data, axis=-1)
    
    return filtered_data


def filter_epochs_by_bands(epochs, fs, frequency_bands=FREQUENCY_BANDS):
    """
    Filter epochs into different frequency bands.
    
    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_channels, n_samples)
        Epoched data
    fs : float
        Sampling frequency
    frequency_bands : dict
        Dictionary mapping band names to (low, high) frequency tuples
        
    Returns
    -------
    filtered_epochs : dict
        Dictionary mapping band names to filtered epochs
        Each value has shape (n_epochs, n_channels, n_samples)
    """
    filtered_epochs = {}
    
    print(f"\nFiltering epochs into frequency bands:")
    
    for band_name, (low_freq, high_freq) in frequency_bands.items():
        print(f"  {band_name}: {low_freq}-{high_freq} Hz")
        filtered_epochs[band_name] = bandpass_filter(epochs, fs, low_freq, high_freq)
    
    return filtered_epochs


def process_subject_epochs(data, fs):
    """
    Complete epoch processing pipeline for a single subject.
    
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
        Continuous EEG data
    fs : float
        Sampling frequency
        
    Returns
    -------
    filtered_epochs : dict
        Dictionary mapping band names to filtered epochs
        Each value has shape (n_epochs, n_channels, n_samples)
    """
    # Create epochs
    epochs = create_epochs(data, fs)
    
    # Filter into frequency bands
    filtered_epochs = filter_epochs_by_bands(epochs, fs)
    
    return filtered_epochs


def prepare_epochs_for_connectivity(filtered_epochs, band_name):
    """
    Prepare epochs from a specific band for connectivity analysis.
    
    Parameters
    ----------
    filtered_epochs : dict
        Output from filter_epochs_by_bands
    band_name : str
        Name of frequency band
        
    Returns
    -------
    epochs_array : ndarray, shape (n_epochs, n_channels, n_samples)
        Epochs ready for connectivity analysis
    """
    return filtered_epochs[band_name]

###
connectivity.py:
"""
Connectivity analysis using various methods
"""

import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time, phase_slope_index
from config import CONNECTIVITY_METHODS, FMIN, FMAX

def compute_plv(epochs, fs, fmin, fmax):
    """
    Compute Phase Locking Value (PLV) connectivity.
    
    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_channels, n_samples)
        Epoched data
    fs : float
        Sampling frequency
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
        
    Returns
    -------
    connectivity : ndarray, shape (n_channels, n_channels)
        PLV connectivity matrix
    """
    # Convert to MNE Epochs object
    info = mne.create_info(
        ch_names=[f'Ch{i}' for i in range(epochs.shape[1])],
        sfreq=fs,
        ch_types='eeg'
    )
    epochs_mne = mne.EpochsArray(epochs, info, verbose=False)
    
    # Compute PLV
    con = spectral_connectivity_epochs(
        epochs_mne,
        method='plv',
        mode='multitaper',
        sfreq=fs,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        verbose='ERROR'
    )
    
    # Get connectivity matrix (average across frequency)
    connectivity_matrix = con.get_data(output='dense')
    # print(f'{connectivity_matrix.shape=}')
    connectivity_matrix = np.mean(connectivity_matrix, axis=2)  # Average across freqs
    connectivity_matrix += connectivity_matrix.T

    return connectivity_matrix


def compute_psi(epochs, fs, fmin, fmax):
    """
    Compute Phase Slope Index (PSI) connectivity.
    
    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_channels, n_samples)
        Epoched data
    fs : float
        Sampling frequency
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
        
    Returns
    -------
    connectivity : ndarray, shape (n_channels, n_channels)
        PSI connectivity matrix (directed)
    """
    # Convert to MNE Epochs object
    info = mne.create_info(
        ch_names=[f'Ch{i}' for i in range(epochs.shape[1])],
        sfreq=fs,
        ch_types='eeg'
    )
    epochs_mne = mne.EpochsArray(epochs, info, verbose=False)
    
    # Compute PSI
    con = phase_slope_index(
        epochs_mne,
        mode='multitaper',
        sfreq=fs,
        fmin=fmin,
        fmax=fmax,
        verbose='ERROR'
    )

    # Get connectivity matrix
    connectivity_matrix = con.get_data(output='dense')
    connectivity_matrix = np.mean(connectivity_matrix, axis=2)  # Average across freqs

    psi_pos = np.zeros_like(connectivity_matrix)
    neg = connectivity_matrix < 0
    psi_pos[neg.T] = -connectivity_matrix[neg]   
    pos = connectivity_matrix > 0
    psi_pos[pos] = connectivity_matrix[pos]

    return psi_pos


def compute_granger_causality(epochs, fs, fmin, fmax):
    """
    Compute Spectral Granger Causality connectivity.
    
    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_channels, n_samples)
        Epoched data
    fs : float
        Sampling frequency
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
        
    Returns
    -------
    connectivity : ndarray, shape (n_channels, n_channels)
        Granger Causality connectivity matrix (directed)
    """
    # Convert to MNE Epochs object
    n_ch = epochs.shape[1]

    info = mne.create_info(
        ch_names=[f'Ch{i}' for i in range(epochs.shape[1])],
        sfreq=fs,
        ch_types='eeg'
    )
    epochs_mne = mne.EpochsArray(epochs, info, verbose=False)
    
    sources, targets = np.where(~np.eye(n_ch, dtype=bool))
    seeds   = [[int(i)] for i in sources]
    targs   = [[int(j)] for j in targets]
    indices = (seeds, targs)
    # indices = (sources.tolist(), targets.tolist())
    # print(f'{indices=}')
    
    # Compute Granger Causality
    con = spectral_connectivity_epochs(
        epochs_mne,
        method='gc',
        mode='multitaper',
        indices=indices,
        sfreq=fs,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        verbose='ERROR'
    )
    
    # Get connectivity matrix
    vals = con.get_data()
    vals = vals[:, 0] if vals.ndim == 2 else vals  # handle (n_conn, 1)

    gc_mat = np.full((n_ch, n_ch), np.nan, float)
    gc_mat[sources, targets] = vals
    np.fill_diagonal(gc_mat, 0.0)
    return gc_mat

def compute_granger_causality_tr(epochs, fs, fmin, fmax):
    """
    Compute Time Teversed Spectral Granger Causality connectivity.
    
    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_channels, n_samples)
        Epoched data
    fs : float
        Sampling frequency
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
        
    Returns
    -------
    connectivity : ndarray, shape (n_channels, n_channels)
        Time Reversed Granger Causality connectivity matrix (directed)
    """
    # Convert to MNE Epochs object
    n_ch = epochs.shape[1]

    info = mne.create_info(
        ch_names=[f'Ch{i}' for i in range(epochs.shape[1])],
        sfreq=fs,
        ch_types='eeg'
    )
    epochs_mne = mne.EpochsArray(epochs, info, verbose=False)
    
    sources, targets = np.where(~np.eye(n_ch, dtype=bool))
    seeds   = [[int(i)] for i in sources]
    targs   = [[int(j)] for j in targets]
    indices = (seeds, targs)
    # indices = (sources.tolist(), targets.tolist())
    # print(f'{indices=}')
    
    # Compute Time Reversed Granger Causality
    con = spectral_connectivity_epochs(
        epochs_mne,
        method='gc_tr',
        mode='multitaper',
        indices=indices,
        sfreq=fs,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        verbose='ERROR'
    )
    
    # Get connectivity matrix
    vals = con.get_data()
    vals = vals[:, 0] if vals.ndim == 2 else vals  # handle (n_conn, 1)

    gc_mat = np.full((n_ch, n_ch), np.nan, float)
    gc_mat[sources, targets] = vals
    np.fill_diagonal(gc_mat, 0.0)
    return gc_mat

def compute_pdc(epochs, fs, fmin, fmax):
    raise NotImplementedError


def compute_connectivity_for_band(filtered_epochs, band_name, fs, method='plv'):
    """
    Compute connectivity for a specific frequency band and method.
    
    Parameters
    ----------
    filtered_epochs : dict
        Dictionary of filtered epochs per band
    band_name : str
        Name of frequency band
    fs : float
        Sampling frequency
    method : str
        Connectivity method ('plv', 'psi', 'gc', 'pdc')
        
    Returns
    -------
    connectivity_matrix : ndarray, shape (n_channels, n_channels)
        Average connectivity matrix across epochs
    """
    epochs = filtered_epochs[band_name]
    
    # Get frequency range for this band
    from config import FREQUENCY_BANDS
    fmin, fmax = FREQUENCY_BANDS[band_name]
    
    # Compute connectivity based on method
    if method == 'plv':
        connectivity = compute_plv(epochs, fs, fmin, fmax)
    elif method == 'psi':
        connectivity = compute_psi(epochs, fs, fmin, fmax)
    elif method == 'gc':
        connectivity = compute_granger_causality(epochs, fs, fmin, fmax)
    elif method == 'gc_tr':
        connectivity = compute_granger_causality_tr(epochs, fs, fmin, fmax)
    elif method == 'pdc':
        connectivity = compute_pdc(epochs, fs, fmin, fmax)
    else:
        raise ValueError(f"Unknown connectivity method: {method}")
    
    return connectivity


def normalize_connectivity_matrix(matrix):
    """
    Normalize connectivity matrix to [0, 1] range.
    
    Parameters
    ----------
    matrix : ndarray, shape (n_channels, n_channels)
        Connectivity matrix
        
    Returns
    -------
    normalized : ndarray
        Normalized connectivity matrix
    """
    # Min-max normalization
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    
    if max_val - min_val > 0:
        normalized = (matrix - min_val) / (max_val - min_val)
    else:
        normalized = matrix
    
    return normalized


def compute_all_connectivity(filtered_epochs, fs, methods=CONNECTIVITY_METHODS):
    """
    Compute connectivity for all methods and all frequency bands.
    
    Parameters
    ----------
    filtered_epochs : dict
        Dictionary of filtered epochs per band
    fs : float
        Sampling frequency
    methods : list
        List of connectivity methods to compute
        
    Returns
    -------
    connectivity_results : dict
        Nested dictionary: {method: {band: normalized_matrix}}
    """
    from config import FREQUENCY_BANDS
    
    connectivity_results = {}
    
    for method in methods:
        print(f"\nComputing {method.upper()} connectivity:")
        connectivity_results[method] = {}
        
        for band_name in FREQUENCY_BANDS.keys():
            print(f"  {band_name}...", end=' ')
            
            try:
                conn_matrix = compute_connectivity_for_band(
                    filtered_epochs, band_name, fs, method
                )
                
                # Normalize
                conn_matrix_normalized = normalize_connectivity_matrix(conn_matrix)
                
                connectivity_results[method][band_name] = conn_matrix_normalized
                print("✓")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                # Store zeros as placeholder
                n_channels = filtered_epochs[band_name].shape[1]
                connectivity_results[method][band_name] = np.zeros((n_channels, n_channels))
    
    return connectivity_results

###
network_measures.py:
"""
Network measures computation using Brain Connectivity Toolbox (bctpy)
"""

import numpy as np
import bct
from config import NETWORK_MEASURES

def _prepare_weighted_directed_matrix(adjacency_matrix):
    """
    Prepare a weighted directed adjacency matrix for metric computation.
    """
    W = np.array(adjacency_matrix, dtype=float, copy=True)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(W, 0.0)
    # Most connectivity measures are non-negative; clip tiny negatives/noise.
    W[W < 0] = 0.0
    return W


def _to_length_matrix(weight_matrix):
    """
    Convert weights to connection lengths for shortest-path metrics.
    """
    L = np.full_like(weight_matrix, np.inf, dtype=float)
    positive = weight_matrix > 0
    L[positive] = 1.0 / weight_matrix[positive]
    np.fill_diagonal(L, 0.0)
    return L


def _global_efficiency_from_distance(distance_matrix):
    """
    Compute global efficiency from a shortest-path distance matrix.
    """
    D = np.array(distance_matrix, dtype=float, copy=True)
    np.fill_diagonal(D, np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_D = 1.0 / D
    inv_D[~np.isfinite(inv_D)] = 0.0

    n = D.shape[0]
    if n <= 1:
        return np.nan
    return np.sum(inv_D) / (n * (n - 1))


def compute_global_efficiency(adjacency_matrix):
    """
    Compute global efficiency of the network.
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    efficiency : float
        Global efficiency
    """
    W = _prepare_weighted_directed_matrix(adjacency_matrix)
    L = _to_length_matrix(W)
    D = bct.distance_wei(L)[0]
    return _global_efficiency_from_distance(D)


def compute_local_efficiency(adjacency_matrix):
    """
    Compute local efficiency (average across nodes).
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    efficiency : float
        Average local efficiency
    """
    W = _prepare_weighted_directed_matrix(adjacency_matrix)
    n = W.shape[0]
    if n <= 2:
        return np.nan

    local_eff = np.full(n, np.nan, dtype=float)

    for i in range(n):
        # Directed neighborhood: nodes with incoming OR outgoing edge to i.
        nbr_mask = (W[i, :] > 0) | (W[:, i] > 0)
        nbr_mask[i] = False
        nbr_idx = np.where(nbr_mask)[0]

        if nbr_idx.size < 2:
            continue

        subW = W[np.ix_(nbr_idx, nbr_idx)]
        subL = _to_length_matrix(subW)
        subD = bct.distance_wei(subL)[0]
        local_eff[i] = _global_efficiency_from_distance(subD)

    if np.all(np.isnan(local_eff)):
        return np.nan
    return np.nanmean(local_eff)


def compute_clustering_coefficient(adjacency_matrix):
    """
    Compute average clustering coefficient.
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    clustering : float
        Average clustering coefficient
    """
    W = _prepare_weighted_directed_matrix(adjacency_matrix)
    cc = bct.clustering_coef_wd(W)
    return np.mean(cc)


def compute_transitivity(adjacency_matrix):
    """
    Compute transitivity of the network.
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    transitivity : float
        Network transitivity
    """
    W = _prepare_weighted_directed_matrix(adjacency_matrix)
    return bct.transitivity_wd(W)


def compute_modularity(adjacency_matrix):
    """
    Compute modularity using Louvain algorithm.
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    modularity : float
        Modularity Q value
    """
    try:
        W = _prepare_weighted_directed_matrix(adjacency_matrix)
        _, Q = bct.modularity_dir(W)
        return Q
    except:
        try:
            W = _prepare_weighted_directed_matrix(adjacency_matrix)
            _, Q = bct.community_louvain(W)
            return Q
        except:
            # If modularity computation fails, return NaN
            return np.nan


def compute_degree(adjacency_matrix):
    """
    Compute average weighted degree.
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    degree : float
        Average degree
    """
    W = _prepare_weighted_directed_matrix(adjacency_matrix)
    strengths = bct.strengths_dir(W)
    return np.mean(strengths)


def compute_betweenness_centrality(adjacency_matrix):
    """
    Compute average betweenness centrality.
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    betweenness : float
        Average betweenness centrality
    """
    # Convert to connection-length matrix (inverse weights)
    # Avoid division by zero
    W = _prepare_weighted_directed_matrix(adjacency_matrix)
    length_matrix = np.zeros_like(W, dtype=float)
    positive = W > 0
    length_matrix[positive] = 1.0 / W[positive]
    bc = bct.betweenness_wei(length_matrix)
    return np.mean(bc)


def compute_rich_club(adjacency_matrix, k=None):
    """
    Compute rich club coefficient.
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
    k : int, optional
        Degree threshold. If None, uses median degree.
        
    Returns
    -------
    rich_club : float
        Rich club coefficient at degree k
    """
    try:
        W = _prepare_weighted_directed_matrix(adjacency_matrix)
        _, _, degrees = bct.degrees_dir(W)
        if k is None:
            k = int(np.median(degrees))

        rc = bct.rich_club_wd(W, klevel=k)
        
        if len(rc) > 0 and not np.isnan(rc[0]):
            return rc[0]
        else:
            return np.nan
    except:
        return np.nan


def compute_assortativity(adjacency_matrix):
    """
    Compute assortativity coefficient.
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    assortativity : float
        Assortativity coefficient
    """
    try:
        W = _prepare_weighted_directed_matrix(adjacency_matrix)
        A = (W > 0).astype(float)
        # Directed assortativity: average over out-in, in-out, out-out, in-in.
        vals = []
        for flag in (1, 2, 3, 4):
            try:
                vals.append(float(bct.assortativity_bin(A, flag=flag)))
            except:
                continue
        if len(vals) == 0:
            return np.nan
        return np.mean(vals)
    except:
        return np.nan


def compute_spectral_radius(adjacency_matrix):
    """
    Compute spectral radius (largest eigenvalue).
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    spectral_radius : float
        Largest eigenvalue magnitude
    """
    W = _prepare_weighted_directed_matrix(adjacency_matrix)
    eigenvalues = np.linalg.eigvals(W)
    return np.max(np.abs(eigenvalues))


def compute_small_worldness(adjacency_matrix):
    """
    Compute small-worldness coefficient.
    
    Small-worldness = (C/C_random) / (L/L_random)
    where C is clustering and L is path length
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    small_worldness : float
        Small-worldness coefficient
    """
    try:
        # Real network measures
        W = _prepare_weighted_directed_matrix(adjacency_matrix)
        C_real = np.mean(bct.clustering_coef_wd(W))
        
        # Convert to length matrix for path length computation
        length_matrix = _to_length_matrix(W)
        D = bct.distance_wei(length_matrix)[0]
        finite_D = D[np.isfinite(D) & (D > 0)]
        L_real = np.mean(finite_D) if finite_D.size > 0 else np.nan
        
        # Generate random network with same density
        n_nodes = adjacency_matrix.shape[0]
        n_edges = np.sum(W > 0)
        density = n_edges / (n_nodes * (n_nodes - 1))
        density = float(np.clip(density, 0.0, 1.0))
        
        # Random network
        rand_matrix = np.random.rand(n_nodes, n_nodes)
        threshold = np.percentile(rand_matrix, (1 - density) * 100)
        rand_matrix = np.where(rand_matrix > threshold, rand_matrix, 0.0)
        np.fill_diagonal(rand_matrix, 0)
        
        C_rand = np.mean(bct.clustering_coef_wd(rand_matrix))
        
        rand_length = _to_length_matrix(rand_matrix)
        D_rand = bct.distance_wei(rand_length)[0]
        finite_D_rand = D_rand[np.isfinite(D_rand) & (D_rand > 0)]
        L_rand = np.mean(finite_D_rand) if finite_D_rand.size > 0 else np.nan
        
        # Small-worldness
        gamma = C_real / C_rand if C_rand > 0 else np.nan
        lambda_ = L_real / L_rand if L_rand > 0 else np.nan
        sigma = gamma / lambda_ if lambda_ > 0 else np.nan
        
        return sigma
    except:
        return np.nan


def compute_diameter(adjacency_matrix):
    """
    Compute network diameter (longest shortest path).
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    diameter : float
        Network diameter
    """
    try:
        # Convert to length matrix
        W = _prepare_weighted_directed_matrix(adjacency_matrix)
        length_matrix = _to_length_matrix(W)
        D = bct.distance_wei(length_matrix)[0]
        
        # Get maximum finite distance
        finite_distances = D[np.isfinite(D) & (D > 0)]
        if len(finite_distances) > 0:
            return np.max(finite_distances)
        else:
            return np.nan
    except:
        return np.nan


measure_functions = {
    'global_efficiency': compute_global_efficiency,
    'local_efficiency': compute_local_efficiency,
    'clustering_coefficient': compute_clustering_coefficient,
    'transitivity': compute_transitivity,
    'modularity': compute_modularity,
    'degree': compute_degree,
    'betweenness_centrality': compute_betweenness_centrality,
    'rich_club': compute_rich_club,
    'assortativity': compute_assortativity,
    'spectral_radius': compute_spectral_radius,
    'small_worldness': compute_small_worldness,
    'diameter': compute_diameter
}


def compute_all_network_measures(adjacency_matrix):
    """
    Compute all network measures for a single connectivity matrix.
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    measures : dict
        Dictionary of measure names and values
    """
    measures = {}
    

    for measure_name, func in measure_functions.items():
        try:
            measures[measure_name] = func(adjacency_matrix)
        except Exception as e:
            print(f"  Warning: Failed to compute {measure_name}: {e}")
            measures[measure_name] = np.nan
    
    return measures


def compute_network_measures_for_subjects(connectivity_matrices_dict, band_names):
    """
    Compute network measures for all subjects, bands, and methods.
    
    Parameters
    ----------
    connectivity_matrices_dict : dict
        Dictionary with structure:
        {group: {subject_id: {method: {band: matrix}}}}
    band_names : list
        List of frequency band names
        
    Returns
    -------
    network_measures : dict
        Dictionary with structure:
        {group: {subject_id: {method: {band: {measure: value}}}}}
    """
    network_measures = {}
    
    for group in connectivity_matrices_dict.keys():
        network_measures[group] = {}
        
        for subject_id, subject_data in connectivity_matrices_dict[group].items():
            print(f"\nComputing network measures for {subject_id} ({group})")
            network_measures[group][subject_id] = {}
            
            for method in subject_data.keys():
                network_measures[group][subject_id][method] = {}
                
                for band in band_names:
                    conn_matrix = subject_data[method][band]
                    
                    print(f"  {method} - {band}...", end=' ')
                    measures = compute_all_network_measures(conn_matrix)
                    network_measures[group][subject_id][method][band] = measures
                    print("✓")
    
    return network_measures

###
statistics_utils.py:
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
###
visualization.py:
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
    
    # plt.show()


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
            print(f'{pvalue_matrix=}')
            
            # Plot p-values
            im = axes[idx].imshow(pvalue_matrix, cmap=CMAP_PVALUE, 
                                 aspect='auto', vmin=0, vmax=0.2)
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
    
    # plt.show()


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
    
    # plt.show()


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
    
    # plt.show()


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
    
    # plt.show()


def _chunk_items(items, chunk_size=4):
    """Split list-like items into fixed-size chunks."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def plot_top_feature_sets_per_band(top_features_by_band, output_path=None, panels_per_figure=4):
    """
    Visualization 5 (band-wise): top triplets table per band, grouped as 4 panels per figure.

    Parameters
    ----------
    top_features_by_band : dict
        {band_name: top_features_df}
    output_path : str, optional
        Base output path for figure files
    panels_per_figure : int
        Maximum number of band panels in each figure
    """
    band_items = list(top_features_by_band.items())
    chunks = _chunk_items(band_items, chunk_size=panels_per_figure)

    for part_idx, chunk in enumerate(chunks, start=1):
        n_panels = len(chunk)
        n_cols = 2 if n_panels > 1 else 1
        n_rows = 2 if n_panels > 2 else 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
        axes = np.array(axes).reshape(-1)

        for ax in axes:
            ax.axis('off')

        for ax_idx, (band, top_df) in enumerate(chunk):
            ax = axes[ax_idx]
            ax.axis('off')

            table = ax.table(
                cellText=top_df.values,
                colLabels=top_df.columns,
                cellLoc='left',
                loc='center',
                colWidths=[0.08, 0.64, 0.28]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1, 1.4)

            for col_idx in range(len(top_df.columns)):
                table[(0, col_idx)].set_facecolor('#4CAF50')
                table[(0, col_idx)].set_text_props(weight='bold', color='white')

            ax.set_title(f'{band.upper()} Band', fontsize=12, fontweight='bold', pad=8)

        fig.suptitle('Top Feature Triplets per Frequency Band', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.95))

        if output_path:
            if len(chunks) == 1:
                save_path = output_path
            else:
                base, ext = output_path.rsplit('.', 1)
                save_path = f"{base}_part{part_idx}.{ext}"
            plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Saved: {save_path}")


def plot_feature_importance_per_band(best_triplets_by_band, output_path=None, panels_per_figure=4, top_n=None):
    """
    Visualization 6 (band-wise): feature importance bars per band, grouped as 4 panels per figure.

    Parameters
    ----------
    best_triplets_by_band : dict
        {band_name: {'feature_names': [...], 'coefficients': [...]}}
    output_path : str, optional
        Base output path for figure files
    panels_per_figure : int
        Maximum number of band panels in each figure
    top_n : int, optional
        Limit to top-N absolute coefficients per band
    """
    band_items = list(best_triplets_by_band.items())
    chunks = _chunk_items(band_items, chunk_size=panels_per_figure)

    for part_idx, chunk in enumerate(chunks, start=1):
        n_panels = len(chunk)
        n_cols = 2 if n_panels > 1 else 1
        n_rows = 2 if n_panels > 2 else 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10))
        axes = np.array(axes).reshape(-1)

        for ax_idx, ax in enumerate(axes):
            if ax_idx >= n_panels:
                ax.axis('off')

        for ax_idx, (band, best_triplet) in enumerate(chunk):
            ax = axes[ax_idx]
            feature_names = best_triplet['feature_names']
            coefficients = np.array(best_triplet['coefficients'])
            sorted_indices = np.argsort(np.abs(coefficients))[::-1]
            if top_n is not None and top_n > 0:
                sorted_indices = sorted_indices[:top_n]

            sorted_features = [feature_names[i] for i in sorted_indices]
            sorted_coeffs = [coefficients[i] for i in sorted_indices]
            colors = ['#4CAF50' if c > 0 else '#F44336' for c in sorted_coeffs]

            y_pos = np.arange(len(sorted_features))
            ax.barh(y_pos, sorted_coeffs, color=colors, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_features, fontsize=8)
            ax.invert_yaxis()
            ax.axvline(x=0, color='black', linewidth=0.8)
            ax.grid(axis='x', alpha=0.3)
            ax.set_title(
                f"{band.upper()} (acc={best_triplet['accuracy']:.3f})",
                fontsize=11,
                fontweight='bold'
            )
            ax.set_xlabel('Coefficient')

        fig.suptitle('Best Triplet Feature Importance per Frequency Band', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.95))

        if output_path:
            if len(chunks) == 1:
                save_path = output_path
            else:
                base, ext = output_path.rsplit('.', 1)
                save_path = f"{base}_part{part_idx}.{ext}"
            plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Saved: {save_path}")


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
    
    # plt.show()


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

###
classification.py
"""
Classification and feature selection utilities
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from config import (
    N_FEATURES_COMBINATION, N_FOLDS, N_TOP_FEATURES, RANDOM_STATE,
    CLASSIFICATION_MODEL, CLASSIFICATION_C
)
from tqdm import tqdm


def _build_classifier(model_type, random_state=RANDOM_STATE, c_value=CLASSIFICATION_C):
    """Create a linear classifier that exposes coefficients for feature importance."""
    if model_type == 'linear_svm':
        return LinearSVC(C=c_value, random_state=random_state, max_iter=5000)
    if model_type == 'logistic':
        return LogisticRegression(C=c_value, random_state=random_state, max_iter=1000)
    raise ValueError(
        f"Unsupported model_type '{model_type}'. Use 'linear_svm' or 'logistic'."
    )

def evaluate_feature_triplet(X, y, feature_indices, n_folds=N_FOLDS, random_state=RANDOM_STATE):
    """
    Evaluate a triplet of features using cross-validation.
    
    Parameters
    ----------
    X : ndarray, shape (n_subjects, n_features)
        Feature matrix
    y : ndarray, shape (n_subjects,)
        Labels
    feature_indices : tuple
        Indices of the 3 features to use
    n_folds : int
        Number of cross-validation folds
    random_state : int
        Random seed
        
    Returns
    -------
    mean_accuracy : float
        Average accuracy across folds
    std_accuracy : float
        Standard deviation of accuracy
    coefficients : ndarray
        Logistic regression coefficients (from last fold)
    """
    # Select features
    X_subset = X[:, feature_indices]
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    accuracies = []
    all_coefficients = []
    
    for train_idx, test_idx in skf.split(X_subset, y):
        X_train, X_test = X_subset[train_idx], X_subset[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train logistic regression
        clf = LogisticRegression(random_state=random_state, max_iter=1000)
        clf.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = clf.predict(X_test_scaled)
        
        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        all_coefficients.append(clf.coef_[0])
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    # Return coefficients from last fold (representative)
    coefficients = all_coefficients[-1]
    
    return mean_accuracy, std_accuracy, coefficients


def find_best_feature_triplets(X, y, feature_names, n_top=N_TOP_FEATURES, verbose=True):
    """
    Find the best triplets of features for classification.
    
    Parameters
    ----------
    X : ndarray, shape (n_subjects, n_features)
        Feature matrix
    y : ndarray, shape (n_subjects,)
        Labels
    feature_names : list
        List of feature names
    n_top : int
        Number of top triplets to return
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with top feature triplets and their accuracies
    all_results : list
        List of all results (for further analysis)
    """
    n_features = X.shape[1]
    # print(f'{n_features=}')
    # Generate all combinations of 3 features
    all_triplets = list(combinations(range(n_features), N_FEATURES_COMBINATION))
    n_triplets = len(all_triplets)
    
    if verbose:
        print(f"\nEvaluating {n_triplets} feature triplets...")
        print(f"Using {N_FOLDS}-fold cross-validation")
        print("=" * 80)
    
    results = []
    
    for i, triplet in tqdm(enumerate(all_triplets)):
        if verbose and (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{n_triplets} ({(i+1)/n_triplets*100:.1f}%)")
        
        # Evaluate triplet
        mean_acc, std_acc, coeffs = evaluate_feature_triplet(X, y, triplet)
        
        # Store results
        triplet_names = [feature_names[idx] for idx in triplet]
        results.append({
            'triplet_indices': triplet,
            'triplet_names': triplet_names,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'coefficients': coeffs
        })
    
    # Sort by accuracy
    results.sort(key=lambda x: x['mean_accuracy'], reverse=True)
    
    # Create DataFrame with top results
    top_results = results[:n_top]
    
    df_data = []
    for rank, result in enumerate(top_results, 1):
        df_data.append({
            'Rank': rank,
            'Features': ' + '.join(result['triplet_names']),
            'Accuracy': f"{result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}"
        })
    
    results_df = pd.DataFrame(df_data)
    
    if verbose:
        print("\n" + "=" * 80)
        print("TOP 10 FEATURE TRIPLETS:")
        print("=" * 80)
        print(results_df.to_string(index=False))
        print("=" * 80)
    
    return results_df, results


def evaluate_all_features(
    X,
    y,
    model_type=CLASSIFICATION_MODEL,
    n_folds=N_FOLDS,
    random_state=RANDOM_STATE,
    c_value=CLASSIFICATION_C
):
    """
    Evaluate a model using all features with cross-validation.

    Parameters
    ----------
    X : ndarray, shape (n_subjects, n_features)
        Feature matrix
    y : ndarray, shape (n_subjects,)
        Labels
    model_type : str
        'linear_svm' or 'logistic'
    n_folds : int
        Number of cross-validation folds
    random_state : int
        Random seed
    c_value : float
        Regularization strength for linear models

    Returns
    -------
    mean_accuracy : float
        Average accuracy across folds
    std_accuracy : float
        Standard deviation of accuracy
    coefficients : ndarray
        Mean coefficients across folds
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    accuracies = []
    all_coefficients = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = _build_classifier(model_type, random_state=random_state, c_value=c_value)
        clf.fit(X_train_scaled, y_train)

        y_pred = clf.predict(X_test_scaled)
        accuracies.append(accuracy_score(y_test, y_pred))
        all_coefficients.append(clf.coef_[0])

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    coefficients = np.mean(all_coefficients, axis=0)

    return mean_accuracy, std_accuracy, coefficients


def get_best_triplet_details(results_list, rank=1):
    """
    Get detailed information about a specific ranked triplet.
    
    Parameters
    ----------
    results_list : list
        Output from find_best_feature_triplets (all_results)
    rank : int
        Rank of the triplet to retrieve (1-indexed)
        
    Returns
    -------
    triplet_info : dict
        Dictionary with triplet details
    """
    if rank < 1 or rank > len(results_list):
        raise ValueError(f"Rank must be between 1 and {len(results_list)}")
    
    result = results_list[rank - 1]
    
    return {
        'rank': rank,
        'feature_names': result['triplet_names'],
        'feature_indices': result['triplet_indices'],
        'accuracy': result['mean_accuracy'],
        'accuracy_std': result['std_accuracy'],
        'coefficients': result['coefficients']
    }


def perform_final_classification(X, y, feature_indices, random_state=RANDOM_STATE):
    """
    Perform final classification with best features on entire dataset.
    
    Parameters
    ----------
    X : ndarray, shape (n_subjects, n_features)
        Feature matrix
    y : ndarray, shape (n_subjects,)
        Labels
    feature_indices : tuple or list
        Indices of features to use
    random_state : int
        Random seed
        
    Returns
    -------
    model : LogisticRegression
        Trained model
    scaler : StandardScaler
        Fitted scaler
    train_accuracy : float
        Training accuracy
    """
    # Select features
    X_subset = X[:, feature_indices]
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)
    
    # Train model
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_scaled, y)
    
    # Compute training accuracy
    y_pred = model.predict(X_scaled)
    train_accuracy = accuracy_score(y, y_pred)
    
    return model, scaler, train_accuracy


def perform_final_classification_all_features(
    X,
    y,
    model_type=CLASSIFICATION_MODEL,
    random_state=RANDOM_STATE,
    c_value=CLASSIFICATION_C
):
    """
    Train a final model on all features for reporting and inspection.

    Returns
    -------
    model : classifier
        Trained model
    scaler : StandardScaler
        Fitted scaler
    train_accuracy : float
        Training accuracy
    coefficients : ndarray
        Coefficients from the fitted model
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = _build_classifier(model_type, random_state=random_state, c_value=c_value)
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    train_accuracy = accuracy_score(y, y_pred)

    return model, scaler, train_accuracy, model.coef_[0]


def analyze_feature_importance(coefficients, feature_names):
    """
    Analyze and rank feature importance from logistic regression.
    
    Parameters
    ----------
    coefficients : ndarray
        Logistic regression coefficients
    feature_names : list
        Names of features
        
    Returns
    -------
    importance_df : pd.DataFrame
        DataFrame with feature importance analysis
    """
    abs_coeffs = np.abs(coefficients)
    sorted_indices = np.argsort(abs_coeffs)[::-1]
    
    df_data = []
    for idx in sorted_indices:
        df_data.append({
            'Feature': feature_names[idx],
            'Coefficient': coefficients[idx],
            'Abs_Coefficient': abs_coeffs[idx],
            'Importance_Rank': len(df_data) + 1
        })
    
    importance_df = pd.DataFrame(df_data)
    
    return importance_df


def create_classification_report(X, y, feature_names, results_list, output_path=None):
    """
    Create a comprehensive classification report.
    
    Parameters
    ----------
    X : ndarray
        Feature matrix
    y : ndarray
        Labels
    feature_names : list
        Feature names
    results_list : list
        All results from feature selection
    output_path : str, optional
        Path to save report
        
    Returns
    -------
    report_dict : dict
        Dictionary with report information
    """
    # Get best triplet
    best = get_best_triplet_details(results_list, rank=1)
    
    # Train final model
    model, scaler, train_acc = perform_final_classification(
        X, y, best['feature_indices']
    )
    
    # Analyze importance
    importance_df = analyze_feature_importance(
        best['coefficients'], best['feature_names']
    )
    
    report_dict = {
        'best_triplet': best,
        'importance_df': importance_df,
        'train_accuracy': train_acc,
        'model': model,
        'scaler': scaler
    }
    
    if output_path:
        # Save report as text
        with open(output_path, 'w') as f:
            f.write("CLASSIFICATION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Best Feature Triplet (Rank 1):\n")
            f.write(f"  Features: {', '.join(best['feature_names'])}\n")
            f.write(f"  Cross-validation Accuracy: {best['accuracy']:.4f} ± {best['accuracy_std']:.4f}\n")
            f.write(f"  Training Accuracy: {train_acc:.4f}\n\n")
            
            f.write("Feature Importance:\n")
            f.write(importance_df.to_string(index=False))
            f.write("\n\n")
            
        print(f"Saved classification report: {output_path}")
    
    return report_dict


def create_full_feature_report(
    X,
    y,
    feature_names,
    model_type=CLASSIFICATION_MODEL,
    c_value=CLASSIFICATION_C,
    cv_accuracy=None,
    cv_accuracy_std=None,
    cv_coefficients=None,
    output_path=None
):
    """
    Create a classification report for models trained on all features.
    """
    if cv_accuracy is None or cv_accuracy_std is None or cv_coefficients is None:
        cv_accuracy, cv_accuracy_std, cv_coefficients = evaluate_all_features(
            X,
            y,
            model_type=model_type,
            c_value=c_value
        )

    model, scaler, train_acc, final_coefficients = perform_final_classification_all_features(
        X,
        y,
        model_type=model_type,
        c_value=c_value
    )

    importance_df = analyze_feature_importance(cv_coefficients, feature_names)

    report_dict = {
        'cv_accuracy': cv_accuracy,
        'cv_accuracy_std': cv_accuracy_std,
        'train_accuracy': train_acc,
        'importance_df': importance_df,
        'model': model,
        'scaler': scaler,
        'coefficients': final_coefficients,
        'model_type': model_type
    }

    if output_path:
        with open(output_path, 'w') as f:
            f.write("CLASSIFICATION ANALYSIS REPORT (ALL METRICS)\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Model Type: {model_type}\n")
            f.write(f"Cross-validation Accuracy: {cv_accuracy:.4f} ± {cv_accuracy_std:.4f}\n")
            f.write(f"Training Accuracy: {train_acc:.4f}\n\n")

            f.write("Feature Importance (by absolute coefficient):\n")
            f.write(importance_df.to_string(index=False))
            f.write("\n\n")

        print(f"Saved classification report: {output_path}")

    return report_dict

###

###
and now optimization part: 


run_optimization.py:
"""
Main script to run NSGA-II optimization for EEG connectivity
"""
import os
import sys
import numpy as np
from datetime import datetime

# Import configuration
from config import (
    OUTPUT_DIR, FREQUENCY_BANDS, SELECTED_METHOD
)
from optimization_config import (
    OPTIMIZATION_MEASURES, OPTIMIZATION_OUTPUT_DIR,
    OPTIMIZATION_RESULTS_FILE, OPTIMIZATION_FIGURES_DIR, OPTIMIZATION_N_JOBS,
    OPTIMIZATION_TOP_K, OPTIMIZATION_MODE
)

# Import optimization modules
from eeg_optimization import create_optimizer_from_config
from optimization_visualization import (
    plot_optimization_summary, create_optimization_report
)

# Import data loading (assuming these exist in your main pipeline)
from data_loader import load_group_data


def create_output_directories():
    """Create output directories for optimization results."""
    dirs = [
        OPTIMIZATION_OUTPUT_DIR,
        OPTIMIZATION_FIGURES_DIR
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print(f"Output directories created")


def load_data_for_optimization():
    """
    Load all required data for optimization.
    
    Returns
    -------
    connectivity_matrices : dict
        Pre-computed connectivity matrices
    network_measures : dict
        Pre-computed network measures
    subject_data : dict
        Raw subject data for baseline activation
    channel_names : list
        EEG channel names
    """
    print("\n" + "="*80)
    print("LOADING DATA FOR OPTIMIZATION")
    print("="*80)
    
    # Load connectivity matrices
    connectivity_path = os.path.join(OUTPUT_DIR, 'data', 'connectivity_matrices.npy')
    if not os.path.exists(connectivity_path):
        raise FileNotFoundError(f"Connectivity matrices not found at: {connectivity_path}")
    
    connectivity_matrices = np.load(connectivity_path, allow_pickle=True).item()
    print(f"✓ Loaded connectivity matrices")
    
    # Load network measures
    measures_path = os.path.join(OUTPUT_DIR, 'data', 'network_measures.npy')
    if not os.path.exists(measures_path):
        raise FileNotFoundError(f"Network measures not found at: {measures_path}")
    
    network_measures = np.load(measures_path, allow_pickle=True).item()
    print(f"✓ Loaded network measures")
    
    # Load raw subject data for baseline activation
    # This should be loaded from your original data files
    print(f"\nLoading raw EEG data for baseline activation...")
    
    # You need to specify your data paths
    # Replace these with your actual paths from config
    from config import HC_DATA_PATH, PATIENT_DATA_PATH
    
    healthy_data = load_group_data(HC_DATA_PATH, group_name="Healthy")
    patient_data = load_group_data(PATIENT_DATA_PATH, group_name="Patient")
    
    # Create subject_data dict
    subject_data = {}
    
    for subject in healthy_data:
        subject_data[subject['subject_id']] = {
            'data': subject['data'],
            'fs': subject['fs'],
            'channels': subject['channels'],
            'group': 'Healthy'
        }
    
    for subject in patient_data:
        subject_data[subject['subject_id']] = {
            'data': subject['data'],
            'fs': subject['fs'],
            'channels': subject['channels'],
            'group': 'Patient'
        }
    
    print(f"✓ Loaded raw data for {len(subject_data)} subjects")
    
    # Get channel names (assuming all subjects have same channels)
    first_subject = list(subject_data.values())[0]
    channel_names = first_subject['channels']
    
    print(f"✓ Number of channels: {len(channel_names)}")
    
    return connectivity_matrices, network_measures, subject_data, channel_names


def verify_optimization_requirements(connectivity_matrices, network_measures):
    """Verify that required data exists for optimization."""
    print("\n" + "="*80)
    print("VERIFYING OPTIMIZATION REQUIREMENTS")
    print("="*80)
    
    # Check that selected method exists
    print(f"\nSelected connectivity method: {SELECTED_METHOD}")
    
    # Check Patient group exists
    if 'Patient' not in network_measures:
        raise ValueError("Patient group not found in network measures!")
    
    patient_subjects = list(network_measures['Patient'].keys())
    print(f"Number of patient subjects: {len(patient_subjects)}")
    
    # Check that Healthy group exists
    if 'Healthy' not in network_measures:
        raise ValueError("Healthy group not found in network measures!")
    
    healthy_subjects = list(network_measures['Healthy'].keys())
    print(f"Number of healthy subjects: {len(healthy_subjects)}")
    
    # Verify measures exist
    print(f"\nOptimization measures: {OPTIMIZATION_MEASURES}")
    
    # Check at least one subject has required measures
    sample_subject = patient_subjects[0]
    sample_band = list(FREQUENCY_BANDS.keys())[0]
    
    if SELECTED_METHOD not in network_measures['Patient'][sample_subject]:
        raise ValueError(f"Method {SELECTED_METHOD} not found in network measures!")
    
    if sample_band not in network_measures['Patient'][sample_subject][SELECTED_METHOD]:
        raise ValueError(f"Band {sample_band} not found in network measures!")
    
    available_measures = list(network_measures['Patient'][sample_subject][SELECTED_METHOD][sample_band].keys())
    print(f"\nAvailable measures in data: {available_measures}")
    
    # Check that optimization measures exist
    for measure in OPTIMIZATION_MEASURES:
        if measure not in available_measures:
            raise ValueError(f"Optimization measure '{measure}' not found in data!")
    
    print("\n✓ All requirements verified!")


def main():
    """Main optimization pipeline."""
    
    print("\n" + "="*80)
    print("NSGA-II OPTIMIZATION PIPELINE FOR EEG CONNECTIVITY")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Optimization mode: {OPTIMIZATION_MODE}")
    
    # Create output directories
    create_output_directories()
    
    # Load data
    try:
        connectivity_matrices, network_measures, subject_data, channel_names = \
            load_data_for_optimization()
    except Exception as e:
        print(f"\nERROR loading data: {str(e)}")
        print("\nMake sure you have run the main pipeline first to generate:")
        print("  - connectivity_matrices.npy")
        print("  - network_measures.npy")
        return
    
    # Verify requirements
    try:
        verify_optimization_requirements(connectivity_matrices, network_measures)
    except Exception as e:
        print(f"\nERROR in verification: {str(e)}")
        return
    
    # Create optimizer
    print("\n" + "="*80)
    print("CREATING OPTIMIZER")
    print("="*80)
    
    optimizer = create_optimizer_from_config(
        connectivity_matrices=connectivity_matrices,
        network_measures=network_measures,
        subject_data=subject_data,
        frequency_bands=FREQUENCY_BANDS,
        channel_names=channel_names,
        selected_method=SELECTED_METHOD
    )
    
    print(f"\n✓ Optimizer created successfully")
    print(f"  - Connectivity method: {SELECTED_METHOD}")
    print(f"  - Optimization measures: {', '.join(OPTIMIZATION_MEASURES)}")
    print(f"  - Number of nodes: {optimizer.n_nodes}")
    print(f"  - Number of bands: {optimizer.n_bands}")
    
    # Run optimization for all patients
    print("\n" + "="*80)
    print("RUNNING OPTIMIZATION")
    print("="*80)
    effective_workers = (os.cpu_count()-1 or 1) if OPTIMIZATION_N_JOBS is None else max(1, int(OPTIMIZATION_N_JOBS))
    print(f"Optimization workers requested: {OPTIMIZATION_N_JOBS} (effective: {effective_workers})")
    
    try:
        optimization_results = optimizer.optimize_all_patients(
            verbose=True,
            n_jobs=OPTIMIZATION_N_JOBS
        )
    except Exception as e:
        print(f"\nERROR during optimization: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    results_path = os.path.join(OPTIMIZATION_OUTPUT_DIR, OPTIMIZATION_RESULTS_FILE)
    optimizer.save_results(results_path)
    
    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        plot_optimization_summary(
            optimization_results=optimization_results,
            channel_names=channel_names,
            band_names=list(FREQUENCY_BANDS.keys()),
            optimization_measures=OPTIMIZATION_MEASURES,
            output_dir=OPTIMIZATION_FIGURES_DIR,
            top_k=OPTIMIZATION_TOP_K
        )
    except Exception as e:
        print(f"\nERROR creating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Create text report
    print("\n" + "="*80)
    print("GENERATING REPORT")
    print("="*80)
    
    try:
        report_path = os.path.join(OPTIMIZATION_OUTPUT_DIR, 'optimization_report.txt')
        create_optimization_report(
            optimization_results=optimization_results,
            channel_names=channel_names,
            band_names=list(FREQUENCY_BANDS.keys()),
            optimization_measures=OPTIMIZATION_MEASURES,
            optimization_directions=optimizer.optimization_directions,
            output_path=report_path,
            top_k=OPTIMIZATION_TOP_K
        )
    except Exception as e:
        print(f"\nERROR creating report: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("OPTIMIZATION PIPELINE COMPLETE")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to:")
    print(f"  - Optimization results: {results_path}")
    print(f"  - Figures: {OPTIMIZATION_FIGURES_DIR}")
    print(f"  - Report: {report_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

###
optimization_config.py:
"""
Configuration for NSGA-II optimization of EEG connectivity
"""

# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================

# Network measures to optimize (select as many as needed from available measures)
# OPTIMIZATION_MEASURES = [
#     # 'global_efficiency',
#     'betweenness_centrality', 
#     # 'small_worldness'
#     # 'modularity',
#     'clustering_coefficient',
#     'degree',
# ]

OPTIMIZATION_MEASURES = [
    'global_efficiency',
    'betweenness_centrality', 
    'small_worldness'
]

# NSGA-II Algorithm parameters (using pymoo)
NSGA_CONFIG = {
    'population_size': 100,           # Population size for NSGA-II
    'n_generations': 50,              # Number of generations
    'crossover_prob': 0.9,            # Crossover probability
    'crossover_eta': 15.0,            # Distribution index for SBX crossover
    'mutation_prob': None,            # Mutation probability (None = 1/n_var)
    'mutation_eta': 20.0,             # Distribution index for polynomial mutation
    'seed': None,                     # Random seed for reproducibility (None = random)
}
OPTIMIZATION_N_JOBS = None  # None: use all available CPU cores, 1: disable multiprocessing

# Number of top-ranked Pareto solutions to keep per subject (used for weighted summaries)
OPTIMIZATION_TOP_K = 5

# Ranking pool for distance-based selection (grid + NSGA)
# - True: rank and select from Pareto front only
# - False: rank and select from all solutions (including dominated)
GRID_USE_PARETO_ONLY = True

# Optimization mode
# - 'nsga': NSGA-II with continuous stimulation duration/amplitude
# - 'grid': exhaustive node x band evaluation using fixed SIMULATION_CONFIG values
OPTIMIZATION_MODE = 'nsga'

# Objective mode (how objectives are computed)
# - 'directional': maximize/minimize based on Patient vs Healthy direction
# - 'distance_to_gt': minimize distance to Healthy mean (ground truth)
OPTIMIZATION_OBJECTIVE_MODE = 'distance_to_gt'

# State-space simulation parameters
# SIMULATION_CONFIG = {
#     'stimulation_duration': 1,      # Stimulation duration in seconds
#     'stimulation_amplitude': 1,     # Stimulation amplitude
#     'dt': 0.001,                      # Time step for simulation (seconds)
#     'stability_constant': 0.01,       # Constant for A matrix normalization (c in A/(c+lambda))
#     'leak': 0,                      # Identity damping for A' = A - leak * I
# }
SIMULATION_CONFIG = {
    'stimulation_duration': 10,      # Stimulation duration in seconds
    'stimulation_amplitude': 1,     # Stimulation amplitude
    'dt': 0.01,                      # Time step for simulation (seconds)
    'stability_constant': 0.01,       # Constant for A matrix normalization (c in A/(c+lambda))
    'leak': 1,                      # Identity damping for A' = A - leak * I
}

# Optimization bounds for stimulation parameters
# STIMULATION_DURATION_BOUNDS = (1, 20)
# STIMULATION_AMPLITUDE_BOUNDS = (0.03, 0.3)
# STIMULATION_LEAK_BOUNDS = (0.0, 2.0)
STIMULATION_DURATION_BOUNDS = (1, 20)
STIMULATION_AMPLITUDE_BOUNDS = (0.1, 2)
STIMULATION_LEAK_BOUNDS = (0.0, 2.0)

# Plasticity parameters
PLASTICITY_CONFIG = {
    'plasticity_enabled': True,       # Enable plasticity-based connectivity updates
    'plasticity_scaling': 1.0,        # Scaling factor for plasticity updates
}

# Debug/plotting defaults
OPTIMIZATION_DEBUG_SUBJECT = 'MDD S2  EC'

# Output paths for optimization
OPTIMIZATION_OUTPUT_DIR = 'results-MDD/optimization-nsga-distance_to_gt'
OPTIMIZATION_RESULTS_FILE = 'optimization_results.npy'
OPTIMIZATION_FIGURES_DIR = 'results-MDD/optimization-nsga-distance_to_gt/optimization/figures'

###
eeg_optimization.py:
"""
Complete EEG optimization pipeline using NSGA-II
"""
import os
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed

from optimization_config import (
    OPTIMIZATION_MEASURES, NSGA_CONFIG, SIMULATION_CONFIG, PLASTICITY_CONFIG, OPTIMIZATION_TOP_K,
    STIMULATION_DURATION_BOUNDS, STIMULATION_AMPLITUDE_BOUNDS, STIMULATION_LEAK_BOUNDS, OPTIMIZATION_MODE,
    GRID_USE_PARETO_ONLY, OPTIMIZATION_OBJECTIVE_MODE
)
from state_space_simulation import run_full_simulation
from plasticity import compute_plasticity_effect


_WORKER_OPTIMIZER = None
_WORKER_VERBOSE = False


def _init_optimizer_worker(optimizer, verbose):
    """Initialize each process with an optimizer instance."""
    global _WORKER_OPTIMIZER, _WORKER_VERBOSE
    _WORKER_OPTIMIZER = optimizer
    _WORKER_VERBOSE = verbose


def _optimize_subject_worker(subject_id: str):
    """Run optimization for one subject in worker process."""
    result = _WORKER_OPTIMIZER.optimize_subject(subject_id, verbose=_WORKER_VERBOSE)
    return subject_id, result


class EEGOptimizer:
    """
    EEG connectivity optimization using NSGA-II with state-space simulation.
    
    For each patient subject, optimizes stimulation parameters (node and frequency band)
    to improve network measures toward healthy controls.
    """
    
    def __init__(self,
                 connectivity_matrices: Dict,
                 network_measures: Dict,
                 subject_data: Dict,
                 frequency_bands: Dict,
                 channel_names: List[str],
                 selected_method: str,
                 optimization_measures: List[str],
                 optimization_mode: str = None,
                 objective_mode: str = None,
                 nsga_config: Dict = None,
                 simulation_config: Dict = None,
                 plasticity_config: Dict = None):
        """
        Initialize EEG optimizer.
        
        Parameters
        ----------
        connectivity_matrices : dict
            Nested dict: connectivity_matrices[group][subject][method][band]
        network_measures : dict
            Nested dict: network_measures[group][subject][method][band][measure]
        subject_data : dict
            Dict mapping subject_id to raw EEG data for computing baseline activation
        frequency_bands : dict
            Dict mapping band names to (low_freq, high_freq) tuples
        channel_names : list of str
            Names of EEG channels/nodes
        selected_method : str
            Connectivity method to use (e.g., 'plv', 'pdc', 'gc', 'psi')
        optimization_measures : list of str
            Names of network measures to optimize
        optimization_mode : str
            Optimization mode: 'nsga' (continuous) or 'grid' (discrete)
        objective_mode : str
            Objective mode: 'directional' or 'distance_to_gt'
        nsga_config : dict
            NSGA-II configuration parameters
        simulation_config : dict
            State-space simulation parameters
        plasticity_config : dict
            Plasticity update parameters
        """
        self.connectivity_matrices = connectivity_matrices
        self.network_measures = network_measures
        self.subject_data = subject_data
        self.frequency_bands = frequency_bands
        self.channel_names = channel_names
        self.selected_method = selected_method
        self.optimization_measures = optimization_measures
        if len(self.optimization_measures) == 0:
            raise ValueError("optimization_measures must contain at least one measure.")

        self.optimization_mode = (optimization_mode or OPTIMIZATION_MODE).strip().lower()
        if self.optimization_mode not in ("nsga", "grid"):
            raise ValueError(
                "optimization_mode must be 'nsga' or 'grid'. "
                f"Got: {self.optimization_mode!r}"
            )

        self.objective_mode = (objective_mode or OPTIMIZATION_OBJECTIVE_MODE).strip().lower()
        if self.objective_mode not in ("directional", "distance_to_gt"):
            raise ValueError(
                "objective_mode must be 'directional' or 'distance_to_gt'. "
                f"Got: {self.objective_mode!r}"
            )
        
        # Configuration
        self.nsga_config = nsga_config or NSGA_CONFIG
        self.simulation_config = simulation_config or SIMULATION_CONFIG
        self.plasticity_config = plasticity_config or PLASTICITY_CONFIG
        
        # Derived parameters
        self.n_nodes = len(channel_names)
        self.n_bands = len(frequency_bands)
        self.band_names = list(frequency_bands.keys())
        
        # Determine optimization directions (minimize or maximize)
        self.optimization_directions = self._determine_optimization_directions()
        # Fixed normalization baselines (one per optimization measure)
        self.healthy_measure_baselines = self._compute_healthy_measure_baselines()
        
        # Store results
        self.optimization_results = {}

    def _compute_healthy_measure_baselines(self) -> Dict[str, float]:
        """
        Compute constant normalization baselines from Healthy subjects.
        
        For each optimization measure, baseline is the average value across all
        Healthy subjects and all configured frequency bands for selected_method.
        
        Returns
        -------
        baselines : dict
            Mapping from measure name to baseline scalar
        """
        baselines = {}
        eps = 1e-10

        for measure in self.optimization_measures:
            healthy_values = []
            for subject_id in self.network_measures['Healthy'].keys():
                for band in self.band_names:
                    if self.selected_method in self.network_measures['Healthy'][subject_id]:
                        if band in self.network_measures['Healthy'][subject_id][self.selected_method]:
                            if measure in self.network_measures['Healthy'][subject_id][self.selected_method][band]:
                                val = self.network_measures['Healthy'][subject_id][self.selected_method][band][measure]
                                if np.isfinite(val):
                                    healthy_values.append(float(val))

            baseline = float(np.mean(healthy_values)) if healthy_values else 1.0
            if abs(baseline) < eps:
                print(f"  Warning: Healthy baseline for {measure} is near zero ({baseline:.4e}); using 1.0")
                baseline = 1.0

            baselines[measure] = baseline
            print(f"  Baseline ({measure}): {baseline:.6f}")

        return baselines
    
    def _determine_optimization_directions(self) -> Dict[str, str]:
        """
        Determine whether to minimize or maximize each measure.
        
        Based on comparing average measure values between Patient and Healthy groups:
        - If Patient avg > Healthy avg: MINIMIZE (reduce patient values toward healthy)
        - If Patient avg < Healthy avg: MAXIMIZE (increase patient values toward healthy)
        
        Returns
        -------
        directions : dict
            Mapping from measure name to 'minimize' or 'maximize'
        """
        directions = {}
        
        for measure in self.optimization_measures:
            # Compute average measure values for each group
            patient_values = []
            healthy_values = []
            
            # Collect all values across bands for this measure
            for subject_id in self.network_measures['Patient'].keys():
                for band in self.band_names:
                    if self.selected_method in self.network_measures['Patient'][subject_id]:
                        if band in self.network_measures['Patient'][subject_id][self.selected_method]:
                            if measure in self.network_measures['Patient'][subject_id][self.selected_method][band]:
                                val = self.network_measures['Patient'][subject_id][self.selected_method][band][measure]
                                patient_values.append(val)
            
            for subject_id in self.network_measures['Healthy'].keys():
                for band in self.band_names:
                    if self.selected_method in self.network_measures['Healthy'][subject_id]:
                        if band in self.network_measures['Healthy'][subject_id][self.selected_method]:
                            if measure in self.network_measures['Healthy'][subject_id][self.selected_method][band]:
                                val = self.network_measures['Healthy'][subject_id][self.selected_method][band][measure]
                                healthy_values.append(val)
            
            # Compute averages
            patient_avg = np.mean(patient_values) if patient_values else 0.0
            healthy_avg = np.mean(healthy_values) if healthy_values else 0.0
            
            # Determine direction
            if patient_avg > healthy_avg:
                directions[measure] = 'minimize'
            else:
                directions[measure] = 'maximize'
            
            print(f"  {measure}: Patient avg = {patient_avg:.4f}, Healthy avg = {healthy_avg:.4f} "
                  f"-> {directions[measure].upper()}")
        
        return directions
    
    def _compute_baseline_activation(self, subject_id: str) -> np.ndarray:
        """
        Compute baseline activation (average over time) for a subject.
        
        Parameters
        ----------
        subject_id : str
            Subject identifier
            
        Returns
        -------
        baseline : ndarray, shape (n_nodes,)
            Average activation for each node
        """
        if subject_id not in self.subject_data:
            # If data not available, use random baseline
            print(f"    Warning: No raw data for {subject_id}, using random baseline")
            # return np.random.randn(self.n_nodes)  # mean=0, std=1
            raise RuntimeError(f"No raw data for subject: {subject_id}")

        # Get raw data
        data = self.subject_data[subject_id]['data']  # shape: (n_channels, n_samples)

        # Compute mean over time
        baseline = np.mean(data, axis=1)
        
        # Z-score normalization (mean=0, std=1)
        # baseline = (baseline - np.mean(baseline)) / (np.std(baseline) + 1e-10)
        baseline = (baseline - baseline.min()) / (baseline.max() - baseline.min() + 1e-10)
        baseline = baseline * 0.9 + 0.1
        return baseline
    
    def _create_evaluation_function(self, subject_id: str, baseline_activation: np.ndarray):
        """
        Create evaluation function for NSGA-II for a specific subject.
        
        Parameters
        ----------
        subject_id : str
            Patient subject identifier
        baseline_activation : ndarray
            Baseline node activations
            
        Returns
        -------
        evaluate_func : callable
            Function that takes (node, band) and returns objectives array
        evaluate_with_details : callable
            Function that returns (objectives, measure_values)
        """
        def evaluate(
            node: int,
            band_idx: int,
            stimulation_duration: float = None,
            stimulation_amplitude: float = None,
            stimulation_leak: float = None
        ) -> np.ndarray:
            objectives, _ = self._evaluate_solution_details(
                subject_id=subject_id,
                baseline_activation=baseline_activation,
                node=node,
                band_idx=band_idx,
                stimulation_duration=stimulation_duration,
                stimulation_amplitude=stimulation_amplitude,
                stimulation_leak=stimulation_leak
            )
            return objectives

        def evaluate_with_details(
            node: int,
            band_idx: int,
            stimulation_duration: float = None,
            stimulation_amplitude: float = None,
            stimulation_leak: float = None
        ) -> Tuple[np.ndarray, np.ndarray]:
            return self._evaluate_solution_details(
                subject_id=subject_id,
                baseline_activation=baseline_activation,
                node=node,
                band_idx=band_idx,
                stimulation_duration=stimulation_duration,
                stimulation_amplitude=stimulation_amplitude,
                stimulation_leak=stimulation_leak
            )

        return evaluate, evaluate_with_details

    def _compute_objectives_from_measures(self, measure_values: List[float]) -> np.ndarray:
        """Convert raw measures into objectives based on objective mode."""
        objectives = []
        for measure_name, measure_value in zip(self.optimization_measures, measure_values):
            baseline = float(self.healthy_measure_baselines[measure_name])
            denom = baseline if abs(baseline) > 1e-10 else 1.0

            if self.objective_mode == "distance_to_gt":
                objectives.append(abs(float(measure_value) - baseline) / abs(denom))
            else:
                normalized_value = float(measure_value) / denom
                if self.optimization_directions[measure_name] == 'maximize':
                    objectives.append(-normalized_value)
                else:
                    objectives.append(normalized_value)

        return np.array(objectives, dtype=float)

    def _evaluate_solution_details(
        self,
        subject_id: str,
        baseline_activation: np.ndarray,
        node: int,
        band_idx: int,
        stimulation_duration: float = None,
        stimulation_amplitude: float = None,
        stimulation_leak: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate objectives and return raw measure values."""
        from network_measures import measure_functions

        band_name = self.band_names[band_idx]

        if stimulation_duration is None:
            stimulation_duration = float(self.simulation_config['stimulation_duration'])
        if stimulation_amplitude is None:
            stimulation_amplitude = float(self.simulation_config['stimulation_amplitude'])
        if stimulation_leak is None:
            stimulation_leak = float(self.simulation_config.get('leak', 0.0))

        original_matrix = self.connectivity_matrices['Patient'][subject_id][self.selected_method][band_name]

        sim_results = run_full_simulation(
            adjacency_matrix=original_matrix,
            baseline_activation=baseline_activation,
            stimulation_node=node,
            stimulation_duration=stimulation_duration,
            stimulation_amplitude=stimulation_amplitude,
            dt=self.simulation_config['dt'],
            stability_constant=self.simulation_config['stability_constant'],
            leak=stimulation_leak
        )

        if self.plasticity_config['plasticity_enabled']:
            updated_matrix = compute_plasticity_effect(
                adjacency_matrix=original_matrix,
                activation_ratios=sim_results['activation_ratios'],
                normalize=True,
                scaling=self.plasticity_config['plasticity_scaling']
            )
        else:
            updated_matrix = original_matrix

        measure_values = []
        for measure_name in self.optimization_measures:
            measure_func = measure_functions[measure_name]
            measure_value = measure_func(updated_matrix)
            measure_values.append(float(measure_value))

        objectives = self._compute_objectives_from_measures(measure_values)
        return objectives, np.array(measure_values, dtype=float)

    def _extract_initial_metrics(self, subject_id: str, band_name: str) -> Optional[np.ndarray]:
        """Extract baseline metrics from precomputed network measures."""
        try:
            band_data = self.network_measures['Patient'][subject_id][self.selected_method][band_name]
        except KeyError:
            return None

        values = []
        for measure_name in self.optimization_measures:
            if measure_name not in band_data:
                return None
            value = float(band_data[measure_name])
            if not np.isfinite(value):
                return None
            values.append(value)

        return np.array(values, dtype=float)

    def _compute_pareto_front(self, solutions: List[Dict]) -> List[Dict]:
        """
        Compute Pareto front for a list of solutions (minimization objectives).

        Parameters
        ----------
        solutions : list of dict
            Solutions with 'objectives' arrays

        Returns
        -------
        pareto_front : list of dict
            Non-dominated solutions
        """
        if not solutions:
            return []

        objectives = [np.asarray(sol['objectives'], dtype=float) for sol in solutions]
        n_solutions = len(solutions)
        is_dominated = [False] * n_solutions

        for i in range(n_solutions):
            if is_dominated[i]:
                continue
            for j in range(n_solutions):
                if i == j or is_dominated[i]:
                    continue
                if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    is_dominated[i] = True
                    break

        return [sol for idx, sol in enumerate(solutions) if not is_dominated[idx]]

    def _select_best_solution(self, best_front: List[Dict]) -> Dict:
        """Select the best solution from a candidate list by distance to ideal point."""
        if not best_front:
            return None
        if len(best_front) == 1:
            return best_front[0]

        objectives = np.array([sol['objectives'] for sol in best_front], dtype=float)
        if self.objective_mode == "distance_to_gt":
            ideal_point = np.zeros(objectives.shape[1], dtype=float)
        else:
            ideal_point = objectives.min(axis=0)
        distances = np.linalg.norm(objectives - ideal_point, axis=1)
        best_idx = int(np.argmin(distances))
        return best_front[best_idx]

    def _optimize_subject_grid(self, subject_id: str, evaluate_func: Callable, verbose: bool = True):
        """
        Exhaustive evaluation over node x band combinations (no NSGA).

        Returns
        -------
        best_front : list of dict
        history : list (empty)
        solutions : list of dict
        """
        duration = float(self.simulation_config['stimulation_duration'])
        amplitude = float(self.simulation_config['stimulation_amplitude'])
        leak = float(self.simulation_config.get('leak', 0.0))

        if verbose:
            print("Using discrete grid search over node x band")
            print(f"  Fixed stimulation duration: {duration}")
            print(f"  Fixed stimulation amplitude: {amplitude}")
            print(f"  Fixed leak: {leak}")

        solutions = []
        for node in range(self.n_nodes):
            for band_idx in range(self.n_bands):
                objectives, measure_values = evaluate_func(node, band_idx)
                solutions.append({
                    'node': node,
                    'band': band_idx,
                    'band_name': self.band_names[band_idx],
                    'stimulation_duration': None,
                    'stimulation_amplitude': None,
                    'leak': leak,
                    'objectives': objectives,
                    'measure_values': measure_values
                })

        best_front = self._compute_pareto_front(solutions)
        history = []
        return best_front, history, solutions

    def _rank_solutions(self, best_front: List[Dict], top_k: int) -> List[Dict]:
        """
        Rank solutions by distance to ideal point and keep top-k.

        Parameters
        ----------
        best_front : list of dict
            Candidate solutions
        top_k : int
            Number of solutions to keep

        Returns
        -------
        ranked : list of dict
            Ranked solutions with added 'rank', 'distance', and 'strength'
        """
        if not best_front:
            return []

        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = len(best_front)

        top_k = max(1, top_k)
        top_k = min(top_k, len(best_front))

        objectives = np.array([sol['objectives'] for sol in best_front])
        if self.objective_mode == "distance_to_gt":
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
                'band_name': sol['band_name'],
                'stimulation_duration': sol.get('stimulation_duration'),
                'stimulation_amplitude': sol.get('stimulation_amplitude'),
                'leak': sol.get('leak'),
                'objectives': sol['objectives'],
                'distance': float(distances[idx]),
                'rank': rank,
                'strength': 1.0 / float(rank)
            })

        return ranked
    
    def optimize_subject(self, subject_id: str, verbose: bool = True) -> Dict:
        """
        Run optimization for a single patient subject.
        
        Parameters
        ----------
        subject_id : str
            Patient subject identifier
        verbose : bool
            Print progress information
            
        Returns
        -------
        results : dict
            Optimization results including:
            - 'best_front': Pareto-optimal solutions
            - 'all_solutions': All evaluated solutions (Pareto + dominated)
            - 'best_solution': Single best solution
            - 'history': Optimization history
            - 'baseline_activation': Baseline activation used
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"OPTIMIZING SUBJECT: {subject_id}")
            print(f"{'='*80}")
        
        # Compute baseline activation
            print(f"Optimization mode: {self.optimization_mode.upper()}")
        baseline_activation = self._compute_baseline_activation(subject_id)
        # print(f'{baseline_activation=}')
        # min_nonzero = baseline_activation[baseline_activation > 0].min()
        # baseline_activation[baseline_activation == 0] = min_nonzero
        # print(f'{baseline_activation=}\n\n')

        if verbose:
            print(f"Baseline activation computed: mean={np.mean(baseline_activation):.4f}, "
                  f"std={np.std(baseline_activation):.4f}")
        
        # Create evaluation function
        evaluate_func, evaluate_with_details = self._create_evaluation_function(
            subject_id, baseline_activation
        )
        
        if self.optimization_mode == "grid":
            best_front, history, solutions = self._optimize_subject_grid(
                subject_id, evaluate_with_details, verbose=verbose
            )
            all_solutions = solutions
            ranking_pool = best_front if GRID_USE_PARETO_ONLY else solutions
            best_solution = self._select_best_solution(ranking_pool)
        else:
            from nsga_optimizer import NSGAIIOptimizer

            # Create optimizer
            optimizer = NSGAIIOptimizer(
                n_nodes=self.n_nodes,
                n_bands=self.n_bands,
                band_names=self.band_names,
                evaluate_func=evaluate_func,
                n_objectives=len(self.optimization_measures),
                duration_bounds=STIMULATION_DURATION_BOUNDS,
                amplitude_bounds=STIMULATION_AMPLITUDE_BOUNDS,
                leak_bounds=STIMULATION_LEAK_BOUNDS,
                population_size=self.nsga_config['population_size'],
                n_generations=self.nsga_config['n_generations'],
                crossover_prob=self.nsga_config['crossover_prob'],
                mutation_prob=self.nsga_config['mutation_prob'],
                # tournament_size=self.nsga_config['tournament_size']
            )
            
            # Run optimization
            best_front, history = optimizer.optimize(verbose=verbose)

            all_solutions = getattr(optimizer, "all_solutions", None) or best_front
            
            # Get single best solution (closest to GT or ideal point)
            ranking_pool = best_front if GRID_USE_PARETO_ONLY else all_solutions
            best_solution = self._select_best_solution(ranking_pool)

        # Rank top solutions using distance-to-ideal (strength = 1 / rank)
        top_solutions = self._rank_solutions(ranking_pool, OPTIMIZATION_TOP_K)

        initial_metrics = None
        final_metrics = None
        if best_solution is not None:
            band_idx = int(best_solution['band'])
            band_name = best_solution.get('band_name') or self.band_names[band_idx]
            initial_metrics = self._extract_initial_metrics(subject_id, band_name)

            if 'measure_values' in best_solution:
                final_metrics = np.array(best_solution['measure_values'], dtype=float)
            else:
                objectives, measure_values = evaluate_with_details(
                    node=int(best_solution['node']),
                    band_idx=band_idx,
                    stimulation_duration=best_solution.get('stimulation_duration'),
                    stimulation_amplitude=best_solution.get('stimulation_amplitude'),
                    stimulation_leak=best_solution.get('leak')
                )
                best_solution['objectives'] = objectives
                best_solution['measure_values'] = measure_values.tolist()
                final_metrics = measure_values
        
        # Package results
        results = {
            'subject_id': subject_id,
            'best_front': best_front,
            'all_solutions': all_solutions,
            'best_solution': best_solution,
            'top_solutions': top_solutions,
            'top_k': OPTIMIZATION_TOP_K,
            'history': history,
            'baseline_activation': baseline_activation,
            'n_nodes': self.n_nodes,
            'n_bands': self.n_bands,
            'band_names': self.band_names,
            'channel_names': self.channel_names,
            'optimization_mode': self.optimization_mode,
            'objective_mode': self.objective_mode,
            'optimization_measures': list(self.optimization_measures),
            'optimization_directions': dict(self.optimization_directions),
            'healthy_measure_baselines': dict(self.healthy_measure_baselines),
            'initial_metrics': initial_metrics.tolist() if initial_metrics is not None else None,
            'final_metrics': final_metrics.tolist() if final_metrics is not None else None
        }
        
        if verbose:
            print(f"\nBest solution:")
            print(f"  Node: {best_solution['node']} ({self.channel_names[best_solution['node']]})")
            print(f"  Band: {self.band_names[best_solution['band']]}")
            print(f"  Objectives: {best_solution['objectives']}")
        
        return results
    
    def optimize_all_patients(self, verbose: bool = True, n_jobs: int = None) -> Dict:
        """
        Run optimization for all patient subjects.
        
        Parameters
        ----------
        verbose : bool
            Print progress information
        n_jobs : int, optional
            Number of parallel worker processes. If None, uses all available cores.
            Use 1 to force sequential execution.
            
        Returns
        -------
        all_results : dict
            Mapping from subject_id to optimization results
        """
        print(f"\n{'='*80}")
        print(f"OPTIMIZING ALL PATIENT SUBJECTS")
        print(f"{'='*80}")
        print(f"Connectivity method: {self.selected_method.upper()}")
        print(f"Optimization measures: {', '.join(self.optimization_measures)}")
        print(f"\nOptimization directions:")
        
        # Get patient subject IDs
        patient_subjects = list(self.network_measures['Patient'].keys())
        
        print(f"\nTotal patient subjects: {len(patient_subjects)}")

        total_subjects = len(patient_subjects)
        requested_workers = n_jobs
        max_workers = (os.cpu_count() or 1) if requested_workers is None else max(1, int(requested_workers))
        max_workers = min(max_workers, total_subjects) if total_subjects > 0 else 1

        all_results = {}
        if max_workers <= 1 or total_subjects <= 1:
            print(f"Running optimization sequentially (workers={max_workers})")
            for i, subject_id in enumerate(patient_subjects):
                if verbose:
                    print(f"\n[{i+1}/{total_subjects}] ", end="")
                try:
                    results = self.optimize_subject(subject_id, verbose=verbose)
                    all_results[subject_id] = results
                except Exception as e:
                    print(f"ERROR optimizing {subject_id}: {str(e)}")
                    continue
        else:
            print(f"Running optimization in parallel with {max_workers} processes...")
            # Avoid interleaved detailed logs from multiple worker processes.
            subject_verbose = False
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_optimizer_worker,
                initargs=(self, subject_verbose)
            ) as executor:
                future_to_subject = {
                    executor.submit(_optimize_subject_worker, subject_id): subject_id
                    for subject_id in patient_subjects
                }

                for i, future in enumerate(as_completed(future_to_subject), start=1):
                    subject_id = future_to_subject[future]
                    try:
                        _, results = future.result()
                        all_results[subject_id] = results
                        print(f"[{i}/{total_subjects}] Completed {subject_id}")
                    except Exception as e:
                        print(f"[{i}/{total_subjects}] ERROR optimizing {subject_id}: {str(e)}")
                        continue
        
        self.optimization_results = all_results
        
        print(f"\n{'='*80}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"Successfully optimized: {len(all_results)}/{len(patient_subjects)} subjects")
        print(f"{'='*80}")
        
        return all_results
    
    def save_results(self, output_path: str):
        """Save optimization results to file."""
        np.save(output_path, self.optimization_results, allow_pickle=True)
        print(f"\nOptimization results saved to: {output_path}")
    
    @staticmethod
    def load_results(input_path: str) -> Dict:
        """Load optimization results from file."""
        results = np.load(input_path, allow_pickle=True).item()
        print(f"Optimization results loaded from: {input_path}")
        return results


def create_optimizer_from_config(connectivity_matrices: Dict,
                                network_measures: Dict,
                                subject_data: Dict,
                                frequency_bands: Dict,
                                channel_names: List[str],
                                selected_method: str) -> EEGOptimizer:
    """
    Create EEGOptimizer instance from configuration files.
    
    Parameters
    ----------
    connectivity_matrices : dict
        Connectivity matrices
    network_measures : dict
        Network measures
    subject_data : dict
        Subject raw data
    frequency_bands : dict
        Frequency band definitions
    channel_names : list
        Channel names
    selected_method : str
        Selected connectivity method
        
    Returns
    -------
    optimizer : EEGOptimizer
        Configured optimizer instance
    """
    optimizer = EEGOptimizer(
        connectivity_matrices=connectivity_matrices,
        network_measures=network_measures,
        subject_data=subject_data,
        frequency_bands=frequency_bands,
        channel_names=channel_names,
        selected_method=selected_method,
        optimization_measures=OPTIMIZATION_MEASURES,
        optimization_mode=OPTIMIZATION_MODE,
        objective_mode=OPTIMIZATION_OBJECTIVE_MODE,
        nsga_config=NSGA_CONFIG,
        simulation_config=SIMULATION_CONFIG,
        plasticity_config=PLASTICITY_CONFIG
    )
    
    return optimizer
###
optimization_visualization.py:
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
    for _, results in optimization_results.items():
        if isinstance(results, dict):
            if objective_mode is None:
                objective_mode = results.get('objective_mode')
            if healthy_baselines is None:
                healthy_baselines = results.get('healthy_measure_baselines')
            if stored_measures is None:
                stored_measures = results.get('optimization_measures')
        if objective_mode or healthy_baselines or stored_measures:
            break

    if stored_measures:
        optimization_measures = list(stored_measures)

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
        f.write(f"Total subjects optimized: {len(optimization_results)}\n")
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

###
another seperate side file which runs independently:
mdd_vs_healthy_raw_eeg_classification.py:
"""
MDD vs Healthy classification from raw EEG.

This script loads raw EEG (.set), extracts common spectral features
(band power, relative band power, Hjorth parameters, spectral entropy),
and evaluates linear models with nested cross-validation.
"""

import os
import numpy as np
import pandas as pd
from scipy import signal, integrate

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score
)

from config import (
    HC_DATA_PATH,
    PATIENT_DATA_PATH,
    OUTPUT_DIR,
    FREQUENCY_BANDS,
    FMIN,
    FMAX,
    EPOCH_DURATION,
    OVERLAP,
    N_FOLDS,
    RANDOM_STATE
)
from data_loader import load_group_data, verify_data_consistency
from signal_processing import create_epochs

# ============================================================================
# SETTINGS
# ============================================================================
USE_FEATURE_CACHE = True
FEATURE_CACHE_PATH = os.path.join(
    OUTPUT_DIR, 'data', 'mdd_vs_hc_raw_eeg_features.npz'
)
MAX_SUBJECTS_PER_GROUP = None  # set to an int for quick tests

MODEL_SPECS = {
    'linear_svm': {
        'estimator': LinearSVC(
            class_weight='balanced',
            max_iter=5000,
            random_state=RANDOM_STATE
        ),
        'param_grid': {'clf__C': [0.01, 0.1, 1.0, 10.0]}
    },
    'logistic': {
        'estimator': LogisticRegression(
            class_weight='balanced',
            solver='liblinear',
            max_iter=2000,
            random_state=RANDOM_STATE
        ),
        'param_grid': {'clf__C': [0.01, 0.1, 1.0, 10.0]}
    }
}

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def _compute_psd_welch(epochs, fs):
    n_samples = epochs.shape[-1]
    nperseg = min(int(fs * 2), n_samples)
    freqs, psd = signal.welch(epochs, fs=fs, nperseg=nperseg, axis=-1)
    return freqs, psd


def _bandpower_from_psd(psd, freqs, band):
    low, high = band
    idx = (freqs >= low) & (freqs <= high)
    if not np.any(idx):
        return np.zeros(psd.shape[:-1])
    if hasattr(np, "trapezoid"):
        return np.trapezoid(psd[..., idx], freqs[idx], axis=-1)
    if hasattr(np, "trapz"):
        return np.trapz(psd[..., idx], freqs[idx], axis=-1)
    return integrate.trapezoid(psd[..., idx], freqs[idx], axis=-1)


def _spectral_entropy(psd, freqs, band):
    low, high = band
    idx = (freqs >= low) & (freqs <= high)
    if not np.any(idx):
        return np.zeros(psd.shape[:-1])
    psd_band = psd[..., idx]
    psd_sum = np.sum(psd_band, axis=-1, keepdims=True)
    prob = np.divide(psd_band, psd_sum, out=np.zeros_like(psd_band), where=psd_sum > 0)
    entropy = -np.sum(prob * np.log(prob + 1e-12), axis=-1)
    return entropy


def _hjorth_parameters(epochs):
    diff1 = np.diff(epochs, axis=-1)
    diff2 = np.diff(diff1, axis=-1)

    var0 = np.var(epochs, axis=-1)
    var1 = np.var(diff1, axis=-1)
    var2 = np.var(diff2, axis=-1)

    activity = var0
    mobility = np.sqrt(np.divide(var1, var0, out=np.zeros_like(var1), where=var0 > 0))
    complexity = np.sqrt(np.divide(var2, var1, out=np.zeros_like(var2), where=var1 > 0))
    complexity = np.divide(complexity, mobility, out=np.zeros_like(complexity), where=mobility > 0)

    return activity, mobility, complexity


def extract_subject_features(data, fs, channel_names):
    epochs = create_epochs(data, fs, epoch_duration=EPOCH_DURATION, overlap=OVERLAP)
    freqs, psd = _compute_psd_welch(epochs, fs)

    total_power = _bandpower_from_psd(psd, freqs, (FMIN, FMAX))

    features = []
    feature_names = []

    for band_name, band in FREQUENCY_BANDS.items():
        band_power = _bandpower_from_psd(psd, freqs, band)
        log_power = np.log10(band_power + 1e-12)
        rel_power = np.divide(band_power, total_power, out=np.zeros_like(band_power), where=total_power > 0)

        mean_log_power = np.mean(log_power, axis=0)
        mean_rel_power = np.mean(rel_power, axis=0)

        for ch_idx, ch_name in enumerate(channel_names):
            features.append(mean_log_power[ch_idx])
            feature_names.append(f"log_power_{band_name}_{ch_name}")
            features.append(mean_rel_power[ch_idx])
            feature_names.append(f"rel_power_{band_name}_{ch_name}")

    activity, mobility, complexity = _hjorth_parameters(epochs)
    activity = np.mean(activity, axis=0)
    mobility = np.mean(mobility, axis=0)
    complexity = np.mean(complexity, axis=0)

    for ch_idx, ch_name in enumerate(channel_names):
        features.append(activity[ch_idx])
        feature_names.append(f"hjorth_activity_{ch_name}")
        features.append(mobility[ch_idx])
        feature_names.append(f"hjorth_mobility_{ch_name}")
        features.append(complexity[ch_idx])
        feature_names.append(f"hjorth_complexity_{ch_name}")

    entropy = _spectral_entropy(psd, freqs, (FMIN, FMAX))
    entropy = np.mean(entropy, axis=0)

    for ch_idx, ch_name in enumerate(channel_names):
        features.append(entropy[ch_idx])
        feature_names.append(f"spectral_entropy_{ch_name}")

    return np.array(features, dtype=float), feature_names


def build_dataset():
    if USE_FEATURE_CACHE and os.path.exists(FEATURE_CACHE_PATH):
        cache = np.load(FEATURE_CACHE_PATH, allow_pickle=True)
        return cache['X'], cache['y'], cache['feature_names'].tolist(), cache['subject_ids'].tolist()

    healthy_data = load_group_data(HC_DATA_PATH, group_name="Healthy")
    patient_data = load_group_data(PATIENT_DATA_PATH, group_name="Patient")

    if MAX_SUBJECTS_PER_GROUP is not None:
        healthy_data = healthy_data[:MAX_SUBJECTS_PER_GROUP]
        patient_data = patient_data[:MAX_SUBJECTS_PER_GROUP]

    all_data = healthy_data + patient_data
    if not verify_data_consistency(all_data):
        raise ValueError("Data consistency check failed")

    X_list = []
    y_list = []
    subject_ids = []
    feature_names_ref = None

    for subject in healthy_data:
        features, feature_names = extract_subject_features(
            subject['data'], subject['fs'], subject['channels']
        )
        if feature_names_ref is None:
            feature_names_ref = feature_names
        X_list.append(features)
        y_list.append(0)
        subject_ids.append(subject['subject_id'])

    for subject in patient_data:
        features, feature_names = extract_subject_features(
            subject['data'], subject['fs'], subject['channels']
        )
        if feature_names_ref is None:
            feature_names_ref = feature_names
        X_list.append(features)
        y_list.append(1)
        subject_ids.append(subject['subject_id'])

    X = np.vstack(X_list)
    y = np.array(y_list)

    os.makedirs(os.path.dirname(FEATURE_CACHE_PATH), exist_ok=True)
    np.savez(
        FEATURE_CACHE_PATH,
        X=X,
        y=y,
        feature_names=np.array(feature_names_ref, dtype=object),
        subject_ids=np.array(subject_ids, dtype=object)
    )

    return X, y, feature_names_ref, subject_ids

# ============================================================================
# MODELING
# ============================================================================

def _get_score_values(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return None


def nested_cv_evaluate(X, y, model_name, model_spec, n_splits=N_FOLDS):
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=max(3, n_splits - 2), shuffle=True, random_state=RANDOM_STATE)

    fold_rows = []
    all_coeffs = []
    best_params = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model_spec['estimator'])
        ])

        grid = GridSearchCV(
            pipeline,
            model_spec['param_grid'],
            scoring='balanced_accuracy',
            cv=inner_cv,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        best_params.append(grid.best_params_)

        y_pred = best_model.predict(X_test)
        scores = _get_score_values(best_model, X_test)

        fold_metrics = {
            'fold': fold_idx,
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        if scores is not None:
            fold_metrics['roc_auc'] = roc_auc_score(y_test, scores)
        else:
            fold_metrics['roc_auc'] = np.nan

        train_pred = best_model.predict(X_train)
        fold_metrics['train_accuracy'] = accuracy_score(y_train, train_pred)
        fold_rows.append(fold_metrics)

        if hasattr(best_model.named_steps['clf'], 'coef_'):
            all_coeffs.append(best_model.named_steps['clf'].coef_[0])

    fold_df = pd.DataFrame(fold_rows)
    mean_metrics = fold_df.mean(numeric_only=True).to_dict()
    std_metrics = fold_df.std(numeric_only=True).to_dict()

    coeffs_mean = np.mean(all_coeffs, axis=0) if all_coeffs else None

    summary = {
        'model': model_name,
        'mean_accuracy': mean_metrics.get('accuracy', np.nan),
        'std_accuracy': std_metrics.get('accuracy', np.nan),
        'mean_balanced_accuracy': mean_metrics.get('balanced_accuracy', np.nan),
        'std_balanced_accuracy': std_metrics.get('balanced_accuracy', np.nan),
        'mean_f1': mean_metrics.get('f1', np.nan),
        'std_f1': std_metrics.get('f1', np.nan),
        'mean_roc_auc': mean_metrics.get('roc_auc', np.nan),
        'std_roc_auc': std_metrics.get('roc_auc', np.nan),
        'mean_train_accuracy': mean_metrics.get('train_accuracy', np.nan),
        'std_train_accuracy': std_metrics.get('train_accuracy', np.nan)
    }

    return summary, fold_df, coeffs_mean, best_params


def fit_final_model(X, y, model_spec):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model_spec['estimator'])
    ])

    grid = GridSearchCV(
        pipeline,
        model_spec['param_grid'],
        scoring='balanced_accuracy',
        cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1
    )
    grid.fit(X, y)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X)
    train_accuracy = accuracy_score(y, y_pred)

    coefficients = None
    if hasattr(best_model.named_steps['clf'], 'coef_'):
        coefficients = best_model.named_steps['clf'].coef_[0]

    return best_model, train_accuracy, coefficients, grid.best_params_


# ============================================================================
# MAIN
# ============================================================================

def main():
    output_root = os.path.join(OUTPUT_DIR, 'classification_mdd_vs_hc_raw_eeg')
    data_dir = os.path.join(output_root, 'data')
    report_dir = os.path.join(output_root, 'reports')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    X, y, feature_names, subject_ids = build_dataset()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Subjects: {len(subject_ids)}")
    print(f"Class 0 (Healthy): {np.sum(y == 0)}")
    print(f"Class 1 (Patient): {np.sum(y == 1)}")

    summary_rows = []

    for model_name, model_spec in MODEL_SPECS.items():
        print(f"\nEvaluating model: {model_name}")
        summary, fold_df, coeffs_mean, best_params = nested_cv_evaluate(
            X, y, model_name, model_spec
        )
        summary_rows.append(summary)

        fold_df.to_csv(
            os.path.join(data_dir, f'cv_folds_{model_name}.csv'),
            index=False
        )

        if coeffs_mean is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coeffs_mean,
                'abs_coefficient': np.abs(coeffs_mean)
            }).sort_values(by='abs_coefficient', ascending=False)

            importance_df.to_csv(
                os.path.join(data_dir, f'feature_importance_{model_name}.csv'),
                index=False
            )

        _, train_acc, final_coeffs, best_params_full = fit_final_model(
            X, y, model_spec
        )

        report_path = os.path.join(report_dir, f'classification_report_{model_name}.txt')
        with open(report_path, 'w') as f:
            f.write("MDD vs Healthy Classification Report (Raw EEG)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Best Params (outer CV mean): {best_params[:3]}...\n")
            f.write(f"Best Params (full data): {best_params_full}\n\n")

            f.write("Nested CV Metrics (mean ± std):\n")
            f.write(f"  Accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}\n")
            f.write(
                f"  Balanced Accuracy: {summary['mean_balanced_accuracy']:.4f} ± "
                f"{summary['std_balanced_accuracy']:.4f}\n"
            )
            f.write(f"  F1: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}\n")
            f.write(f"  ROC AUC: {summary['mean_roc_auc']:.4f} ± {summary['std_roc_auc']:.4f}\n\n")

            f.write(f"Train Accuracy (full data): {train_acc:.4f}\n\n")

            if final_coeffs is not None:
                top_idx = np.argsort(np.abs(final_coeffs))[::-1][:20]
                f.write("Top 20 Features (by |coef|):\n")
                for idx in top_idx:
                    f.write(f"  {feature_names[idx]}: {final_coeffs[idx]:.6f}\n")

        print(f"Saved report: {report_path}")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by='mean_balanced_accuracy', ascending=False
    )
    summary_df.to_csv(os.path.join(data_dir, 'model_summary.csv'), index=False)
    print(f"Saved summary: {os.path.join(data_dir, 'model_summary.csv')}")


if __name__ == "__main__":
    main()
###
another file  (again separate) there is that addes to metrics and improves accuracy:
mdd_vs_healthy_classification.py:
"""
Standalone MDD vs Healthy classification pipeline.

This script loads EEG data (or precomputed connectivity), builds
network-measure features across all bands, and adds node-level
measures (per-channel) for richer classification features. Models
are evaluated with nested cross-validation to limit overfitting.
"""

import os
import numpy as np
import pandas as pd
import bct

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score
)

from config import (
    HC_DATA_PATH,
    PATIENT_DATA_PATH,
    OUTPUT_DIR,
    FREQUENCY_BANDS,
    SELECTED_METHOD,
    NETWORK_MEASURES,
    N_FOLDS,
    RANDOM_STATE
)
from data_loader import load_group_data, verify_data_consistency
from signal_processing import process_subject_epochs
from connectivity import compute_all_connectivity
from network_measures import compute_all_network_measures, compute_network_measures_for_subjects
from statistics_utils import extract_features_for_classification

# ============================================================================
# SCRIPT SETTINGS
# ============================================================================
USE_PRECOMPUTED_CONNECTIVITY = True
PRECOMPUTED_CONNECTIVITY_PATH = os.path.join(
    OUTPUT_DIR, 'data', 'connectivity_matrices.npy'
)
SAVE_CONNECTIVITY = True
MAX_SUBJECTS_PER_GROUP = None  # set to an int for quick tests

INCLUDE_NODE_LEVEL_FEATURES = True
INCLUDE_NODE_SUMMARY_FEATURES = True
NODE_MEASURES = (
    'strength_in',
    'strength_out',
    'strength_total',
    'strength_balance',
    'clustering',
    'betweenness'
)
NODE_SUMMARY_STATS = (
    'mean',
    'std',
    'median',
    'max',
    'min'
)

GLOBAL_EXTRA_MEASURES = (
    'density',
    'mean_weight',
    'std_weight',
    'median_weight',
    'max_weight',
    'min_weight',
    'cv_weight',
    'char_path_length'
)

FEATURE_SETS = {
    'global_only': {
        'include_node_summary': False
    },
    'global_plus_node_summary': {
        'include_node_summary': True
    }
}

MODEL_SPECS = {
    'linear_svm': {
        'estimator': LinearSVC(
            class_weight='balanced',
            max_iter=5000,
            random_state=RANDOM_STATE
        ),
        'param_grid': {
            'clf__C': [0.001, 0.005, 0.01, 0.05, 0.1]
        }
    }
}

# ============================================================================
# HELPERS
# ============================================================================

def _ensure_group_order(grouped_dict):
    """Ensure consistent label ordering: Healthy -> Patient."""
    if 'Healthy' in grouped_dict and 'Patient' in grouped_dict:
        return {
            'Healthy': grouped_dict['Healthy'],
            'Patient': grouped_dict['Patient']
        }
    return grouped_dict


def _prepare_weighted_matrix(adjacency_matrix):
    """Prepare a weighted matrix for node-level metrics."""
    W = np.array(adjacency_matrix, dtype=float, copy=True)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(W, 0.0)
    W[W < 0] = 0.0
    return W


def _to_length_matrix(weight_matrix):
    """Convert weights to lengths for shortest-path metrics."""
    L = np.full_like(weight_matrix, np.inf, dtype=float)
    positive = weight_matrix > 0
    L[positive] = 1.0 / weight_matrix[positive]
    np.fill_diagonal(L, 0.0)
    return L


def _compute_node_measures(adjacency_matrix):
    """Compute node-level measures for a connectivity matrix."""
    W = _prepare_weighted_matrix(adjacency_matrix)

    strengths = bct.strengths_dir(W)
    if isinstance(strengths, tuple) and len(strengths) == 2:
        strength_in, strength_out = strengths
    else:
        strength_in = np.array(strengths)
        strength_out = np.array(strengths)

    strength_total = strength_in + strength_out
    strength_balance = strength_out - strength_in

    clustering = bct.clustering_coef_wd(W)
    length_matrix = _to_length_matrix(W)
    betweenness = bct.betweenness_wei(length_matrix)

    return {
        'strength_in': strength_in,
        'strength_out': strength_out,
        'strength_total': strength_total,
        'strength_balance': strength_balance,
        'clustering': clustering,
        'betweenness': betweenness
    }


def _compute_summary_stats(values):
    values = np.array(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    stats = {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'max': float(np.max(values)),
        'min': float(np.min(values))
    }
    return stats


def _compute_global_weight_stats(adjacency_matrix):
    W = _prepare_weighted_matrix(adjacency_matrix)
    n_nodes = W.shape[0]
    if n_nodes <= 1:
        return {name: 0.0 for name in GLOBAL_EXTRA_MEASURES}

    mask = ~np.eye(n_nodes, dtype=bool)
    values = W[mask]
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    nonzero = values[values > 0]
    density = float(nonzero.size) / float(values.size) if values.size else 0.0

    if values.size == 0:
        mean_weight = std_weight = median_weight = max_weight = min_weight = 0.0
    else:
        mean_weight = float(np.mean(values))
        std_weight = float(np.std(values))
        median_weight = float(np.median(values))
        max_weight = float(np.max(values))
        min_weight = float(np.min(values))

    cv_weight = float(std_weight / mean_weight) if mean_weight > 0 else 0.0

    length_matrix = _to_length_matrix(W)
    distances = bct.distance_wei(length_matrix)[0]
    finite_distances = distances[np.isfinite(distances) & (distances > 0)]
    char_path_length = float(np.mean(finite_distances)) if finite_distances.size else 0.0

    return {
        'density': density,
        'mean_weight': mean_weight,
        'std_weight': std_weight,
        'median_weight': median_weight,
        'max_weight': max_weight,
        'min_weight': min_weight,
        'cv_weight': cv_weight,
        'char_path_length': char_path_length
    }


def load_or_compute_network_measures():
    """Load precomputed network measures or compute them from raw EEG."""
    if USE_PRECOMPUTED_CONNECTIVITY and os.path.exists(PRECOMPUTED_CONNECTIVITY_PATH):
        print(f"Loading precomputed connectivity: {PRECOMPUTED_CONNECTIVITY_PATH}")
        connectivity_matrices = np.load(
            PRECOMPUTED_CONNECTIVITY_PATH, allow_pickle=True
        ).item()
        connectivity_matrices = _ensure_group_order(connectivity_matrices)
        return compute_network_measures_for_subjects(
            connectivity_matrices,
            list(FREQUENCY_BANDS.keys())
        )

    print("Precomputed connectivity not found or disabled. Computing from raw data...")
    healthy_data = load_group_data(HC_DATA_PATH, group_name="Healthy")
    patient_data = load_group_data(PATIENT_DATA_PATH, group_name="Patient")

    if MAX_SUBJECTS_PER_GROUP is not None:
        healthy_data = healthy_data[:MAX_SUBJECTS_PER_GROUP]
        patient_data = patient_data[:MAX_SUBJECTS_PER_GROUP]

    all_data = healthy_data + patient_data
    if not verify_data_consistency(all_data):
        raise ValueError("Data consistency check failed")

    all_subjects_filtered = {}
    for group_data, group_name in [(healthy_data, "Healthy"), (patient_data, "Patient")]:
        all_subjects_filtered[group_name] = {}
        for subject in group_data:
            subject_id = subject['subject_id']
            print(f"Processing {subject_id} ({group_name})...")
            filtered_epochs = process_subject_epochs(subject['data'], subject['fs'])
            all_subjects_filtered[group_name][subject_id] = {
                'filtered_epochs': filtered_epochs,
                'fs': subject['fs'],
                'channels': subject['channels']
            }

    connectivity_matrices = {group: {} for group in all_subjects_filtered.keys()}
    for group_name, subjects_dict in all_subjects_filtered.items():
        for subject_id, subject_data in subjects_dict.items():
            connectivity_matrices[group_name][subject_id] = compute_all_connectivity(
                subject_data['filtered_epochs'],
                subject_data['fs'],
                methods=[SELECTED_METHOD]
            )

    if SAVE_CONNECTIVITY:
        os.makedirs(os.path.dirname(PRECOMPUTED_CONNECTIVITY_PATH), exist_ok=True)
        np.save(PRECOMPUTED_CONNECTIVITY_PATH, connectivity_matrices, allow_pickle=True)
        print(f"Saved connectivity: {PRECOMPUTED_CONNECTIVITY_PATH}")

    network_measures = compute_network_measures_for_subjects(
        connectivity_matrices,
        list(FREQUENCY_BANDS.keys())
    )
    network_measures = _ensure_group_order(network_measures)
    return network_measures


def load_or_compute_connectivity():
    """Load precomputed connectivity matrices or compute them from raw EEG."""
    if USE_PRECOMPUTED_CONNECTIVITY and os.path.exists(PRECOMPUTED_CONNECTIVITY_PATH):
        print(f"Loading precomputed connectivity: {PRECOMPUTED_CONNECTIVITY_PATH}")
        connectivity_matrices = np.load(
            PRECOMPUTED_CONNECTIVITY_PATH, allow_pickle=True
        ).item()
        return _ensure_group_order(connectivity_matrices), None

    print("Precomputed connectivity not found or disabled. Computing from raw data...")
    healthy_data = load_group_data(HC_DATA_PATH, group_name="Healthy")
    patient_data = load_group_data(PATIENT_DATA_PATH, group_name="Patient")

    if MAX_SUBJECTS_PER_GROUP is not None:
        healthy_data = healthy_data[:MAX_SUBJECTS_PER_GROUP]
        patient_data = patient_data[:MAX_SUBJECTS_PER_GROUP]

    all_data = healthy_data + patient_data
    if not verify_data_consistency(all_data):
        raise ValueError("Data consistency check failed")

    all_subjects_filtered = {}
    channel_names = None
    for group_data, group_name in [(healthy_data, "Healthy"), (patient_data, "Patient")]:
        all_subjects_filtered[group_name] = {}
        for subject in group_data:
            subject_id = subject['subject_id']
            if channel_names is None:
                channel_names = subject['channels']
            print(f"Processing {subject_id} ({group_name})...")
            filtered_epochs = process_subject_epochs(subject['data'], subject['fs'])
            all_subjects_filtered[group_name][subject_id] = {
                'filtered_epochs': filtered_epochs,
                'fs': subject['fs'],
                'channels': subject['channels']
            }

    connectivity_matrices = {group: {} for group in all_subjects_filtered.keys()}
    for group_name, subjects_dict in all_subjects_filtered.items():
        for subject_id, subject_data in subjects_dict.items():
            connectivity_matrices[group_name][subject_id] = compute_all_connectivity(
                subject_data['filtered_epochs'],
                subject_data['fs'],
                methods=[SELECTED_METHOD]
            )

    if SAVE_CONNECTIVITY:
        os.makedirs(os.path.dirname(PRECOMPUTED_CONNECTIVITY_PATH), exist_ok=True)
        np.save(PRECOMPUTED_CONNECTIVITY_PATH, connectivity_matrices, allow_pickle=True)
        print(f"Saved connectivity: {PRECOMPUTED_CONNECTIVITY_PATH}")

    return connectivity_matrices, channel_names


def build_feature_matrix(network_measures):
    """Extract all measures across all bands into a feature matrix."""
    band_names = list(FREQUENCY_BANDS.keys())
    X, y, feature_names, subject_ids = extract_features_for_classification(
        network_measures,
        NETWORK_MEASURES,
        band_names,
        SELECTED_METHOD
    )
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, feature_names, subject_ids


def build_feature_matrix_from_connectivity(
    connectivity_matrices,
    include_node_summary=False,
    channel_names=None
):
    """Build feature matrix from connectivity matrices with node-level measures."""
    band_names = list(FREQUENCY_BANDS.keys())
    group_labels = {name: idx for idx, name in enumerate(connectivity_matrices.keys())}

    X_list = []
    y_list = []
    subject_ids = []

    feature_names = None
    n_nodes = None

    for group_name, group_data in connectivity_matrices.items():
        for subject_id, subject_data in group_data.items():
            features = []

            for band in band_names:
                if SELECTED_METHOD not in subject_data:
                    raise ValueError(f"Missing method '{SELECTED_METHOD}' for {subject_id}")
                if band not in subject_data[SELECTED_METHOD]:
                    raise ValueError(f"Missing band '{band}' for {subject_id}")

                matrix = subject_data[SELECTED_METHOD][band]
                if n_nodes is None:
                    n_nodes = matrix.shape[0]

                global_measures = compute_all_network_measures(matrix)
                for measure_name in NETWORK_MEASURES:
                    features.append(global_measures.get(measure_name, 0.0))

                global_stats = _compute_global_weight_stats(matrix)
                for extra_name in GLOBAL_EXTRA_MEASURES:
                    features.append(global_stats.get(extra_name, 0.0))

                if include_node_summary:
                    node_measures = _compute_node_measures(matrix)
                    for node_measure in NODE_MEASURES:
                        values = node_measures.get(node_measure, np.zeros(n_nodes))
                        summary_stats = _compute_summary_stats(values)
                        for stat_name in NODE_SUMMARY_STATS:
                            features.append(summary_stats.get(stat_name, 0.0))

            X_list.append(features)
            y_list.append(group_labels[group_name])
            subject_ids.append(subject_id)

            if feature_names is None:
                feature_names = []
                for band in band_names:
                    for measure_name in NETWORK_MEASURES:
                        feature_names.append(f"{measure_name}_{band}")

                    for extra_name in GLOBAL_EXTRA_MEASURES:
                        feature_names.append(f"{extra_name}_{band}")

                    if include_node_summary:
                        for node_measure in NODE_MEASURES:
                            for stat_name in NODE_SUMMARY_STATS:
                                feature_names.append(f"{node_measure}_{stat_name}_{band}")

    X = np.array(X_list, dtype=float)
    y = np.array(y_list)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, feature_names, subject_ids


def _get_score_values(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return None


def nested_cv_evaluate(X, y, model_name, model_spec, n_splits=N_FOLDS):
    """Nested CV evaluation with an inner grid search."""
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=max(3, n_splits - 2), shuffle=True, random_state=RANDOM_STATE)

    fold_rows = []
    all_coeffs = []
    best_params = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model_spec['estimator'])
        ])

        grid = GridSearchCV(
            pipeline,
            model_spec['param_grid'],
            scoring='balanced_accuracy',
            cv=inner_cv,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        best_params.append(grid.best_params_)

        y_pred = best_model.predict(X_test)
        scores = _get_score_values(best_model, X_test)

        fold_metrics = {
            'fold': fold_idx,
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        if scores is not None:
            fold_metrics['roc_auc'] = roc_auc_score(y_test, scores)
        else:
            fold_metrics['roc_auc'] = np.nan

        train_pred = best_model.predict(X_train)
        fold_metrics['train_accuracy'] = accuracy_score(y_train, train_pred)
        fold_rows.append(fold_metrics)

        if hasattr(best_model.named_steps['clf'], 'coef_'):
            all_coeffs.append(best_model.named_steps['clf'].coef_[0])

    fold_df = pd.DataFrame(fold_rows)
    mean_metrics = fold_df.mean(numeric_only=True).to_dict()
    std_metrics = fold_df.std(numeric_only=True).to_dict()

    coeffs_mean = np.mean(all_coeffs, axis=0) if all_coeffs else None

    summary = {
        'model': model_name,
        'mean_accuracy': mean_metrics.get('accuracy', np.nan),
        'std_accuracy': std_metrics.get('accuracy', np.nan),
        'mean_balanced_accuracy': mean_metrics.get('balanced_accuracy', np.nan),
        'std_balanced_accuracy': std_metrics.get('balanced_accuracy', np.nan),
        'mean_f1': mean_metrics.get('f1', np.nan),
        'std_f1': std_metrics.get('f1', np.nan),
        'mean_roc_auc': mean_metrics.get('roc_auc', np.nan),
        'std_roc_auc': std_metrics.get('roc_auc', np.nan),
        'mean_train_accuracy': mean_metrics.get('train_accuracy', np.nan),
        'std_train_accuracy': std_metrics.get('train_accuracy', np.nan)
    }

    return summary, fold_df, coeffs_mean, best_params


def fit_final_model(X, y, model_spec):
    """Fit a final model on full data with grid search."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model_spec['estimator'])
    ])

    grid = GridSearchCV(
        pipeline,
        model_spec['param_grid'],
        scoring='balanced_accuracy',
        cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1
    )
    grid.fit(X, y)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X)
    train_accuracy = accuracy_score(y, y_pred)
    coefficients = None
    if hasattr(best_model.named_steps['clf'], 'coef_'):
        coefficients = best_model.named_steps['clf'].coef_[0]
    return best_model, train_accuracy, coefficients, grid.best_params_


# ============================================================================
# MAIN
# ============================================================================

def main():
    output_root = os.path.join(OUTPUT_DIR, 'classification_mdd_vs_hc')
    data_dir = os.path.join(output_root, 'data')
    report_dir = os.path.join(output_root, 'reports')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    connectivity_matrices, channel_names = load_or_compute_connectivity()

    summary_rows = []

    for feature_set_name, feature_set in FEATURE_SETS.items():
        include_node_summary = feature_set['include_node_summary']
        if include_node_summary and not INCLUDE_NODE_SUMMARY_FEATURES:
            continue

        X, y, feature_names, subject_ids = build_feature_matrix_from_connectivity(
            connectivity_matrices,
            include_node_summary=include_node_summary,
            channel_names=channel_names
        )

        print(f"\nFeature set: {feature_set_name}")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Subjects: {len(subject_ids)}")
        print(f"Class 0 (Healthy): {np.sum(y == 0)}")
        print(f"Class 1 (Patient): {np.sum(y == 1)}")

        for model_name, model_spec in MODEL_SPECS.items():
            print(f"\nEvaluating model: {model_name} ({feature_set_name})")
            summary, fold_df, coeffs_mean, best_params = nested_cv_evaluate(
                X, y, model_name, model_spec
            )
            summary.update({
                'feature_set': feature_set_name,
                'n_features': X.shape[1]
            })
            summary_rows.append(summary)

            fold_df.to_csv(
                os.path.join(data_dir, f'cv_folds_{feature_set_name}_{model_name}.csv'),
                index=False
            )

            if coeffs_mean is not None:
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': coeffs_mean,
                    'abs_coefficient': np.abs(coeffs_mean)
                }).sort_values(by='abs_coefficient', ascending=False)

                importance_df.to_csv(
                    os.path.join(data_dir, f'feature_importance_{feature_set_name}_{model_name}.csv'),
                    index=False
                )

            _, train_acc, final_coeffs, best_params_full = fit_final_model(
                X, y, model_spec
            )

            report_path = os.path.join(
                report_dir,
                f'classification_report_{feature_set_name}_{model_name}.txt'
            )
            with open(report_path, 'w') as f:
                f.write("MDD vs Healthy Classification Report\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Feature Set: {feature_set_name}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Number of Features: {X.shape[1]}\n")
                f.write(f"Best Params (outer CV mean): {best_params[:3]}...\n")
                f.write(f"Best Params (full data): {best_params_full}\n\n")

                f.write("Nested CV Metrics (mean ± std):\n")
                f.write(f"  Accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}\n")
                f.write(
                    f"  Balanced Accuracy: {summary['mean_balanced_accuracy']:.4f} ± "
                    f"{summary['std_balanced_accuracy']:.4f}\n"
                )
                f.write(f"  F1: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}\n")
                f.write(f"  ROC AUC: {summary['mean_roc_auc']:.4f} ± {summary['std_roc_auc']:.4f}\n\n")

                f.write(f"Train Accuracy (full data): {train_acc:.4f}\n\n")

                if final_coeffs is not None:
                    top_idx = np.argsort(np.abs(final_coeffs))[::-1][:20]
                    f.write("Top 20 Features (by |coef|):\n")
                    for idx in top_idx:
                        f.write(
                            f"  {feature_names[idx]}: {final_coeffs[idx]:.6f}\n"
                        )

            print(f"Saved report: {report_path}")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by='mean_balanced_accuracy', ascending=False
    )
    summary_df.to_csv(os.path.join(data_dir, 'model_summary.csv'), index=False)
    print(f"Saved summary: {os.path.join(data_dir, 'model_summary.csv')}")


if __name__ == "__main__":
    main()
###

a little bit better accuracy though. (node level metrics overfitted and did not returned better. globals where a little bit better final cv accuracy.... )

all files done. 



ok files done. 

there are some thing i wanna do and some problems. 
first, this optimization, results are not satisfing. maybe i did something wrong (i mean in code. not general logic). fix them
second, the code is not maintainable anymore. not modular enough with folder structure and ... 
third, about the logic, i have some ideas. 
the classification accuracy of tripplets or even more like 10 metrics is low. 
I wanna use new metrics, on top of old metrics to improve. 
and i guess i need another method of selecting final features. 
actually there is another issue with optimization and features. the optimization is supposed to result in final band and node. about the band, i am doing a selection before optimization. so i should not get biased to a band, when performing feature selection. in final feature pool, i need equal count of features for each band (but maybe different for each band.) 
the feature selection based on classification accuracy i think is fine. but accuracy should be high. but classification should be another method. i don't think tripplet thing is enough. maybe increase tripplets or just global svm. i don't know. 

final objectives should be 2 or one per each band (configurable). (previous to optimization, report global accuracy, per band, per metric... separately in plots)

so i want you to re-write the whole project. with these modifications. 