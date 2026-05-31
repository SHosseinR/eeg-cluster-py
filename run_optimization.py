"""
Main script to run NSGA-II optimization for EEG connectivity
"""
import math
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List

# Import configuration
from config import (
    OUTPUT_DIR, FREQUENCY_BANDS, SELECTED_METHOD
)
from optimization_config import (
    OPTIMIZATION_MEASURES, OPTIMIZATION_OUTPUT_DIR,
    OPTIMIZATION_RESULTS_FILE, OPTIMIZATION_FIGURES_DIR, OPTIMIZATION_N_JOBS,
    OPTIMIZATION_TOP_K, OPTIMIZATION_MODE,
    OPTIMIZATION_PER_BAND, OPTIMIZATION_MEASURES_BY_BAND,
    PATIENT_REJECTION_PERCENT, PATIENT_REJECTION_RANKING_FILE
)

# Import optimization modules
from eeg_optimization import create_optimizer_from_config
from optimization_visualization import (
    plot_optimization_summary, create_optimization_report,
    plot_candidate_region_statistics
)
from statistics_utils import compute_candidate_region_selection_stats

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


def _get_measures_for_band(band_name: str) -> List[str]:
    if OPTIMIZATION_MEASURES_BY_BAND:
        return OPTIMIZATION_MEASURES_BY_BAND.get(band_name, OPTIMIZATION_MEASURES)
    return OPTIMIZATION_MEASURES


def apply_patient_rejection_filter(connectivity_matrices, network_measures, subject_data):
    """Filter patient dictionaries using the SVM-margin rejection ranking."""
    rejection_percent = float(PATIENT_REJECTION_PERCENT)
    if rejection_percent <= 0:
        print("\nPatient rejection filter: disabled (PATIENT_REJECTION_PERCENT=0)")
        return connectivity_matrices, network_measures, subject_data

    if rejection_percent < 0 or rejection_percent > 100:
        raise ValueError(
            f"PATIENT_REJECTION_PERCENT must be between 0 and 100; got {rejection_percent}"
        )

    ranking_path = PATIENT_REJECTION_RANKING_FILE.format(OUTPUT_DIR=OUTPUT_DIR)
    if not os.path.exists(ranking_path):
        raise FileNotFoundError(
            "Patient rejection ranking file not found: "
            f"{ranking_path}\nRun plot_svm_margin_rejection_3d.py first, "
            "or set PATIENT_REJECTION_PERCENT = 0."
        )

    ranking_df = pd.read_csv(ranking_path)
    required_columns = {'subject_id', 'aggregate_rank', 'aggregate_margin_percentile'}
    missing_columns = required_columns.difference(ranking_df.columns)
    if missing_columns:
        raise ValueError(
            f"Ranking file is missing required columns {sorted(missing_columns)}: {ranking_path}"
        )

    ranking_df = ranking_df.sort_values(
        by=['aggregate_rank', 'aggregate_margin_percentile', 'subject_id'],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    available_patient_ids = set(network_measures.get('Patient', {}).keys())
    ranked_patient_ids = [
        sid for sid in ranking_df['subject_id'].astype(str).tolist()
        if sid in available_patient_ids
    ]
    if not ranked_patient_ids:
        raise ValueError(
            f"No ranked patient IDs from {ranking_path} match network_measures['Patient']."
        )
    unranked_patient_ids = sorted(available_patient_ids.difference(ranked_patient_ids))
    if unranked_patient_ids:
        raise ValueError(
            "Some patient subjects are missing from the SVM-margin ranking file: "
            f"{unranked_patient_ids}\nRegenerate the ranking with plot_svm_margin_rejection_3d.py "
            "using the same network_measures.npy before running filtered optimization."
        )

    reject_n = int(math.ceil(rejection_percent / 100.0 * len(ranked_patient_ids)))
    reject_n = min(max(reject_n, 0), len(ranked_patient_ids))
    rejected_ids = set(ranked_patient_ids[:reject_n])
    retained_ids = [sid for sid in ranked_patient_ids if sid not in rejected_ids]

    print("\n" + "=" * 80)
    print("PATIENT REJECTION FILTER")
    print("=" * 80)
    print(f"Ranking file: {ranking_path}")
    print(f"Configured rejection percent: {rejection_percent:g}%")
    print(f"Rejected patients: {len(rejected_ids)}/{len(ranked_patient_ids)}")
    print(f"Retained patients: {len(retained_ids)}")
    print(f"Rejected IDs: {sorted(rejected_ids)}")
    print(f"Retained IDs: {retained_ids}")

    network_measures = dict(network_measures)
    connectivity_matrices = dict(connectivity_matrices)
    network_measures['Patient'] = {
        sid: data
        for sid, data in network_measures.get('Patient', {}).items()
        if sid in retained_ids
    }
    connectivity_matrices['Patient'] = {
        sid: data
        for sid, data in connectivity_matrices.get('Patient', {}).items()
        if sid in retained_ids
    }
    subject_data = {
        sid: data
        for sid, data in subject_data.items()
        if data.get('group') != 'Patient' or sid in retained_ids
    }

    return connectivity_matrices, network_measures, subject_data


def save_candidate_region_selection_stats(
    optimization_results,
    channel_names,
    output_dir,
    figures_dir=None
):
    """Save selection-frequency tests for candidate stimulation electrodes."""
    stats_df = compute_candidate_region_selection_stats(
        optimization_results,
        channel_names
    )
    stats_path = os.path.join(output_dir, 'candidate_region_selection_stats.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved candidate-region selection statistics: {stats_path}")

    figure_paths = []
    if figures_dir is not None:
        figure_paths = plot_candidate_region_statistics(
            stats_df,
            channel_names,
            figures_dir,
            prefix='final_target_statistics'
        )

    return stats_path, figure_paths


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
    if OPTIMIZATION_PER_BAND:
        print("\nOptimization measures by band:")
        for band_name in FREQUENCY_BANDS.keys():
            measures = _get_measures_for_band(band_name)
            print(f"  - {band_name}: {measures}")
    else:
        print(f"\nOptimization measures: {OPTIMIZATION_MEASURES}")
    
    # Check at least one subject has required measures
    sample_subject = patient_subjects[0]
    sample_band = list(FREQUENCY_BANDS.keys())[0]
    
    if SELECTED_METHOD not in network_measures['Patient'][sample_subject]:
        raise ValueError(f"Method {SELECTED_METHOD} not found in network measures!")
    
    if sample_band not in network_measures['Patient'][sample_subject][SELECTED_METHOD]:
        raise ValueError(f"Band {sample_band} not found in network measures!")
    
    sample_available = list(network_measures['Patient'][sample_subject][SELECTED_METHOD][sample_band].keys())
    print(f"\nAvailable measures in data: {sample_available}")

    # Check that optimization measures exist
    if OPTIMIZATION_PER_BAND:
        for band_name in FREQUENCY_BANDS.keys():
            available_measures = list(
                network_measures['Patient'][sample_subject][SELECTED_METHOD][band_name].keys()
            )
            measures = _get_measures_for_band(band_name)
            for measure in measures:
                if measure not in available_measures:
                    raise ValueError(
                        f"Optimization measure '{measure}' not found in data for band '{band_name}'!"
                    )
    else:
        for measure in OPTIMIZATION_MEASURES:
            if measure not in sample_available:
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

    try:
        connectivity_matrices, network_measures, subject_data = apply_patient_rejection_filter(
            connectivity_matrices,
            network_measures,
            subject_data
        )
    except Exception as e:
        print(f"\nERROR applying patient rejection filter: {str(e)}")
        return
    
    # Verify requirements
    try:
        verify_optimization_requirements(connectivity_matrices, network_measures)
    except Exception as e:
        print(f"\nERROR in verification: {str(e)}")
        return
    
    if OPTIMIZATION_PER_BAND:
        results_by_band = {}
        combined_results = {}
        optimization_directions = {}

        print("\n" + "="*80)
        print("CREATING PER-BAND OPTIMIZERS")
        print("="*80)

        effective_workers = (os.cpu_count() - 1 or 1) if OPTIMIZATION_N_JOBS is None else max(1, int(OPTIMIZATION_N_JOBS))
        print(f"Optimization workers requested: {OPTIMIZATION_N_JOBS} (effective: {effective_workers})")

        for band_idx, band_name in enumerate(FREQUENCY_BANDS.keys()):
            band_measures = _get_measures_for_band(band_name)
            print("\n" + "-"*80)
            print(f"Band: {band_name}")
            print(f"Measures: {band_measures}")
            print("-"*80)

            optimizer = create_optimizer_from_config(
                connectivity_matrices=connectivity_matrices,
                network_measures=network_measures,
                subject_data=subject_data,
                frequency_bands=FREQUENCY_BANDS,
                channel_names=channel_names,
                selected_method=SELECTED_METHOD,
                optimization_measures=band_measures,
                fixed_band_name=band_name
            )

            try:
                optimization_results = optimizer.optimize_all_patients(
                    verbose=True,
                    n_jobs=OPTIMIZATION_N_JOBS
                )
            except Exception as e:
                print(f"\nERROR during optimization for band {band_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                return

            results_by_band[band_name] = optimization_results
            optimization_directions[band_name] = optimizer.optimization_directions

            for subject_id, result in optimization_results.items():
                combined_key = f"{subject_id}::{band_name}"
                combined_results[combined_key] = result

            band_results_path = os.path.join(
                OPTIMIZATION_OUTPUT_DIR,
                f"{band_name}_{OPTIMIZATION_RESULTS_FILE}"
            )
            optimizer.save_results(band_results_path)

        print("\n" + "="*80)
        print("SAVING COMBINED RESULTS")
        print("="*80)
        results_path = os.path.join(OPTIMIZATION_OUTPUT_DIR, OPTIMIZATION_RESULTS_FILE)
        np.save(results_path, combined_results, allow_pickle=True)

        plot_measures = _get_measures_for_band(next(iter(FREQUENCY_BANDS.keys())))

        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)

        try:
            plot_optimization_summary(
                optimization_results=combined_results,
                channel_names=channel_names,
                band_names=list(FREQUENCY_BANDS.keys()),
                optimization_measures=plot_measures,
                output_dir=OPTIMIZATION_FIGURES_DIR,
                top_k=OPTIMIZATION_TOP_K
            )
        except Exception as e:
            print(f"\nERROR creating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()

        print("\n" + "="*80)
        print("GENERATING REPORT")
        print("="*80)

        report_path = os.path.join(OPTIMIZATION_OUTPUT_DIR, 'optimization_report.txt')
        try:
            create_optimization_report(
                optimization_results=combined_results,
                channel_names=channel_names,
                band_names=list(FREQUENCY_BANDS.keys()),
                optimization_measures=plot_measures,
                optimization_directions={},
                output_path=report_path,
                top_k=OPTIMIZATION_TOP_K
            )
        except Exception as e:
            print(f"\nERROR creating report: {str(e)}")
            import traceback
            traceback.print_exc()

        try:
            candidate_stats_path, candidate_stats_figures = save_candidate_region_selection_stats(
                combined_results,
                channel_names,
                OPTIMIZATION_OUTPUT_DIR,
                figures_dir=OPTIMIZATION_FIGURES_DIR
            )
        except Exception as e:
            candidate_stats_path = None
            candidate_stats_figures = []
            print(f"\nERROR saving candidate-region statistics: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
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

        report_path = os.path.join(OPTIMIZATION_OUTPUT_DIR, 'optimization_report.txt')
        try:
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

        try:
            candidate_stats_path, candidate_stats_figures = save_candidate_region_selection_stats(
                optimization_results,
                channel_names,
                OPTIMIZATION_OUTPUT_DIR,
                figures_dir=OPTIMIZATION_FIGURES_DIR
            )
        except Exception as e:
            candidate_stats_path = None
            candidate_stats_figures = []
            print(f"\nERROR saving candidate-region statistics: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("OPTIMIZATION PIPELINE COMPLETE")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to:")
    print(f"  - Optimization results: {results_path}")
    if OPTIMIZATION_PER_BAND:
        print(f"  - Per-band results: {OPTIMIZATION_OUTPUT_DIR}/*_{OPTIMIZATION_RESULTS_FILE}")
    print(f"  - Figures: {OPTIMIZATION_FIGURES_DIR}")
    print(f"  - Report: {report_path}")
    if 'candidate_stats_path' in locals() and candidate_stats_path:
        print(f"  - Candidate-region stats: {candidate_stats_path}")
    if 'candidate_stats_figures' in locals() and candidate_stats_figures:
        print("  - Candidate-region statistic figures:")
        for figure_path in candidate_stats_figures:
            print(f"    - {figure_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
