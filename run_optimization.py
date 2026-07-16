"""
Main script to run NSGA-II optimization for EEG connectivity
"""
import math
import os
import sys
os.environ.setdefault("MPLBACKEND", "Agg")
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
    PATIENT_REJECTION_PERCENT, PATIENT_REJECTION_RANKING_FILE,
    PATIENT_REJECTION_PERCENT_BY_BAND, PATIENT_REJECTION_RANKING_FILE_BY_BAND
)

# Import optimization modules
from eeg_optimization import create_optimizer_from_config
from optimization_visualization import (
    plot_optimization_summary, create_optimization_report,
    plot_candidate_region_statistics,
    plot_weighted_rank_region_statistics
)
from statistics_utils import (
    compute_candidate_region_selection_stats,
    compute_candidate_region_weighted_rank_stats
)

# Import data loading (assuming these exist in your main pipeline)
from data_loader import load_subject_epochs
from channel_metadata import get_display_channel_names


def create_output_directories():
    """Create output directories for optimization results."""
    dirs = [
        OPTIMIZATION_OUTPUT_DIR,
        OPTIMIZATION_FIGURES_DIR
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print(f"Output directories created")


def _compute_baseline_activation(data):
    """Compute the optimizer's baseline node activation from raw EEG data."""
    baseline = np.mean(data, axis=1)
    baseline = (baseline - baseline.min()) / (baseline.max() - baseline.min() + 1e-10)
    return baseline * 0.9 + 0.1


def _load_group_baseline_data(data_path, group_name):
    """Load subjects one at a time and keep only optimizer baseline vectors."""
    subject_folders = [f.path for f in os.scandir(data_path) if f.is_dir()]
    subject_folders = sorted(subject_folders)

    if not subject_folders:
        raise ValueError(f"No subject folders found in {data_path}")

    print(f"\nLoading {group_name} baseline activations from: {data_path}")
    print(f"Found {len(subject_folders)} subjects")

    subject_data = {}
    for i, subject_folder in enumerate(subject_folders, start=1):
        subject_id = os.path.basename(subject_folder)
        print(f"[{i}/{len(subject_folders)}] Baseline for {subject_id}")

        try:
            data, fs, channels, channel_metadata = load_subject_epochs(subject_folder)
        except Exception as e:
            print(f"  ERROR loading {subject_id}: {str(e)}")
            continue

        subject_data[subject_id] = {
            'baseline_activation': _compute_baseline_activation(data),
            'fs': fs,
            'channels': channels,
            'channel_names': channels,
            'channel_display_names': channel_metadata['channel_display_names'],
            'channel_metadata': channel_metadata,
            'group': group_name
        }
        del data

    print(f"Loaded {len(subject_data)}/{len(subject_folders)} {group_name} baseline activations")
    return subject_data


def _load_subject_optimization_results(result_dir):
    """Load per-subject optimization result files into the usual result dict."""
    results = {}
    for result_file in sorted(os.listdir(result_dir)):
        if not result_file.endswith('.npy'):
            continue
        result_path = os.path.join(result_dir, result_file)
        result = np.load(result_path, allow_pickle=True).item()
        subject_id = result.get('subject_id') or os.path.splitext(result_file)[0]
        results[subject_id] = result
    print(f"Loaded {len(results)} per-subject optimization results from: {result_dir}")
    return results


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
    channel_display_names : list
        Plot/report channel labels
    channel_metadata : dict
        Full channel metadata audit record
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
    
    subject_data = {}
    subject_data.update(_load_group_baseline_data(HC_DATA_PATH, "Healthy"))
    subject_data.update(_load_group_baseline_data(PATIENT_DATA_PATH, "Patient"))

    print(f"Loaded baseline activations for {len(subject_data)} subjects")

    # Get channel names (assuming all subjects have same channels)
    first_subject = list(subject_data.values())[0]
    channel_names = first_subject['channels']
    channel_metadata = first_subject.get('channel_metadata') or {
        'channel_names': channel_names,
        'channel_display_names': first_subject.get('channel_display_names', channel_names),
    }
    channel_display_names = get_display_channel_names(channel_metadata, n_nodes=len(channel_names))
    
    print(f"✓ Number of channels: {len(channel_names)}")
    
    return (
        connectivity_matrices,
        network_measures,
        subject_data,
        channel_names,
        channel_display_names,
        channel_metadata
    )


def _get_measures_for_band(band_name: str) -> List[str]:
    if OPTIMIZATION_MEASURES_BY_BAND:
        return OPTIMIZATION_MEASURES_BY_BAND.get(band_name, OPTIMIZATION_MEASURES)
    return OPTIMIZATION_MEASURES


def _format_rejection_ranking_path(path_template, band_name=None):
    return path_template.format(
        OUTPUT_DIR=OUTPUT_DIR,
        band=band_name,
        band_name=band_name
    )


def _filter_to_retained_patients(connectivity_matrices, network_measures, subject_data, retained_ids):
    retained_ids = set(retained_ids)
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


def _get_patient_rejection_percent(band_name=None):
    if band_name is not None:
        return float(PATIENT_REJECTION_PERCENT_BY_BAND.get(band_name, 0))
    return float(PATIENT_REJECTION_PERCENT)


def apply_patient_rejection_filter(connectivity_matrices, network_measures, subject_data, band_name=None):
    """Filter patient dictionaries using the SVM-margin rejection ranking."""
    rejection_percent = _get_patient_rejection_percent(band_name)
    filter_label = f"{band_name} band" if band_name else "global"
    if rejection_percent <= 0:
        print(f"\nPatient rejection filter disabled for {filter_label} (rejection percent=0)")
        return connectivity_matrices, network_measures, subject_data

    if rejection_percent < 0 or rejection_percent > 100:
        raise ValueError(
            f"Patient rejection percent must be between 0 and 100; got {rejection_percent}"
        )

    if band_name is None:
        ranking_path = _format_rejection_ranking_path(PATIENT_REJECTION_RANKING_FILE)
        required_columns = {'subject_id', 'aggregate_rank', 'aggregate_margin_percentile'}
        sort_columns = ['aggregate_rank', 'aggregate_margin_percentile', 'subject_id']
    else:
        ranking_path = _format_rejection_ranking_path(
            PATIENT_REJECTION_RANKING_FILE_BY_BAND,
            band_name=band_name
        )
        required_columns = {'subject_id', 'rank', 'margin_percentile'}
        sort_columns = ['rank', 'margin_percentile', 'subject_id']

    if not os.path.exists(ranking_path):
        raise FileNotFoundError(
            "Patient rejection ranking file not found: "
            f"{ranking_path}\nRun plot_svm_margin_rejection_3d.py first, "
            "or set the corresponding rejection percent to 0."
        )

    ranking_df = pd.read_csv(ranking_path)
    missing_columns = required_columns.difference(ranking_df.columns)
    if missing_columns:
        raise ValueError(
            f"Ranking file is missing required columns {sorted(missing_columns)}: {ranking_path}"
        )

    ranking_df = ranking_df.sort_values(
        by=sort_columns,
        ascending=[True] * len(sort_columns)
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
    print(f"Scope: {filter_label}")
    print(f"Ranking file: {ranking_path}")
    print(f"Configured rejection percent: {rejection_percent:g}%")
    print(f"Rejected patients: {len(rejected_ids)}/{len(ranked_patient_ids)}")
    print(f"Retained patients: {len(retained_ids)}")
    print(f"Rejected IDs: {sorted(rejected_ids)}")
    print(f"Retained IDs: {retained_ids}")

    return _filter_to_retained_patients(
        connectivity_matrices,
        network_measures,
        subject_data,
        retained_ids
    )


def save_candidate_region_selection_stats(
    optimization_results,
    channel_names,
    output_dir,
    figures_dir=None,
    file_prefix=None,
    figure_prefix=None,
    label=None
):
    """Save selection-frequency tests for candidate stimulation electrodes."""
    output_label = f" ({label})" if label else ""
    file_stem = f"{file_prefix}_" if file_prefix else ""
    hard_figure_prefix = (
        f"{figure_prefix}_hard_best_solution_target_statistics"
        if figure_prefix else
        "hard_best_solution_target_statistics"
    )
    weighted_figure_prefix = (
        f"{figure_prefix}_rank_weighted_target_statistics"
        if figure_prefix else
        "rank_weighted_target_statistics"
    )

    stats_df = compute_candidate_region_selection_stats(
        optimization_results,
        channel_names
    )
    stats_path = os.path.join(
        output_dir,
        f'{file_stem}candidate_region_selection_stats.csv'
    )
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved hard best-solution candidate-region statistics{output_label}: {stats_path}")

    weighted_stats_df = compute_candidate_region_weighted_rank_stats(
        optimization_results,
        channel_names
    )
    weighted_stats_path = os.path.join(
        output_dir,
        f'{file_stem}candidate_region_weighted_rank_stats.csv'
    )
    weighted_stats_df.to_csv(weighted_stats_path, index=False)
    print(f"Saved rank-weighted candidate-region statistics{output_label}: {weighted_stats_path}")

    figure_paths = []
    if figures_dir is not None:
        statistics_scope = figure_prefix if figure_prefix else "overall"
        statistics_dir = os.path.join(
            figures_dir, "target_statistics", statistics_scope
        )
        figure_paths = plot_candidate_region_statistics(
            stats_df,
            channel_names,
            statistics_dir,
            prefix=hard_figure_prefix
        )
        figure_paths.extend(plot_weighted_rank_region_statistics(
            weighted_stats_df,
            channel_names,
            statistics_dir,
            prefix=weighted_figure_prefix
        ))

    return {
        'label': label,
        'hard_stats_path': stats_path,
        'weighted_stats_path': weighted_stats_path,
        'figure_paths': figure_paths
    }


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
        connectivity_matrices, network_measures, subject_data, channel_names, channel_display_names, channel_metadata = \
            load_data_for_optimization()
    except Exception as e:
        print(f"\nERROR loading data: {str(e)}")
        print("\nMake sure you have run the main pipeline first to generate:")
        print("  - connectivity_matrices.npy")
        print("  - network_measures.npy")
        return

    if not OPTIMIZATION_PER_BAND:
        try:
            connectivity_matrices, network_measures, subject_data = apply_patient_rejection_filter(
                connectivity_matrices,
                network_measures,
                subject_data
            )
        except Exception as e:
            print(f"\nERROR applying patient rejection filter: {str(e)}")
            return
    else:
        configured_band_rejections = {
            band_name: percent
            for band_name, percent in PATIENT_REJECTION_PERCENT_BY_BAND.items()
            if float(percent) > 0
        }
        if configured_band_rejections:
            print("\nPer-band patient rejection configured:")
            for band_name in FREQUENCY_BANDS.keys():
                percent = float(PATIENT_REJECTION_PERCENT_BY_BAND.get(band_name, 0))
                print(f"  - {band_name}: {percent:g}%")
        elif float(PATIENT_REJECTION_PERCENT) > 0:
            print(
                "\nPATIENT_REJECTION_PERCENT is ignored in per-band mode when "
                "PATIENT_REJECTION_PERCENT_BY_BAND is all zero."
            )
    
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
        candidate_stats_outputs = []

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

            try:
                band_connectivity_matrices, band_network_measures, band_subject_data = apply_patient_rejection_filter(
                    connectivity_matrices,
                    network_measures,
                    subject_data,
                    band_name=band_name
                )
            except Exception as e:
                print(f"\nERROR applying patient rejection filter for band {band_name}: {str(e)}")
                return

            optimizer = create_optimizer_from_config(
                connectivity_matrices=band_connectivity_matrices,
                network_measures=band_network_measures,
                subject_data=band_subject_data,
                frequency_bands=FREQUENCY_BANDS,
                channel_names=channel_names,
                channel_display_names=channel_display_names,
                channel_metadata=channel_metadata,
                selected_method=SELECTED_METHOD,
                optimization_measures=band_measures,
                fixed_band_name=band_name
            )

            try:
                subject_result_dir = os.path.join(
                    OPTIMIZATION_OUTPUT_DIR,
                    f"{band_name}_subject_results"
                )
                optimizer.optimize_all_patients(
                    verbose=True,
                    n_jobs=OPTIMIZATION_N_JOBS,
                    result_dir=subject_result_dir,
                    return_results=False
                )
                optimization_results = _load_subject_optimization_results(subject_result_dir)
                optimizer.optimization_results = optimization_results
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

            try:
                candidate_stats_outputs.append(save_candidate_region_selection_stats(
                    optimization_results,
                    channel_display_names,
                    OPTIMIZATION_OUTPUT_DIR,
                    figures_dir=OPTIMIZATION_FIGURES_DIR,
                    file_prefix=band_name,
                    figure_prefix=band_name,
                    label=f"{band_name} band"
                ))
            except Exception as e:
                print(f"\nERROR saving candidate-region statistics for band {band_name}: {str(e)}")
                import traceback
                traceback.print_exc()

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
                channel_names=channel_display_names,
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
                channel_names=channel_display_names,
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
            candidate_stats_outputs.append(save_candidate_region_selection_stats(
                combined_results,
                channel_display_names,
                OPTIMIZATION_OUTPUT_DIR,
                figures_dir=OPTIMIZATION_FIGURES_DIR,
                label="combined across bands"
            ))
        except Exception as e:
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
            channel_display_names=channel_display_names,
            channel_metadata=channel_metadata,
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
            subject_result_dir = os.path.join(
                OPTIMIZATION_OUTPUT_DIR,
                "subject_results"
            )
            optimizer.optimize_all_patients(
                verbose=True,
                n_jobs=OPTIMIZATION_N_JOBS,
                result_dir=subject_result_dir,
                return_results=False
            )
            optimization_results = _load_subject_optimization_results(subject_result_dir)
            optimizer.optimization_results = optimization_results
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
                channel_names=channel_display_names,
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
                channel_names=channel_display_names,
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
            candidate_stats_outputs = [save_candidate_region_selection_stats(
                optimization_results,
                channel_display_names,
                OPTIMIZATION_OUTPUT_DIR,
                figures_dir=OPTIMIZATION_FIGURES_DIR
            )]
        except Exception as e:
            candidate_stats_outputs = []
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
    if 'candidate_stats_outputs' in locals() and candidate_stats_outputs:
        print("  - Candidate-region statistics:")
        for stats_output in candidate_stats_outputs:
            label = stats_output.get('label') or 'overall'
            if stats_output.get('hard_stats_path'):
                print(f"    - {label} hard best-solution stats: {stats_output['hard_stats_path']}")
            if stats_output.get('weighted_stats_path'):
                print(f"    - {label} rank-weighted stats: {stats_output['weighted_stats_path']}")

        figure_count = sum(len(stats_output.get('figure_paths', [])) for stats_output in candidate_stats_outputs)
        if figure_count:
            print(f"  - Candidate-region statistic figures: {figure_count} files in {OPTIMIZATION_FIGURES_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
