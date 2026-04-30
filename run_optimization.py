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
    OPTIMIZATION_TOP_K
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
    effective_workers = (os.cpu_count() or 1) if OPTIMIZATION_N_JOBS is None else max(1, int(OPTIMIZATION_N_JOBS))
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
