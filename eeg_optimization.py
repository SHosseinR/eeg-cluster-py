"""
Complete EEG optimization pipeline using NSGA-II
"""
import os
import numpy as np
from typing import Dict, List, Tuple
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed

from optimization_config import (
    OPTIMIZATION_MEASURES, NSGA_CONFIG, SIMULATION_CONFIG, PLASTICITY_CONFIG
)
from state_space_simulation import run_full_simulation
from plasticity import compute_plasticity_effect
from nsga_optimizer import NSGAIIOptimizer


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
            return np.random.rand(self.n_nodes)
        
        # Get raw data
        data = self.subject_data[subject_id]['data']  # shape: (n_channels, n_samples)
        
        # Compute mean over time
        baseline = np.mean(data, axis=1)
        
        # Normalize to reasonable range
        baseline = (baseline - np.min(baseline)) / (np.max(baseline) - np.min(baseline) + 1e-10)
        
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
        """
        # Import network measure functions
        from network_measures import measure_functions
        
        def evaluate(node: int, band_idx: int) -> np.ndarray:
            """
            Evaluate objectives for given stimulation parameters.
            
            Parameters
            ----------
            node : int
                Stimulation node index
            band_idx : int
                Frequency band index
                
            Returns
            -------
            objectives : ndarray, shape (n_objectives,)
                Objective values (to be minimized)
            """
            band_name = self.band_names[band_idx]
            
            # Get original connectivity matrix
            original_matrix = self.connectivity_matrices['Patient'][subject_id][self.selected_method][band_name]
            
            # Run simulation
            sim_results = run_full_simulation(
                adjacency_matrix=original_matrix,
                baseline_activation=baseline_activation,
                stimulation_node=node,
                stimulation_duration=self.simulation_config['stimulation_duration'],
                stimulation_amplitude=self.simulation_config['stimulation_amplitude'],
                dt=self.simulation_config['dt'],
                stability_constant=self.simulation_config['stability_constant']
            )
            
            # Apply plasticity to update connectivity
            if self.plasticity_config['plasticity_enabled']:
                updated_matrix = compute_plasticity_effect(
                    adjacency_matrix=original_matrix,
                    activation_ratios=sim_results['activation_ratios'],
                    normalize=True,
                    scaling=self.plasticity_config['plasticity_scaling']
                )
            else:
                updated_matrix = original_matrix
            
            # Compute network measures on updated matrix
            objectives = []
            for measure_name in self.optimization_measures:
                measure_func = measure_functions[measure_name]
                # print(f'{updated_matrix=}')
                measure_value = measure_func(updated_matrix)
                baseline = self.healthy_measure_baselines[measure_name]
                normalized_value = measure_value / baseline
                
                # Convert to minimization problem
                if self.optimization_directions[measure_name] == 'maximize':
                    # Negate for maximization
                    objectives.append(-normalized_value)
                else:
                    objectives.append(normalized_value)
            
            return np.array(objectives)
        
        return evaluate
    
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
            - 'best_solution': Single best solution
            - 'history': Optimization history
            - 'baseline_activation': Baseline activation used
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"OPTIMIZING SUBJECT: {subject_id}")
            print(f"{'='*80}")
        
        # Compute baseline activation
        baseline_activation = self._compute_baseline_activation(subject_id)
        
        if verbose:
            print(f"Baseline activation computed: mean={np.mean(baseline_activation):.4f}, "
                  f"std={np.std(baseline_activation):.4f}")
        
        # Create evaluation function
        evaluate_func = self._create_evaluation_function(subject_id, baseline_activation)
        
        # Create optimizer
        optimizer = NSGAIIOptimizer(
            n_nodes=self.n_nodes,
            n_bands=self.n_bands,
            band_names=self.band_names,
            evaluate_func=evaluate_func,
            n_objectives=len(self.optimization_measures),
            population_size=self.nsga_config['population_size'],
            n_generations=self.nsga_config['n_generations'],
            crossover_prob=self.nsga_config['crossover_prob'],
            mutation_prob=self.nsga_config['mutation_prob'],
            # tournament_size=self.nsga_config['tournament_size']
        )
        
        # Run optimization
        best_front, history = optimizer.optimize(verbose=verbose)
        
        # Get single best solution (closest to ideal point)
        best_solution = optimizer.get_best_solution()
        
        # Package results
        results = {
            'subject_id': subject_id,
            'best_front': best_front,
            'best_solution': best_solution,
            'history': history,
            'baseline_activation': baseline_activation,
            'n_nodes': self.n_nodes,
            'n_bands': self.n_bands,
            'band_names': self.band_names,
            'channel_names': self.channel_names
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
        nsga_config=NSGA_CONFIG,
        simulation_config=SIMULATION_CONFIG,
        plasticity_config=PLASTICITY_CONFIG
    )
    
    return optimizer
