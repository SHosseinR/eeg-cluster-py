"""
Complete EEG optimization pipeline using NSGA-II
"""
import os
import csv
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed

from optimization_config import (
    OPTIMIZATION_MEASURES, NSGA_CONFIG, SIMULATION_CONFIG, PLASTICITY_CONFIG, OPTIMIZATION_TOP_K,
    STIMULATION_DURATION_BOUNDS, STIMULATION_AMPLITUDE_BOUNDS, STIMULATION_LEAK_BOUNDS,
    ACTIVATION_RATIO_FEASIBILITY_BOUNDS, OPTIMIZATION_MODE,
    GRID_USE_PARETO_ONLY, OPTIMIZATION_OBJECTIVE_MODE
)
from state_space_simulation import run_full_simulation
from plasticity import compute_plasticity_effect
from classification_score.band_connectivity_classifier import (
    BandConnectivityClassifier,
    matrix_change_rms,
    matrix_manifold_rms,
    matrix_ood_rms,
    predict_patient_probability,
    vectorize_band_matrix,
)


_WORKER_OPTIMIZER = None
_WORKER_VERBOSE = False
_WORKER_RESULT_DIR = None


def _safe_result_filename(subject_id: str) -> str:
    safe = str(subject_id).replace(os.sep, "_").replace("/", "_")
    return f"{safe}.npy"


def _save_subject_result(result_dir: str, subject_id: str, result: Dict) -> str:
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, _safe_result_filename(subject_id))
    np.save(result_path, result, allow_pickle=True)
    return result_path


def _init_optimizer_worker(optimizer, verbose, result_dir=None):
    """Initialize each process with an optimizer instance."""
    global _WORKER_OPTIMIZER, _WORKER_VERBOSE, _WORKER_RESULT_DIR
    _WORKER_OPTIMIZER = optimizer
    _WORKER_VERBOSE = verbose
    _WORKER_RESULT_DIR = result_dir


def _optimize_subject_worker(subject_id: str):
    """Run optimization for one subject in worker process."""
    result = _WORKER_OPTIMIZER.optimize_subject(subject_id, verbose=_WORKER_VERBOSE)
    if _WORKER_RESULT_DIR:
        return subject_id, _save_subject_result(_WORKER_RESULT_DIR, subject_id, result)
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
                 channel_display_names: Optional[List[str]] = None,
                 channel_metadata: Optional[Dict] = None,
                 fixed_band_name: Optional[str] = None,
                 optimization_mode: str = None,
                 objective_mode: str = None,
                 classifier_bundle: Optional[BandConnectivityClassifier] = None,
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
        channel_display_names : list of str, optional
            Plot/report labels for EEG channels/nodes
        channel_metadata : dict, optional
            Full channel metadata audit record
        selected_method : str
            Connectivity method to use (e.g., 'plv', 'pdc', 'gc', 'psi')
        optimization_measures : list of str
            Names of network measures to optimize
        fixed_band_name : str, optional
            If provided, restrict optimization to this band (band is fixed)
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
        self.channel_display_names = channel_display_names or list(channel_names)
        if len(self.channel_display_names) != len(channel_names):
            self.channel_display_names = list(channel_names)
        self.channel_metadata = channel_metadata or {
            'channel_names': list(channel_names),
            'channel_display_names': list(self.channel_display_names),
        }
        self.selected_method = selected_method
        self.classifier_bundle = classifier_bundle
        self.optimization_measures = list(optimization_measures or [])
        if not self.optimization_measures and classifier_bundle is None:
            raise ValueError("optimization_measures must contain at least one measure.")

        self.optimization_mode = (optimization_mode or OPTIMIZATION_MODE).strip().lower()
        if self.optimization_mode not in ("nsga", "grid"):
            raise ValueError(
                "optimization_mode must be 'nsga' or 'grid'. "
                f"Got: {self.optimization_mode!r}"
            )

        self.objective_mode = (objective_mode or OPTIMIZATION_OBJECTIVE_MODE).strip().lower()
        if self.objective_mode not in (
            "directional", "distance_to_gt", "classifier_patient_probability"
        ):
            raise ValueError(
                "objective_mode must be 'directional', 'distance_to_gt', or "
                "'classifier_patient_probability'. "
                f"Got: {self.objective_mode!r}"
            )
        if self.objective_mode == "classifier_patient_probability":
            if self.classifier_bundle is None:
                raise ValueError("Classifier-probability mode requires a fitted band classifier")
            if not self.classifier_bundle.accepted_for_optimization:
                raise ValueError(
                    f"The {self.classifier_bundle.band} classifier did not pass the "
                    "optimization evidence gate"
                )
            if self.selected_method != self.classifier_bundle.method:
                raise ValueError("Connectivity method does not match the fitted classifier")
            if list(self.channel_names) != self.classifier_bundle.channel_names:
                raise ValueError("Optimizer channel order does not match the fitted classifier")
            self.optimization_measures = ["patient_probability"]
        
        # Configuration
        self.nsga_config = nsga_config or NSGA_CONFIG
        self.simulation_config = simulation_config or SIMULATION_CONFIG
        self.plasticity_config = plasticity_config or PLASTICITY_CONFIG
        ratio_lower, ratio_upper = ACTIVATION_RATIO_FEASIBILITY_BOUNDS
        if (
            not np.isfinite(ratio_lower)
            or not np.isfinite(ratio_upper)
            or ratio_lower <= 0
            or ratio_lower >= ratio_upper
        ):
            raise ValueError(
                "ACTIVATION_RATIO_FEASIBILITY_BOUNDS must contain finite, "
                "positive, increasing values."
            )
        
        # Derived parameters
        self.n_nodes = len(channel_names)
        self.band_names = list(frequency_bands.keys())
        self.fixed_band_name = fixed_band_name
        self.fixed_band_index = None
        if fixed_band_name is not None:
            if fixed_band_name not in self.band_names:
                raise ValueError(
                    f"fixed_band_name '{fixed_band_name}' not found in frequency_bands"
                )
            self.fixed_band_index = self.band_names.index(fixed_band_name)
            self.n_bands = 1
        else:
            self.n_bands = len(self.band_names)
        if self.objective_mode == "classifier_patient_probability":
            if self.fixed_band_name is None:
                raise ValueError("Classifier-probability optimization must run separately per band")
            if self.fixed_band_name != self.classifier_bundle.band:
                raise ValueError("Fixed optimization band does not match classifier band")
        
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
        if self.objective_mode == "classifier_patient_probability":
            return {"patient_probability": 0.0}
        baselines = {}
        eps = 1e-10

        bands = [self.fixed_band_name] if self.fixed_band_name is not None else self.band_names

        for measure in self.optimization_measures:
            healthy_values = []
            for subject_id in self.network_measures['Healthy'].keys():
                for band in bands:
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
        if self.objective_mode == "classifier_patient_probability":
            return {"patient_probability": "minimize"}
        directions = {}
        
        bands = [self.fixed_band_name] if self.fixed_band_name is not None else self.band_names

        for measure in self.optimization_measures:
            # Compute average measure values for each group
            patient_values = []
            healthy_values = []
            
            # Collect all values across bands for this measure
            for subject_id in self.network_measures['Patient'].keys():
                for band in bands:
                    if self.selected_method in self.network_measures['Patient'][subject_id]:
                        if band in self.network_measures['Patient'][subject_id][self.selected_method]:
                            if measure in self.network_measures['Patient'][subject_id][self.selected_method][band]:
                                val = self.network_measures['Patient'][subject_id][self.selected_method][band][measure]
                                patient_values.append(val)
            
            for subject_id in self.network_measures['Healthy'].keys():
                for band in bands:
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

        if 'baseline_activation' in self.subject_data[subject_id]:
            return np.asarray(
                self.subject_data[subject_id]['baseline_activation'],
                dtype=float
            )

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
            if self.fixed_band_index is not None:
                band_idx = self.fixed_band_index
            objectives, _, details = self._evaluate_solution_details(
                subject_id=subject_id,
                baseline_activation=baseline_activation,
                node=node,
                band_idx=band_idx,
                stimulation_duration=stimulation_duration,
                stimulation_amplitude=stimulation_amplitude,
                stimulation_leak=stimulation_leak
            )
            return objectives, details['constraint_values']

        def evaluate_with_details(
            node: int,
            band_idx: int,
            stimulation_duration: float = None,
            stimulation_amplitude: float = None,
            stimulation_leak: float = None
        ) -> Tuple[np.ndarray, np.ndarray, Dict]:
            if self.fixed_band_index is not None:
                band_idx = self.fixed_band_index
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
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Evaluate objectives, raw measures, and trajectory feasibility."""
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
            leak=stimulation_leak,
            return_trajectory=False,
        )

        raw_ratios = np.asarray(sim_results['raw_activation_ratios'], dtype=float)
        ratio_lower, ratio_upper = ACTIVATION_RATIO_FEASIBILITY_BOUNDS
        constraint_values = [
            float(ratio_lower) - float(np.min(raw_ratios)),
            float(np.max(raw_ratios)) - float(ratio_upper),
        ]
        feasibility_details = {
            'raw_activation_ratio_min': float(np.min(raw_ratios)),
            'raw_activation_ratio_max': float(np.max(raw_ratios)),
            'stimulation_polarity': (
                'suppression' if stimulation_amplitude < -1e-10 else
                'enhancement' if stimulation_amplitude > 1e-10 else
                'zero'
            ),
        }

        if self.plasticity_config['plasticity_enabled']:
            updated_matrix = compute_plasticity_effect(
                adjacency_matrix=original_matrix,
                activation_ratios=sim_results['activation_ratios'],
                # The classifier was fitted to natural coherence values.  A
                # candidate-wise min-max transform would change its feature
                # space and make its probability meaningless.
                normalize=self.objective_mode != "classifier_patient_probability",
                scaling=self.plasticity_config['plasticity_scaling']
            )
        else:
            updated_matrix = original_matrix

        if self.objective_mode == "classifier_patient_probability":
            edges = vectorize_band_matrix(updated_matrix)[0]
            ood_rms = matrix_ood_rms(self.classifier_bundle, updated_matrix)
            manifold_rms = matrix_manifold_rms(self.classifier_bundle, updated_matrix)
            local_change_rms = matrix_change_rms(
                self.classifier_bundle, original_matrix, updated_matrix
            )
            patient_probability = predict_patient_probability(
                self.classifier_bundle,
                updated_matrix,
                channel_names=self.channel_names,
            )
            constraint_values.extend([
                -float(np.min(edges)),
                float(np.max(edges)) - 1.0,
                ood_rms - float(self.classifier_bundle.ood_rms_threshold),
                manifold_rms - float(self.classifier_bundle.manifold_rms_threshold),
                local_change_rms - float(self.classifier_bundle.local_change_rms_threshold),
            ])
            measure_values = [patient_probability]
            objectives = np.array([patient_probability], dtype=float)
            feasibility_details.update({
                'patient_probability': patient_probability,
                'healthy_probability': 1.0 - patient_probability,
                'classifier_ood_rms': ood_rms,
                'classifier_ood_threshold': float(self.classifier_bundle.ood_rms_threshold),
                'classifier_manifold_rms': manifold_rms,
                'classifier_manifold_threshold': float(self.classifier_bundle.manifold_rms_threshold),
                'classifier_local_change_rms': local_change_rms,
                'classifier_local_change_threshold': float(self.classifier_bundle.local_change_rms_threshold),
                'updated_connectivity_min': float(np.min(edges)),
                'updated_connectivity_max': float(np.max(edges)),
                # The details dictionary is discarded during candidate
                # evaluation. It is persisted only when the selected best
                # solution is re-evaluated below, enabling fixed-projection
                # before/after visualizations without rerunning simulation.
                'updated_connectivity_matrix': updated_matrix,
            })
        else:
            measure_values = []
            for measure_name in self.optimization_measures:
                measure_func = measure_functions[measure_name]
                measure_value = measure_func(updated_matrix)
                measure_values.append(float(measure_value))
            objectives = self._compute_objectives_from_measures(measure_values)

        constraint_values = np.asarray(constraint_values, dtype=float)
        constraint_violation = float(np.sum(np.maximum(constraint_values, 0.0)))
        feasibility_details.update({
            'constraint_values': constraint_values,
            'constraint_violation': constraint_violation,
            'feasible': bool(constraint_violation <= 1e-10),
        })
        return objectives, np.array(measure_values, dtype=float), feasibility_details

    def _extract_initial_metrics(self, subject_id: str, band_name: str) -> Optional[np.ndarray]:
        """Extract baseline metrics from precomputed network measures."""
        if self.objective_mode == "classifier_patient_probability":
            matrix = self.connectivity_matrices['Patient'][subject_id][self.selected_method][band_name]
            probability = predict_patient_probability(
                self.classifier_bundle, matrix, channel_names=self.channel_names
            )
            return np.array([probability], dtype=float)
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

        solutions = [sol for sol in solutions if sol.get('feasible', True)]
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
        best_front = [sol for sol in best_front if sol.get('feasible', True)]
        if not best_front:
            return None
        if len(best_front) == 1:
            return best_front[0]

        objectives = np.array([sol['objectives'] for sol in best_front], dtype=float)
        if self.objective_mode in ("distance_to_gt", "classifier_patient_probability"):
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
                objectives, measure_values, details = evaluate_func(node, band_idx)
                global_band_idx = (
                    self.fixed_band_index
                    if self.fixed_band_index is not None
                    else band_idx
                )
                solution = {
                    'node': node,
                    'band': global_band_idx,
                    'band_name': self.band_names[global_band_idx],
                    'stimulation_duration': duration,
                    'stimulation_amplitude': amplitude,
                    'leak': leak,
                    'objectives': objectives,
                    'measure_values': measure_values
                }
                solution.update(details)
                solutions.append(solution)

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
        best_front = [sol for sol in best_front if sol.get('feasible', True)]
        if not best_front:
            return []

        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = len(best_front)

        top_k = max(1, top_k)
        top_k = min(top_k, len(best_front))

        objectives = np.array([sol['objectives'] for sol in best_front])
        if self.objective_mode in ("distance_to_gt", "classifier_patient_probability"):
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
                'measure_values': sol.get('measure_values'),
                'constraint_values': sol.get('constraint_values'),
                'constraint_violation': sol.get('constraint_violation', 0.0),
                'feasible': sol.get('feasible', True),
                'raw_activation_ratio_min': sol.get('raw_activation_ratio_min'),
                'raw_activation_ratio_max': sol.get('raw_activation_ratio_max'),
                'stimulation_polarity': sol.get('stimulation_polarity'),
                'patient_probability': sol.get('patient_probability'),
                'healthy_probability': sol.get('healthy_probability'),
                'classifier_ood_rms': sol.get('classifier_ood_rms'),
                'classifier_ood_threshold': sol.get('classifier_ood_threshold'),
                'classifier_manifold_rms': sol.get('classifier_manifold_rms'),
                'classifier_local_change_rms': sol.get('classifier_local_change_rms'),
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
                n_constraints=(7 if self.objective_mode == "classifier_patient_probability" else 2),
                activation_ratio_bounds=ACTIVATION_RATIO_FEASIBILITY_BOUNDS,
                duration_bounds=STIMULATION_DURATION_BOUNDS,
                amplitude_bounds=STIMULATION_AMPLITUDE_BOUNDS,
                leak_bounds=STIMULATION_LEAK_BOUNDS,
                population_size=self.nsga_config['population_size'],
                n_generations=self.nsga_config['n_generations'],
                crossover_prob=self.nsga_config['crossover_prob'],
                crossover_eta=self.nsga_config.get('crossover_eta', 15.0),
                mutation_prob=self.nsga_config['mutation_prob'],
                mutation_eta=self.nsga_config.get('mutation_eta', 20.0),
                seed=self.nsga_config.get('seed'),
                # tournament_size=self.nsga_config['tournament_size']
                fixed_band_index=self.fixed_band_index
            )
            
            # Run optimization
            best_front, history = optimizer.optimize(verbose=verbose)

            all_solutions = getattr(optimizer, "all_solutions", None) or best_front
            
            # Get single best solution (closest to GT or ideal point)
            best_pool = best_front if GRID_USE_PARETO_ONLY else all_solutions
            best_solution = self._select_best_solution(best_pool)
            if self.objective_mode == "classifier_patient_probability":
                # A one-objective Pareto front normally contains only its
                # minimum. Use the feasible final population for meaningful
                # top-K alternatives without counting every historical repeat.
                ranking_pool = (
                    getattr(optimizer, "final_population_solutions", None)
                    or best_front
                )
            else:
                ranking_pool = best_pool

        if best_solution is None:
            raise RuntimeError(
                "No feasible stimulation solution was found. Adjust "
                "STIMULATION_AMPLITUDE_BOUNDS, stimulation duration/leak bounds, "
                "or ACTIVATION_RATIO_FEASIBILITY_BOUNDS."
            )

        # Rank top solutions using distance-to-ideal (strength = 1 / rank)
        top_solutions = self._rank_solutions(ranking_pool, OPTIMIZATION_TOP_K)

        initial_metrics = None
        final_metrics = None
        if best_solution is not None:
            band_idx = int(best_solution['band'])
            band_name = best_solution.get('band_name') or self.band_names[band_idx]
            initial_metrics = self._extract_initial_metrics(subject_id, band_name)

            if self.objective_mode == "classifier_patient_probability":
                # Re-evaluate exactly one selected solution so its complete
                # connectivity matrix and trust diagnostics are saved. NSGA's
                # candidate records intentionally remain lightweight.
                objectives, measure_values, details = evaluate_with_details(
                    node=int(best_solution['node']),
                    band_idx=band_idx,
                    stimulation_duration=best_solution.get('stimulation_duration'),
                    stimulation_amplitude=best_solution.get('stimulation_amplitude'),
                    stimulation_leak=best_solution.get('leak')
                )
                best_solution['objectives'] = objectives
                best_solution['measure_values'] = measure_values.tolist()
                best_solution.update(details)
                final_metrics = measure_values
            elif 'measure_values' in best_solution:
                final_metrics = np.array(best_solution['measure_values'], dtype=float)
            else:
                objectives, measure_values, details = evaluate_with_details(
                    node=int(best_solution['node']),
                    band_idx=band_idx,
                    stimulation_duration=best_solution.get('stimulation_duration'),
                    stimulation_amplitude=best_solution.get('stimulation_amplitude'),
                    stimulation_leak=best_solution.get('leak')
                )
                best_solution['objectives'] = objectives
                best_solution['measure_values'] = measure_values.tolist()
                best_solution.update(details)
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
            'channel_display_names': self.channel_display_names,
            'channel_metadata': self.channel_metadata,
            'optimization_mode': self.optimization_mode,
            'objective_mode': self.objective_mode,
            'optimization_measures': list(self.optimization_measures),
            'optimization_directions': dict(self.optimization_directions),
            'stimulation_amplitude_bounds': tuple(STIMULATION_AMPLITUDE_BOUNDS),
            'activation_ratio_feasibility_bounds': tuple(ACTIVATION_RATIO_FEASIBILITY_BOUNDS),
            'healthy_measure_baselines': dict(self.healthy_measure_baselines),
            'classifier_band': self.classifier_bundle.band if self.classifier_bundle else None,
            'classifier_model': self.classifier_bundle.model_name if self.classifier_bundle else None,
            'classifier_cv_metrics': dict(self.classifier_bundle.cv_metrics) if self.classifier_bundle else None,
            'classifier_best_params': dict(self.classifier_bundle.best_params) if self.classifier_bundle else None,
            'classifier_fit_diagnostics': dict(self.classifier_bundle.fit_diagnostics) if self.classifier_bundle else None,
            'fixed_band_name': self.fixed_band_name,
            'fixed_band_index': self.fixed_band_index,
            'initial_metrics': initial_metrics.tolist() if initial_metrics is not None else None,
            'final_metrics': final_metrics.tolist() if final_metrics is not None else None
        }
        
        if verbose and best_solution is not None:
            print(f"\nBest solution:")
            node_idx = int(best_solution['node'])
            print(f"  Node: {node_idx} ({self.channel_display_names[node_idx]})")
            print(f"  Band: {self.band_names[best_solution['band']]}")
            print(f"  Objectives: {best_solution['objectives']}")
        
        return results
    
    def optimize_all_patients(
        self,
        verbose: bool = True,
        n_jobs: int = None,
        result_dir: str = None,
        return_results: bool = True
    ) -> Dict:
        """
        Run optimization for all patient subjects.
        
        Parameters
        ----------
        verbose : bool
            Print progress information
        n_jobs : int, optional
            Number of parallel worker processes. If None, uses all available cores.
            Use 1 to force sequential execution.
        result_dir : str, optional
            If provided, save each subject result to this directory as soon as it
            finishes.
        return_results : bool
            If False with result_dir set, keep only result paths in memory during
            the optimization run.
            
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

        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
            print(f"Per-subject optimization results will be saved to: {result_dir}")
        
        # Get patient subject IDs
        patient_subjects = list(self.network_measures['Patient'].keys())
        
        print(f"\nTotal patient subjects: {len(patient_subjects)}")

        total_subjects = len(patient_subjects)
        requested_workers = n_jobs
        max_workers = (os.cpu_count()-1 or 1) if requested_workers is None else max(1, int(requested_workers))
        max_workers = min(max_workers, total_subjects) if total_subjects > 0 else 1

        all_results = {}
        optimization_failures = {}
        if max_workers <= 1 or total_subjects <= 1:
            print(f"Running optimization sequentially (workers={max_workers})")
            for i, subject_id in enumerate(patient_subjects):
                if verbose:
                    print(f"\n[{i+1}/{total_subjects}] ", end="")
                try:
                    results = self.optimize_subject(subject_id, verbose=verbose)
                    if result_dir:
                        result_path = _save_subject_result(result_dir, subject_id, results)
                        all_results[subject_id] = results if return_results else result_path
                    else:
                        all_results[subject_id] = results
                except Exception as e:
                    print(f"ERROR optimizing {subject_id}: {str(e)}")
                    optimization_failures[str(subject_id)] = str(e)
                    continue
        else:
            print(f"Running optimization in parallel with {max_workers} processes...")
            # Avoid interleaved detailed logs from multiple worker processes.
            subject_verbose = False
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_optimizer_worker,
                initargs=(self, subject_verbose, result_dir)
            ) as executor:
                future_to_subject = {
                    executor.submit(_optimize_subject_worker, subject_id): subject_id
                    for subject_id in patient_subjects
                }

                for i, future in enumerate(as_completed(future_to_subject), start=1):
                    subject_id = future_to_subject[future]
                    try:
                        _, result_or_path = future.result()
                        if result_dir and return_results:
                            result_or_path = np.load(result_or_path, allow_pickle=True).item()
                        all_results[subject_id] = result_or_path
                        print(f"[{i}/{total_subjects}] Completed {subject_id}")
                    except Exception as e:
                        print(f"[{i}/{total_subjects}] ERROR optimizing {subject_id}: {str(e)}")
                        optimization_failures[str(subject_id)] = str(e)
                        continue
        
        self.optimization_results = all_results
        self.optimization_failures = optimization_failures
        if result_dir:
            failure_path = os.path.join(result_dir, 'optimization_failures.csv')
            with open(failure_path, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=['subject_id', 'error'])
                writer.writeheader()
                for failed_subject, error in sorted(optimization_failures.items()):
                    writer.writerow({'subject_id': failed_subject, 'error': error})
            print(f"Optimization failure manifest: {failure_path}")
        
        print(f"\n{'='*80}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"Successfully optimized: {len(all_results)}/{len(patient_subjects)} subjects")
        print(f"Subjects without a feasible saved solution: {len(optimization_failures)}")
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
                                selected_method: str,
                                optimization_measures: Optional[List[str]] = None,
                                channel_display_names: Optional[List[str]] = None,
                                channel_metadata: Optional[Dict] = None,
                                fixed_band_name: Optional[str] = None,
                                classifier_bundle: Optional[BandConnectivityClassifier] = None) -> EEGOptimizer:
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
    channel_display_names : list, optional
        Display channel labels for plots/reports
    channel_metadata : dict, optional
        Full channel metadata audit record
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
        channel_display_names=channel_display_names,
        channel_metadata=channel_metadata,
        selected_method=selected_method,
        optimization_measures=optimization_measures or OPTIMIZATION_MEASURES,
        fixed_band_name=fixed_band_name,
        optimization_mode=OPTIMIZATION_MODE,
        objective_mode=OPTIMIZATION_OBJECTIVE_MODE,
        classifier_bundle=classifier_bundle,
        nsga_config=NSGA_CONFIG,
        simulation_config=SIMULATION_CONFIG,
        plasticity_config=PLASTICITY_CONFIG
    )
    
    return optimizer
