"""
NSGA-II optimizer using pymoo library for EEG connectivity optimization
"""
import inspect
import numpy as np
from typing import Callable, List, Dict

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination

from stimulation_models import (
    DYNAMICS_FREE_STIMULATION_MODELS,
    STIMULATION_MODELS,
)


class ZeroAmplitudeAnchorSampling(FloatRandomSampling):
    """Random sampling with one guaranteed no-stimulation candidate when allowed."""

    def __init__(self, amplitude_index: int):
        super().__init__()
        self.amplitude_index = int(amplitude_index)

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        samples = super()._do(
            problem, n_samples, *args, random_state=random_state, **kwargs
        )
        if (
            n_samples > 0
            and problem.xl[self.amplitude_index] <= 0.0
            and problem.xu[self.amplitude_index] >= 0.0
        ):
            samples[0] = (problem.xl + problem.xu) / 2.0
            samples[0, 0] = 0.0
            samples[0, self.amplitude_index] = 0.0
        return samples


class EEGOptimizationProblem(Problem):
    """
    pymoo Problem definition for EEG optimization.
    
    ``state_space`` decision variables are node, optional band, duration,
    amplitude, and leak. Dynamics-free models remove duration and leak,
    leaving node, optional band, and one signed stimulation amount.
    
    Objectives:
    - f[0] ... f[n-1]: Network measures to optimize
    """
    
    def __init__(self, 
                 n_nodes: int,
                 n_bands: int,
                 evaluate_func: Callable,
                 duration_bounds: tuple = (0.0, 2.0),
                 amplitude_bounds: tuple = (0.0, 2.0),
                 leak_bounds: tuple = (0.0, 2.0),
                 n_objectives: int = 3,
                 n_constraints: int = 0,
                 fixed_band_index: int = None,
                 stimulation_model: str = "state_space"):
        """
        Initialize EEG optimization problem.
        
        Parameters
        ----------
        n_nodes : int
            Number of nodes (EEG channels)
        n_bands : int
            Number of frequency bands
        evaluate_func : callable
            Function to evaluate objectives: func(node, band, duration, amplitude, leak) -> objectives array
        n_objectives : int
            Number of objectives to optimize (default: 3)
        """
        self.evaluate_func = evaluate_func
        self.fixed_band_index = fixed_band_index
        self.stimulation_model = str(stimulation_model).strip().lower()
        if self.stimulation_model not in STIMULATION_MODELS:
            raise ValueError(
                f"stimulation_model must be one of {sorted(STIMULATION_MODELS)}"
            )
        self._accepts_continuous = self._check_accepts_continuous(evaluate_func)
        self._accepts_leak = self._check_accepts_leak(evaluate_func)
        
        duration_min, duration_max = duration_bounds
        amplitude_min, amplitude_max = amplitude_bounds
        leak_min, leak_max = leak_bounds
        for name, lower, upper in (
            ("duration", duration_min, duration_max),
            ("amplitude", amplitude_min, amplitude_max),
            ("leak", leak_min, leak_max),
        ):
            if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
                raise ValueError(
                    f"Invalid {name} bounds ({lower}, {upper}); bounds must be "
                    "finite and ordered as (minimum, maximum)."
                )

        if self.stimulation_model in DYNAMICS_FREE_STIMULATION_MODELS:
            if fixed_band_index is None:
                super().__init__(
                    n_var=3,  # node, band, signed stimulation amount
                    n_obj=n_objectives,
                    n_ieq_constr=int(n_constraints),
                    xl=np.array([0.0, 0.0, float(amplitude_min)]),
                    xu=np.array([
                        float(n_nodes - 1),
                        float(n_bands - 1),
                        float(amplitude_max),
                    ]),
                )
            else:
                super().__init__(
                    n_var=2,  # node, signed stimulation amount
                    n_obj=n_objectives,
                    n_ieq_constr=int(n_constraints),
                    xl=np.array([0.0, float(amplitude_min)]),
                    xu=np.array([float(n_nodes - 1), float(amplitude_max)]),
                )
        elif fixed_band_index is None:
            # Define problem with band as a decision variable
            super().__init__(
                n_var=5,  # node, band, stimulation_duration, stimulation_amplitude, leak
                n_obj=n_objectives,  # Number of objectives
                n_ieq_constr=int(n_constraints),
                xl=np.array([0.0, 0.0, float(duration_min), float(amplitude_min), float(leak_min)]),
                xu=np.array([
                    float(n_nodes - 1),
                    float(n_bands - 1),
                    float(duration_max),
                    float(amplitude_max),
                    float(leak_max)
                ])
            )
        else:
            # Band is fixed; remove band from decision variables
            super().__init__(
                n_var=4,  # node, stimulation_duration, stimulation_amplitude, leak
                n_obj=n_objectives,
                n_ieq_constr=int(n_constraints),
                xl=np.array([0.0, float(duration_min), float(amplitude_min), float(leak_min)]),
                xu=np.array([
                    float(n_nodes - 1),
                    float(duration_max),
                    float(amplitude_max),
                    float(leak_max)
                ])
            )
    
    def _check_accepts_continuous(self, func: Callable) -> bool:
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return True

        params = list(signature.parameters.values())
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
            return True

        positional_params = [
            p for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        return len(positional_params) >= 4

    def _check_accepts_leak(self, func: Callable) -> bool:
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return True

        params = list(signature.parameters.values())
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
            return True

        positional_params = [
            p for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        return len(positional_params) >= 5

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate objectives for a population.
        
        Parameters
        ----------
        X : ndarray
            Population matching the configured stimulation-model layout.
        out : dict
            Output dictionary to store objectives
        """
        # Evaluate each individual
        objectives = []
        constraints = []
        for x in X:
            node = int(np.clip(np.round(x[0]), 0, self.xu[0]))
            if self.stimulation_model in DYNAMICS_FREE_STIMULATION_MODELS:
                if self.fixed_band_index is None:
                    band = int(np.clip(np.round(x[1]), 0, self.xu[1]))
                    amplitude = float(x[2])
                else:
                    band = int(self.fixed_band_index)
                    amplitude = float(x[1])
                duration = None
                leak = None
            elif self.fixed_band_index is None:
                band = int(np.clip(np.round(x[1]), 0, self.xu[1]))
                duration = float(x[2])
                amplitude = float(x[3])
                leak = float(x[4])
            else:
                band = int(self.fixed_band_index)
                duration = float(x[1])
                amplitude = float(x[2])
                leak = float(x[3])
            if self.stimulation_model in DYNAMICS_FREE_STIMULATION_MODELS:
                evaluation = self.evaluate_func(node, band, None, amplitude, None)
            elif self._accepts_leak:
                evaluation = self.evaluate_func(node, band, duration, amplitude, leak)
            elif self._accepts_continuous:
                evaluation = self.evaluate_func(node, band, duration, amplitude)
            else:
                evaluation = self.evaluate_func(node, band)

            if isinstance(evaluation, tuple) and len(evaluation) == 2:
                obj, constraint_values = evaluation
            else:
                obj = evaluation
                constraint_values = np.zeros(self.n_ieq_constr, dtype=float)
            objectives.append(obj)
            if self.n_ieq_constr:
                constraint_values = np.asarray(constraint_values, dtype=float).reshape(-1)
                if constraint_values.size != self.n_ieq_constr:
                    raise ValueError(
                        f"Expected {self.n_ieq_constr} constraint values, got "
                        f"{constraint_values.size}."
                    )
                constraints.append(constraint_values)
        
        # Store objectives
        out["F"] = np.array(objectives)
        if self.n_ieq_constr:
            out["G"] = np.array(constraints)


class NSGAIIOptimizer:
    """
    NSGA-II optimizer using pymoo for EEG connectivity optimization.
    
    Wraps pymoo's NSGA2 algorithm for easier integration with EEG pipeline.
    """
    
    def __init__(self,
                 n_nodes: int,
                 n_bands: int,
                 band_names: List[str],
                 evaluate_func: Callable,
                 duration_bounds: tuple = (0.0, 2.0),
                 amplitude_bounds: tuple = (0.0, 2.0),
                 leak_bounds: tuple = (0.0, 2.0),
                 n_objectives: int = 3,
                 n_constraints: int = 0,
                 activation_ratio_bounds: tuple = None,
                 population_size: int = 100,
                 n_generations: int = 50,
                 crossover_prob: float = 0.9,
                 crossover_eta: float = 15.0,
                 mutation_prob: float = None,
                 mutation_eta: float = 20.0,
                 seed: int = None,
                 verbose: bool = True,
                 fixed_band_index: int = None,
                 stimulation_model: str = "state_space"):
        """
        Initialize NSGA-II optimizer using pymoo.
        
        Parameters
        ----------
        n_nodes : int
            Number of nodes in the network
        n_bands : int
            Number of frequency bands
        band_names : list of str
            Names of frequency bands
        evaluate_func : callable
            Function to evaluate objectives: func(node, band, duration, amplitude, leak) -> objectives array
        n_objectives : int
            Number of objectives to optimize (default: 3)
        population_size : int
            Population size (default: 100)
        n_generations : int
            Number of generations (default: 50)
        crossover_prob : float
            Crossover probability (default: 0.9)
        crossover_eta : float
            Crossover distribution index for SBX (default: 15.0)
        mutation_prob : float
            Mutation probability (default: 1/n_var)
        mutation_eta : float
            Mutation distribution index for PM (default: 20.0)
        seed : int
            Random seed for reproducibility (default: None)
        verbose : bool
            Print progress (default: True)
        """
        self.n_nodes = n_nodes
        self.n_bands = n_bands
        self.band_names = band_names
        self.evaluate_func = evaluate_func
        self.duration_bounds = duration_bounds
        self.amplitude_bounds = amplitude_bounds
        self.leak_bounds = leak_bounds
        self.n_objectives = int(n_objectives)
        if self.n_objectives < 1:
            raise ValueError("n_objectives must be >= 1.")
        self.population_size = population_size
        self.n_generations = n_generations
        self.seed = seed
        self.verbose = verbose
        self.fixed_band_index = fixed_band_index
        self.stimulation_model = str(stimulation_model).strip().lower()
        self.n_constraints = int(n_constraints)
        self.activation_ratio_bounds = activation_ratio_bounds
        
        # Create problem
        self.problem = EEGOptimizationProblem(
            n_nodes=n_nodes,
            n_bands=n_bands,
            evaluate_func=evaluate_func,
            duration_bounds=duration_bounds,
            amplitude_bounds=amplitude_bounds,
            leak_bounds=leak_bounds,
            n_objectives=self.n_objectives,
            n_constraints=self.n_constraints,
            fixed_band_index=fixed_band_index,
            stimulation_model=self.stimulation_model,
        )
        
        # Set default mutation probability if not specified
        if mutation_prob is None:
            mutation_prob = 1.0 / self.problem.n_var
        
        # Create algorithm
        if self.stimulation_model in DYNAMICS_FREE_STIMULATION_MODELS:
            amplitude_index = 2 if fixed_band_index is None else 1
        else:
            amplitude_index = 3 if fixed_band_index is None else 2
        self.algorithm = NSGA2(
            pop_size=population_size,
            sampling=ZeroAmplitudeAnchorSampling(amplitude_index),
            crossover=SBX(prob=crossover_prob, eta=crossover_eta, vtype=float),
            mutation=PM(prob=mutation_prob, eta=mutation_eta, vtype=float),
            eliminate_duplicates=True
        )
        
        # Termination criterion
        self.termination = get_termination("n_gen", n_generations)
        
        # Results
        self.result = None
        self.best_front = None
        self.history = []
        self.all_solutions = []
        self.final_population_solutions = []

    def _build_solutions(
        self,
        X: np.ndarray,
        F: np.ndarray,
        G: np.ndarray = None,
    ) -> List[Dict]:
        if X is None or F is None:
            return []

        if X.ndim == 1:
            X = X.reshape(1, -1)
            F = F.reshape(1, -1)
        if G is None:
            G = np.zeros((len(X), self.n_constraints), dtype=float)
        elif G.ndim == 1:
            G = G.reshape(1, -1)

        solutions = []
        for x, f, g in zip(X, F, G):
            node = int(np.clip(np.round(x[0]), 0, self.problem.xu[0]))
            if self.stimulation_model in DYNAMICS_FREE_STIMULATION_MODELS:
                if self.fixed_band_index is None:
                    band = int(np.clip(np.round(x[1]), 0, self.problem.xu[1]))
                    amplitude = float(x[2])
                else:
                    band = int(self.fixed_band_index)
                    amplitude = float(x[1])
                duration = None
                leak = None
            elif self.fixed_band_index is None:
                band = int(np.clip(np.round(x[1]), 0, self.problem.xu[1]))
                duration = float(x[2])
                amplitude = float(x[3])
                leak = float(x[4])
            else:
                band = int(self.fixed_band_index)
                duration = float(x[1])
                amplitude = float(x[2])
                leak = float(x[3])
            band_name = self.band_names[band] if band < len(self.band_names) else None
            constraint_values = np.asarray(g, dtype=float).reshape(-1)
            constraint_violation = float(np.sum(np.maximum(constraint_values, 0.0)))
            solution = {
                'node': node,
                'band': band,
                'band_name': band_name,
                'stimulation_duration': duration,
                'stimulation_amplitude': amplitude,
                'stimulation_total_change': (
                    amplitude
                    if self.stimulation_model == "static_adjacency"
                    else None
                ),
                'stimulation_activation_amount': (
                    amplitude
                    if self.stimulation_model == "adjacency_activation"
                    else None
                ),
                'stimulation_model': self.stimulation_model,
                'leak': leak,
                'objectives': f,
                'constraint_values': constraint_values,
                'constraint_violation': constraint_violation,
                'feasible': bool(constraint_violation <= 1e-10),
                'stimulation_polarity': (
                    'suppression' if amplitude < -1e-10 else
                    'enhancement' if amplitude > 1e-10 else
                    'zero'
                )
            }
            if self.activation_ratio_bounds is not None and constraint_values.size >= 2:
                ratio_min, ratio_max = self.activation_ratio_bounds
                solution['raw_activation_ratio_min'] = float(ratio_min - constraint_values[0])
                solution['raw_activation_ratio_max'] = float(ratio_max + constraint_values[1])
            solutions.append(solution)

        return solutions
    
    def optimize(self, verbose: bool = None):
        """
        Run NSGA-II optimization using pymoo.
        
        Parameters
        ----------
        verbose : bool, optional
            Override instance verbose setting
            
        Returns
        -------
        best_front : list of dict
            Pareto-optimal solutions, each with keys:
            - 'node': Stimulation node
            - 'band': Frequency band
            - 'band_name': Name of frequency band
            - 'objectives': Objective values
        history : list of dict
            History of optimization (objectives per generation)

        Notes
        -----
        Populates `self.all_solutions` with all evaluated solutions collected
        from optimization history (or final population if history is unavailable).
        """
        if verbose is None:
            verbose = self.verbose
        
        if verbose:
            print(f"\nStarting NSGA-II optimization with pymoo...")
            print(f"  Population size: {self.population_size}")
            print(f"  Generations: {self.n_generations}")
            print(f"  Objectives: {self.n_objectives}")
            print(f"  Nodes: {self.n_nodes}")
            print(f"  Bands: {self.n_bands}")
        
        # Run optimization
        self.result = minimize(
            self.problem,
            self.algorithm,
            self.termination,
            seed=self.seed,
            verbose=verbose,
            save_history=True
        )
        
        # Extract Pareto front
        self.best_front = self._build_solutions(
            self.result.X, self.result.F, getattr(self.result, 'G', None)
        )
        final_population = getattr(self.result, 'pop', None)
        if final_population is not None:
            self.final_population_solutions = self._build_solutions(
                final_population.get("X"),
                final_population.get("F"),
                final_population.get("G"),
            )

        # Collect all solutions from history (fallback to final population)
        all_X = []
        all_F = []
        all_G = []
        if hasattr(self.result, 'history') and self.result.history is not None:
            for h in self.result.history:
                pop = getattr(h, 'pop', None)
                if pop is None:
                    continue
                X = pop.get("X")
                F = pop.get("F")
                if X is None or F is None:
                    continue
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                    F = F.reshape(1, -1)
                all_X.append(X)
                all_F.append(F)
                G = pop.get("G")
                if G is None:
                    G = np.zeros((len(X), self.n_constraints), dtype=float)
                elif G.ndim == 1:
                    G = G.reshape(1, -1)
                all_G.append(G)

        if not all_X and getattr(self.result, 'pop', None) is not None:
            X = self.result.pop.get("X")
            F = self.result.pop.get("F")
            if X is not None and F is not None:
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                    F = F.reshape(1, -1)
                all_X = [X]
                all_F = [F]
                G = self.result.pop.get("G")
                if G is None:
                    G = np.zeros((len(X), self.n_constraints), dtype=float)
                elif G.ndim == 1:
                    G = G.reshape(1, -1)
                all_G = [G]

        if all_X:
            all_X = np.vstack(all_X)
            all_F = np.vstack(all_F)
            all_G = np.vstack(all_G)
            self.all_solutions = self._build_solutions(all_X, all_F, all_G)
        else:
            self.all_solutions = list(self.best_front)
        
        # Extract history
        self.history = []
        if hasattr(self.result, 'history') and self.result.history is not None:
            for i, h in enumerate(self.result.history):
                if h.opt is not None:
                    self.history.append({
                        'generation': i,
                        'best_front_size': len(h.opt),
                        'best_objectives': h.opt.get("F")
                    })
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"  Final Pareto front size: {len(self.best_front)}")
        
        return self.best_front, self.history
    
    def get_best_solution(self, preference_weights=None):
        """
        Get single best solution from Pareto front.
        
        Parameters
        ----------
        preference_weights : array-like, optional
            Weights for each objective (for weighted sum approach)
            If None, selects solution closest to ideal point
            
        Returns
        -------
        best_solution : dict
            Best solution with keys: 'node', 'band', 'band_name', 'objectives'
        """
        if not self.best_front:
            return None
        
        if len(self.best_front) == 1:
            return self.best_front[0]
        
        # Extract objectives
        objectives = np.array([sol['objectives'] for sol in self.best_front])
        
        if preference_weights is not None:
            # Weighted sum approach
            preference_weights = np.asarray(preference_weights, dtype=float)
            if preference_weights.shape[0] != objectives.shape[1]:
                raise ValueError(
                    f"preference_weights length ({preference_weights.shape[0]}) must match "
                    f"number of objectives ({objectives.shape[1]})."
                )
            weighted_sums = np.sum(preference_weights * objectives, axis=1)
            best_idx = np.argmin(weighted_sums)
        else:
            # Distance to ideal point based on Pareto-front minima
            ideal_point = objectives.min(axis=0)
            distances = np.linalg.norm(objectives - ideal_point, axis=1)
            best_idx = np.argmin(distances)
        
        return self.best_front[best_idx]
    
    def get_optimization_summary(self):
        """
        Get summary statistics of optimization results.
        
        Returns
        -------
        summary : dict
            Summary with keys:
            - 'n_solutions': Number of Pareto-optimal solutions
            - 'best_solution': Best solution (closest to ideal)
            - 'objective_ranges': Min/max for each objective
            - 'node_distribution': Count of each node in Pareto front
            - 'band_distribution': Count of each band in Pareto front
        """
        if not self.best_front:
            return None
        
        # Extract data
        nodes = [sol['node'] for sol in self.best_front]
        bands = [sol['band'] for sol in self.best_front]
        objectives = np.array([sol['objectives'] for sol in self.best_front])
        
        # Compute distributions
        node_counts = np.bincount(nodes, minlength=self.n_nodes)
        band_counts = np.bincount(bands, minlength=self.n_bands)
        
        # Objective ranges
        obj_ranges = {
            f'obj_{i}': {
                'min': float(objectives[:, i].min()),
                'max': float(objectives[:, i].max()),
                'mean': float(objectives[:, i].mean())
            }
            for i in range(objectives.shape[1])
        }
        
        summary = {
            'n_solutions': len(self.best_front),
            'best_solution': self.get_best_solution(),
            'objective_ranges': obj_ranges,
            'node_distribution': node_counts.tolist(),
            'band_distribution': band_counts.tolist()
        }
        
        return summary


# Example usage
if __name__ == "__main__":
    # Define a simple test function
    def test_evaluate(node, band, duration, amplitude, leak):
        """Test evaluation function (4 objectives)."""
        # Objective 1: prefer lower node indices
        obj1 = float(node)
        # Objective 2: prefer higher band indices
        obj2 = float(5 - band)
        # Objective 3: prefer node + band = 5
        obj3 = abs(node + band - 5)
        # Objective 4: prefer middle node index
        obj4 = abs(node - 4.5)
        return np.array([obj1, obj2, obj3, obj4])
    
    # Create optimizer
    print("Creating NSGA-II optimizer with pymoo...")
    optimizer = NSGAIIOptimizer(
        n_nodes=10,
        n_bands=5,
        band_names=['delta', 'theta', 'alpha', 'beta', 'gamma'],
        evaluate_func=test_evaluate,
        duration_bounds=(0.0, 2.0),
        amplitude_bounds=(0.0, 2.0),
        leak_bounds=(0.0, 2.0),
        n_objectives=4,
        population_size=50,
        n_generations=30,
        seed=42
    )
    
    # Run optimization
    print("\nRunning optimization...")
    best_front, history = optimizer.optimize(verbose=True)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nPareto front size: {len(best_front)}")
    
    print("\nPareto-optimal solutions:")
    for i, sol in enumerate(best_front[:5]):  # Show first 5
        print(f"  Solution {i+1}: Node={sol['node']}, Band={sol['band_name']}, "
              f"Objectives={sol['objectives']}")
    
    # Get best solution
    best = optimizer.get_best_solution()
    print(f"\nBest solution (closest to ideal):")
    print(f"  Node: {best['node']}")
    print(f"  Band: {best['band_name']}")
    print(f"  Objectives: {best['objectives']}")
    
    # Get summary
    summary = optimizer.get_optimization_summary()
    print(f"\nSummary:")
    print(f"  Number of Pareto-optimal solutions: {summary['n_solutions']}")
    print(f"  Objective ranges:")
    for obj, ranges in summary['objective_ranges'].items():
        print(f"    {obj}: [{ranges['min']:.3f}, {ranges['max']:.3f}], mean={ranges['mean']:.3f}")
