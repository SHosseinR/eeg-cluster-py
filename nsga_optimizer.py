"""
NSGA-II optimizer using pymoo library for EEG connectivity optimization
"""
import numpy as np
from typing import Callable, List, Dict

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination


class EEGOptimizationProblem(Problem):
    """
    pymoo Problem definition for EEG optimization.
    
    Decision variables:
    - x[0]: Stimulation node (integer, 0 to n_nodes-1)
    - x[1]: Frequency band (integer, 0 to n_bands-1)
    
    Objectives:
    - f[0], f[1], ..., f[n-1]: Network measures to optimize (dynamic number)
    """
    
    def __init__(self, 
                 n_nodes: int,
                 n_bands: int,
                 evaluate_func: Callable,
                 n_objectives: int = 3):
        """
        Initialize EEG optimization problem.
        
        Parameters
        ----------
        n_nodes : int
            Number of nodes (EEG channels)
        n_bands : int
            Number of frequency bands
        evaluate_func : callable
            Function to evaluate objectives: func(node, band) -> objectives array
        n_objectives : int
            Number of objectives to optimize (default: 3)
        """
        self.evaluate_func = evaluate_func
        
        # Define problem
        super().__init__(
            n_var=2,  # Two decision variables: node and band
            n_obj=n_objectives,  # Number of objectives
            n_constr=0,  # No constraints
            xl=np.array([0, 0]),  # Lower bounds
            xu=np.array([n_nodes - 1, n_bands - 1]),  # Upper bounds
            type_var=np.int64  # Integer variables
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate objectives for a population.
        
        Parameters
        ----------
        X : ndarray, shape (n_pop, 2)
            Population of solutions (node, band pairs)
        out : dict
            Output dictionary to store objectives
        """
        # Evaluate each individual
        objectives = []
        for x in X:
            node = int(x[0])
            band = int(x[1])
            obj = self.evaluate_func(node, band)
            objectives.append(obj)
        
        # Store objectives
        out["F"] = np.array(objectives)


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
                 population_size: int = 100,
                 n_generations: int = 50,
                 crossover_prob: float = 0.9,
                 crossover_eta: float = 15.0,
                 mutation_prob: float = None,
                 mutation_eta: float = 20.0,
                 seed: int = None,
                 verbose: bool = True):
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
            Function to evaluate objectives: func(node, band) -> objectives array
        population_size : int
            Population size (default: 100)
        n_generations : int
            Number of generations (default: 50)
        crossover_prob : float
            Crossover probability (default: 0.9)
        crossover_eta : float
            Crossover distribution index for SBX (default: 15.0)
        mutation_prob : float
            Mutation probability (default: 1/n_var = 0.5 for 2 variables)
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
        self.population_size = population_size
        self.n_generations = n_generations
        self.seed = seed
        self.verbose = verbose
        
        # Store number of objectives (will be set when creating problem)
        self.n_objectives = None
        
        # Problem will be created when we know n_objectives
        self.problem = None
        
    def set_problem(self, n_objectives: int):
        """
        Create the optimization problem with specified number of objectives.
        
        Parameters
        ----------
        n_objectives : int
            Number of objectives to optimize
        """
        self.n_objectives = n_objectives
        self.problem = EEGOptimizationProblem(
            n_nodes=self.n_nodes,
            n_bands=self.n_bands,
            evaluate_func=self.evaluate_func,
            n_objectives=n_objectives
        )
        
        # Set default mutation probability if not specified
        if mutation_prob is None:
            mutation_prob = 1.0 / self.problem.n_var  # 0.5 for 2 variables
        
        # Create algorithm
        self.algorithm = NSGA2(
            pop_size=population_size,
            sampling=IntegerRandomSampling(),
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
        """
        if verbose is None:
            verbose = self.verbose
        
        # Ensure problem is created
        if self.problem is None:
            raise RuntimeError("Problem not initialized. Call set_problem() first.")
        
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
        self.best_front = []
        if self.result.X is not None:
            X = self.result.X
            F = self.result.F
            
            # Handle single solution (1D array)
            if X.ndim == 1:
                X = X.reshape(1, -1)
                F = F.reshape(1, -1)
            
            for x, f in zip(X, F):
                node = int(x[0])
                band = int(x[1])
                solution = {
                    'node': node,
                    'band': band,
                    'band_name': self.band_names[band],
                    'objectives': f
                }
                self.best_front.append(solution)
        
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
            weighted_sums = np.sum(preference_weights * objectives, axis=1)
            best_idx = np.argmin(weighted_sums)
        else:
            # Distance to ideal point (all objectives minimized to 0)
            distances = np.linalg.norm(objectives, axis=1)
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
    def test_evaluate(node, band):
        """Test evaluation function (3 objectives)."""
        # Objective 1: prefer lower node indices
        obj1 = float(node)
        # Objective 2: prefer higher band indices
        obj2 = float(5 - band)
        # Objective 3: prefer node + band = 5
        obj3 = abs(node + band - 5)
        return np.array([obj1, obj2, obj3])
    
    # Create optimizer
    print("Creating NSGA-II optimizer with pymoo...")
    optimizer = NSGAIIOptimizer(
        n_nodes=10,
        n_bands=5,
        band_names=['delta', 'theta', 'alpha', 'beta', 'gamma'],
        evaluate_func=test_evaluate,
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

