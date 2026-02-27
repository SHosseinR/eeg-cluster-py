"""
NSGA-II optimizer for EEG connectivity optimization
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Callable, Dict
import copy


@dataclass
class Individual:
    """Represents an individual solution in the population."""
    node: int  # Stimulation node (0-indexed)
    band: int  # Frequency band index (0-indexed)
    objectives: np.ndarray = None  # Objective values
    rank: int = None  # Pareto rank
    crowding_distance: float = 0.0  # Crowding distance
    
    def dominates(self, other):
        """Check if this individual dominates another (for minimization)."""
        if self.objectives is None or other.objectives is None:
            return False
        
        # For minimization: self dominates other if:
        # - self is no worse in all objectives
        # - self is strictly better in at least one objective
        no_worse = np.all(self.objectives <= other.objectives)
        strictly_better = np.any(self.objectives < other.objectives)
        
        return no_worse and strictly_better


class NSGAIIOptimizer:
    """
    NSGA-II optimizer for multi-objective optimization.
    
    Optimizes stimulation node and frequency band to optimize
    network measures.
    """
    
    def __init__(self, 
                 n_nodes: int,
                 n_bands: int,
                 band_names: List[str],
                 evaluate_func: Callable,
                 population_size: int = 100,
                 n_generations: int = 50,
                 crossover_prob: float = 0.9,
                 mutation_prob: float = 0.1,
                 tournament_size: int = 3,
                 random_seed: int = None):
        """
        Initialize NSGA-II optimizer.
        
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
            Size of population (default: 100)
        n_generations : int
            Number of generations (default: 50)
        crossover_prob : float
            Probability of crossover (default: 0.9)
        mutation_prob : float
            Probability of mutation (default: 0.1)
        tournament_size : int
            Tournament size for selection (default: 3)
        random_seed : int
            Random seed for reproducibility (default: None)
        """
        self.n_nodes = n_nodes
        self.n_bands = n_bands
        self.band_names = band_names
        self.evaluate_func = evaluate_func
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.population = []
        self.best_front = []
        self.history = []
    
    def initialize_population(self):
        """Create initial random population."""
        self.population = []
        for _ in range(self.population_size):
            node = np.random.randint(0, self.n_nodes)
            band = np.random.randint(0, self.n_bands)
            individual = Individual(node=node, band=band)
            self.population.append(individual)
    
    def evaluate_population(self, population: List[Individual]):
        """Evaluate objectives for all individuals in population."""
        for ind in population:
            if ind.objectives is None:
                ind.objectives = self.evaluate_func(ind.node, ind.band)
    
    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """
        Fast non-dominated sorting algorithm.
        
        Returns list of fronts, where front[0] is the Pareto front.
        """
        # Count domination for each individual
        domination_count = [0] * len(population)
        dominated_solutions = [[] for _ in range(len(population))]
        
        # Find domination relationships
        for i, ind_i in enumerate(population):
            for j, ind_j in enumerate(population):
                if i != j:
                    if ind_i.dominates(ind_j):
                        dominated_solutions[i].append(j)
                    elif ind_j.dominates(ind_i):
                        domination_count[i] += 1
        
        # Create fronts
        fronts = [[]]
        for i, count in enumerate(domination_count):
            if count == 0:
                population[i].rank = 0
                fronts[0].append(population[i])
        
        # Build remaining fronts
        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for ind in fronts[current_front]:
                ind_idx = population.index(ind)
                for dominated_idx in dominated_solutions[ind_idx]:
                    domination_count[dominated_idx] -= 1
                    if domination_count[dominated_idx] == 0:
                        population[dominated_idx].rank = current_front + 1
                        next_front.append(population[dominated_idx])
            current_front += 1
            fronts.append(next_front)
        
        # Remove empty last front
        if len(fronts[-1]) == 0:
            fronts.pop()
        
        return fronts
    
    def calculate_crowding_distance(self, front: List[Individual]):
        """Calculate crowding distance for individuals in a front."""
        n = len(front)
        
        if n == 0:
            return
        
        # Initialize distances
        for ind in front:
            ind.crowding_distance = 0.0
        
        # If only 1 or 2 individuals, they get infinite distance
        if n <= 2:
            for ind in front:
                ind.crowding_distance = float('inf')
            return
        
        # Get number of objectives
        n_objectives = len(front[0].objectives)
        
        # Calculate distance for each objective
        for obj_idx in range(n_objectives):
            # Sort by objective value
            front_sorted = sorted(front, key=lambda x: x.objectives[obj_idx])
            
            # Boundary points get infinite distance
            front_sorted[0].crowding_distance = float('inf')
            front_sorted[-1].crowding_distance = float('inf')
            
            # Calculate distance for interior points
            obj_min = front_sorted[0].objectives[obj_idx]
            obj_max = front_sorted[-1].objectives[obj_idx]
            obj_range = obj_max - obj_min
            
            if obj_range > 0:
                for i in range(1, n - 1):
                    distance = (front_sorted[i + 1].objectives[obj_idx] - 
                               front_sorted[i - 1].objectives[obj_idx]) / obj_range
                    front_sorted[i].crowding_distance += distance
    
    def tournament_selection(self, population: List[Individual]) -> Individual:
        """Select individual using tournament selection."""
        # Randomly select tournament_size individuals
        tournament = np.random.choice(population, size=self.tournament_size, replace=False)
        
        # Select best based on rank and crowding distance
        best = tournament[0]
        for ind in tournament[1:]:
            if ind.rank < best.rank:
                best = ind
            elif ind.rank == best.rank and ind.crowding_distance > best.crowding_distance:
                best = ind
        
        return best
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Single-point crossover."""
        if np.random.random() < self.crossover_prob:
            # Crossover node from parent1, band from parent2
            child1 = Individual(node=parent1.node, band=parent2.band)
            child2 = Individual(node=parent2.node, band=parent1.band)
        else:
            # No crossover, copy parents
            child1 = Individual(node=parent1.node, band=parent1.band)
            child2 = Individual(node=parent2.node, band=parent2.band)
        
        return child1, child2
    
    def mutate(self, individual: Individual):
        """Mutate individual."""
        # Mutate node
        if np.random.random() < self.mutation_prob:
            individual.node = np.random.randint(0, self.n_nodes)
        
        # Mutate band
        if np.random.random() < self.mutation_prob:
            individual.band = np.random.randint(0, self.n_bands)
    
    def create_offspring(self) -> List[Individual]:
        """Create offspring population through selection, crossover, and mutation."""
        offspring = []
        
        while len(offspring) < self.population_size:
            # Select parents
            parent1 = self.tournament_selection(self.population)
            parent2 = self.tournament_selection(self.population)
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutation
            self.mutate(child1)
            self.mutate(child2)
            
            offspring.extend([child1, child2])
        
        # Trim to population size
        offspring = offspring[:self.population_size]
        
        return offspring
    
    def select_next_generation(self, combined_population: List[Individual]) -> List[Individual]:
        """Select next generation using elitism."""
        # Fast non-dominated sort
        fronts = self.fast_non_dominated_sort(combined_population)
        
        # Calculate crowding distance for each front
        for front in fronts:
            self.calculate_crowding_distance(front)
        
        # Select next generation
        next_generation = []
        for front in fronts:
            if len(next_generation) + len(front) <= self.population_size:
                # Add entire front
                next_generation.extend(front)
            else:
                # Sort by crowding distance and add best
                front_sorted = sorted(front, key=lambda x: x.crowding_distance, reverse=True)
                remaining = self.population_size - len(next_generation)
                next_generation.extend(front_sorted[:remaining])
                break
        
        return next_generation
    
    def optimize(self, verbose=True):
        """
        Run NSGA-II optimization.
        
        Parameters
        ----------
        verbose : bool
            Print progress information (default: True)
            
        Returns
        -------
        best_front : list of Individual
            Pareto-optimal solutions (best front)
        history : list of dict
            History of optimization (objectives per generation)
        """
        if verbose:
            print(f"\nStarting NSGA-II optimization...")
            print(f"  Population size: {self.population_size}")
            print(f"  Generations: {self.n_generations}")
            print(f"  Nodes: {self.n_nodes}")
            print(f"  Bands: {self.n_bands}")
        
        # Initialize population
        self.initialize_population()
        self.evaluate_population(self.population)
        
        # Evolution loop
        for generation in range(self.n_generations):
            # Create offspring
            offspring = self.create_offspring()
            
            # Evaluate offspring
            self.evaluate_population(offspring)
            
            # Combine parent and offspring
            combined = self.population + offspring
            
            # Select next generation
            self.population = self.select_next_generation(combined)
            
            # Store history
            fronts = self.fast_non_dominated_sort(self.population)
            best_front_objs = np.array([ind.objectives for ind in fronts[0]])
            self.history.append({
                'generation': generation,
                'best_front_size': len(fronts[0]),
                'best_objectives': best_front_objs
            })
            
            if verbose and (generation + 1) % 10 == 0:
                print(f"  Generation {generation + 1}/{self.n_generations}: "
                      f"Pareto front size = {len(fronts[0])}")
        
        # Get final Pareto front
        fronts = self.fast_non_dominated_sort(self.population)
        self.best_front = fronts[0]
        
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
        best_individual : Individual
            Best solution based on preference
        """
        if not self.best_front:
            return None
        
        if preference_weights is not None:
            # Weighted sum approach
            weighted_sums = []
            for ind in self.best_front:
                weighted_sum = np.sum(preference_weights * ind.objectives)
                weighted_sums.append(weighted_sum)
            best_idx = np.argmin(weighted_sums)
        else:
            # Distance to ideal point (all objectives minimized to 0)
            distances = []
            for ind in self.best_front:
                distance = np.linalg.norm(ind.objectives)
                distances.append(distance)
            best_idx = np.argmin(distances)
        
        return self.best_front[best_idx]


# Example usage
if __name__ == "__main__":
    # Define a simple test function
    def test_evaluate(node, band):
        """Test evaluation function (2 objectives)."""
        # Objective 1: prefer lower node indices
        obj1 = node
        # Objective 2: prefer higher band indices
        obj2 = 5 - band
        return np.array([obj1, obj2])
    
    # Create optimizer
    optimizer = NSGAIIOptimizer(
        n_nodes=10,
        n_bands=5,
        band_names=['delta', 'theta', 'alpha', 'beta', 'gamma'],
        evaluate_func=test_evaluate,
        population_size=50,
        n_generations=30
    )
    
    # Run optimization
    best_front, history = optimizer.optimize(verbose=True)
    
    # Print Pareto front
    print("\nPareto front solutions:")
    for i, ind in enumerate(best_front):
        print(f"  Solution {i + 1}: Node={ind.node}, Band={ind.band}, "
              f"Objectives={ind.objectives}")
