"""
Test script for optimization module components
"""
import numpy as np
import sys

def test_state_space_simulation():
    """Test state-space simulation components."""
    print("\n" + "="*80)
    print("Testing State-Space Simulation")
    print("="*80)
    
    from state_space_simulation import (
        normalize_adjacency_matrix,
        create_control_matrix,
        create_stimulation_signal,
        simulate_eeg_dynamics,
        run_full_simulation
    )
    
    # Create test data
    n_nodes = 5
    A = np.random.rand(n_nodes, n_nodes) * 0.5
    A = (A + A.T) / 2  # Symmetric
    xbar = np.random.rand(n_nodes) * 0.5 + 0.5
    
    print(f"  Test matrix shape: {A.shape}")
    print(f"  Test baseline shape: {xbar.shape}")
    
    # Test normalization
    A_norm = normalize_adjacency_matrix(A)
    w, _ = np.linalg.eig(A_norm)
    max_eigenvalue = np.abs(w).max()
    print(f"  ✓ Matrix normalized (max eigenvalue: {max_eigenvalue:.4f})")
    assert max_eigenvalue < 1.0, "Eigenvalue should be < 1 for stability"
    
    # Test control matrix
    B = create_control_matrix(n_nodes, stimulation_node=2)
    print(f"  ✓ Control matrix created: {B.shape}")
    assert B[2, 2] == 1.0, "Stimulation node should be 1"
    assert np.sum(B) == 1.0, "Only one node should be stimulated"
    
    # Test stimulation signal
    U = create_stimulation_signal(n_nodes, 2, duration=1.0, dt=0.001)
    print(f"  ✓ Stimulation signal created: {U.shape}")
    assert U.shape[1] == 1000, "Should have 1000 timesteps"
    
    # Test simulation
    results = run_full_simulation(A, xbar, stimulation_node=2)
    print(f"  ✓ Simulation completed")
    print(f"    - Trajectory shape: {results['trajectory'].shape}")
    print(f"    - Final state shape: {results['final_state'].shape}")
    print(f"    - Activation ratios shape: {results['activation_ratios'].shape}")
    
    assert results['final_state'].shape[0] == n_nodes
    assert results['activation_ratios'].shape[0] == n_nodes
    
    print("\n✓ State-space simulation tests PASSED")
    return True


def test_plasticity():
    """Test plasticity components."""
    print("\n" + "="*80)
    print("Testing Plasticity Module")
    print("="*80)
    
    from plasticity import (
        apply_plasticity_updates,
        normalize_connectivity_matrix,
        compute_plasticity_effect
    )
    
    # Create test data
    n_nodes = 5
    A = np.random.rand(n_nodes, n_nodes) * 0.5
    A = (A + A.T) / 2
    activation_ratios = np.array([1.2, 0.8, 1.5, 1.0, 0.9])
    
    print(f"  Test matrix shape: {A.shape}")
    print(f"  Activation ratios: {activation_ratios}")
    
    # Test plasticity updates
    updated = apply_plasticity_updates(A, activation_ratios)
    print(f"  ✓ Plasticity updates applied")
    assert updated.shape == A.shape
    
    # Test normalization
    normalized = normalize_connectivity_matrix(updated)
    print(f"  ✓ Matrix normalized")
    assert np.min(normalized) >= 0 and np.max(normalized) <= 1
    
    # Test full pipeline
    final = compute_plasticity_effect(A, activation_ratios, normalize=True)
    print(f"  ✓ Full plasticity pipeline executed")
    assert final.shape == A.shape
    
    print("\n✓ Plasticity tests PASSED")
    return True


def test_nsga_optimizer():
    """Test NSGA-II optimizer."""
    print("\n" + "="*80)
    print("Testing NSGA-II Optimizer")
    print("="*80)
    
    from nsga_optimizer import NSGAIIOptimizer
    
    # Define simple test evaluation function
    def test_evaluate(node, band):
        """Test function with 4 objectives."""
        return np.array([
            float(node),
            float(5 - 2 * band),
            float(abs(node + band - 5)),
            float(abs(node - 4.5))
        ])
    
    # Create optimizer
    optimizer = NSGAIIOptimizer(
        n_nodes=10,
        n_bands=5,
        band_names=['delta', 'theta', 'alpha', 'beta', 'gamma'],
        evaluate_func=test_evaluate,
        n_objectives=4,
        population_size=20,
        n_generations=10
    )
    
    print(f"  ✓ Optimizer created")
    print(f"    - Population size: {optimizer.population_size}")
    print(f"    - Generations: {optimizer.n_generations}")
    print(f"    - Nodes: {optimizer.n_nodes}")
    print(f"    - Bands: {optimizer.n_bands}")
    print(f"    - Objectives: {optimizer.problem.n_obj}")
    
    # # Test population initialization
    # optimizer.initialize_population()
    # print(f"  ✓ Population initialized: {len(optimizer.population)} individuals")
    # assert len(optimizer.population) == optimizer.population_size
    
    # # Test evaluation
    # optimizer.evaluate_population(optimizer.population)
    # print(f"  ✓ Population evaluated")
    # assert all(ind.objectives is not None for ind in optimizer.population)
    
    # # Test non-dominated sorting
    # fronts = optimizer.fast_non_dominated_sort(optimizer.population)
    # print(f"  ✓ Non-dominated sorting: {len(fronts)} fronts")
    # assert len(fronts) > 0
    
    # # Test crowding distance
    # if len(fronts[0]) > 0:
    #     optimizer.calculate_crowding_distance(fronts[0])
    #     print(f"  ✓ Crowding distance calculated")
    
    # Run short optimization
    print("\n  Running optimization (10 generations)...")
    best_front, history = optimizer.optimize(verbose=False)
    print(f"  ✓ Optimization completed")
    print(f"    - Pareto front size: {len(best_front)}")
    print(f"    - History length: {len(history)}")
    
    assert len(best_front) > 0
    assert len(history) == optimizer.n_generations
    assert optimizer.problem.n_obj == 4
    assert len(best_front[0]['objectives']) == 4
    
    # Get best solution
    best = optimizer.get_best_solution()
    print(f"  ✓ Best solution: Node={best['node']}, Band={best['band']}")
    
    print("\n✓ NSGA-II optimizer tests PASSED")
    return True


def test_full_pipeline():
    """Test integration of all components."""
    print("\n" + "="*80)
    print("Testing Full Integration")
    print("="*80)
    
    # Create synthetic data
    n_nodes = 5
    n_bands = 3
    
    # Create adjacency matrices
    connectivity_matrices = {
        'Patient': {
            'P1': {
                'plv': {
                    'delta': np.random.rand(n_nodes, n_nodes) * 0.5,
                    'theta': np.random.rand(n_nodes, n_nodes) * 0.5,
                    'alpha': np.random.rand(n_nodes, n_nodes) * 0.5,
                }
            }
        },
        'Healthy': {
            'H1': {
                'plv': {
                    'delta': np.random.rand(n_nodes, n_nodes) * 0.5,
                    'theta': np.random.rand(n_nodes, n_nodes) * 0.5,
                    'alpha': np.random.rand(n_nodes, n_nodes) * 0.5,
                }
            }
        }
    }
    
    # Make matrices symmetric
    for group in connectivity_matrices.values():
        for subj in group.values():
            for method in subj.values():
                for band in method.keys():
                    mat = method[band]
                    method[band] = (mat + mat.T) / 2
    
    # Create network measures
    network_measures = {
        'Patient': {
            'P1': {
                'plv': {
                    'delta': {'global_efficiency': 0.3, 'clustering_coefficient': 0.4, 'modularity': 0.5},
                    'theta': {'global_efficiency': 0.3, 'clustering_coefficient': 0.4, 'modularity': 0.5},
                    'alpha': {'global_efficiency': 0.3, 'clustering_coefficient': 0.4, 'modularity': 0.5},
                }
            }
        },
        'Healthy': {
            'H1': {
                'plv': {
                    'delta': {'global_efficiency': 0.4, 'clustering_coefficient': 0.5, 'modularity': 0.6},
                    'theta': {'global_efficiency': 0.4, 'clustering_coefficient': 0.5, 'modularity': 0.6},
                    'alpha': {'global_efficiency': 0.4, 'clustering_coefficient': 0.5, 'modularity': 0.6},
                }
            }
        }
    }
    
    # Create subject data
    subject_data = {
        'P1': {
            'data': np.random.rand(n_nodes, 1000),
            'fs': 250,
            'channels': [f'Ch{i}' for i in range(n_nodes)],
            'group': 'Patient'
        },
        'H1': {
            'data': np.random.rand(n_nodes, 1000),
            'fs': 250,
            'channels': [f'Ch{i}' for i in range(n_nodes)],
            'group': 'Healthy'
        }
    }
    
    print("  ✓ Synthetic data created")
    
    # Import optimization module
    from eeg_optimization import EEGOptimizer
    
    # Create optimizer
    optimizer = EEGOptimizer(
        connectivity_matrices=connectivity_matrices,
        network_measures=network_measures,
        subject_data=subject_data,
        frequency_bands={'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13)},
        channel_names=[f'Ch{i}' for i in range(n_nodes)],
        selected_method='plv',
        optimization_measures=['global_efficiency', 'clustering_coefficient', 'modularity'],
        nsga_config={'population_size': 10, 'n_generations': 5, 
                    'crossover_prob': 0.9, 'mutation_prob': 0.1, 'tournament_size': 3}
    )
    
    print("  ✓ Optimizer created")
    
    # Optimize single subject
    print("\n  Running optimization for single subject (5 generations)...")
    results = optimizer.optimize_subject('P1', verbose=False)
    
    print(f"  ✓ Optimization completed")
    print(f"    - Best node: {results['best_solution']['node']}")
    print(f"    - Best band: {results['best_solution']['band']}")
    print(f"    - Objectives: {results['best_solution']['objectives']}")
    
    assert results['best_solution'] is not None
    assert len(results['best_front']) > 0
    
    print("\n✓ Full integration tests PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("OPTIMIZATION MODULE TEST SUITE")
    print("="*80)
    
    tests = [
        ("State-Space Simulation", test_state_space_simulation),
        ("Plasticity", test_plasticity),
        ("NSGA-II Optimizer", test_nsga_optimizer),
        ("Full Integration", test_full_pipeline),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            success = test_func()
            results[name] = "PASSED" if success else "FAILED"
        except Exception as e:
            print(f"\n✗ {name} test FAILED with error:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            results[name] = "FAILED"
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for name, result in results.items():
        status = "✓" if result == "PASSED" else "✗"
        print(f"  {status} {name}: {result}")
    
    all_passed = all(r == "PASSED" for r in results.values())
    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
