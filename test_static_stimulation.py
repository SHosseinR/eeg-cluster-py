"""Focused tests for the dynamics-free adjacency stimulation model."""

from __future__ import annotations

import unittest

import numpy as np

from nsga_optimizer import EEGOptimizationProblem, NSGAIIOptimizer
from stimulation_models import apply_static_adjacency_stimulation


class StaticAdjacencyStimulationTests(unittest.TestCase):
    def test_incident_change_is_weight_scaled_symmetric_and_conserved(self):
        matrix = np.array(
            [
                [0.0, 0.2, 0.6],
                [0.2, 0.0, 0.4],
                [0.6, 0.4, 0.0],
            ]
        )
        updated, details = apply_static_adjacency_stimulation(
            matrix, 0, 0.8, edge_scope="incident"
        )
        delta = updated - matrix
        np.testing.assert_allclose(updated, updated.T)
        self.assertAlmostEqual(float(np.sum(np.abs(delta))), 0.8)
        self.assertAlmostEqual(delta[0, 2] / delta[0, 1], 3.0)
        self.assertEqual(delta[1, 2], 0.0)
        self.assertAlmostEqual(details["realized_total_change_l1"], 0.8)

    def test_negative_amount_weakens_existing_edges(self):
        matrix = np.array(
            [[0.0, 0.2, 0.6], [0.2, 0.0, 0.4], [0.6, 0.4, 0.0]]
        )
        updated, _ = apply_static_adjacency_stimulation(matrix, 0, -0.4)
        self.assertTrue(np.all(updated[0, 1:] < matrix[0, 1:]))
        self.assertAlmostEqual(float(np.sum(np.abs(updated - matrix))), 0.4)

    def test_directed_scope_changes_only_requested_axis(self):
        matrix = np.array(
            [[0.0, 0.2, 0.4], [0.5, 0.0, 0.3], [0.8, 0.6, 0.0]]
        )
        outgoing, _ = apply_static_adjacency_stimulation(
            matrix, 1, 0.3, edge_scope="outgoing"
        )
        delta = outgoing - matrix
        self.assertTrue(np.any(delta[1, :] != 0.0))
        np.testing.assert_allclose(delta[[0, 2], :], 0.0)
        self.assertAlmostEqual(float(np.sum(np.abs(delta))), 0.3)

    def test_static_problem_removes_duration_and_leak(self):
        seen = []

        def evaluate(node, band, duration, amount, leak):
            seen.append((node, band, duration, amount, leak))
            return np.array([amount**2])

        problem = EEGOptimizationProblem(
            n_nodes=4,
            n_bands=1,
            evaluate_func=evaluate,
            amplitude_bounds=(-2.0, 2.0),
            n_objectives=1,
            fixed_band_index=0,
            stimulation_model="static_adjacency",
        )
        self.assertEqual(problem.n_var, 2)
        output = {}
        problem._evaluate(np.array([[2.0, -0.75]]), output)
        self.assertEqual(seen, [(2, 0, None, -0.75, None)])
        np.testing.assert_allclose(output["F"], [[0.5625]])

    def test_static_nsga_solution_records_new_contract(self):
        def evaluate(node, band, duration, amount, leak):
            return np.array([(amount - 0.25) ** 2])

        optimizer = NSGAIIOptimizer(
            n_nodes=3,
            n_bands=1,
            band_names=["alpha"],
            evaluate_func=evaluate,
            amplitude_bounds=(-1.0, 1.0),
            n_objectives=1,
            population_size=12,
            n_generations=4,
            seed=7,
            fixed_band_index=0,
            stimulation_model="static_adjacency",
        )
        front, _ = optimizer.optimize(verbose=False)
        self.assertTrue(front)
        self.assertTrue(all(item["stimulation_duration"] is None for item in front))
        self.assertTrue(all(item["leak"] is None for item in front))
        self.assertTrue(
            all(
                item["stimulation_total_change"]
                == item["stimulation_amplitude"]
                for item in front
            )
        )


if __name__ == "__main__":
    unittest.main()
