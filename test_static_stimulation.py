"""Focused tests for the dynamics-free adjacency stimulation model."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import numpy as np

from nsga_optimizer import EEGOptimizationProblem, NSGAIIOptimizer
import plot_subject_activation_and_adjacency
from state_space_simulation import normalize_adjacency_matrix
from stimulation_models import (
    apply_static_adjacency_stimulation,
    run_adjacency_activation_stimulation,
)


class StaticAdjacencyStimulationTests(unittest.TestCase):
    def test_adjacency_activation_profiles_share_hybrid_contract(self):
        config_dir = Path(__file__).resolve().parent / "dataset_configs"
        profiles = (
            "tdbrain_coherence_adjacency_activation_signed_no_rejection_logistic.toml",
            "first_paper_coherence_adjacency_activation_signed_no_rejection_logistic.toml",
        )
        output_directories = []
        for profile in profiles:
            with (config_dir / profile).open("rb") as stream:
                settings = tomllib.load(stream)
            optimization = settings["optimization"]
            self.assertEqual(settings["classification"]["models"], ["logistic_l2"])
            self.assertEqual(
                optimization["stimulation_model"],
                "adjacency_activation",
            )
            self.assertEqual(
                optimization["stimulation_activation_amount_bounds"],
                [-3.0, 3.0],
            )
            self.assertEqual(
                optimization["adjacency_activation_neighbor_scale"],
                1.0,
            )
            self.assertEqual(optimization["patient_rejection_percent"], 0)
            output_directories.append(settings["output_directory"])
        self.assertTrue(
            all(
                path.startswith(
                    "./results-adjact-signed-norej-logistic/"
                )
                for path in output_directories
            )
        )

    def test_adjacency_activation_is_direct_plus_one_hop_only(self):
        matrix = np.array(
            [
                [0.0, 0.6, 0.0],
                [0.6, 0.0, 0.4],
                [0.0, 0.4, 0.0],
            ]
        )
        baseline = np.array([0.5, 0.6, 0.7])
        amount = 0.2
        neighbor_scale = 0.75
        result = run_adjacency_activation_stimulation(
            matrix,
            baseline,
            stimulation_node=0,
            stimulation_amount=amount,
            neighbor_scale=neighbor_scale,
            stability_constant=0.01,
        )
        normalized = normalize_adjacency_matrix(matrix, 0.01)
        expected_delta = amount * (
            np.array([1.0, 0.0, 0.0])
            + neighbor_scale * normalized[:, 0]
        )
        np.testing.assert_allclose(result["activation_change"], expected_delta)
        np.testing.assert_allclose(
            result["final_state"],
            baseline + expected_delta,
        )
        self.assertAlmostEqual(result["activation_change"][0], amount)
        # Node 2 has a two-hop path from node 0 but no direct edge; unlike the
        # state-space model, the adjacency-activation mode does not reach it.
        self.assertEqual(result["activation_change"][2], 0.0)
        self.assertIsNone(result["trajectory"])

    def test_adjacency_activation_nsga_removes_duration_and_leak(self):
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
            stimulation_model="adjacency_activation",
        )
        self.assertEqual(problem.n_var, 2)
        output = {}
        problem._evaluate(np.array([[2.0, -0.75]]), output)
        self.assertEqual(seen, [(2, 0, None, -0.75, None)])
        np.testing.assert_allclose(output["F"], [[0.5625]])

    def test_adjacency_activation_subject_plot_writes_activation_and_edges(self):
        matrix = np.array(
            [
                [0.0, 0.3, 0.2],
                [0.3, 0.0, 0.4],
                [0.2, 0.4, 0.0],
            ]
        )
        results = {
            "P1::alpha": {
                "subject_id": "P1",
                "best_solution": {
                    "node": 0,
                    "band": 0,
                    "band_name": "alpha",
                    "stimulation_model": "adjacency_activation",
                    "stimulation_duration": None,
                    "stimulation_amplitude": 0.1,
                    "stimulation_activation_amount": 0.1,
                    "leak": None,
                },
                "band_names": ["alpha"],
                "baseline_activation": np.array([0.5, 0.6, 0.7]),
                "channel_names": ["A", "B", "C"],
                "stimulation_model": "adjacency_activation",
                "adjacency_activation_neighbor_scale": 1.0,
            }
        }
        connectivity = {
            "Patient": {"P1": {"coh": {"alpha": matrix}}},
        }
        with tempfile.TemporaryDirectory() as directory:
            results_path = os.path.join(directory, "results.npy")
            connectivity_path = os.path.join(directory, "connectivity.npy")
            output_dir = os.path.join(directory, "figures")
            np.save(results_path, results, allow_pickle=True)
            np.save(connectivity_path, connectivity, allow_pickle=True)
            argv = [
                "plot_subject_activation_and_adjacency.py",
                "--subject",
                "P1",
                "--results",
                results_path,
                "--connectivity",
                connectivity_path,
                "--output-dir",
                output_dir,
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    plot_subject_activation_and_adjacency,
                    "SELECTED_METHOD",
                    "coh",
                ),
            ):
                plot_subject_activation_and_adjacency.main()
            self.assertEqual(
                set(os.listdir(output_dir)),
                {
                    "P1_alpha_activation_change.png",
                    "P1_alpha_act_heatmap.png",
                    "P1_alpha_adj_compare.png",
                },
            )

    def test_logistic_full_pipeline_profiles_share_static_contract(self):
        config_dir = Path(__file__).resolve().parent / "dataset_configs"
        profiles = (
            "tdbrain_coherence_static_signed_no_rejection_logistic.toml",
            "first_paper_coherence_static_signed_no_rejection_logistic.toml",
        )
        output_directories = []
        for profile in profiles:
            with (config_dir / profile).open("rb") as stream:
                settings = tomllib.load(stream)
            optimization = settings["optimization"]
            self.assertEqual(settings["classification"]["models"], ["logistic_l2"])
            self.assertEqual(optimization["stimulation_model"], "static_adjacency")
            self.assertEqual(optimization["static_edge_scope"], "incident")
            self.assertEqual(
                optimization["stimulation_total_change_bounds"], [-3.0, 3.0]
            )
            self.assertEqual(optimization["patient_rejection_percent"], 0)
            self.assertNotIn("analysis_input_directory", optimization)
            output_directories.append(settings["output_directory"])
        self.assertTrue(
            all(
                path.startswith("./results-static-signed-no-rejection-logistic/")
                for path in output_directories
            )
        )

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
