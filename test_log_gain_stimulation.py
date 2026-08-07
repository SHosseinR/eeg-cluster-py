"""Focused tests for TD-BRAIN multiplicative log-gain stimulation."""

from __future__ import annotations

from pathlib import Path
import tomllib
import unittest

import numpy as np

from nsga_optimizer import EEGOptimizationProblem
from plasticity import (
    apply_log_gain_plasticity_updates,
    apply_plasticity_updates,
)
from state_space_simulation import normalize_adjacency_matrix
from stimulation_models import (
    DYNAMICS_FREE_STIMULATION_MODELS,
    compute_band_rms,
    run_adjacency_activation_log_gain_stimulation,
    run_adjacency_activation_stimulation,
)


class LogGainStimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = np.array(
            [
                [0.0, 0.4, 0.2],
                [0.4, 0.0, 0.3],
                [0.2, 0.3, 0.0],
            ],
            dtype=float,
        )
        self.baseline = np.array([2.0, 3.0, 4.0], dtype=float)

    def test_band_rms_uses_all_epochs_and_samples_per_channel(self) -> None:
        epochs = np.array(
            [
                [[3.0, 4.0], [0.0, 0.0]],
                [[0.0, 0.0], [6.0, 8.0]],
            ]
        )
        expected = np.sqrt(
            [
                (3.0**2 + 4.0**2) / 4.0,
                (6.0**2 + 8.0**2) / 4.0,
            ]
        )
        np.testing.assert_allclose(compute_band_rms(epochs), expected)
        np.testing.assert_allclose(
            compute_band_rms(np.array([[3.0, 4.0], [6.0, 8.0]])),
            [np.sqrt(12.5), np.sqrt(50.0)],
        )

    def test_zero_gain_is_identity_for_activation_and_connectivity(self) -> None:
        result = run_adjacency_activation_log_gain_stimulation(
            self.matrix,
            self.baseline,
            stimulation_node=1,
            log_gain=0.0,
        )
        np.testing.assert_allclose(result["activation_ratios"], 1.0)
        np.testing.assert_allclose(result["final_state"], self.baseline)
        np.testing.assert_allclose(result["activation_change"], 0.0)
        updated = apply_log_gain_plasticity_updates(
            self.matrix,
            result["activation_ratios"],
        )
        np.testing.assert_allclose(updated, self.matrix)

    def test_positive_and_negative_gain_use_exact_exponential_ratio(self) -> None:
        gain = np.log(2.0)
        positive = run_adjacency_activation_log_gain_stimulation(
            self.matrix,
            self.baseline,
            stimulation_node=0,
            log_gain=gain,
            neighbor_scale=1.0,
        )
        negative = run_adjacency_activation_log_gain_stimulation(
            self.matrix,
            self.baseline,
            stimulation_node=0,
            log_gain=-gain,
            neighbor_scale=1.0,
        )
        normalized = normalize_adjacency_matrix(self.matrix, 0.01)
        unit = np.array([1.0, 0.0, 0.0])
        spatial_profile = unit + normalized @ unit
        expected = np.exp(gain * spatial_profile)
        np.testing.assert_allclose(positive["log_gain_spatial_profile"], spatial_profile)
        np.testing.assert_allclose(positive["activation_ratios"], expected)
        np.testing.assert_allclose(positive["final_state"], self.baseline * expected)
        np.testing.assert_allclose(
            negative["activation_ratios"],
            1.0 / expected,
        )
        self.assertAlmostEqual(positive["activation_ratios"][0], 2.0)
        self.assertGreater(positive["final_state"][1], self.baseline[1])
        self.assertLess(negative["final_state"][1], self.baseline[1])

    def test_plasticity_fraction_zero_and_one_match_closed_form(self) -> None:
        ratios = np.array([2.0, 0.5, 1.5])
        exponent = 1.25
        target = self.matrix * np.power(np.outer(ratios, ratios), exponent)
        fraction_zero = apply_log_gain_plasticity_updates(
            self.matrix,
            ratios,
            plasticity_exponent=exponent,
            plasticity_fraction=0.0,
        )
        fraction_one = apply_log_gain_plasticity_updates(
            self.matrix,
            ratios,
            plasticity_exponent=exponent,
            plasticity_fraction=1.0,
        )
        np.testing.assert_allclose(fraction_zero, self.matrix)
        np.testing.assert_allclose(fraction_one, target)

    def test_shape_symmetry_and_finite_values_are_preserved(self) -> None:
        stimulation = run_adjacency_activation_log_gain_stimulation(
            self.matrix,
            self.baseline,
            stimulation_node=2,
            log_gain=1.0,
        )
        updated = apply_log_gain_plasticity_updates(
            self.matrix,
            stimulation["activation_ratios"],
        )
        self.assertEqual(updated.shape, self.matrix.shape)
        self.assertTrue(np.all(np.isfinite(stimulation["activation_ratios"])))
        self.assertTrue(np.all(np.isfinite(stimulation["final_state"])))
        self.assertTrue(np.all(np.isfinite(updated)))
        np.testing.assert_allclose(updated, updated.T)

    def test_non_finite_inputs_and_outputs_fail_clearly(self) -> None:
        bad_epochs = np.ones((1, 3, 2))
        bad_epochs[0, 1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compute_band_rms(bad_epochs)
        with self.assertRaisesRegex(ValueError, "log_gain must be finite"):
            run_adjacency_activation_log_gain_stimulation(
                self.matrix,
                self.baseline,
                stimulation_node=0,
                log_gain=np.inf,
            )
        with self.assertRaisesRegex(ValueError, "non-finite activation ratios"):
            run_adjacency_activation_log_gain_stimulation(
                self.matrix,
                self.baseline,
                stimulation_node=0,
                log_gain=1e308,
            )
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            apply_log_gain_plasticity_updates(
                self.matrix,
                np.array([1.0, 0.0, 1.0]),
            )

    def test_log_gain_uses_dynamics_free_nsga_layout(self) -> None:
        seen = []

        def evaluate(node, band, duration, log_gain, leak):
            seen.append((node, band, duration, log_gain, leak))
            return np.array([log_gain**2])

        problem = EEGOptimizationProblem(
            n_nodes=3,
            n_bands=1,
            evaluate_func=evaluate,
            amplitude_bounds=(-2.302585093, 2.302585093),
            n_objectives=1,
            fixed_band_index=0,
            stimulation_model="adjacency_activation_log_gain",
        )
        self.assertEqual(problem.n_var, 2)
        output = {}
        problem._evaluate(np.array([[2.0, -0.5]]), output)
        self.assertEqual(seen, [(2, 0, None, -0.5, None)])
        np.testing.assert_allclose(output["F"], [[0.25]])
        self.assertIn(
            "adjacency_activation_log_gain",
            DYNAMICS_FREE_STIMULATION_MODELS,
        )

    def test_legacy_activation_and_plasticity_contract_is_unchanged(self) -> None:
        amount = 0.2
        neighbor_scale = 0.75
        result = run_adjacency_activation_stimulation(
            self.matrix,
            self.baseline,
            stimulation_node=0,
            stimulation_amount=amount,
            neighbor_scale=neighbor_scale,
            stability_constant=0.01,
        )
        normalized = normalize_adjacency_matrix(self.matrix, 0.01)
        expected_change = amount * (
            np.array([1.0, 0.0, 0.0])
            + neighbor_scale * normalized[:, 0]
        )
        np.testing.assert_allclose(result["activation_change"], expected_change)
        ratios = np.array([1.2, 0.8, 1.0])
        np.testing.assert_allclose(
            apply_plasticity_updates(self.matrix, ratios, scaling=1.0),
            self.matrix * np.outer(ratios, ratios),
        )

    def test_tdbrain_profile_is_isolated_and_reuses_cached_analysis(self) -> None:
        path = (
            Path(__file__).resolve().parent
            / "dataset_configs"
            / "tdbrain_coherence_adjacency_activation_log_gain_signed_no_rejection_logistic.toml"
        )
        with path.open("rb") as stream:
            profile = tomllib.load(stream)
        optimization = profile["optimization"]
        self.assertEqual(
            optimization["stimulation_model"],
            "adjacency_activation_log_gain",
        )
        self.assertEqual(
            optimization["log_gain_bounds"],
            [-2.302585093, 2.302585093],
        )
        self.assertEqual(optimization["neighbor_scale"], 1.0)
        self.assertEqual(optimization["plasticity_exponent"], 1.0)
        self.assertEqual(optimization["plasticity_fraction"], 1.0)
        self.assertEqual(optimization["patient_rejection_percent"], 0)
        self.assertEqual(
            profile["optimization_output_subdirectory"],
            "optimization-log-gain",
        )
        self.assertEqual(
            Path(profile["output_directory"]),
            Path(optimization["analysis_input_directory"]),
        )


if __name__ == "__main__":
    unittest.main()
