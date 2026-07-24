"""Tests for the band classifier probability optimization objective."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression

from classification_score.band_connectivity_classifier import BandConnectivityClassifier
from eeg_optimization import EEGOptimizer
from state_space_simulation import (
    normalize_adjacency_matrix,
    run_full_simulation,
    simulate_eeg_final_state,
)


def _bundle() -> BandConnectivityClassifier:
    X = np.array([[0.1, 0.15, 0.2], [0.7, 0.75, 0.8]])
    estimator = LogisticRegression().fit(X, [0, 1])
    return BandConnectivityClassifier(
        estimator=estimator,
        band="alpha",
        method="coh",
        channel_names=["A", "B", "C"],
        model_name="logistic_l2",
        n_features=3,
        best_params={},
        cv_metrics={"roc_auc": 0.8},
        accepted_for_optimization=True,
        acceptance_reasons=[],
        feature_mean=np.mean(X, axis=0),
        feature_scale=np.std(X, axis=0, ddof=1),
        ood_rms_threshold=100.0,
        training_min=0.1,
        training_max=0.8,
    )


def _optimizer(bundle: BandConnectivityClassifier) -> EEGOptimizer:
    matrix = np.array([[0.0, 0.7, 0.75], [0.7, 0.0, 0.8], [0.75, 0.8, 0.0]])
    connectivity = {
        "Healthy": {},
        "Patient": {"P1": {"coh": {"alpha": matrix}}},
    }
    return EEGOptimizer(
        connectivity_matrices=connectivity,
        network_measures={"Healthy": {}, "Patient": {"P1": {}}},
        subject_data={"P1": {"baseline_activation": np.array([0.5, 0.6, 0.7])}},
        frequency_bands={"delta": (1, 4), "alpha": (8, 13), "beta": (13, 30)},
        channel_names=["A", "B", "C"],
        selected_method="coh",
        optimization_measures=[],
        fixed_band_name="alpha",
        objective_mode="classifier_patient_probability",
        classifier_bundle=bundle,
        stimulation_model="state_space",
    )


class ClassifierProbabilityOptimizationTests(unittest.TestCase):
    def test_fast_final_state_matches_legacy_trajectory(self):
        matrix = np.array([[0.0, 0.2, 0.1], [0.2, 0.0, 0.3], [0.1, 0.3, 0.0]])
        baseline = np.array([0.4, 0.6, 0.8])
        kwargs = dict(
            adjacency_matrix=matrix,
            baseline_activation=baseline,
            stimulation_node=1,
            stimulation_duration=2.0,
            stimulation_amplitude=-0.35,
            dt=0.01,
            stability_constant=0.01,
            leak=0.7,
        )
        legacy = run_full_simulation(**kwargs, return_trajectory=True)
        fast = run_full_simulation(**kwargs, return_trajectory=False)
        np.testing.assert_allclose(fast["final_state"], legacy["final_state"], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            fast["raw_activation_ratios"], legacy["raw_activation_ratios"], rtol=1e-10, atol=1e-10
        )
        self.assertIsNone(fast["trajectory"])

    def test_objective_is_patient_probability_and_preserves_natural_scale(self):
        optimizer = _optimizer(_bundle())
        original = optimizer.connectivity_matrices["Patient"]["P1"]["coh"]["alpha"]
        with patch("eeg_optimization.compute_plasticity_effect", return_value=original) as update:
            objectives, values, details = optimizer._evaluate_solution_details(
                "P1", np.array([0.5, 0.6, 0.7]), 0, 1, 1.0, 0.1, 1.0
            )
        self.assertEqual(objectives.shape, (1,))
        np.testing.assert_allclose(objectives, values)
        self.assertEqual(details["constraint_values"].shape, (7,))
        self.assertAlmostEqual(details["healthy_probability"], 1.0 - objectives[0])
        self.assertFalse(update.call_args.kwargs["normalize"])

    def test_classifier_band_must_match_fixed_band(self):
        bundle = _bundle()
        bundle.band = "beta"
        with self.assertRaisesRegex(ValueError, "does not match"):
            _optimizer(bundle)

    def test_static_model_bypasses_dynamics_and_uses_five_classifier_constraints(self):
        optimizer = _optimizer(_bundle())
        optimizer.stimulation_model = "static_adjacency"
        with patch("eeg_optimization.run_full_simulation") as simulation:
            objectives, values, details = optimizer._evaluate_solution_details(
                "P1", np.ones(3), 0, 1, None, -0.2, None
            )
        simulation.assert_not_called()
        np.testing.assert_allclose(objectives, values)
        self.assertEqual(details["constraint_values"].shape, (5,))
        self.assertEqual(details["stimulation_total_change"], -0.2)
        self.assertIsNone(details["raw_activation_ratio_min"])


if __name__ == "__main__":
    unittest.main()
