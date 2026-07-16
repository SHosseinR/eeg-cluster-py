"""Synthetic ground-truth tests for standalone connectivity estimators."""

from __future__ import annotations

import unittest

import numpy as np

from classification_score.connectivity_methods import (
    compute_conditional_var_connectivity,
    compute_envelope_connectivity,
    compute_fourier_connectivity,
    split_half_reliability,
)
from classification_score.selected_connectivity_model import (
    vectorize_connectivity_matrices,
)


class FourierConnectivityTests(unittest.TestCase):
    def test_zero_lag_copy_is_rejected_by_lag_resistant_measures(self) -> None:
        rng = np.random.default_rng(11)
        epochs, samples, fs = 16, 500, 100.0
        time = np.arange(samples) / fs
        data = np.empty((epochs, 3, samples))
        for epoch in range(epochs):
            phase = rng.uniform(0, 2 * np.pi)
            shared = np.sin(2 * np.pi * 10 * time + phase) + rng.normal(size=samples)
            data[epoch, 0] = shared + 0.01 * rng.normal(size=samples)
            data[epoch, 1] = shared + 0.01 * rng.normal(size=samples)
            data[epoch, 2] = rng.normal(size=samples)

        matrices = compute_fourier_connectivity(data, fs, 8.0, 13.0)

        self.assertGreater(matrices["coherence"][0, 1], 0.95)
        self.assertGreater(matrices["plv"][0, 1], 0.95)
        self.assertLess(matrices["imaginary_coherence"][0, 1], 0.10)
        self.assertLess(matrices["wpli2_debiased"][0, 1], 0.15)

    def test_delayed_oscillator_has_correct_phase_lead_direction(self) -> None:
        rng = np.random.default_rng(17)
        epochs, samples, fs = 20, 600, 100.0
        time = np.arange(samples) / fs
        data = np.empty((epochs, 3, samples))
        for epoch in range(epochs):
            phase = rng.uniform(0, 2 * np.pi)
            source = np.sin(2 * np.pi * 10 * time + phase)
            data[epoch, 0] = source + 0.15 * rng.normal(size=samples)
            data[epoch, 1] = np.roll(source, 2) + 0.15 * rng.normal(size=samples)
            data[epoch, 2] = rng.normal(size=samples)

        matrix = compute_fourier_connectivity(
            data, fs, 8.0, 13.0, methods=("directed_wpli",)
        )["directed_wpli"]

        self.assertGreater(matrix[0, 1], 0.50)
        self.assertEqual(matrix[1, 0], 0.0)
        self.assertGreater(matrix[0, 1], matrix[0, 2] + 0.30)


class ConditionalVarTests(unittest.TestCase):
    def test_recovers_planted_conditional_var_chain(self) -> None:
        rng = np.random.default_rng(23)
        epochs, samples = 18, 500
        data = np.zeros((epochs, 3, samples))
        for epoch in range(epochs):
            innovations = rng.normal(scale=0.5, size=(3, samples))
            for sample in range(2, samples):
                data[epoch, 0, sample] = (
                    0.70 * data[epoch, 0, sample - 1] + innovations[0, sample]
                )
                data[epoch, 1, sample] = (
                    0.55 * data[epoch, 1, sample - 1]
                    + 0.50 * data[epoch, 0, sample - 1]
                    + innovations[1, sample]
                )
                data[epoch, 2, sample] = (
                    0.50 * data[epoch, 2, sample - 1]
                    + 0.45 * data[epoch, 1, sample - 1]
                    + innovations[2, sample]
                )

        matrices, diagnostics = compute_conditional_var_connectivity(
            data,
            fs=100.0,
            fmin=1.0,
            fmax=30.0,
            target_sfreq=100.0,
            lag_ms=20.0,
            ridge_alpha=1.0,
        )

        self.assertEqual(diagnostics.order, 2)
        self.assertLess(abs(diagnostics.residual_lag1_correlation), 0.05)
        for method in ("conditional_var_wald", "pdc", "dtf"):
            matrix = matrices[method]
            self.assertGreater(matrix[0, 1], matrix[1, 0] * 3)
            self.assertGreater(matrix[1, 2], matrix[2, 1] * 3)
            self.assertEqual(float(np.trace(matrix)), 0.0)


class UtilityTests(unittest.TestCase):
    def test_envelope_outputs_are_symmetric_and_finite(self) -> None:
        rng = np.random.default_rng(29)
        epochs = rng.normal(size=(4, 4, 300))
        for matrix in compute_envelope_connectivity(epochs).values():
            np.testing.assert_allclose(matrix, matrix.T)
            self.assertTrue(np.all(np.isfinite(matrix)))
            self.assertEqual(float(np.trace(matrix)), 0.0)

    def test_identical_split_halves_have_perfect_reliability(self) -> None:
        matrix = np.array([[0.0, 0.2, 0.4], [0.2, 0.0, 0.8], [0.4, 0.8, 0.0]])
        result = split_half_reliability(matrix, matrix, directed=False)
        self.assertEqual(result["edge_spearman"], 1.0)
        self.assertEqual(result["edge_pearson"], 1.0)
        self.assertEqual(result["normalized_edge_error"], 0.0)
        self.assertEqual(result["top10pct_jaccard"], 1.0)

    def test_selected_model_vectorization_preserves_band_and_edge_contract(self) -> None:
        matrices = {
            "delta": np.array([[0.0, 0.1, 0.2], [0.1, 0.0, 0.3], [0.2, 0.3, 0.0]]),
            "alpha": np.array([[0.0, 0.4, 0.5], [0.4, 0.0, 0.6], [0.5, 0.6, 0.0]]),
            "beta": np.array([[0.0, 0.7, 0.8], [0.7, 0.0, 0.9], [0.8, 0.9, 0.0]]),
        }
        natural = vectorize_connectivity_matrices(
            matrices, directed=False, transformation="natural_edges"
        )
        np.testing.assert_allclose(
            natural, [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
        )
        centered = vectorize_connectivity_matrices(
            matrices, directed=False, transformation="within_subject_centered"
        ).reshape(3, 3)
        np.testing.assert_allclose(np.mean(centered, axis=1), 0.0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
