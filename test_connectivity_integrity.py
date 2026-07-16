"""Focused tests for connectivity-v2 production integrity behavior."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import connectivity
from signal_processing import process_subject_epochs


class ConnectivityNormalizationTests(unittest.TestCase):
    def test_none_preserves_absolute_scale(self) -> None:
        matrix = np.array([[0.0, 0.2], [0.4, 0.0]])
        np.testing.assert_array_equal(
            connectivity.normalize_connectivity_matrix(matrix, mode="none"), matrix
        )

    def test_minmax_discards_scalar_strength(self) -> None:
        matrix = np.array([[0.0, 0.2], [0.5, 0.0]])
        first = connectivity.normalize_connectivity_matrix(matrix, mode="minmax")
        second = connectivity.normalize_connectivity_matrix(10 * matrix, mode="minmax")
        np.testing.assert_allclose(first, second)

    def test_nonfinite_matrix_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            connectivity.normalize_connectivity_matrix(
                np.array([[0.0, np.nan], [0.2, 0.0]]), mode="none"
            )


class ConnectivityDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(3)
        self.filtered = {
            "delta": rng.normal(size=(3, 2, 128)),
            "alpha": rng.normal(size=(3, 2, 128)),
            "beta": rng.normal(size=(3, 2, 128)),
        }

    def test_broadband_profile_requires_broadband_epochs(self) -> None:
        with patch.object(connectivity, "SPECTRAL_CONNECTIVITY_INPUT", "broadband"):
            with self.assertRaisesRegex(ValueError, "Broadband epochs are required"):
                connectivity.compute_connectivity_for_band(
                    self.filtered, "alpha", 100.0, "plv"
                )

    def test_error_policy_raise_does_not_create_zero_scientific_data(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "method=not_a_method"):
            connectivity.compute_all_connectivity(
                self.filtered,
                100.0,
                methods=["not_a_method"],
                normalization="none",
                error_policy="raise",
            )

    def test_legacy_zero_policy_is_explicit(self) -> None:
        result = connectivity.compute_all_connectivity(
            self.filtered,
            100.0,
            methods=["not_a_method"],
            normalization="none",
            error_policy="zeros",
        )
        self.assertTrue(np.all(result["not_a_method"]["alpha"] == 0))


class SignalProcessingTests(unittest.TestCase):
    def test_can_persist_unfiltered_epochs_for_spectral_estimators(self) -> None:
        rng = np.random.default_rng(5)
        data = rng.normal(size=(2, 2000))
        filtered, broadband = process_subject_epochs(
            data, 100.0, return_broadband=True
        )
        self.assertEqual(broadband.shape, (2, 2, 1000))
        self.assertEqual(set(filtered), {"delta", "alpha", "beta"})
        self.assertFalse(np.allclose(filtered["alpha"], broadband))


if __name__ == "__main__":
    unittest.main()
