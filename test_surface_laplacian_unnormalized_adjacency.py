"""Focused tests for Laplacian preprocessing and raw adjacency propagation."""

from __future__ import annotations

from pathlib import Path
import tomllib
import unittest

import mne
import numpy as np

from signal_processing import apply_surface_laplacian
from stimulation_models import (
    prepare_adjacency_propagation_matrix,
    run_adjacency_activation_log_gain_stimulation,
)


class SurfaceLaplacianUnnormalizedAdjacencyTests(unittest.TestCase):
    def test_surface_laplacian_preserves_shape_order_and_finite_values(self) -> None:
        channel_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
        info = mne.create_info(channel_names, sfreq=100.0, ch_types="eeg")
        raw = mne.io.RawArray(
            np.random.default_rng(42).normal(size=(len(channel_names), 500)) * 1e-6,
            info,
            verbose=False,
        )
        original = raw.get_data().copy()

        transformed = apply_surface_laplacian(
            raw,
            montage="standard_1005",
            copy=True,
        )

        self.assertEqual(transformed.ch_names, channel_names)
        self.assertEqual(transformed.get_data().shape, original.shape)
        self.assertTrue(np.all(np.isfinite(transformed.get_data())))
        self.assertFalse(np.allclose(transformed.get_data(), original, rtol=1e-6, atol=0.0))
        np.testing.assert_array_equal(raw.get_data(), original)

    def test_none_mode_preserves_off_diagonal_connectivity_scale(self) -> None:
        matrix = np.array(
            [
                [2.0, 0.4, 0.2],
                [0.4, 3.0, 0.3],
                [0.2, 0.3, 4.0],
            ]
        )
        propagation = prepare_adjacency_propagation_matrix(
            matrix,
            normalization="none",
        )
        expected = matrix.copy()
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_array_equal(propagation, expected)

        result = run_adjacency_activation_log_gain_stimulation(
            matrix,
            np.ones(3),
            stimulation_node=0,
            log_gain=np.log(2.0),
            adjacency_normalization="none",
        )
        spatial_profile = np.array([1.0, 0.4, 0.2])
        np.testing.assert_allclose(result["log_gain_spatial_profile"], spatial_profile)
        np.testing.assert_allclose(
            result["activation_ratios"],
            np.exp(np.log(2.0) * spatial_profile),
        )
        self.assertEqual(result["adjacency_propagation_normalization"], "none")

    def test_legacy_default_remains_spectral_radius_normalization(self) -> None:
        matrix = np.array([[0.0, 0.4], [0.4, 0.0]])
        default = prepare_adjacency_propagation_matrix(matrix)
        expected = matrix / (0.4 + 0.01)
        np.testing.assert_allclose(default, expected)

    def test_new_profile_enables_both_experiment_changes(self) -> None:
        profile_path = (
            Path(__file__).resolve().parent
            / "dataset_configs"
            / "tdbrain_coherence_surface_laplacian_adjacency_activation_log_gain_unnormalized_signed_no_rejection_logistic.toml"
        )
        with profile_path.open("rb") as stream:
            profile = tomllib.load(stream)

        self.assertTrue(profile["preprocessing"]["surface_laplacian"]["enabled"])
        self.assertEqual(
            profile["preprocessing"]["surface_laplacian"]["montage"],
            "standard_1005",
        )
        self.assertEqual(
            profile["optimization"]["adjacency_propagation_normalization"],
            "none",
        )
        self.assertEqual(profile["connectivity"]["normalization"], "none")
        self.assertEqual(
            Path(profile["output_directory"]),
            Path(profile["optimization"]["analysis_input_directory"]),
        )


if __name__ == "__main__":
    unittest.main()
