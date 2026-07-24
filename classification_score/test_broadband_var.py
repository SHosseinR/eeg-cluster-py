"""Synthetic ground-truth tests for broadband VAR/PDC/DTF."""

from __future__ import annotations

import unittest

import numpy as np

from classification_score.broadband_var import fit_subject_connectivity


class BroadbandVarTests(unittest.TestCase):
    def test_recovers_stable_whitened_directed_chain(self) -> None:
        rng = np.random.default_rng(31)
        epochs, channels, samples = 14, 3, 400
        data = np.zeros((epochs, channels, samples))
        for epoch in range(epochs):
            noise = rng.normal(scale=0.5, size=(channels, samples))
            for sample in range(2, samples):
                data[epoch, 0, sample] = (
                    0.70 * data[epoch, 0, sample - 1] + noise[0, sample]
                )
                data[epoch, 1, sample] = (
                    0.55 * data[epoch, 1, sample - 1]
                    + 0.50 * data[epoch, 0, sample - 1]
                    + noise[1, sample]
                )
                data[epoch, 2, sample] = (
                    0.50 * data[epoch, 2, sample - 1]
                    + 0.45 * data[epoch, 1, sample - 1]
                    + noise[2, sample]
                )

        connectivity, diagnostics = fit_subject_connectivity(
            data,
            100.0,
            lag_ms_candidates=(10.0, 20.0),
            ridge_candidates=(1.0, 10.0),
        )

        self.assertTrue(diagnostics["stable"])
        self.assertTrue(diagnostics["residual_whiteness_pass"])
        self.assertLess(diagnostics["residual_lag1_abs_mean"], 0.05)
        self.assertGreater(
            diagnostics["pdc_time_reversal_direction_agreement"], 0.5
        )
        self.assertGreater(diagnostics["pdc_time_reversal_spearman"], 0.0)
        self.assertGreater(diagnostics["dtf_time_reversal_spearman"], 0.0)
        self.assertTrue(
            all("full_spectral_radius" in row for row in diagnostics["selection_grid"])
        )
        for method in ("pdc", "dtf"):
            matrix = connectivity["full"][method]["beta"]
            self.assertGreater(matrix[0, 1], matrix[1, 0] * 5)
            self.assertGreater(matrix[1, 2], matrix[2, 1] * 5)
            self.assertEqual(float(np.trace(matrix)), 0.0)


if __name__ == "__main__":
    unittest.main()
