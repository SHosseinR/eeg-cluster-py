import unittest

import numpy as np
import pandas as pd

from generate_tdbrain_log_gain_paper_results import (
    AgeResidualizer,
    baseline_pca_projection,
    target_concentration_statistics,
)


class AgeResidualizerTests(unittest.TestCase):
    def test_training_age_slope_is_removed_and_age_column_dropped(self):
        age = np.arange(20.0, 30.0)
        edges = np.column_stack([2.0 * age + 3.0, -0.5 * age + 7.0])
        transformed = AgeResidualizer().fit_transform(np.column_stack([edges, age]))
        self.assertEqual(transformed.shape, edges.shape)
        for column in range(transformed.shape[1]):
            centered = transformed[:, column] - transformed[:, column].mean()
            self.assertAlmostEqual(float(centered @ (age - age.mean())), 0.0, places=6)


class TargetConcentrationTests(unittest.TestCase):
    def test_concentrated_targets_are_more_extreme_than_uniform_null(self):
        channels = ["A", "B", "C", "D"]
        rows = []
        for band in ("delta", "alpha", "beta"):
            for index in range(40):
                rows.append({"subject_id": f"s{index}", "band": band, "target_label": "A"})
        stats, counts = target_concentration_statistics(
            pd.DataFrame(rows), channels, simulations=2_000, bootstraps=500
        )
        self.assertTrue((stats["top_fraction"] == 1.0).all())
        self.assertTrue((stats["normalized_entropy"] == 0.0).all())
        self.assertTrue((stats["uniform_null_p_max"] < 0.01).all())
        self.assertEqual(int(counts.loc[counts["channel"] == "A", "count"].sum()), 120)


class ProjectionTests(unittest.TestCase):
    def test_candidate_projection_reuses_baseline_scaler_and_pca(self):
        rng = np.random.default_rng(7)
        baseline = rng.normal(size=(20, 6))
        candidate = baseline[:4] + 0.2
        baseline_xy, candidate_xy, explained = baseline_pca_projection(
            baseline, candidate
        )
        self.assertEqual(baseline_xy.shape, (20, 2))
        self.assertEqual(candidate_xy.shape, (4, 2))
        self.assertEqual(explained.shape, (2,))
        self.assertTrue(np.isfinite(candidate_xy).all())
        self.assertGreater(float(explained.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
