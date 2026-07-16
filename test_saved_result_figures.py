import shutil
import unittest
from pathlib import Path
import uuid

import numpy as np

from figure_paths import organize_profile_figures
from plot_group_metric_space_3d import (
    extract_group_points, healthy_relative, select_kmeans,
)
from plot_modularity_reordered_connectivity import (
    best_louvain_partition, modularity_node_order,
)
from plot_top_selected_nodes import aggregate_node_selections
from plot_weighted_selection_target_3d import (
    _marker_radius, _marker_sizes, _validity_score,
    aggregate_validity_weighted_scores,
)
from saved_results_utils import SavedDatasetProfile


class ModularityOrderingTests(unittest.TestCase):
    def test_partition_and_order_are_deterministic_and_contiguous(self):
        matrix = np.array([
            [0, 8, .1, .1], [7, 0, .1, .1],
            [.1, .1, 0, 9], [.1, .1, 8, 0],
        ], dtype=float)
        first_ci, first_q = best_louvain_partition(matrix, n_restarts=12)
        second_ci, second_q = best_louvain_partition(matrix, n_restarts=12)
        np.testing.assert_array_equal(first_ci, second_ci)
        self.assertAlmostEqual(first_q, second_q)
        order, normalized = modularity_node_order(matrix, first_ci)
        ordered_modules = normalized[order]
        for module in np.unique(ordered_modules):
            locations = np.flatnonzero(ordered_modules == module)
            np.testing.assert_array_equal(locations, np.arange(locations[0], locations[-1] + 1))
        self.assertEqual(sorted(order.tolist()), [0, 1, 2, 3])


class MetricSpaceTests(unittest.TestCase):
    def test_healthy_relative_and_nonfinite_filtering(self):
        target = [2.0, 4.0, 0.0]
        np.testing.assert_allclose(
            healthy_relative(np.array([[3.0, 2.0, 1.0]]), target),
            [[0.5, -0.5, 1.0]],
        )
        network = {
            "Healthy": {
                "good": {"gc": {"alpha": {"a": 3, "b": 2, "c": 1}}},
                "bad": {"gc": {"alpha": {"a": np.nan, "b": 2, "c": 1}}},
            }
        }
        points, ids, skipped = extract_group_points(
            network, "Healthy", "gc", "alpha", ["a", "b", "c"], target,
        )
        self.assertEqual(ids, ["good"])
        self.assertEqual(skipped, 1)
        np.testing.assert_allclose(points, [[0.5, -0.5, 1.0]])

    def test_kmeans_selection_is_deterministic(self):
        rng = np.random.default_rng(3)
        points = np.vstack([rng.normal(-3, .15, (12, 3)), rng.normal(3, .15, (12, 3))])
        labels_a, k_a, scores_a = select_kmeans(points)
        labels_b, k_b, scores_b = select_kmeans(points)
        self.assertEqual(k_a, 2)
        self.assertEqual(k_a, k_b)
        np.testing.assert_array_equal(labels_a, labels_b)
        np.testing.assert_allclose(scores_a["silhouette_score"], scores_b["silhouette_score"])


class WeightedValidityTests(unittest.TestCase):
    def test_radius_uses_only_clipped_median_improvement(self):
        self.assertEqual(_validity_score(-0.5), 0.0)
        self.assertEqual(_validity_score(1.8), 1.0)
        self.assertLess(_marker_radius(0.0), _marker_radius(0.8))
        sizes = _marker_sizes(4, 0.5)
        self.assertEqual(len(np.unique(sizes)), 1)

    def test_combined_score_applies_each_band_factor_before_summing(self):
        band_scores = {
            "delta": np.array([1.0, 2.0]),
            "alpha": np.array([4.0, 1.0]),
            "beta": np.array([10.0, 10.0]),
        }
        validity = {
            "delta": {"median": 0.5, "n_subjects": 4},
            "alpha": {"median": 1.2, "n_subjects": 3},
            "beta": {"median": -0.3, "n_subjects": 2},
        }
        raw, weighted, factors = aggregate_validity_weighted_scores(
            band_scores, validity,
        )
        np.testing.assert_allclose(raw, [15.0, 13.0])
        np.testing.assert_allclose(weighted, [4.5, 2.0])
        self.assertEqual(factors, {"delta": 0.5, "alpha": 1.0, "beta": 0.0})


class TopSelectionTests(unittest.TestCase):
    def test_hard_and_rank_weighted_scores(self):
        results = {
            "s1": {
                "channel_names": ["A", "B", "C"],
                "best_solution": {"node": 1},
                "top_solutions": [
                    {"node": 1, "rank": 1, "strength": 1.0},
                    {"node": 2, "rank": 2},
                ],
            },
            "s2": {
                "channel_names": ["A", "B", "C"],
                "best_solution": {"node": 2},
                "top_solutions": [{"node": 2, "rank": 1}],
            },
        }
        aggregate = aggregate_node_selections(results)
        np.testing.assert_allclose(aggregate["hard"], [0, 1, 1])
        np.testing.assert_allclose(aggregate["weighted"], [0, 1, 1.5])
        self.assertEqual(aggregate["n_units"], 2)


class OrganizerTests(unittest.TestCase):
    def _temporary_root(self) -> Path:
        root = Path(__file__).resolve().parent / "results-comparison" / f"test-{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def _profile(self, root: Path) -> SavedDatasetProfile:
        output = root / "results-test"
        return SavedDatasetProfile(
            config_path=root / "test.toml", config_name="test.toml", label="Test",
            output_dir=output, optimization_dir=output / "optimization-run",
        )

    def test_organizer_is_idempotent_and_writes_manifests(self):
        profile = self._profile(self._temporary_root())
        profile.main_figures_dir.mkdir(parents=True)
        profile.optimization_figures_dir.mkdir(parents=True)
        (profile.main_figures_dir / "viz1_connectivity_matrices.png").write_bytes(b"main")
        (profile.optimization_figures_dir / "metric_shift_3d_alpha.png").write_bytes(b"shift")
        (profile.optimization_figures_dir / "unclassified.png").write_bytes(b"misc")
        first = organize_profile_figures(profile, ["alpha"])
        second = organize_profile_figures(profile, ["alpha"])
        self.assertEqual(first["moved"], 3)
        self.assertEqual(second["moved"], 0)
        self.assertTrue((profile.main_figures_dir / "connectivity" / "viz1_connectivity_matrices.png").exists())
        self.assertTrue((profile.optimization_figures_dir / "metric_space" / "shifts" / "metric_shift_3d_alpha.png").exists())
        self.assertTrue((profile.optimization_figures_dir / "misc" / "unclassified.png").exists())
        self.assertTrue((profile.optimization_figures_dir / "figure_manifest.csv").exists())

    def test_collision_is_reported(self):
        profile = self._profile(self._temporary_root())
        profile.main_figures_dir.mkdir(parents=True)
        destination = profile.main_figures_dir / "connectivity" / "viz1_a.png"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"old")
        (profile.main_figures_dir / "viz1_a.png").write_bytes(b"new")
        with self.assertRaises(FileExistsError):
            organize_profile_figures(profile, ["alpha"])


if __name__ == "__main__":
    unittest.main()
