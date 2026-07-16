"""Focused synthetic tests for the standalone classification benchmark."""

from __future__ import annotations

import unittest

import numpy as np

from data_features import COMMON_CHANNELS, _connectivity_features, _subject_epoch_features, align_feature_matrix
from modeling import nested_oof_evaluate, probability_metrics


class FeatureTests(unittest.TestCase):
    def test_epoch_feature_shapes_and_finite_values(self) -> None:
        rng = np.random.default_rng(4)
        channels = list(COMMON_CHANNELS)
        epochs = {
            "delta": rng.normal(size=(4, len(channels), 64)),
            "alpha": rng.normal(size=(4, len(channels), 64)),
            "beta": rng.normal(size=(4, len(channels), 64)),
        }
        values, names = _subject_epoch_features(epochs, channels)
        self.assertEqual(values["spectral_channel"].shape, (285,))
        self.assertEqual(values["covariance_logcorr"].shape, (570,))
        self.assertEqual(values["covariance_common_logcorr"].shape, (570,))
        for key in values:
            self.assertEqual(values[key].size, len(names[key]))
            self.assertFalse(np.any(np.isinf(values[key])))

    def test_directed_connectivity_keeps_both_directions(self) -> None:
        matrix = np.array([[0.0, 1.0], [2.0, 0.0]])
        values, names = _connectivity_features({"gc": {"alpha": matrix}}, ["F3", "F4"])
        self.assertEqual(values["connectivity_edges"].tolist(), [1.0, 2.0])
        self.assertEqual(
            names["connectivity_edges"], ["edge_alpha_F3_to_F4", "edge_alpha_F4_to_F3"]
        )

    def test_feature_alignment(self) -> None:
        aligned = align_feature_matrix(np.array([[1.0, 2.0]]), ["b", "a"], ["a", "b"])
        np.testing.assert_array_equal(aligned, [[2.0, 1.0]])


class ModelingTests(unittest.TestCase):
    def test_nested_oof_predicts_each_subject_once(self) -> None:
        rng = np.random.default_rng(5)
        y = np.repeat([0, 1], 15)
        X = rng.normal(size=(30, 8))
        X[:, 0] += y * 1.2
        summary, predictions = nested_oof_evaluate(
            X,
            y,
            feature_set="synthetic",
            model_name="logistic_l2",
            mode="quick",
            outer_splits=3,
            repeats=1,
            inner_splits=2,
        )
        self.assertEqual(len(predictions), len(y))
        self.assertEqual(predictions["subject_id"].nunique(), len(y))
        self.assertGreater(summary["roc_auc"], 0.5)

    def test_probability_metrics_perfect(self) -> None:
        metrics = probability_metrics(np.array([0, 0, 1, 1]), np.array([0.01, 0.1, 0.9, 0.99]))
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
