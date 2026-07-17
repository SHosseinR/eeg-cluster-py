"""Focused tests for the one-band connectivity classifier contract."""

from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from classification_score.band_connectivity_classifier import (
    BandConnectivityClassifier,
    matrix_ood_rms,
    predict_patient_probability,
    select_band_models,
    vectorize_band_matrix,
)


class BandConnectivityClassifierTests(unittest.TestCase):
    def test_selection_prefers_calibration_within_auc_tolerance(self):
        summary = pd.DataFrame([
            {"band": "alpha", "model": "linear_svm_sigmoid", "roc_auc": 0.82, "brier": 0.20, "ece_10": 0.14},
            {"band": "alpha", "model": "logistic_l2", "roc_auc": 0.805, "brier": 0.17, "ece_10": 0.07},
            {"band": "alpha", "model": "extra_trees", "roc_auc": 0.79, "brier": 0.15, "ece_10": 0.05},
        ])
        self.assertEqual(select_band_models(summary), {"alpha": "logistic_l2"})

    def test_undirected_vector_has_one_edge_per_pair(self):
        matrix = np.array([[0.0, 0.2, 0.3], [0.2, 0.0, 0.4], [0.3, 0.4, 0.0]])
        np.testing.assert_allclose(vectorize_band_matrix(matrix), [[0.2, 0.3, 0.4]])

    def test_rejects_asymmetric_matrix(self):
        with self.assertRaisesRegex(ValueError, "symmetric"):
            vectorize_band_matrix(np.array([[0.0, 0.2], [0.1, 0.0]]))

    def test_prediction_and_ood_use_only_one_matrix(self):
        X = np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])
        estimator = LogisticRegression().fit(X, [0, 1])
        bundle = BandConnectivityClassifier(
            estimator=estimator,
            band="alpha",
            method="coh",
            channel_names=["A", "B", "C"],
            model_name="logistic_l2",
            n_features=3,
            best_params={},
            cv_metrics={},
            accepted_for_optimization=True,
            acceptance_reasons=[],
            feature_mean=np.mean(X, axis=0),
            feature_scale=np.std(X, axis=0, ddof=1),
            ood_rms_threshold=2.0,
            training_min=0.1,
            training_max=0.9,
        )
        matrix = np.array([[0.0, 0.7, 0.8], [0.7, 0.0, 0.9], [0.8, 0.9, 0.0]])
        probability = predict_patient_probability(bundle, matrix, channel_names=["A", "B", "C"])
        self.assertGreater(probability, 0.5)
        self.assertGreaterEqual(matrix_ood_rms(bundle, matrix), 0.0)
        with self.assertRaisesRegex(ValueError, "Channel order"):
            predict_patient_probability(bundle, matrix, channel_names=["B", "A", "C"])

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bundle.joblib"
            joblib.dump(bundle, path)
            loaded = joblib.load(path)
            self.assertEqual(loaded.band, "alpha")


if __name__ == "__main__":
    unittest.main()
