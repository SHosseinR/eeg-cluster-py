"""Fast contract tests for optional PyTorch graph classifiers."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone

from classification_score.neural_graph_models import (
    TorchGraphClassifier,
    edge_vectors_to_symmetric_matrices,
    infer_undirected_node_count,
)


class NeuralGraphModelTests(unittest.TestCase):
    @staticmethod
    def _synthetic_edges(n_samples: int = 36, n_nodes: int = 5):
        rng = np.random.default_rng(12)
        y = np.tile([0, 1], n_samples // 2)
        matrices = rng.uniform(0.05, 0.25, size=(n_samples, n_nodes, n_nodes))
        matrices = (matrices + matrices.transpose(0, 2, 1)) / 2.0
        diagonal = np.arange(n_nodes)
        matrices[:, diagonal, diagonal] = 0.0
        matrices[y == 1, 0, 1] += 0.55
        matrices[y == 1, 1, 0] += 0.55
        upper = np.triu_indices(n_nodes, k=1)
        return matrices[:, upper[0], upper[1]], y

    def test_triangle_round_trip_contract(self):
        X, _ = self._synthetic_edges()
        self.assertEqual(infer_undirected_node_count(X.shape[1]), 5)
        matrices = edge_vectors_to_symmetric_matrices(X)
        np.testing.assert_allclose(matrices, matrices.transpose(0, 2, 1))
        np.testing.assert_allclose(np.diagonal(matrices, axis1=1, axis2=2), 0.0)

    def test_single_band_models_fit_predict_and_clone(self):
        X, y = self._synthetic_edges()
        for model_name in ("gcn", "brainnetcnn"):
            estimator = TorchGraphClassifier(
                model_name=model_name,
                hidden_dim=8,
                max_epochs=30,
                patience=6,
                batch_size=12,
                random_state=3,
            )
            fitted = clone(estimator).fit(X, y)
            probability = fitted.predict_proba(X[:4])
            self.assertEqual(probability.shape, (4, 2))
            np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-6)
            self.assertTrue(np.all(np.isfinite(probability)))
            with tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / f"{model_name}.joblib"
                joblib.dump(fitted, path)
                restored = joblib.load(path)
                np.testing.assert_allclose(
                    restored.predict_proba(X[:4]), probability, atol=1e-7
                )

    def test_three_band_fusion_contract(self):
        X, y = self._synthetic_edges()
        fused = np.concatenate([X, X * 0.8, X * 0.6], axis=1)
        estimator = TorchGraphClassifier(
            model_name="gcn",
            n_bands=3,
            hidden_dim=8,
            max_epochs=20,
            patience=5,
            batch_size=12,
            random_state=4,
        ).fit(fused, y)
        self.assertEqual(estimator.n_bands_, 3)
        self.assertEqual(estimator.predict_proba(fused[:3]).shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
