"""Scikit-learn-compatible neural classifiers for connectivity graphs.

PyTorch is imported only when a neural model is fitted, so the established
classical classifier pipeline remains usable without the optional dependency.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def infer_undirected_node_count(n_edges: int) -> int:
    """Return ``n`` for an upper triangle containing ``n * (n - 1) / 2`` edges."""

    n_edges = int(n_edges)
    n_nodes = int(round((1.0 + math.sqrt(1.0 + 8.0 * n_edges)) / 2.0))
    if n_nodes * (n_nodes - 1) // 2 != n_edges:
        raise ValueError(
            f"{n_edges} features are not an undirected upper triangle"
        )
    return n_nodes


def edge_vectors_to_symmetric_matrices(
    X: np.ndarray, n_nodes: int | None = None
) -> np.ndarray:
    """Rebuild zero-diagonal symmetric matrices from upper-triangle rows."""

    rows = check_array(X, dtype=float, ensure_2d=True)
    n_nodes = infer_undirected_node_count(rows.shape[1]) if n_nodes is None else int(n_nodes)
    expected = n_nodes * (n_nodes - 1) // 2
    if rows.shape[1] != expected:
        raise ValueError(f"Expected {expected} edges for {n_nodes} nodes")
    matrices = np.zeros((rows.shape[0], n_nodes, n_nodes), dtype=np.float32)
    upper = np.triu_indices(n_nodes, k=1)
    matrices[:, upper[0], upper[1]] = rows.astype(np.float32)
    matrices[:, upper[1], upper[0]] = rows.astype(np.float32)
    return matrices


def _torch_components(
    model_name: str,
    n_nodes: int,
    n_bands: int,
    hidden_dim: int,
    dropout: float,
):
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Neural graph classifiers require PyTorch. Install "
            "requirements_gnn.txt before selecting gcn or brainnetcnn."
        ) from exc

    class SpectralGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.node_embedding = nn.Parameter(
                torch.randn(n_nodes, hidden_dim) / math.sqrt(hidden_dim)
            )
            self.graph_conv = nn.Linear(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Sequential(
                nn.Linear(2 * hidden_dim * n_bands, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, adjacency):
            batch_size = adjacency.shape[0]
            adjacency = adjacency.reshape(
                batch_size * n_bands, n_nodes, n_nodes
            )
            identity = torch.eye(
                n_nodes, dtype=adjacency.dtype, device=adjacency.device
            ).expand(batch_size * n_bands, -1, -1)
            weights = torch.clamp(adjacency, min=0.0) + identity
            degree = weights.sum(dim=2).clamp_min(1e-7)
            inv_sqrt = degree.rsqrt()
            normalized = inv_sqrt.unsqueeze(2) * weights * inv_sqrt.unsqueeze(1)
            h = self.node_embedding.expand(batch_size * n_bands, -1, -1)
            h = torch.relu(torch.matmul(normalized, h))
            h = self.dropout(h)
            h = torch.relu(self.graph_conv(torch.matmul(normalized, h)))
            pooled = torch.cat((h.mean(dim=1), h.amax(dim=1)), dim=1)
            pooled = pooled.reshape(batch_size, 2 * hidden_dim * n_bands)
            return self.classifier(pooled).squeeze(1)

    class BrainNetCNN(nn.Module):
        def __init__(self):
            super().__init__()
            edge_channels = hidden_dim
            node_channels = 2 * hidden_dim
            self.row_filter = nn.Conv2d(
                n_bands, edge_channels, kernel_size=(1, n_nodes)
            )
            self.column_filter = nn.Conv2d(
                n_bands, edge_channels, kernel_size=(n_nodes, 1)
            )
            self.edge_to_node = nn.Conv2d(
                edge_channels, node_channels, kernel_size=(1, n_nodes)
            )
            self.node_to_graph = nn.Conv2d(
                node_channels, node_channels, kernel_size=(n_nodes, 1)
            )
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(node_channels, 1)

        def forward(self, adjacency):
            x = adjacency
            row = self.row_filter(x).expand(-1, -1, -1, n_nodes)
            column = self.column_filter(x).expand(-1, -1, n_nodes, -1)
            edges = self.dropout(torch.relu(row + column))
            nodes = self.dropout(torch.relu(self.edge_to_node(edges)))
            graph = torch.relu(self.node_to_graph(nodes)).flatten(1)
            return self.classifier(self.dropout(graph)).squeeze(1)

    if model_name == "gcn":
        return torch, SpectralGCN()
    if model_name == "brainnetcnn":
        return torch, BrainNetCNN()
    raise ValueError("model_name must be 'gcn' or 'brainnetcnn'")


class TorchGraphClassifier(ClassifierMixin, BaseEstimator):
    """Binary graph classifier with fold-internal validation and early stopping."""

    def __init__(
        self,
        *,
        model_name: str = "gcn",
        n_bands: int = 1,
        hidden_dim: int = 24,
        dropout: float = 0.35,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-3,
        max_epochs: int = 200,
        patience: int = 25,
        batch_size: int = 32,
        validation_fraction: float = 0.15,
        random_state: int = 42,
        torch_num_threads: int = 1,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.n_bands = n_bands
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.torch_num_threads = torch_num_threads
        self.verbose = verbose

    def _prepare_matrices(self, X: np.ndarray, *, fitting: bool) -> np.ndarray:
        rows = check_array(X, dtype=float, ensure_2d=True)
        edges_per_band = self.n_nodes_ * (self.n_nodes_ - 1) // 2
        reshaped = rows.reshape(len(rows), self.n_bands_, edges_per_band)
        matrices = np.stack(
            [
                edge_vectors_to_symmetric_matrices(
                    reshaped[:, band, :], self.n_nodes_
                )
                for band in range(self.n_bands_)
            ],
            axis=1,
        )
        if self.model_name == "brainnetcnn":
            if fitting:
                self.matrix_mean_ = float(np.mean(matrices))
                scale = float(np.std(matrices))
                self.matrix_scale_ = scale if scale > 1e-7 else 1.0
            matrices = (matrices - self.matrix_mean_) / self.matrix_scale_
            diagonal = np.arange(self.n_nodes_)
            matrices[:, :, diagonal, diagonal] = 0.0
        return matrices.astype(np.float32, copy=False)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit one graph model; all preprocessing is learned from this fold."""

        X, y = check_X_y(X, y, dtype=float, ensure_2d=True)
        classes = np.unique(y)
        if not np.array_equal(classes, np.array([0, 1])):
            raise ValueError("TorchGraphClassifier requires binary labels 0 and 1")
        if self.model_name not in {"gcn", "brainnetcnn"}:
            raise ValueError("model_name must be 'gcn' or 'brainnetcnn'")
        self.n_features_in_ = X.shape[1]
        self.n_bands_ = int(self.n_bands)
        if self.n_bands_ < 1 or X.shape[1] % self.n_bands_:
            raise ValueError("n_bands must divide the number of edge features")
        self.n_nodes_ = infer_undirected_node_count(
            X.shape[1] // self.n_bands_
        )
        self.classes_ = classes

        indices = np.arange(len(y))
        class_counts = np.bincount(y, minlength=2)
        can_validate = (
            self.validation_fraction > 0.0
            and len(y) >= 12
            and int(class_counts.min()) >= 2
        )
        if can_validate:
            train_indices, valid_indices = train_test_split(
                indices,
                test_size=float(self.validation_fraction),
                stratify=y,
                random_state=int(self.random_state),
            )
        else:
            train_indices = indices
            valid_indices = indices

        matrices = self._prepare_matrices(X, fitting=True)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Neural graph classifiers require PyTorch. Install "
                "requirements_gnn.txt before selecting gcn or brainnetcnn."
            ) from exc
        torch.set_num_threads(max(1, int(self.torch_num_threads)))
        torch.manual_seed(int(self.random_state))
        torch, model = _torch_components(
            self.model_name,
            self.n_nodes_,
            self.n_bands_,
            int(self.hidden_dim),
            float(self.dropout),
        )
        model = model.to("cpu")

        labels = torch.as_tensor(y, dtype=torch.float32)
        data = torch.as_tensor(matrices, dtype=torch.float32)
        train_y = y[train_indices]
        positives = max(1, int(np.sum(train_y == 1)))
        negatives = max(1, int(np.sum(train_y == 0)))
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(float(negatives / positives))
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )

        rng = np.random.default_rng(int(self.random_state))
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        stale_epochs = 0
        epochs_run = 0
        batch_size = max(1, int(self.batch_size))
        for epoch in range(max(1, int(self.max_epochs))):
            model.train()
            shuffled = rng.permutation(train_indices)
            for start in range(0, len(shuffled), batch_size):
                batch = shuffled[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = model(data[batch])
                loss = criterion(logits, labels[batch])
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                valid_logits = model(data[valid_indices])
                valid_loss = float(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        valid_logits, labels[valid_indices]
                    ).item()
                )
            epochs_run = epoch + 1
            if valid_loss < best_loss - 1e-5:
                best_loss = valid_loss
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= max(1, int(self.patience)):
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.model_state_ = {
            name: value.detach().cpu().numpy()
            for name, value in model.state_dict().items()
        }
        self.n_epochs_ = epochs_run
        self.best_validation_loss_ = best_loss

        # Temperature is estimated only from the fold-internal validation part.
        with torch.no_grad():
            validation_logits = (
                model(data[valid_indices]).detach().cpu().numpy().astype(float)
            )
        validation_y = y[valid_indices].astype(float)
        temperatures = np.logspace(-0.7, 0.7, 41)
        losses = []
        for temperature in temperatures:
            probability = 1.0 / (1.0 + np.exp(-validation_logits / temperature))
            probability = np.clip(probability, 1e-7, 1.0 - 1e-7)
            losses.append(
                float(
                    -np.mean(
                        validation_y * np.log(probability)
                        + (1.0 - validation_y) * np.log(1.0 - probability)
                    )
                )
            )
        self.temperature_ = float(temperatures[int(np.argmin(losses))])
        return self

    def _ensure_runtime_model(self):
        if hasattr(self, "model_"):
            return self.model_
        torch, model = _torch_components(
            self.model_name,
            self.n_nodes_,
            self.n_bands_,
            int(self.hidden_dim),
            float(self.dropout),
        )
        state = {
            name: torch.as_tensor(value)
            for name, value in self.model_state_.items()
        }
        model.load_state_dict(state)
        model.eval()
        self.model_ = model.to("cpu")
        return self.model_

    def __getstate__(self):
        """Serialize tensor arrays, not function-local PyTorch module classes."""

        state = self.__dict__.copy()
        state.pop("model_", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return temperature-scaled Patient logits."""

        check_is_fitted(self, ["model_state_", "temperature_", "n_nodes_"])
        X = check_array(X, dtype=float, ensure_2d=True)
        matrices = self._prepare_matrices(X, fitting=False)
        import torch

        data = torch.as_tensor(matrices, dtype=torch.float32)
        output = []
        batch_size = max(1, int(self.batch_size))
        model = self._ensure_runtime_model()
        model.eval()
        with torch.no_grad():
            for start in range(0, len(data), batch_size):
                logits = model(data[start : start + batch_size])
                output.append(logits.detach().cpu().numpy())
        return np.concatenate(output).astype(float) / float(self.temperature_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return columns ``P(Healthy)`` and ``P(Patient)``."""

        logits = self.decision_function(X)
        patient = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        return np.column_stack((1.0 - patient, patient))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
