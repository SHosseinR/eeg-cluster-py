"""Dynamics-free stimulation transforms for connectivity matrices."""

from __future__ import annotations

import numpy as np


STATIC_EDGE_SCOPES = {"incident", "incoming", "outgoing"}


def apply_static_adjacency_stimulation(
    adjacency_matrix: np.ndarray,
    stimulation_node: int,
    total_change: float,
    *,
    edge_scope: str = "incident",
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Distribute a signed total change across a stimulated node's edges.

    The absolute size of each edge update is proportional to the edge's
    original absolute weight. Positive ``total_change`` strengthens existing
    edge magnitudes and negative values weaken them. The diagonal is never
    changed. For a symmetric matrix, ``incident`` preserves symmetry.

    Parameters
    ----------
    adjacency_matrix
        Finite square connectivity matrix with axes ``(source, target)``.
    stimulation_node
        Zero-based channel index.
    total_change
        Signed L1 amount distributed over all selected matrix entries.
    edge_scope
        ``"outgoing"`` changes the selected row, ``"incoming"`` changes the
        selected column, and ``"incident"`` changes both.

    Returns
    -------
    updated_matrix, diagnostics
        The transformed matrix and realized-change audit values.
    """

    matrix = np.asarray(adjacency_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"adjacency_matrix must be square; got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("adjacency_matrix contains non-finite values")
    if edge_scope not in STATIC_EDGE_SCOPES:
        raise ValueError(
            f"edge_scope must be one of {sorted(STATIC_EDGE_SCOPES)}; "
            f"got {edge_scope!r}"
        )
    node = int(stimulation_node)
    if node < 0 or node >= matrix.shape[0]:
        raise IndexError(
            f"stimulation_node={node} is outside [0, {matrix.shape[0] - 1}]"
        )
    amount = float(total_change)
    if not np.isfinite(amount):
        raise ValueError("total_change must be finite")

    selected = np.zeros_like(matrix, dtype=bool)
    if edge_scope in {"outgoing", "incident"}:
        selected[node, :] = True
    if edge_scope in {"incoming", "incident"}:
        selected[:, node] = True
    np.fill_diagonal(selected, False)

    selected_weights = matrix[selected]
    scale_total = float(np.sum(np.abs(selected_weights)))
    delta = np.zeros_like(matrix)
    if abs(amount) > 0.0:
        if scale_total <= np.finfo(float).eps:
            raise ValueError(
                f"Node {node} has no non-zero {edge_scope} weights to scale"
            )
        # Moving in the sign of an existing edge strengthens its magnitude;
        # a negative amount moves toward zero. This is identical to ordinary
        # additive increase/decrease for the non-negative coherence profiles.
        delta[selected] = (
            amount
            * np.sign(selected_weights)
            * np.abs(selected_weights)
            / scale_total
        )

    updated = matrix + delta
    realized_l1 = float(np.sum(np.abs(delta)))
    diagnostics: dict[str, float | str] = {
        "static_edge_scope": edge_scope,
        "requested_total_change": amount,
        "realized_total_change_l1": realized_l1,
        "selected_original_weight_l1": scale_total,
        "changed_matrix_entries": float(np.count_nonzero(delta)),
    }
    return updated, diagnostics
