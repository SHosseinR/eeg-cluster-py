"""Dynamics-free stimulation transforms for EEG connectivity optimization."""

from __future__ import annotations

import numpy as np

from state_space_simulation import (
    compute_activation_changes,
    normalize_adjacency_matrix,
)


STATIC_EDGE_SCOPES = {"incident", "incoming", "outgoing"}
STIMULATION_MODELS = {
    "state_space",
    "static_adjacency",
    "adjacency_activation",
}
DYNAMICS_FREE_STIMULATION_MODELS = {
    "static_adjacency",
    "adjacency_activation",
}


def run_adjacency_activation_stimulation(
    adjacency_matrix: np.ndarray,
    baseline_activation: np.ndarray,
    stimulation_node: int,
    stimulation_amount: float,
    *,
    neighbor_scale: float = 1.0,
    stability_constant: float = 0.01,
) -> dict[str, np.ndarray | float | str | None]:
    """Compute direct-plus-one-hop activation without state-space dynamics.

    For node ``k``, this implements the finite activation change

    ``delta_x = amount * (e_k + neighbor_scale * A_norm @ e_k)``

    where ``A_norm`` uses the same spectral normalization as the original
    state-space model and has a zero diagonal. Consequently, ``amount`` is
    the signed activation change applied directly to the selected node.
    Connected nodes receive one-hop changes proportional to column ``k`` of
    the normalized adjacency matrix. No duration, leak, trajectory, or
    higher-order adjacency powers are involved.

    The returned activation ratios use the original state-space ratio
    calculation and clipping contract, so callers can apply the unchanged
    Hebbian-like plasticity stage.

    Parameters
    ----------
    adjacency_matrix
        Finite square connectivity matrix with axes ``(source, target)``.
    baseline_activation
        Finite baseline node activation vector with shape ``(n_nodes,)``.
    stimulation_node
        Zero-based selected node index.
    stimulation_amount
        Signed direct activation change at the selected node.
    neighbor_scale
        Non-negative multiplier for the adjacency-scaled one-hop response.
    stability_constant
        Positive constant used by the shared spectral normalization.

    Returns
    -------
    dict
        Final activation, clipped and raw ratios, normalized adjacency, and
        audit diagnostics. ``trajectory`` is always ``None``.
    """

    matrix = np.asarray(adjacency_matrix, dtype=float)
    baseline = np.asarray(baseline_activation, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"adjacency_matrix must be square; got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("adjacency_matrix contains non-finite values")
    if baseline.shape != (matrix.shape[0],):
        raise ValueError(
            "baseline_activation must have shape "
            f"({matrix.shape[0]},); got {baseline.shape}"
        )
    if not np.all(np.isfinite(baseline)):
        raise ValueError("baseline_activation contains non-finite values")

    node = int(stimulation_node)
    if node < 0 or node >= matrix.shape[0]:
        raise IndexError(
            f"stimulation_node={node} is outside [0, {matrix.shape[0] - 1}]"
        )
    amount = float(stimulation_amount)
    if not np.isfinite(amount):
        raise ValueError("stimulation_amount must be finite")
    spread_scale = float(neighbor_scale)
    if not np.isfinite(spread_scale) or spread_scale < 0.0:
        raise ValueError("neighbor_scale must be finite and non-negative")
    normalization_constant = float(stability_constant)
    if not np.isfinite(normalization_constant) or normalization_constant <= 0.0:
        raise ValueError("stability_constant must be finite and positive")

    normalized_matrix = normalize_adjacency_matrix(
        matrix,
        stability_constant=normalization_constant,
    )
    direct_change = np.zeros(matrix.shape[0], dtype=float)
    direct_change[node] = amount
    neighbor_change = (
        amount * spread_scale * normalized_matrix[:, node]
    )
    # The shared normalization already clears the diagonal. Enforce the
    # scientific contract explicitly in case its implementation changes.
    neighbor_change[node] = 0.0
    activation_change = direct_change + neighbor_change
    final_state = baseline + activation_change
    activation_ratios, raw_activation_ratios = compute_activation_changes(
        baseline,
        final_state,
        return_raw=True,
    )

    return {
        "trajectory": None,
        "final_state": final_state,
        "activation_change": activation_change,
        "direct_activation_change": float(direct_change[node]),
        "neighbor_activation_change_l1": float(np.sum(np.abs(neighbor_change))),
        "total_activation_change_l1": float(np.sum(np.abs(activation_change))),
        "changed_activation_nodes": float(np.count_nonzero(activation_change)),
        "activation_ratios": activation_ratios,
        "raw_activation_ratios": raw_activation_ratios,
        "normalized_matrix": normalized_matrix,
        "baseline": baseline,
        "stimulation_activation_amount": amount,
        "adjacency_activation_neighbor_scale": spread_scale,
        "adjacency_activation_orientation": "column",
    }


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
