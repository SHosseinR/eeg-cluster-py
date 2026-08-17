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
    "adjacency_activation_log_gain",
}
DYNAMICS_FREE_STIMULATION_MODELS = {
    "static_adjacency",
    "adjacency_activation",
    "adjacency_activation_log_gain",
}
ADJACENCY_PROPAGATION_NORMALIZATIONS = {"none", "spectral_radius"}


def prepare_adjacency_propagation_matrix(
    adjacency_matrix: np.ndarray,
    *,
    normalization: str = "spectral_radius",
    stability_constant: float = 0.01,
) -> np.ndarray:
    """Prepare the one-hop adjacency operator while clearing self-loops.

    ``"spectral_radius"`` preserves the historical division by
    ``max(abs(eigvals(A))) + stability_constant``. ``"none"`` uses the
    connectivity values on their original scale. Neither mode changes channel
    order, matrix orientation, or off-diagonal signs.
    """

    matrix = np.asarray(adjacency_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"adjacency_matrix must be square; got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("adjacency_matrix contains non-finite values")
    mode = str(normalization).strip().lower()
    if mode not in ADJACENCY_PROPAGATION_NORMALIZATIONS:
        raise ValueError(
            "normalization must be 'none' or 'spectral_radius'; "
            f"got {mode!r}"
        )
    if mode == "spectral_radius":
        constant = float(stability_constant)
        if not np.isfinite(constant) or constant <= 0.0:
            raise ValueError("stability_constant must be finite and positive")
        return normalize_adjacency_matrix(
            matrix,
            stability_constant=constant,
        )

    propagation_matrix = matrix.copy()
    np.fill_diagonal(propagation_matrix, 0.0)
    return propagation_matrix


def compute_band_rms(band_filtered_eeg: np.ndarray) -> np.ndarray:
    """Return per-channel RMS amplitude from one band-filtered EEG array.

    Parameters
    ----------
    band_filtered_eeg
        Finite EEG samples with shape ``(epochs, channels, samples)`` or a
        single epoch with shape ``(channels, samples)``.

    Returns
    -------
    ndarray, shape (channels,)
        ``sqrt(mean(EEG_i ** 2))`` over every epoch and sample for channel
        ``i``. Computation uses float64 even when cached epochs are float32.
    """

    eeg = np.asarray(band_filtered_eeg, dtype=np.float64)
    if eeg.ndim == 3:
        if any(size == 0 for size in eeg.shape):
            raise ValueError(
                "band_filtered_eeg dimensions must be non-empty; "
                f"got {eeg.shape}"
            )
        reduction_axes = (0, 2)
    elif eeg.ndim == 2:
        if any(size == 0 for size in eeg.shape):
            raise ValueError(
                "band_filtered_eeg dimensions must be non-empty; "
                f"got {eeg.shape}"
            )
        reduction_axes = 1
    else:
        raise ValueError(
            "band_filtered_eeg must have shape (epochs, channels, samples) "
            f"or (channels, samples); got {eeg.shape}"
        )
    if not np.all(np.isfinite(eeg)):
        raise ValueError("band_filtered_eeg contains non-finite values")

    rms = np.sqrt(np.mean(np.square(eeg), axis=reduction_axes))
    if rms.ndim != 1 or not np.all(np.isfinite(rms)):
        raise ValueError("Band RMS calculation produced invalid values")
    return rms


def run_adjacency_activation_log_gain_stimulation(
    adjacency_matrix: np.ndarray,
    baseline_activation: np.ndarray,
    stimulation_node: int,
    log_gain: float,
    *,
    neighbor_scale: float = 1.0,
    stability_constant: float = 0.01,
    adjacency_normalization: str = "spectral_radius",
) -> dict[str, np.ndarray | float | str | None]:
    """Apply multiplicative direct-plus-one-hop activation gain.

    For selected node ``k``, the spatial profile and activation ratio are

    ``v = e_k + neighbor_scale * A_propagation @ e_k``

    ``R = exp(log_gain * v)``

    The baseline vector is the per-channel RMS of the matching band-filtered
    EEG. The returned final activation is ``E1 = E0 * R``. Connectivity
    plasticity is deliberately performed by the separate log-gain plasticity
    helper so the legacy activation model remains unchanged.
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
    if not np.all(np.isfinite(baseline)) or np.any(baseline < 0.0):
        raise ValueError(
            "baseline_activation must contain finite, non-negative band RMS values"
        )

    node = int(stimulation_node)
    if node < 0 or node >= matrix.shape[0]:
        raise IndexError(
            f"stimulation_node={node} is outside [0, {matrix.shape[0] - 1}]"
        )
    gain = float(log_gain)
    if not np.isfinite(gain):
        raise ValueError("log_gain must be finite")
    spread_scale = float(neighbor_scale)
    if not np.isfinite(spread_scale) or spread_scale < 0.0:
        raise ValueError("neighbor_scale must be finite and non-negative")
    normalization_mode = str(adjacency_normalization).strip().lower()
    propagation_matrix = prepare_adjacency_propagation_matrix(
        matrix,
        normalization=normalization_mode,
        stability_constant=stability_constant,
    )
    unit_vector = np.zeros(matrix.shape[0], dtype=float)
    unit_vector[node] = 1.0
    spatial_profile = unit_vector + spread_scale * (propagation_matrix @ unit_vector)
    with np.errstate(over="ignore", invalid="ignore"):
        activation_ratios = np.exp(gain * spatial_profile)
        final_state = baseline * activation_ratios
    if not np.all(np.isfinite(spatial_profile)):
        raise ValueError("Log-gain spatial profile contains non-finite values")
    if not np.all(np.isfinite(activation_ratios)):
        raise ValueError("exp(log_gain * v) produced non-finite activation ratios")
    if not np.all(np.isfinite(final_state)):
        raise ValueError("Log-gain final activation contains non-finite values")

    activation_change = final_state - baseline
    neighbor_profile = spatial_profile - unit_vector
    return {
        "trajectory": None,
        "final_state": final_state,
        "activation_change": activation_change,
        "activation_ratios": activation_ratios,
        "raw_activation_ratios": activation_ratios.copy(),
        "propagation_matrix": propagation_matrix,
        # Compatibility alias for consumers written before propagation became
        # configurable. In "none" mode this matrix is deliberately unscaled.
        "normalized_matrix": propagation_matrix,
        "adjacency_propagation_normalization": normalization_mode,
        "baseline": baseline,
        "log_gain_spatial_profile": spatial_profile,
        "direct_log_gain_profile": float(spatial_profile[node]),
        "neighbor_log_gain_profile_l1": float(np.sum(np.abs(neighbor_profile))),
        "changed_activation_nodes": float(np.count_nonzero(activation_change)),
        "stimulation_log_gain": gain,
        "log_gain_neighbor_scale": spread_scale,
        "adjacency_activation_orientation": "column",
    }


def run_adjacency_activation_stimulation(
    adjacency_matrix: np.ndarray,
    baseline_activation: np.ndarray,
    stimulation_node: int,
    stimulation_amount: float,
    *,
    neighbor_scale: float = 1.0,
    stability_constant: float = 0.01,
    adjacency_normalization: str = "spectral_radius",
) -> dict[str, np.ndarray | float | str | None]:
    """Compute direct-plus-one-hop activation without state-space dynamics.

    For node ``k``, this implements the finite activation change

    ``delta_x = amount * (e_k + neighbor_scale * A_propagation @ e_k)``

    The propagation matrix always has a zero diagonal. It is either divided
    by spectral radius plus the stability constant (the legacy default) or
    kept on its original connectivity scale. Consequently, ``amount`` is the
    signed activation change applied directly to the selected node. Connected
    nodes receive one-hop changes proportional to column ``k``. No duration,
    leak, trajectory, or higher-order adjacency powers are involved.

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
        Positive constant used only by spectral-radius normalization.
    adjacency_normalization
        ``"spectral_radius"`` for the historical eigenvalue division or
        ``"none"`` to preserve the original connectivity scale.

    Returns
    -------
    dict
        Final activation, clipped and raw ratios, propagation adjacency, and
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
    normalization_mode = str(adjacency_normalization).strip().lower()
    propagation_matrix = prepare_adjacency_propagation_matrix(
        matrix,
        normalization=normalization_mode,
        stability_constant=stability_constant,
    )
    direct_change = np.zeros(matrix.shape[0], dtype=float)
    direct_change[node] = amount
    neighbor_change = (
        amount * spread_scale * propagation_matrix[:, node]
    )
    # Propagation preparation clears the diagonal. Enforce the scientific
    # contract explicitly in case its implementation changes.
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
        "propagation_matrix": propagation_matrix,
        "normalized_matrix": propagation_matrix,
        "adjacency_propagation_normalization": normalization_mode,
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
