"""EEG connectivity estimators and method/band dispatch."""

from __future__ import annotations

import mne
import numpy as np
from mne_connectivity import phase_slope_index, spectral_connectivity_epochs
from scipy import signal

from config import (
    CONNECTIVITY_ERROR_POLICY,
    CONNECTIVITY_METHODS,
    CONNECTIVITY_NORMALIZATION,
    GC_N_LAGS,
    SPECTRAL_CONNECTIVITY_INPUT,
)


BIVARIATE_SPECTRAL_METHODS = {
    "coh": "coh",
    "coherence": "coh",
    "imcoh": "imcoh",
    "ciplv": "ciplv",
    "plv": "plv",
    "wpli2_debiased": "wpli2_debiased",
}
SPECTRAL_METHODS = set(BIVARIATE_SPECTRAL_METHODS) | {"psi", "gc", "gc_tr"}


def _mne_epochs(epochs: np.ndarray, fs: float) -> mne.EpochsArray:
    array = np.asarray(epochs, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"Expected epochs x channels x samples, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Connectivity input contains NaN or infinite samples")
    info = mne.create_info(
        ch_names=[f"Ch{i}" for i in range(array.shape[1])],
        sfreq=fs,
        ch_types="eeg",
    )
    return mne.EpochsArray(array, info, verbose=False)


def compute_bivariate_spectral_connectivity(
    epochs: np.ndarray, fs: float, fmin: float, fmax: float, method: str
) -> np.ndarray:
    """Compute a symmetric bivariate spectral matrix on its natural scale."""

    mne_method = BIVARIATE_SPECTRAL_METHODS.get(method)
    if mne_method is None:
        raise ValueError(f"Unsupported bivariate spectral method: {method}")
    con = spectral_connectivity_epochs(
        _mne_epochs(epochs, fs),
        method=mne_method,
        mode="multitaper",
        sfreq=fs,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        verbose="ERROR",
    )
    # MNE stores bivariate dense output in one triangle only.
    triangle = np.mean(con.get_data(output="dense"), axis=2)
    matrix = triangle + triangle.T
    if method == "imcoh":
        matrix = np.abs(matrix)
    elif method == "wpli2_debiased":
        matrix = np.maximum(matrix, 0.0)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def compute_plv(
    epochs: np.ndarray, fs: float, fmin: float, fmax: float
) -> np.ndarray:
    """Compute phase-locking value connectivity."""

    return compute_bivariate_spectral_connectivity(epochs, fs, fmin, fmax, "plv")


def compute_psi(
    epochs: np.ndarray, fs: float, fmin: float, fmax: float
) -> np.ndarray:
    """Compute nonnegative directed phase-slope index connectivity."""

    con = phase_slope_index(
        _mne_epochs(epochs, fs),
        mode="multitaper",
        sfreq=fs,
        fmin=fmin,
        fmax=fmax,
        verbose="ERROR",
    )
    signed = np.mean(con.get_data(output="dense"), axis=2)
    directed = np.zeros_like(signed)
    negative = signed < 0
    directed[negative.T] = -signed[negative]
    positive = signed > 0
    directed[positive] = signed[positive]
    np.fill_diagonal(directed, 0.0)
    return directed


def _gc_indices(n_channels: int) -> tuple[tuple[list[list[int]], list[list[int]]], np.ndarray, np.ndarray]:
    sources, targets = np.where(~np.eye(n_channels, dtype=bool))
    indices = (
        [[int(source)] for source in sources],
        [[int(target)] for target in targets],
    )
    return indices, sources, targets


def _compute_gc(
    epochs: np.ndarray,
    fs: float,
    fmin: float,
    fmax: float,
    *,
    method: str,
    gc_n_lags: int,
) -> np.ndarray:
    n_channels = epochs.shape[1]
    indices, sources, targets = _gc_indices(n_channels)
    con = spectral_connectivity_epochs(
        _mne_epochs(epochs, fs),
        method=method,
        mode="multitaper",
        indices=indices,
        sfreq=fs,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        gc_n_lags=int(gc_n_lags),
        verbose="ERROR",
    )
    values = con.get_data()
    values = values[:, 0] if values.ndim == 2 else values
    matrix = np.zeros((n_channels, n_channels), dtype=float)
    matrix[sources, targets] = values
    return matrix


def compute_granger_causality(
    epochs: np.ndarray,
    fs: float,
    fmin: float,
    fmax: float,
    gc_n_lags: int = GC_N_LAGS,
) -> np.ndarray:
    """Compute pairwise spectral Granger predictability with explicit lag order."""

    return _compute_gc(
        epochs, fs, fmin, fmax, method="gc", gc_n_lags=gc_n_lags
    )


def compute_granger_causality_tr(
    epochs: np.ndarray,
    fs: float,
    fmin: float,
    fmax: float,
    gc_n_lags: int = GC_N_LAGS,
) -> np.ndarray:
    """Compute pairwise time-reversed spectral Granger predictability."""

    return _compute_gc(
        epochs, fs, fmin, fmax, method="gc_tr", gc_n_lags=gc_n_lags
    )


def compute_aec(epochs: np.ndarray) -> np.ndarray:
    """Compute absolute log amplitude-envelope correlation."""

    array = np.asarray(epochs, dtype=float)
    analytic = signal.hilbert(array, axis=-1)
    envelope = np.log(np.abs(analytic) + np.finfo(float).eps)
    flattened = envelope.transpose(1, 0, 2).reshape(array.shape[1], -1)
    matrix = np.abs(np.corrcoef(flattened))
    np.fill_diagonal(matrix, 0.0)
    return matrix


def compute_pdc(epochs, fs, fmin, fmax):
    """PDC remains available in the standalone conditional-VAR benchmark."""

    raise NotImplementedError(
        "Production PDC is intentionally disabled; use classification_score/"
        "connectivity_benchmark.py --families var and inspect VAR diagnostics."
    )


def compute_connectivity_for_band(
    filtered_epochs: dict[str, np.ndarray],
    band_name: str,
    fs: float,
    method: str = "plv",
    *,
    broadband_epochs: np.ndarray | None = None,
) -> np.ndarray:
    """Compute one method/band matrix, optionally from broadband epochs."""

    from config import FREQUENCY_BANDS

    fmin, fmax = FREQUENCY_BANDS[band_name]
    epochs = filtered_epochs[band_name]
    if method in SPECTRAL_METHODS and SPECTRAL_CONNECTIVITY_INPUT == "broadband":
        if broadband_epochs is None:
            raise ValueError(
                "Broadband epochs are required by this connectivity profile. "
                "Rerun signal-processing stage 2 with the current code."
            )
        epochs = broadband_epochs

    if method in BIVARIATE_SPECTRAL_METHODS:
        return compute_bivariate_spectral_connectivity(epochs, fs, fmin, fmax, method)
    if method == "psi":
        return compute_psi(epochs, fs, fmin, fmax)
    if method == "gc":
        return compute_granger_causality(epochs, fs, fmin, fmax)
    if method == "gc_tr":
        return compute_granger_causality_tr(epochs, fs, fmin, fmax)
    if method == "aec":
        return compute_aec(filtered_epochs[band_name])
    if method == "pdc":
        return compute_pdc(epochs, fs, fmin, fmax)
    raise ValueError(f"Unknown connectivity method: {method}")


def normalize_connectivity_matrix(
    matrix: np.ndarray, mode: str = CONNECTIVITY_NORMALIZATION
) -> np.ndarray:
    """Apply an explicit normalization policy to one square matrix."""

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"Expected square connectivity matrix, got {array.shape}")
    finite = np.isfinite(array)
    if mode == "minmax":
        # Historical profiles keep their finite-edge cleanup for saved-result
        # compatibility. Connectivity-v2 uses mode="none" and fails instead.
        if not np.any(finite):
            return np.zeros_like(array)
        minimum = np.min(array[finite])
        maximum = np.max(array[finite])
        normalized = (
            (array - minimum) / (maximum - minimum)
            if maximum > minimum
            else np.zeros_like(array)
        )
        return np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    if not np.all(finite):
        raise ValueError("Connectivity matrix contains NaN or infinite values")
    if mode == "none":
        return array.copy()
    if mode == "maxabs":
        scale = np.max(np.abs(array))
        return array / scale if scale > 0 else np.zeros_like(array)
    raise ValueError(f"Unknown connectivity normalization mode: {mode}")


def compute_all_connectivity(
    filtered_epochs: dict[str, np.ndarray],
    fs: float,
    methods=CONNECTIVITY_METHODS,
    *,
    broadband_epochs: np.ndarray | None = None,
    normalization: str = CONNECTIVITY_NORMALIZATION,
    error_policy: str = CONNECTIVITY_ERROR_POLICY,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute every configured method/band while preserving the saved schema."""

    from config import FREQUENCY_BANDS

    results: dict[str, dict[str, np.ndarray]] = {}
    for method in methods:
        print(f"\nComputing {method.upper()} connectivity:")
        results[method] = {}
        for band_name in FREQUENCY_BANDS:
            print(f"  {band_name}...", end=" ")
            try:
                matrix = compute_connectivity_for_band(
                    filtered_epochs,
                    band_name,
                    fs,
                    method,
                    broadband_epochs=broadband_epochs,
                )
                results[method][band_name] = normalize_connectivity_matrix(
                    matrix, mode=normalization
                )
                print("ok")
            except Exception as exc:
                print(f"ERROR: {exc}")
                if error_policy == "raise":
                    raise RuntimeError(
                        f"Connectivity failed for method={method}, band={band_name}"
                    ) from exc
                if error_policy != "zeros":
                    raise ValueError(f"Unknown connectivity error policy: {error_policy}")
                n_channels = filtered_epochs[band_name].shape[1]
                results[method][band_name] = np.zeros((n_channels, n_channels))
    return results
