"""Auditable connectivity estimators for the standalone benchmark.

The functions here intentionally do not import the production pipeline.  They
operate on one subject's band-filtered epochs with shape
``(epochs, channels, samples)`` and return square channel matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

import numpy as np
from scipy import signal, stats


UNDIRECTED_METHODS = (
    "coherence",
    "imaginary_coherence",
    "plv",
    "ciplv",
    "ppc",
    "pli",
    "wpli2_debiased",
    "aec",
    "orthogonalized_aec",
)
DIRECTED_PHASE_METHODS = ("dpli", "directed_wpli", "psi")
DIRECTED_VAR_METHODS = ("conditional_var_wald", "pdc", "dtf")


@dataclass(frozen=True)
class VarDiagnostics:
    """Diagnostics for one fitted conditional VAR model."""

    order: int
    target_sfreq: float
    ridge_alpha: float
    n_observations: int
    n_predictors: int
    residual_lag1_correlation: float


def _validate_epochs(epochs: np.ndarray) -> np.ndarray:
    array = np.asarray(epochs, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"Expected epochs x channels x samples, got {array.shape}")
    if array.shape[0] < 2 or array.shape[1] < 2 or array.shape[2] < 8:
        raise ValueError(f"Insufficient epoch data for connectivity: {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Connectivity input contains NaN or infinite samples")
    return array


def _zero_diagonal(matrix: np.ndarray) -> np.ndarray:
    output = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(output, 0.0)
    return output


def _fourier_observations(
    epochs: np.ndarray, fs: float, fmin: float, fmax: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return epoch-frequency observations and per-frequency spectra.

    Returns
    -------
    observations : ndarray, shape (n_epochs * n_freqs, n_channels)
        Complex Fourier coefficients used as approximately independent
        observations for phase-lag estimators.
    spectra : ndarray, shape (n_epochs, n_channels, n_freqs)
        Coefficients retaining frequency identity for PSI.
    frequencies : ndarray, shape (n_freqs,)
        Selected Fourier frequencies in Hz.
    """

    data = _validate_epochs(epochs)
    n_samples = data.shape[-1]
    window = signal.windows.hann(n_samples, sym=False)
    centered = data - np.mean(data, axis=-1, keepdims=True)
    spectra = np.fft.rfft(centered * window, axis=-1)
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / float(fs))
    keep = (frequencies >= fmin) & (frequencies <= fmax) & (frequencies > 0)
    if np.sum(keep) < 2:
        raise ValueError(
            f"Band {fmin:g}-{fmax:g} Hz contains fewer than two Fourier bins at fs={fs:g}"
        )
    selected = spectra[..., keep]
    observations = selected.transpose(0, 2, 1).reshape(-1, data.shape[1])
    return observations, selected, frequencies[keep]


def compute_fourier_connectivity(
    epochs: np.ndarray,
    fs: float,
    fmin: float,
    fmax: float,
    methods: Iterable[str] = (
        "coherence",
        "imaginary_coherence",
        "plv",
        "ciplv",
        "ppc",
        "pli",
        "wpli2_debiased",
        "dpli",
        "directed_wpli",
        "psi",
    ),
) -> dict[str, np.ndarray]:
    """Compute Fourier phase/coherence measures without subject min-max scaling."""

    requested = tuple(dict.fromkeys(methods))
    supported = set(UNDIRECTED_METHODS[:7]) | set(DIRECTED_PHASE_METHODS)
    unknown = sorted(set(requested) - supported)
    if unknown:
        raise ValueError(f"Unsupported Fourier connectivity methods: {unknown}")

    _, spectra, _ = _fourier_observations(epochs, fs, fmin, fmax)
    n_epochs, n_channels, n_frequencies = spectra.shape
    eps = np.finfo(float).eps

    # Estimate phase relationships independently at each frequency. Treating
    # distinct frequencies as exchangeable phase samples can cancel a real
    # relationship merely because its phase slope changes across the band.
    cross_frequency = np.einsum("ecf,edf->fcd", spectra, spectra.conj())
    frequency_power = np.maximum(
        np.real(np.diagonal(cross_frequency, axis1=1, axis2=2)), eps
    )
    frequency_coherency = cross_frequency / np.sqrt(
        frequency_power[:, :, None] * frequency_power[:, None, :]
    )

    amplitudes = np.abs(spectra)
    unit = np.divide(
        spectra,
        amplitudes,
        out=np.zeros_like(spectra),
        where=amplitudes > eps,
    )
    complex_plv_frequency = np.einsum("ecf,edf->fcd", unit, unit.conj()) / n_epochs
    plv_frequency = np.abs(complex_plv_frequency)

    output: dict[str, np.ndarray] = {}
    if "coherence" in requested:
        output["coherence"] = _zero_diagonal(np.mean(np.abs(frequency_coherency), axis=0))
    if "imaginary_coherence" in requested:
        output["imaginary_coherence"] = _zero_diagonal(
            np.mean(np.abs(np.imag(frequency_coherency)), axis=0)
        )
    if "plv" in requested:
        output["plv"] = _zero_diagonal(np.mean(plv_frequency, axis=0))
    if "ciplv" in requested:
        denominator = np.sqrt(
            np.maximum(1.0 - np.square(np.real(complex_plv_frequency)), eps)
        )
        output["ciplv"] = _zero_diagonal(
            np.mean(
                np.clip(np.abs(np.imag(complex_plv_frequency)) / denominator, 0.0, 1.0),
                axis=0,
            )
        )
    if "ppc" in requested:
        ppc = (n_epochs * np.square(plv_frequency) - 1.0) / max(n_epochs - 1, 1)
        output["ppc"] = _zero_diagonal(np.mean(np.clip(ppc, 0.0, 1.0), axis=0))

    lag_methods = {"pli", "wpli2_debiased", "dpli", "directed_wpli"} & set(requested)
    if lag_methods:
        frequency_observations = spectra.transpose(0, 2, 1)
        imaginary = np.imag(
            frequency_observations[:, :, :, None]
            * frequency_observations[:, :, None, :].conj()
        )
        # Pool epoch-frequency observations only for sign/lag statistics; unlike
        # PLV, their signed imaginary cross-spectrum has a shared zero reference.
        sum_imag = np.sum(imaginary, axis=(0, 1))
        sum_abs_imag = np.sum(np.abs(imaginary), axis=(0, 1))
        sum_square_imag = np.sum(np.square(imaginary), axis=(0, 1))
        positive_count = np.sum(imaginary > 0, axis=(0, 1))
        sign_sum = np.sum(np.sign(imaginary), axis=(0, 1))
        n_lag_observations = n_epochs * n_frequencies

        if "pli" in requested:
            output["pli"] = _zero_diagonal(np.abs(sign_sum) / n_lag_observations)
        if "wpli2_debiased" in requested:
            numerator = np.square(sum_imag) - sum_square_imag
            denominator = np.square(sum_abs_imag) - sum_square_imag
            wpli2 = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > eps,
            )
            output["wpli2_debiased"] = _zero_diagonal(np.clip(wpli2, 0.0, 1.0))
        if "dpli" in requested:
            directed_probability = positive_count / n_lag_observations
            output["dpli"] = _zero_diagonal(
                np.clip(2.0 * (directed_probability - 0.5), 0.0, 1.0)
            )
        if "directed_wpli" in requested:
            signed_wpli = np.divide(
                sum_imag,
                sum_abs_imag,
                out=np.zeros_like(sum_imag),
                where=sum_abs_imag > eps,
            )
            output["directed_wpli"] = _zero_diagonal(np.maximum(signed_wpli, 0.0))

    if "psi" in requested:
        signed_psi = np.mean(
            np.imag(
                np.conj(frequency_coherency[:-1]) * frequency_coherency[1:]
            ),
            axis=0,
        )
        output["psi"] = _zero_diagonal(np.maximum(signed_psi, 0.0))

    return output


def compute_envelope_connectivity(
    epochs: np.ndarray, *, include_orthogonalized: bool = True
) -> dict[str, np.ndarray]:
    """Compute amplitude-envelope correlation, optionally with orthogonalization."""

    data = _validate_epochs(epochs)
    analytic = signal.hilbert(data, axis=-1)
    envelope = np.log(np.abs(analytic) + np.finfo(float).eps)
    flattened_envelope = envelope.transpose(1, 0, 2).reshape(data.shape[1], -1)
    aec = np.abs(np.corrcoef(flattened_envelope))
    if not include_orthogonalized:
        return {"aec": _zero_diagonal(aec)}

    centered_envelope = flattened_envelope - np.mean(
        flattened_envelope, axis=1, keepdims=True
    )
    envelope_norm = np.linalg.norm(centered_envelope, axis=1)

    def correlation_with_channel(channel: int, values: np.ndarray) -> float:
        """Pearson correlation against one cached envelope without scipy overhead."""

        centered_values = values - np.mean(values)
        denominator = envelope_norm[channel] * np.linalg.norm(centered_values)
        if denominator <= np.finfo(float).eps:
            return 0.0
        return float(centered_envelope[channel] @ centered_values / denominator)

    n_channels = data.shape[1]
    orthogonalized = np.zeros((n_channels, n_channels), dtype=float)
    for left in range(n_channels):
        left_signal = analytic[:, left, :]
        left_unit_conjugate = np.divide(
            np.conj(left_signal),
            np.abs(left_signal),
            out=np.zeros_like(left_signal),
            where=np.abs(left_signal) > np.finfo(float).eps,
        )
        for right in range(left + 1, n_channels):
            right_signal = analytic[:, right, :]
            right_orthogonal_to_left = np.imag(right_signal * left_unit_conjugate)
            right_orthogonal_envelope = np.log(
                np.abs(right_orthogonal_to_left).reshape(-1) + np.finfo(float).eps
            )
            corr_right_to_left = correlation_with_channel(
                left, right_orthogonal_envelope
            )

            right_unit_conjugate = np.divide(
                np.conj(right_signal),
                np.abs(right_signal),
                out=np.zeros_like(right_signal),
                where=np.abs(right_signal) > np.finfo(float).eps,
            )
            left_orthogonal_to_right = np.imag(left_signal * right_unit_conjugate)
            left_orthogonal_envelope = np.log(
                np.abs(left_orthogonal_to_right).reshape(-1) + np.finfo(float).eps
            )
            corr_left_to_right = correlation_with_channel(
                right, left_orthogonal_envelope
            )
            value = np.nanmean(np.abs([corr_right_to_left, corr_left_to_right]))
            orthogonalized[left, right] = orthogonalized[right, left] = value

    return {
        "aec": _zero_diagonal(aec),
        "orthogonalized_aec": _zero_diagonal(orthogonalized),
    }


def _resample_epochs(epochs: np.ndarray, fs: float, target_sfreq: float) -> np.ndarray:
    if np.isclose(fs, target_sfreq):
        return np.asarray(epochs, dtype=float)
    ratio = Fraction(float(target_sfreq) / float(fs)).limit_denominator(1000)
    return signal.resample_poly(epochs, ratio.numerator, ratio.denominator, axis=-1)


def _lag_design(epochs: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    predictors = []
    targets = []
    for epoch in epochs:
        targets.append(epoch[:, order:].T)
        predictors.append(
            np.concatenate(
                [epoch[:, order - lag : -lag].T for lag in range(1, order + 1)],
                axis=1,
            )
        )
    return np.vstack(predictors), np.vstack(targets)


def _var_residual_lag1(residuals: np.ndarray) -> float:
    if residuals.shape[0] < 3:
        return np.nan
    previous = residuals[:-1].reshape(-1)
    following = residuals[1:].reshape(-1)
    if np.std(previous) == 0 or np.std(following) == 0:
        return 0.0
    return float(np.corrcoef(previous, following)[0, 1])


def compute_conditional_var_connectivity(
    epochs: np.ndarray,
    fs: float,
    fmin: float,
    fmax: float,
    *,
    target_sfreq: float = 100.0,
    lag_ms: float = 100.0,
    ridge_alpha: float = 10.0,
    frequencies_per_band: int = 24,
) -> tuple[dict[str, np.ndarray], VarDiagnostics]:
    """Fit one conditional VAR and derive Wald, PDC, and DTF matrices.

    The Wald score tests the lag block for one source while conditioning on
    every other channel. PDC and DTF are derived from the same fitted VAR.
    Values are raw estimator outputs; no within-subject min-max scaling is
    applied.
    """

    data = _validate_epochs(epochs)
    resampled = _resample_epochs(data, fs, target_sfreq)
    channel_mean = np.mean(resampled, axis=(0, 2), keepdims=True)
    channel_std = np.std(resampled, axis=(0, 2), keepdims=True)
    standardized = np.divide(
        resampled - channel_mean,
        channel_std,
        out=np.zeros_like(resampled),
        where=channel_std > np.finfo(float).eps,
    )
    order = max(1, int(round(lag_ms * target_sfreq / 1000.0)))
    if standardized.shape[-1] <= order + 2:
        raise ValueError(
            f"VAR order {order} is too large for {standardized.shape[-1]} samples per epoch"
        )
    X, Y = _lag_design(standardized, order)
    n_observations, n_predictors = X.shape
    n_channels = Y.shape[1]
    xtx = X.T @ X
    regularized_xtx = xtx + float(ridge_alpha) * np.eye(n_predictors)
    inverse_xtx = np.linalg.pinv(regularized_xtx, hermitian=True)
    coefficients = inverse_xtx @ X.T @ Y
    residuals = Y - X @ coefficients
    residual_variance = np.maximum(
        np.sum(np.square(residuals), axis=0)
        / max(n_observations - n_predictors, 1),
        np.finfo(float).eps,
    )

    wald = np.zeros((n_channels, n_channels), dtype=float)
    for source in range(n_channels):
        block = np.asarray([lag * n_channels + source for lag in range(order)])
        inverse_block = inverse_xtx[np.ix_(block, block)]
        for target in range(n_channels):
            if source == target:
                continue
            beta = coefficients[block, target]
            covariance = residual_variance[target] * inverse_block
            statistic = float(beta @ np.linalg.pinv(covariance, hermitian=True) @ beta / order)
            wald[source, target] = np.log1p(max(statistic, 0.0))

    # A_lag[target, source] in conventional VAR notation.
    lag_matrices = np.stack(
        [
            coefficients[lag * n_channels : (lag + 1) * n_channels].T
            for lag in range(order)
        ]
    )
    frequencies = np.linspace(max(fmin, 0.01), fmax, frequencies_per_band)
    pdc_values = []
    dtf_values = []
    identity = np.eye(n_channels, dtype=complex)
    for frequency in frequencies:
        phase = np.exp(
            -2j * np.pi * frequency * np.arange(1, order + 1) / target_sfreq
        )
        ar_frequency = identity - np.sum(lag_matrices * phase[:, None, None], axis=0)
        pdc_target_source = np.abs(ar_frequency) / np.sqrt(
            np.maximum(np.sum(np.square(np.abs(ar_frequency)), axis=0, keepdims=True), 1e-15)
        )
        transfer = np.linalg.pinv(ar_frequency)
        dtf_target_source = np.abs(transfer) / np.sqrt(
            np.maximum(np.sum(np.square(np.abs(transfer)), axis=1, keepdims=True), 1e-15)
        )
        pdc_values.append(pdc_target_source.T)
        dtf_values.append(dtf_target_source.T)

    matrices = {
        "conditional_var_wald": _zero_diagonal(wald),
        "pdc": _zero_diagonal(np.mean(pdc_values, axis=0)),
        "dtf": _zero_diagonal(np.mean(dtf_values, axis=0)),
    }
    diagnostics = VarDiagnostics(
        order=order,
        target_sfreq=float(target_sfreq),
        ridge_alpha=float(ridge_alpha),
        n_observations=int(n_observations),
        n_predictors=int(n_predictors),
        residual_lag1_correlation=_var_residual_lag1(residuals),
    )
    return matrices, diagnostics


def edge_vector(matrix: np.ndarray, directed: bool) -> np.ndarray:
    """Vectorize a square adjacency matrix without duplicating undirected edges."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square connectivity matrix, got {matrix.shape}")
    if directed:
        return matrix[~np.eye(matrix.shape[0], dtype=bool)]
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def split_half_reliability(
    first: np.ndarray, second: np.ndarray, *, directed: bool
) -> dict[str, float]:
    """Edge-pattern reliability between two independently estimated halves."""

    left = edge_vector(first, directed)
    right = edge_vector(second, directed)
    if np.std(left) == 0 or np.std(right) == 0:
        spearman = 0.0
        pearson = 0.0
    else:
        spearman = float(stats.spearmanr(left, right).statistic)
        pearson = float(np.corrcoef(left, right)[0, 1])
    denominator = np.linalg.norm(left) + np.linalg.norm(right)
    normalized_error = (
        float(2.0 * np.linalg.norm(left - right) / denominator) if denominator > 0 else 0.0
    )
    top_count = max(1, int(round(0.1 * left.size)))
    top_left = set(np.argpartition(left, -top_count)[-top_count:])
    top_right = set(np.argpartition(right, -top_count)[-top_count:])
    union = top_left | top_right
    top_jaccard = len(top_left & top_right) / len(union) if union else 1.0
    return {
        "edge_spearman": spearman,
        "edge_pearson": pearson,
        "normalized_edge_error": normalized_error,
        "top10pct_jaccard": float(top_jaccard),
    }
