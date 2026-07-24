"""Broadband regularized VAR fitting with PDC/DTF validity diagnostics.

The VAR is fitted once to broadband epochs. Frequency-specific PDC and DTF are
then evaluated from that same model, avoiding a separate VAR fit after each
narrow band-pass filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

import numpy as np
from scipy import signal, stats


BANDS = {"delta": (1.0, 4.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}


@dataclass(frozen=True)
class BroadbandVarModel:
    """One fitted broadband VAR; ``A[lag, target, source]``."""

    lag_matrices: np.ndarray
    target_sfreq: float
    order: int
    ridge_alpha: float
    channel_mean: np.ndarray
    channel_std: np.ndarray
    diagnostics: dict[str, float | int | bool]


def validate_epochs(epochs: np.ndarray) -> np.ndarray:
    """Validate epochs with shape ``(epochs, channels, samples)``."""

    array = np.asarray(epochs, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"Expected epochs x channels x samples, got {array.shape}")
    if min(array.shape[:2]) < 2 or array.shape[2] < 16:
        raise ValueError(f"Insufficient broadband data for VAR: {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Broadband VAR input contains NaN or infinite values")
    return array


def resample_epochs(
    epochs: np.ndarray, fs: float, target_sfreq: float = 100.0
) -> np.ndarray:
    """Resample broadband epochs while keeping epoch boundaries intact."""

    array = validate_epochs(epochs)
    if np.isclose(fs, target_sfreq):
        return array.copy()
    ratio = Fraction(float(target_sfreq) / float(fs)).limit_denominator(1000)
    return signal.resample_poly(
        array, ratio.numerator, ratio.denominator, axis=-1
    )


def prepare_broadband_epochs(
    epochs: np.ndarray,
    fs: float,
    *,
    target_sfreq: float = 100.0,
    broadband_fmin: float = 1.0,
    broadband_fmax: float = 45.0,
) -> np.ndarray:
    """Resample and apply one broad stationarity/anti-alias analysis filter."""

    resampled = resample_epochs(epochs, fs, target_sfreq)
    nyquist = target_sfreq / 2.0
    if not 0 < broadband_fmin < broadband_fmax < nyquist:
        raise ValueError(
            f"Invalid broadband range {broadband_fmin}-{broadband_fmax} Hz "
            f"for target sfreq {target_sfreq}"
        )
    sos = signal.butter(
        4,
        [broadband_fmin, broadband_fmax],
        btype="bandpass",
        fs=target_sfreq,
        output="sos",
    )
    return signal.sosfiltfilt(sos, resampled, axis=-1)


def _standardization(
    epochs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(epochs, axis=(0, 2), keepdims=True)
    std = np.std(epochs, axis=(0, 2), keepdims=True)
    if np.any(std <= np.finfo(float).eps):
        bad = np.flatnonzero(std.reshape(-1) <= np.finfo(float).eps)
        raise ValueError(f"Constant broadband channels cannot enter VAR: {bad.tolist()}")
    return (epochs - mean) / std, mean.reshape(-1), std.reshape(-1)


def _apply_standardization(
    epochs: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return (epochs - mean[None, :, None]) / std[None, :, None]


def lag_design(epochs: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Construct a lag design without crossing epoch boundaries."""

    if order < 1 or epochs.shape[-1] <= order + 2:
        raise ValueError(f"VAR order {order} is incompatible with {epochs.shape[-1]} samples")
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


def _fit_coefficients(
    standardized_epochs: np.ndarray, order: int, ridge_alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, Y = lag_design(standardized_epochs, order)
    regularized = X.T @ X + float(ridge_alpha) * np.eye(X.shape[1])
    coefficients = np.linalg.solve(regularized, X.T @ Y)
    residuals = Y - X @ coefficients
    lag_matrices = _lag_matrices_from_coefficients(
        coefficients, standardized_epochs.shape[1], order
    )
    return lag_matrices, residuals, regularized


def _lag_matrices_from_coefficients(
    coefficients: np.ndarray, n_channels: int, order: int
) -> np.ndarray:
    """Convert predictor-by-target coefficients to ``lag,target,source``."""

    return np.stack(
        [
            coefficients[lag * n_channels : (lag + 1) * n_channels].T
            for lag in range(order)
        ]
    )


def _spectral_radius(lag_matrices: np.ndarray) -> float:
    order, n_channels, _ = lag_matrices.shape
    companion = np.zeros((order * n_channels, order * n_channels), dtype=float)
    companion[:n_channels] = np.concatenate(list(lag_matrices), axis=1)
    if order > 1:
        companion[n_channels:, :-n_channels] = np.eye((order - 1) * n_channels)
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def _residual_autocorrelation(
    residuals: np.ndarray,
    *,
    n_epochs: int,
    samples_per_epoch: int,
    max_lag: int,
) -> tuple[float, float, float]:
    shaped = residuals.reshape(n_epochs, samples_per_epoch, residuals.shape[1])
    correlations: list[float] = []
    lag1: list[float] = []
    for lag in range(1, max_lag + 1):
        for channel in range(shaped.shape[2]):
            left = shaped[:, :-lag, channel].reshape(-1)
            right = shaped[:, lag:, channel].reshape(-1)
            if np.std(left) <= np.finfo(float).eps or np.std(right) <= np.finfo(float).eps:
                correlation = 0.0
            else:
                correlation = float(np.corrcoef(left, right)[0, 1])
            correlations.append(abs(correlation))
            if lag == 1:
                lag1.append(abs(correlation))
    return float(np.mean(lag1)), float(np.mean(correlations)), float(np.max(correlations))


def fit_broadband_var(
    epochs: np.ndarray,
    *,
    target_sfreq: float,
    order: int,
    ridge_alpha: float,
) -> BroadbandVarModel:
    """Fit one stable-diagnostic broadband ridge VAR to already resampled epochs."""

    data = validate_epochs(epochs)
    standardized, mean, std = _standardization(data)
    lag_matrices, residuals, regularized = _fit_coefficients(
        standardized, order, ridge_alpha
    )
    lag1, autocorrelation_mean, autocorrelation_max = _residual_autocorrelation(
        residuals,
        n_epochs=data.shape[0],
        samples_per_epoch=data.shape[-1] - order,
        max_lag=min(10, order),
    )
    radius = _spectral_radius(lag_matrices)
    diagnostics: dict[str, float | int | bool] = {
        "n_epochs": int(data.shape[0]),
        "n_channels": int(data.shape[1]),
        "n_observations": int(residuals.shape[0]),
        "n_predictors": int(data.shape[1] * order),
        "spectral_radius": radius,
        "stable": bool(radius < 1.0),
        "residual_lag1_abs_mean": lag1,
        "residual_autocorrelation_abs_mean": autocorrelation_mean,
        "residual_autocorrelation_abs_max": autocorrelation_max,
        "residual_whiteness_pass": bool(lag1 < 0.1 and autocorrelation_mean < 0.1),
        "regularized_condition_number": float(np.linalg.cond(regularized)),
    }
    return BroadbandVarModel(
        lag_matrices=lag_matrices,
        target_sfreq=float(target_sfreq),
        order=int(order),
        ridge_alpha=float(ridge_alpha),
        channel_mean=mean,
        channel_std=std,
        diagnostics=diagnostics,
    )


def select_var_hyperparameters(
    epochs: np.ndarray,
    *,
    target_sfreq: float = 100.0,
    lag_ms_candidates: Iterable[float] = (50.0, 100.0, 200.0),
    ridge_candidates: Iterable[float] = (1.0, 10.0, 100.0),
    validation_fraction: float = 0.25,
) -> tuple[int, float, list[dict[str, float | int | bool]]]:
    """Select order/ridge by held-out-epoch forecasting without cohort labels."""

    data = validate_epochs(epochs)
    if data.shape[0] < 4:
        raise ValueError("At least four epochs are required for within-subject VAR selection")
    n_validation = max(1, int(round(data.shape[0] * validation_fraction)))
    n_validation = min(n_validation, data.shape[0] - 2)
    train = data[:-n_validation]
    validation = data[-n_validation:]
    train_standardized, mean, std = _standardization(train)
    validation_standardized = _apply_standardization(validation, mean, std)
    rows: list[dict[str, float | int | bool]] = []
    for lag_ms in lag_ms_candidates:
        order = max(1, int(round(float(lag_ms) * target_sfreq / 1000.0)))
        X_train, Y_train = lag_design(train_standardized, order)
        X_validation, Y_validation = lag_design(validation_standardized, order)
        eigenvalues, eigenvectors = np.linalg.eigh(X_train.T @ X_train)
        projected_xty = eigenvectors.T @ (X_train.T @ Y_train)
        for ridge_alpha in ridge_candidates:
            coefficients = eigenvectors @ (
                projected_xty / (eigenvalues[:, None] + float(ridge_alpha))
            )
            lag_matrices = _lag_matrices_from_coefficients(
                coefficients, data.shape[1], order
            )
            coefficient_matrix = np.concatenate(list(lag_matrices), axis=1).T
            prediction = X_validation @ coefficient_matrix
            validation_mse = float(np.mean(np.square(Y_validation - prediction)))
            radius = _spectral_radius(lag_matrices)
            rows.append(
                {
                    "lag_ms": float(lag_ms),
                    "order": int(order),
                    "ridge_alpha": float(ridge_alpha),
                    "validation_mse": validation_mse,
                    "spectral_radius": radius,
                    "stable": bool(radius < 1.0),
                }
            )
    stable_rows = [row for row in rows if row["stable"]]
    candidates = stable_rows or rows
    best = min(
        candidates,
        key=lambda row: (
            float(row["validation_mse"]),
            int(row["order"]),
            -float(row["ridge_alpha"]),
        ),
    )
    return int(best["order"]), float(best["ridge_alpha"]), rows


def screen_full_data_stability(
    epochs: np.ndarray,
    candidates: list[dict[str, float | int | bool]],
) -> list[dict[str, float | int | bool]]:
    """Add full-data stability results using one factorization per VAR order."""

    data = validate_epochs(epochs)
    standardized, _, _ = _standardization(data)
    screened = [dict(candidate) for candidate in candidates]
    for order in sorted({int(candidate["order"]) for candidate in screened}):
        X, Y = lag_design(standardized, order)
        eigenvalues, eigenvectors = np.linalg.eigh(X.T @ X)
        projected_xty = eigenvectors.T @ (X.T @ Y)
        for candidate in screened:
            if int(candidate["order"]) != order:
                continue
            ridge_alpha = float(candidate["ridge_alpha"])
            coefficients = eigenvectors @ (
                projected_xty / (eigenvalues[:, None] + ridge_alpha)
            )
            lag_matrices = _lag_matrices_from_coefficients(
                coefficients, data.shape[1], order
            )
            radius = _spectral_radius(lag_matrices)
            candidate["full_spectral_radius"] = radius
            candidate["full_stable"] = bool(radius < 1.0)
    return screened


def frequency_connectivity(
    model: BroadbandVarModel,
    fmin: float,
    fmax: float,
    *,
    frequencies_per_band: int = 24,
) -> dict[str, np.ndarray]:
    """Derive source-by-target PDC and DTF from a broadband VAR."""

    n_channels = model.lag_matrices.shape[1]
    identity = np.eye(n_channels, dtype=complex)
    frequencies = np.linspace(max(float(fmin), 0.01), float(fmax), frequencies_per_band)
    pdc_values = []
    dtf_values = []
    for frequency in frequencies:
        phase = np.exp(
            -2j
            * np.pi
            * frequency
            * np.arange(1, model.order + 1)
            / model.target_sfreq
        )
        ar_frequency = identity - np.sum(
            model.lag_matrices * phase[:, None, None], axis=0
        )
        # Conventional matrices are target x source; persisted matrices are
        # source x target to match the project's directed adjacency contract.
        pdc_target_source = np.abs(ar_frequency) / np.sqrt(
            np.maximum(
                np.sum(np.square(np.abs(ar_frequency)), axis=0, keepdims=True),
                1e-15,
            )
        )
        transfer = np.linalg.inv(ar_frequency)
        dtf_target_source = np.abs(transfer) / np.sqrt(
            np.maximum(
                np.sum(np.square(np.abs(transfer)), axis=1, keepdims=True),
                1e-15,
            )
        )
        pdc_values.append(pdc_target_source.T)
        dtf_values.append(dtf_target_source.T)
    output = {
        "pdc": np.mean(pdc_values, axis=0),
        "dtf": np.mean(dtf_values, axis=0),
    }
    for matrix in output.values():
        np.fill_diagonal(matrix, 0.0)
    return output


def all_band_connectivity(
    model: BroadbandVarModel,
) -> dict[str, dict[str, np.ndarray]]:
    """Return ``{method: {band: source_by_target_matrix}}``."""

    output = {"pdc": {}, "dtf": {}}
    for band, (fmin, fmax) in BANDS.items():
        matrices = frequency_connectivity(model, fmin, fmax)
        for method, matrix in matrices.items():
            output[method][band] = matrix
    return output


def time_reversal_diagnostics(
    forward: dict[str, dict[str, np.ndarray]],
    reversed_connectivity: dict[str, dict[str, np.ndarray]],
) -> dict[str, float]:
    """Measure whether directed asymmetry reverses after reversing time."""

    diagnostics: dict[str, float] = {}
    for method in sorted(forward):
        correlations = []
        agreements = []
        for band in BANDS:
            original_net = forward[method][band] - forward[method][band].T
            reversed_net = (
                reversed_connectivity[method][band]
                - reversed_connectivity[method][band].T
            )
            indices = np.triu_indices(original_net.shape[0], k=1)
            left = original_net[indices]
            right = -reversed_net[indices]
            if np.std(left) > 0 and np.std(right) > 0:
                correlations.append(float(stats.spearmanr(left, right).statistic))
            threshold = max(np.quantile(np.abs(left), 0.5), np.finfo(float).eps)
            informative = (np.abs(left) >= threshold) | (np.abs(right) >= threshold)
            if np.any(informative):
                agreements.append(
                    float(np.mean(np.sign(left[informative]) == np.sign(right[informative])))
                )
        diagnostics[f"{method}_time_reversal_spearman"] = float(np.mean(correlations))
        diagnostics[f"{method}_time_reversal_direction_agreement"] = float(
            np.mean(agreements)
        )
    return diagnostics


def fit_subject_connectivity(
    epochs: np.ndarray,
    fs: float,
    *,
    target_sfreq: float = 100.0,
    broadband_fmin: float = 1.0,
    broadband_fmax: float = 45.0,
    lag_ms_candidates: Iterable[float] = (50.0, 100.0, 200.0),
    ridge_candidates: Iterable[float] = (1.0, 10.0, 100.0, 1000.0),
) -> tuple[
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, float | int | bool | list[dict[str, float | int | bool]]],
]:
    """Fit full/odd/even broadband VARs and a time-reversed diagnostic model."""

    resampled = prepare_broadband_epochs(
        epochs,
        fs,
        target_sfreq=target_sfreq,
        broadband_fmin=broadband_fmin,
        broadband_fmax=broadband_fmax,
    )
    _, _, selection = select_var_hyperparameters(
        resampled,
        target_sfreq=target_sfreq,
        lag_ms_candidates=lag_ms_candidates,
        ridge_candidates=ridge_candidates,
    )
    selection = screen_full_data_stability(resampled, selection)
    ranked_candidates = sorted(
        selection,
        key=lambda row: (
            not bool(row["stable"]),
            float(row["validation_mse"]),
            int(row["order"]),
            -float(row["ridge_alpha"]),
        ),
    )
    stable_candidates = [
        candidate for candidate in ranked_candidates if bool(candidate["full_stable"])
    ]
    if not stable_candidates:
        raise ValueError("No candidate produced a stable full-data broadband VAR")
    selected = stable_candidates[0]
    order = int(selected["order"])
    ridge_alpha = float(selected["ridge_alpha"])
    full_model = fit_broadband_var(
        resampled,
        target_sfreq=target_sfreq,
        order=order,
        ridge_alpha=ridge_alpha,
    )

    split_epochs = {
        "full": resampled,
        "odd": resampled[::2],
        "even": resampled[1::2],
    }
    if min(value.shape[0] for value in split_epochs.values()) < 2:
        raise ValueError("Insufficient odd/even epochs for broadband VAR reliability")
    split_connectivity: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    split_models: dict[str, BroadbandVarModel] = {}
    for split, split_data in split_epochs.items():
        model = (
            full_model
            if split == "full"
            else fit_broadband_var(
                split_data,
                target_sfreq=target_sfreq,
                order=order,
                ridge_alpha=ridge_alpha,
            )
        )
        split_models[split] = model
        split_connectivity[split] = all_band_connectivity(model)
    reversed_model = fit_broadband_var(
        resampled[..., ::-1],
        target_sfreq=target_sfreq,
        order=order,
        ridge_alpha=ridge_alpha,
    )
    reversed_connectivity = all_band_connectivity(reversed_model)
    diagnostics = {
        "selected_order": order,
        "selected_lag_ms": 1000.0 * order / target_sfreq,
        "selected_ridge_alpha": ridge_alpha,
        "target_sfreq": float(target_sfreq),
        "broadband_fmin": float(broadband_fmin),
        "broadband_fmax": float(broadband_fmax),
        **full_model.diagnostics,
        "odd_spectral_radius": split_models["odd"].diagnostics["spectral_radius"],
        "odd_stable": split_models["odd"].diagnostics["stable"],
        "even_spectral_radius": split_models["even"].diagnostics["spectral_radius"],
        "even_stable": split_models["even"].diagnostics["stable"],
        **{
            f"reversed_{key}": value
            for key, value in reversed_model.diagnostics.items()
            if key
            in {
                "spectral_radius",
                "stable",
                "residual_lag1_abs_mean",
                "residual_autocorrelation_abs_mean",
                "residual_whiteness_pass",
            }
        },
        **time_reversal_diagnostics(
            split_connectivity["full"], reversed_connectivity
        ),
        "selection_grid": selection,
    }
    return split_connectivity, diagnostics
