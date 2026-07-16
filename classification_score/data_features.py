"""Load saved EEG artifacts and build subject-level classification features.

This module is deliberately self-contained: it reads the existing pipeline's
stable artifacts but never imports or modifies the main pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any

import numpy as np


GROUPS = ("Healthy", "Patient")
FEATURE_CACHE_VERSION = 2
COMMON_CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
)
LEGACY_CHANNEL_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


@dataclass(frozen=True)
class FeatureDataset:
    """Subject-level labels and named feature matrices."""

    profile: str
    subject_ids: np.ndarray
    groups: np.ndarray
    y: np.ndarray
    matrices: dict[str, np.ndarray]
    feature_names: dict[str, list[str]]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _profile_config(profile: str) -> dict[str, Any]:
    profile_name = profile if profile.endswith(".toml") else f"{profile}.toml"
    path = _repo_root() / "dataset_configs" / profile_name
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _results_root(profile: str) -> Path:
    configured = Path(_profile_config(profile)["output_directory"])
    return configured if configured.is_absolute() else (_repo_root() / configured).resolve()


def _load_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required upstream artifact is missing: {path}")
    value = np.load(path, allow_pickle=True)
    if value.shape != ():
        raise ValueError(f"Expected a saved dictionary at {path}, got shape {value.shape}")
    loaded = value.item()
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a dictionary at {path}, got {type(loaded).__name__}")
    return loaded


def _clean_channel_name(name: str) -> str:
    return str(name).split("-")[0].strip()


def _canonical_channel_name(name: str) -> str:
    clean = _clean_channel_name(name)
    return LEGACY_CHANNEL_ALIASES.get(clean, clean)


def _channel_side(name: str) -> str:
    clean = _clean_channel_name(name)
    digits = "".join(ch for ch in clean if ch.isdigit())
    if clean.lower().endswith("z") or not digits:
        return "M"
    return "L" if int(digits) % 2 else "R"


def _channel_region(name: str) -> str:
    clean = _clean_channel_name(name).upper()
    if clean.startswith(("FP", "AF", "F", "FC")):
        return "frontal"
    if clean.startswith(("CP", "P")):
        return "parietal"
    if clean.startswith("O"):
        return "occipital"
    if clean.startswith("T"):
        return "temporal"
    if clean.startswith("C"):
        return "central"
    return "other"


def _matrix_log_spd(matrix: np.ndarray, floor: float = 1e-7) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, floor)
    return (vectors * np.log(values)) @ vectors.T


def _vectorize_symmetric(matrix: np.ndarray) -> np.ndarray:
    rows, cols = np.triu_indices(matrix.shape[0])
    values = matrix[rows, cols].copy()
    values[rows != cols] *= np.sqrt(2.0)
    return values


def _subject_epoch_features(
    band_epochs: dict[str, np.ndarray], channels: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    bands = sorted(band_epochs)
    n_channels = len(channels)
    power_by_band: dict[str, np.ndarray] = {}
    channel_values: list[float] = []
    channel_names: list[str] = []
    covariance_values: list[float] = []
    covariance_names: list[str] = []
    common_covariance_values: list[float] = []
    common_covariance_names: list[str] = []

    per_band_stats: dict[str, dict[str, np.ndarray]] = {}
    eps = np.finfo(float).tiny
    for band in bands:
        epochs = np.asarray(band_epochs[band], dtype=np.float64)
        if epochs.ndim != 3 or epochs.shape[1] != n_channels:
            raise ValueError(
                f"Band {band!r} has shape {epochs.shape}; expected epochs x {n_channels} x samples"
            )
        epoch_power = np.mean(np.square(epochs), axis=-1)
        log_power = np.log10(np.maximum(epoch_power, eps))
        diff1 = np.diff(epochs, axis=-1)
        diff2 = np.diff(diff1, axis=-1)
        var0 = np.var(epochs, axis=-1)
        var1 = np.var(diff1, axis=-1)
        var2 = np.var(diff2, axis=-1)
        mobility = np.sqrt(np.divide(var1, var0, out=np.zeros_like(var1), where=var0 > 0))
        raw_complexity = np.sqrt(
            np.divide(var2, var1, out=np.zeros_like(var2), where=var1 > 0)
        )
        complexity = np.divide(
            raw_complexity,
            mobility,
            out=np.zeros_like(raw_complexity),
            where=mobility > 0,
        )
        per_band_stats[band] = {
            "log_power": np.median(log_power, axis=0),
            "log_power_iqr": np.percentile(log_power, 75, axis=0)
            - np.percentile(log_power, 25, axis=0),
            "mobility": np.median(mobility, axis=0),
            "complexity": np.median(complexity, axis=0),
        }
        power_by_band[band] = np.median(epoch_power, axis=0)

        flattened = epochs.transpose(1, 0, 2).reshape(n_channels, -1)
        flattened -= flattened.mean(axis=1, keepdims=True)
        covariance = flattened @ flattened.T / max(flattened.shape[1] - 1, 1)
        scale = np.sqrt(np.maximum(np.diag(covariance), eps))
        correlation = covariance / np.outer(scale, scale)
        correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
        correlation = 0.95 * correlation + 0.05 * np.eye(n_channels)
        log_correlation = _matrix_log_spd(correlation)
        covariance_values.extend(_vectorize_symmetric(log_correlation))
        tri_r, tri_c = np.triu_indices(n_channels)
        covariance_names.extend(
            f"logcorr_{band}_{channels[i]}__{channels[j]}" for i, j in zip(tri_r, tri_c)
        )
        canonical_positions = {
            _canonical_channel_name(channel): index for index, channel in enumerate(channels)
        }
        missing_common = [name for name in COMMON_CHANNELS if name not in canonical_positions]
        if missing_common:
            raise ValueError(f"Missing canonical channels needed for portable covariance: {missing_common}")
        common_index = [canonical_positions[name] for name in COMMON_CHANNELS]
        common_correlation = correlation[np.ix_(common_index, common_index)]
        common_log_correlation = _matrix_log_spd(common_correlation)
        common_covariance_values.extend(_vectorize_symmetric(common_log_correlation))
        common_r, common_c = np.triu_indices(len(COMMON_CHANNELS))
        common_covariance_names.extend(
            f"common_logcorr_{band}_{COMMON_CHANNELS[i]}__{COMMON_CHANNELS[j]}"
            for i, j in zip(common_r, common_c)
        )

    total_power = np.sum(np.vstack([power_by_band[band] for band in bands]), axis=0)
    for band in bands:
        relative = np.divide(
            power_by_band[band], total_power, out=np.zeros(n_channels), where=total_power > 0
        )
        stats = per_band_stats[band] | {"relative_power": relative}
        for stat_name in ("log_power", "relative_power", "log_power_iqr", "mobility", "complexity"):
            for channel, value in zip(channels, stats[stat_name]):
                channel_values.append(float(value))
                channel_names.append(f"{stat_name}_{band}_{channel}")

    roi_values: list[float] = []
    roi_names: list[str] = []
    regions = ("frontal", "central", "temporal", "parietal", "occipital")
    sides = ("L", "R", "M")
    channel_regions = np.array([_channel_region(name) for name in channels])
    channel_sides = np.array([_channel_side(name) for name in channels])
    for band in bands:
        relative = np.divide(
            power_by_band[band], total_power, out=np.zeros(n_channels), where=total_power > 0
        )
        stats = per_band_stats[band] | {"relative_power": relative}
        for stat_name in ("log_power", "relative_power", "log_power_iqr", "mobility", "complexity"):
            values = stats[stat_name]
            for region in regions:
                for side in sides:
                    mask = (channel_regions == region) & (channel_sides == side)
                    roi_values.append(float(np.mean(values[mask])) if np.any(mask) else np.nan)
                    roi_names.append(f"roi_{stat_name}_{band}_{region}_{side}")
                left = values[(channel_regions == region) & (channel_sides == "L")]
                right = values[(channel_regions == region) & (channel_sides == "R")]
                asymmetry = float(np.mean(left) - np.mean(right)) if left.size and right.size else np.nan
                roi_values.append(asymmetry)
                roi_names.append(f"roi_{stat_name}_{band}_{region}_LminusR")

    return (
        {
            "spectral_channel": np.asarray(channel_values, dtype=float),
            "spectral_roi": np.asarray(roi_values, dtype=float),
            "covariance_logcorr": np.asarray(covariance_values, dtype=float),
            "covariance_common_logcorr": np.asarray(common_covariance_values, dtype=float),
        },
        {
            "spectral_channel": channel_names,
            "spectral_roi": roi_names,
            "covariance_logcorr": covariance_names,
            "covariance_common_logcorr": common_covariance_names,
        },
    )


def _connectivity_features(
    by_method: dict[str, dict[str, np.ndarray]], channels: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    method = "gc" if "gc" in by_method else sorted(by_method)[0]
    by_band = by_method[method]
    n_channels = len(channels)
    edge_values: list[float] = []
    edge_names: list[str] = []
    topology_values: list[float] = []
    topology_names: list[str] = []
    off_diagonal = ~np.eye(n_channels, dtype=bool)
    for band in sorted(by_band):
        matrix = np.asarray(by_band[band], dtype=float)
        if matrix.shape != (n_channels, n_channels):
            raise ValueError(f"Connectivity {band!r} has incompatible shape {matrix.shape}")
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        edge_values.extend(matrix[off_diagonal])
        edge_names.extend(
            f"edge_{band}_{channels[i]}_to_{channels[j]}"
            for i in range(n_channels)
            for j in range(n_channels)
            if i != j
        )
        out_strength = matrix.sum(axis=1)
        in_strength = matrix.sum(axis=0)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        for label, values in (
            ("out_strength", out_strength),
            ("in_strength", in_strength),
            ("singular_value", singular_values),
        ):
            topology_values.extend(values)
            topology_names.extend(
                f"{label}_{band}_{channels[i] if label != 'singular_value' else i}"
                for i in range(n_channels)
            )
    return (
        {
            "connectivity_edges": np.asarray(edge_values, dtype=float),
            "connectivity_topology": np.asarray(topology_values, dtype=float),
        },
        {
            "connectivity_edges": edge_names,
            "connectivity_topology": topology_names,
        },
    )


def _graph_features(
    by_method: dict[str, dict[str, dict[str, float]]]
) -> tuple[np.ndarray, list[str]]:
    method = "gc" if "gc" in by_method else sorted(by_method)[0]
    values: list[float] = []
    names: list[str] = []
    for band in sorted(by_method[method]):
        measures = by_method[method][band]
        for measure in sorted(measures):
            value = np.asarray(measures[measure], dtype=float)
            if value.size != 1:
                continue
            values.append(float(value.reshape(-1)[0]))
            names.append(f"graph_{band}_{measure}")
    return np.asarray(values, dtype=float), names


def _save_cache(path: Path, dataset: FeatureDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "profile": np.asarray(dataset.profile),
        "subject_ids": dataset.subject_ids,
        "groups": dataset.groups,
        "y": dataset.y,
        "feature_sets": np.asarray(sorted(dataset.matrices), dtype=object),
    }
    for name, matrix in dataset.matrices.items():
        payload[f"X__{name}"] = matrix
        payload[f"names__{name}"] = np.asarray(dataset.feature_names[name], dtype=object)
    np.savez_compressed(path, **payload)


def _load_cache(path: Path) -> FeatureDataset:
    cache = np.load(path, allow_pickle=True)
    feature_sets = cache["feature_sets"].tolist()
    return FeatureDataset(
        profile=str(cache["profile"].item()),
        subject_ids=cache["subject_ids"],
        groups=cache["groups"],
        y=cache["y"].astype(int),
        matrices={name: cache[f"X__{name}"] for name in feature_sets},
        feature_names={name: cache[f"names__{name}"].tolist() for name in feature_sets},
    )


def build_feature_dataset(profile: str, *, force: bool = False) -> FeatureDataset:
    """Build or load all standalone feature families for one dataset profile."""

    normalized_profile = profile.removesuffix(".toml")
    cache_path = (
        Path(__file__).resolve().parent
        / "cache"
        / f"{normalized_profile}_features_v{FEATURE_CACHE_VERSION}.npz"
    )
    if cache_path.exists() and not force:
        return _load_cache(cache_path)

    results_root = _results_root(normalized_profile)
    data_root = results_root / "data"
    index = _load_dict(data_root / "filtered_epochs_index.npy")
    connectivity = _load_dict(data_root / "connectivity_matrices.npy")
    measures = _load_dict(data_root / "network_measures.npy")

    subject_ids: list[str] = []
    group_labels: list[str] = []
    y: list[int] = []
    rows: dict[str, list[np.ndarray]] = {}
    names_ref: dict[str, list[str]] = {}

    for label, group in enumerate(GROUPS):
        common_subjects = sorted(
            set(index.get(group, {}))
            & set(connectivity.get(group, {}))
            & set(measures.get(group, {}))
        )
        if not common_subjects:
            raise ValueError(f"No aligned subjects found for cohort {group!r} in {results_root}")
        for subject_id in common_subjects:
            epoch_path = data_root / "filtered_epochs" / group / f"{subject_id}.npy"
            epoch_payload = _load_dict(epoch_path)
            channels = list(epoch_payload.get("channel_names", epoch_payload.get("channels", [])))
            if not channels:
                raise ValueError(f"No channel order saved for {subject_id} at {epoch_path}")
            epoch_features, epoch_names = _subject_epoch_features(
                epoch_payload["filtered_epochs"], channels
            )
            connectivity_features, connectivity_names = _connectivity_features(
                connectivity[group][subject_id], channels
            )
            graph_values, graph_names = _graph_features(measures[group][subject_id])
            all_features = epoch_features | connectivity_features | {"graph_global": graph_values}
            all_names = epoch_names | connectivity_names | {"graph_global": graph_names}
            for feature_set, values in all_features.items():
                rows.setdefault(feature_set, []).append(values)
                if feature_set not in names_ref:
                    names_ref[feature_set] = all_names[feature_set]
                elif names_ref[feature_set] != all_names[feature_set]:
                    raise ValueError(
                        f"Feature schema changed within {normalized_profile}: {feature_set}, {subject_id}"
                    )
            subject_ids.append(subject_id)
            group_labels.append(group)
            y.append(label)

    matrices = {name: np.vstack(values) for name, values in rows.items()}
    fused_definitions = {
        "eeg_fused": ("spectral_channel", "covariance_logcorr"),
        "eeg_portable_fused": ("spectral_roi", "covariance_common_logcorr"),
        "connectivity_fused": ("connectivity_edges", "connectivity_topology", "graph_global"),
        "all_fused": (
            "spectral_channel",
            "covariance_logcorr",
            "connectivity_edges",
            "connectivity_topology",
            "graph_global",
        ),
    }
    for fused_name, components in fused_definitions.items():
        matrices[fused_name] = np.hstack([matrices[name] for name in components])
        names_ref[fused_name] = [item for name in components for item in names_ref[name]]

    dataset = FeatureDataset(
        profile=normalized_profile,
        subject_ids=np.asarray(subject_ids, dtype=object),
        groups=np.asarray(group_labels, dtype=object),
        y=np.asarray(y, dtype=int),
        matrices=matrices,
        feature_names=names_ref,
    )
    for name, matrix in dataset.matrices.items():
        if matrix.shape[0] != dataset.y.size or matrix.shape[1] != len(dataset.feature_names[name]):
            raise AssertionError(f"Broken feature contract for {name}: {matrix.shape}")
        if np.any(np.isinf(matrix)):
            raise ValueError(f"Infinite values found in feature set {name}")
    _save_cache(cache_path, dataset)
    return dataset


def align_feature_matrix(
    matrix: np.ndarray,
    names: list[str],
    expected_names: list[str],
) -> np.ndarray:
    """Reorder a matrix to a saved schema and fail on missing features."""

    positions = {name: i for i, name in enumerate(names)}
    missing = [name for name in expected_names if name not in positions]
    if missing:
        raise ValueError(f"Missing {len(missing)} required features; first missing: {missing[:5]}")
    return np.asarray(matrix)[:, [positions[name] for name in expected_names]]
