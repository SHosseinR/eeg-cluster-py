"""Validate cached analysis artifacts before optimization-only pipeline runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tomllib

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent


def _profile_path(config_name: str) -> Path:
    candidate = Path(config_name)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / "dataset_configs" / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Dataset config not found: {candidate}")
    return candidate


def _resolve_project_path(value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _load_npy_dict(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing cached {label}: {path}")
    value = np.load(path, allow_pickle=True).item()
    if not isinstance(value, dict):
        raise ValueError(f"Cached {label} is not a dictionary: {path}")
    return value


def validate_cached_analysis(config_name: str) -> dict[str, object]:
    """Validate cached matrices, metadata, epochs, and classifier contracts."""

    profile_path = _profile_path(config_name)
    with profile_path.open("rb") as stream:
        profile = tomllib.load(stream)
    optimization = profile.get("optimization", {})
    output_dir = _resolve_project_path(str(profile["output_directory"]))
    analysis_input_dir = _resolve_project_path(
        str(optimization.get("analysis_input_directory", output_dir))
    )
    data_dir = analysis_input_dir / "data"

    connectivity = _load_npy_dict(
        data_dir / "connectivity_matrices.npy", "connectivity matrices"
    )
    measures = _load_npy_dict(
        data_dir / "network_measures.npy", "network measures"
    )
    analysis_metadata = _load_npy_dict(
        data_dir / "analysis_metadata.npy", "analysis metadata"
    )
    epoch_index = _load_npy_dict(
        data_dir / "filtered_epochs_index.npy", "filtered-epoch index"
    )

    metadata_path = data_dir / "channel_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing cached channel metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as stream:
        channel_metadata = json.load(stream)
    channel_names = list(channel_metadata.get("channel_names", []))
    if not channel_names:
        raise ValueError(f"No channel order in {metadata_path}")

    method = str(profile.get("connectivity", {}).get("selected_method", "gc"))
    bands = tuple(analysis_metadata.get("frequency_bands", {}).keys())
    if not bands:
        raise ValueError("Analysis metadata does not contain frequency bands")
    if analysis_metadata.get("selected_method") != method:
        raise ValueError(
            "Selected connectivity method differs between the profile and "
            "cached analysis metadata"
        )

    cohort_counts: dict[str, int] = {}
    expected_shape = (len(channel_names), len(channel_names))
    for group in ("Healthy", "Patient"):
        if group not in connectivity or group not in measures or group not in epoch_index:
            raise ValueError(f"Cached artifacts are missing cohort {group!r}")
        connectivity_ids = {str(value) for value in connectivity[group]}
        measure_ids = {str(value) for value in measures[group]}
        index_ids = {str(value) for value in epoch_index[group]}
        if connectivity_ids != measure_ids or connectivity_ids != index_ids:
            raise ValueError(
                f"Subject alignment differs across cached {group} artifacts"
            )
        cohort_counts[group] = len(connectivity_ids)
        for subject_id, subject_methods in connectivity[group].items():
            if method not in subject_methods:
                raise ValueError(f"{group}/{subject_id} lacks method {method!r}")
            for band in bands:
                if band not in subject_methods[method]:
                    raise ValueError(f"{group}/{subject_id} lacks band {band!r}")
                matrix = np.asarray(subject_methods[method][band], dtype=float)
                if matrix.shape != expected_shape:
                    raise ValueError(
                        f"Matrix shape mismatch at {group}/{subject_id}/{band}: "
                        f"{matrix.shape} != {expected_shape}"
                    )
                if not np.all(np.isfinite(matrix)):
                    raise ValueError(
                        f"Non-finite matrix at {group}/{subject_id}/{band}"
                    )
                if method == "coh" and not np.allclose(
                    matrix, matrix.T, rtol=1e-5, atol=1e-8
                ):
                    raise ValueError(
                        f"Coherence matrix is not symmetric at {group}/{subject_id}/{band}"
                    )

        epoch_root = data_dir / "filtered_epochs" / group
        for subject_id in sorted(connectivity_ids):
            entry = epoch_index[group][subject_id]
            entry_channels = list(
                entry.get("channel_names", entry.get("channels", []))
            )
            if entry_channels != channel_names:
                raise ValueError(
                    f"Filtered-epoch index channel order differs at {group}/{subject_id}"
                )
            epoch_path = epoch_root / f"{subject_id}.npy"
            if not epoch_path.exists() or epoch_path.stat().st_size <= 0:
                raise FileNotFoundError(
                    f"Missing cached filtered epochs for {group}/{subject_id}: "
                    f"{epoch_path}"
                )

        # Loading two representative payloads verifies the object/array schema;
        # every subject file and indexed channel contract was checked above.
        sample_ids = sorted(connectivity_ids)
        for subject_id in dict.fromkeys((sample_ids[0], sample_ids[-1])):
            epoch_path = epoch_root / f"{subject_id}.npy"
            payload = np.load(epoch_path, allow_pickle=True).item()
            payload_channels = list(
                payload.get("channel_names", payload.get("channels", []))
            )
            if payload_channels != channel_names:
                raise ValueError(f"Epoch payload channel mismatch: {epoch_path}")
            filtered = payload.get("filtered_epochs", {})
            for band in bands:
                array = np.asarray(filtered.get(band))
                if (
                    array.ndim != 3
                    or array.shape[1] != len(channel_names)
                    or not np.all(np.isfinite(array))
                ):
                    raise ValueError(
                        f"Invalid cached epoch array for {subject_id}/{band}: "
                        f"{array.shape}"
                    )

    classifier_dir = analysis_input_dir / str(
        optimization.get(
            "classifier_model_directory",
            "data/connectivity_classifiers/models",
        )
    )
    from classification_score.band_connectivity_classifier import load_band_bundle

    for band in bands:
        bundle_path = classifier_dir / f"{band}_classifier.joblib"
        if not bundle_path.exists():
            raise FileNotFoundError(f"Missing cached classifier: {bundle_path}")
        bundle = load_band_bundle(bundle_path)
        if bundle.band != band or bundle.method != method:
            raise ValueError(f"Classifier band/method mismatch: {bundle_path}")
        if list(bundle.channel_names) != channel_names:
            raise ValueError(f"Classifier channel order mismatch: {bundle_path}")
        if not bundle.accepted_for_optimization:
            raise ValueError(f"Classifier did not pass its evidence gate: {bundle_path}")

    summary = {
        "profile": str(profile_path),
        "output_directory": str(output_dir),
        "analysis_input_directory": str(analysis_input_dir),
        "method": method,
        "bands": list(bands),
        "channels": len(channel_names),
        "cohorts": cohort_counts,
        "classifier_directory": str(classifier_dir),
    }
    print("Cached analysis validation passed:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True)
    args = parser.parse_args()
    validate_cached_analysis(args.dataset_config)


if __name__ == "__main__":
    main()
