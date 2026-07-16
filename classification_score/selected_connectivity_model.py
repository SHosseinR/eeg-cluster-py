"""Train and use the selected connectivity-based probability classifier.

The default selection is TD-BRAIN natural-scale coherence edges with tuned
logistic regression. This favors the larger validation dataset and provides a
calibrated Patient probability while retaining the full edge space.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import joblib
import numpy as np

try:
    from .connectivity_benchmark import (
        BANDS,
        METHOD_DIRECTION,
        _compute_profile_caches,
        _load_benchmark_features,
    )
    from .connectivity_methods import compute_fourier_connectivity, edge_vector
    from .connectivity_sensitivity import connectivity_transformations
    from .modeling import fit_tuned_model
except ImportError:
    from connectivity_benchmark import (
        BANDS,
        METHOD_DIRECTION,
        _compute_profile_caches,
        _load_benchmark_features,
    )
    from connectivity_methods import compute_fourier_connectivity, edge_vector
    from connectivity_sensitivity import connectivity_transformations
    from modeling import fit_tuned_model


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "models" / "tdbrain_coherence_logistic.joblib"
)


@dataclass
class ConnectivityClassifierBundle:
    """Fitted classifier plus its complete connectivity feature contract."""

    estimator: Any
    profile: str
    method: str
    transformation: str
    model_name: str
    bands: dict[str, tuple[float, float]]
    channel_names: list[str]
    directed: bool
    n_features: int
    best_params: dict[str, Any]
    evidence: dict[str, Any]


def _load_subject_payload(path: Path) -> dict[str, Any]:
    value = np.load(path, allow_pickle=True).item()
    if not isinstance(value, dict) or "filtered_epochs" not in value:
        raise ValueError(f"Not a saved filtered-epoch subject payload: {path}")
    return value


def _transform_single(vector: np.ndarray, transformation: str) -> np.ndarray:
    transformed = connectivity_transformations(np.asarray(vector, dtype=float)[None, :])
    if transformation not in transformed:
        raise ValueError(f"Unknown connectivity transformation: {transformation}")
    return transformed[transformation]


def vectorize_connectivity_matrices(
    matrices_by_band: Mapping[str, np.ndarray],
    *,
    directed: bool,
    transformation: str,
) -> np.ndarray:
    """Convert band matrices to the exact one-row classifier feature matrix."""

    missing = [band for band in BANDS if band not in matrices_by_band]
    if missing:
        raise ValueError(f"Missing connectivity bands: {missing}")
    vectors = []
    shape = None
    for band in BANDS:
        matrix = np.asarray(matrices_by_band[band], dtype=float)
        if shape is None:
            shape = matrix.shape
        if matrix.shape != shape or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Inconsistent square matrix for {band}: {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"Non-finite connectivity values in {band}")
        vectors.append(edge_vector(matrix, directed=directed))
    return _transform_single(np.concatenate(vectors), transformation)


def connectivity_from_filtered_epochs(
    filtered_epochs: Mapping[str, np.ndarray], fs: float, method: str
) -> dict[str, np.ndarray]:
    """Compute the selected Fourier connectivity representation for one subject."""

    output = {}
    for band, (fmin, fmax) in BANDS.items():
        output[band] = compute_fourier_connectivity(
            np.asarray(filtered_epochs[band]),
            float(fs),
            fmin,
            fmax,
            methods=(method,),
        )[method]
    return output


def load_bundle(path: str | Path = DEFAULT_MODEL_PATH) -> ConnectivityClassifierBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, ConnectivityClassifierBundle):
        raise TypeError(f"Unexpected connectivity model bundle type: {type(bundle).__name__}")
    return bundle


def predict_patient_probability_from_matrices(
    bundle: ConnectivityClassifierBundle,
    matrices_by_band: Mapping[str, np.ndarray],
    *,
    channel_names: list[str] | None = None,
) -> float:
    """Return P(Patient) after validating matrix and channel contracts."""

    if channel_names is not None and list(channel_names) != bundle.channel_names:
        raise ValueError("Channel order does not match the fitted connectivity classifier")
    X = vectorize_connectivity_matrices(
        matrices_by_band,
        directed=bundle.directed,
        transformation=bundle.transformation,
    )
    if X.shape[1] != bundle.n_features:
        raise ValueError(f"Expected {bundle.n_features} features, got {X.shape[1]}")
    return float(bundle.estimator.predict_proba(X)[0, 1])


def predict_patient_probability_from_epochs(
    bundle: ConnectivityClassifierBundle,
    filtered_epochs: Mapping[str, np.ndarray],
    fs: float,
    *,
    channel_names: list[str] | None = None,
) -> float:
    matrices = connectivity_from_filtered_epochs(filtered_epochs, fs, bundle.method)
    return predict_patient_probability_from_matrices(
        bundle, matrices, channel_names=channel_names
    )


def train_selected_model(
    *,
    profile: str = "tdbrain",
    method: str = "coherence",
    transformation: str = "natural_edges",
    model_name: str = "logistic_l2",
    mode: str = "full",
    n_jobs: int = 1,
    output_path: Path = DEFAULT_MODEL_PATH,
) -> ConnectivityClassifierBundle:
    records, data_root = _compute_profile_caches(
        profile,
        ("fourier", "envelope"),
        n_jobs=n_jobs,
        resume=True,
        max_subjects_per_group=None,
    )
    matrices, y, _, _, _ = _load_benchmark_features(records, data_root, [method])
    X = connectivity_transformations(matrices[method])[transformation]
    estimator, best_params = fit_tuned_model(
        X,
        y,
        model_name=model_name,
        mode=mode,
        inner_splits=5,
        n_jobs=n_jobs,
    )
    first_group, first_subject, _ = records[0]
    payload = _load_subject_payload(
        data_root / "filtered_epochs" / first_group / f"{first_subject}.npy"
    )
    channels = list(payload.get("channel_names", payload.get("channels", [])))
    if not channels:
        raise ValueError("Saved subject payload has no authoritative channel order")
    evidence_root = Path("classification_score") / "results" / "connectivity_benchmark" / profile
    bundle = ConnectivityClassifierBundle(
        estimator=estimator,
        profile=profile,
        method=method,
        transformation=transformation,
        model_name=model_name,
        bands=dict(BANDS),
        channel_names=channels,
        directed=METHOD_DIRECTION[method],
        n_features=int(X.shape[1]),
        best_params=best_params,
        evidence={
            "classification_screen": str(
                evidence_root / "full_screen" / "classification_summary.csv"
            ),
            "topology_sensitivity": str(
                evidence_root / "coherence_topology" / "summary.csv"
            ),
            "age_sex_sensitivity": str(
                evidence_root / "age_sex_matched" / "summary.csv"
            ),
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    metadata = {
        key: value
        for key, value in bundle.__dict__.items()
        if key != "estimator"
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    return bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--profile", default="tdbrain")
    train.add_argument("--method", default="coherence")
    train.add_argument("--transformation", default="natural_edges")
    train.add_argument("--model", default="logistic_l2")
    train.add_argument("--mode", choices=["quick", "full"], default="full")
    train.add_argument("--n-jobs", type=int, default=1)
    train.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    predict = subparsers.add_parser("predict")
    predict.add_argument("subject_epochs", type=Path)
    predict.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "train":
        bundle = train_selected_model(
            profile=args.profile,
            method=args.method,
            transformation=args.transformation,
            model_name=args.model,
            mode=args.mode,
            n_jobs=args.n_jobs,
            output_path=args.output,
        )
        print(f"Saved {bundle.profile} {bundle.method} model to {args.output}")
        print(f"Best parameters: {bundle.best_params}")
        return 0
    bundle = load_bundle(args.model)
    payload = _load_subject_payload(args.subject_epochs)
    channels = list(payload.get("channel_names", payload.get("channels", [])))
    probability = predict_patient_probability_from_epochs(
        bundle,
        payload["filtered_epochs"],
        float(payload["fs"]),
        channel_names=channels,
    )
    print(json.dumps({"patient_probability": probability}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
