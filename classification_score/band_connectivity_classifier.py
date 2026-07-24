"""Band-specific probabilistic classifiers for production connectivity matrices.

Each fitted model consumes exactly one frequency band's independent matrix
edges.  Healthy is class 0 and Patient is class 1 throughout the contract.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import RocCurveDisplay, roc_auc_score

from classification_score.modeling import fit_tuned_model, nested_oof_evaluate


GROUP_LABELS = {"Healthy": 0, "Patient": 1}
DEFAULT_BANDS = ("delta", "alpha", "beta")
DEFAULT_MODELS = ("logistic_l2", "linear_svm_sigmoid", "rbf_svm", "extra_trees")
_CANONICAL_MODULE = "classification_score.band_connectivity_classifier"
if __name__ == "__main__":
    # Keep joblib artifacts importable when this file is executed with `-m`.
    sys.modules[_CANONICAL_MODULE] = sys.modules[__name__]


@dataclass
class BandConnectivityClassifier:
    """Fitted estimator and the complete one-band matrix feature contract."""

    estimator: Any
    band: str
    method: str
    channel_names: list[str]
    model_name: str
    n_features: int
    best_params: dict[str, Any]
    cv_metrics: dict[str, Any]
    accepted_for_optimization: bool
    acceptance_reasons: list[str]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    ood_rms_threshold: float
    training_min: float
    training_max: float
    training_reference_z: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float)
    )
    manifold_rms_threshold: float = float("inf")
    local_change_rms_threshold: float = float("inf")
    fit_diagnostics: dict[str, float] = field(default_factory=dict)
    label_mapping: dict[str, int] = field(default_factory=lambda: dict(GROUP_LABELS))
    directed: bool = False
    feature_representation: str = "upper_triangle_natural_edges"

    def metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable description without the estimator."""

        values = asdict(self)
        values.pop("estimator", None)
        values["feature_mean"] = self.feature_mean.tolist()
        values["feature_scale"] = self.feature_scale.tolist()
        values["training_reference_z_shape"] = list(self.training_reference_z.shape)
        values.pop("training_reference_z", None)
        return values


BandConnectivityClassifier.__module__ = _CANONICAL_MODULE


def vectorize_band_matrix(matrix: np.ndarray, *, directed: bool = False) -> np.ndarray:
    """Return one row of natural-scale off-diagonal matrix features."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Connectivity matrix must be square; got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Connectivity matrix contains non-finite values")
    if directed:
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        return matrix[mask][None, :]
    if not np.allclose(matrix, matrix.T, rtol=1e-5, atol=1e-8):
        raise ValueError("Undirected connectivity classifier requires a symmetric matrix")
    return matrix[np.triu_indices(matrix.shape[0], k=1)][None, :]


def load_connectivity_features(
    connectivity_path: str | Path,
    *,
    method: str,
    bands: Sequence[str] = DEFAULT_BANDS,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Load a saved production dictionary into one independent X per band."""

    payload = np.load(Path(connectivity_path), allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise ValueError("Connectivity artifact is not a dictionary")
    features = {band: [] for band in bands}
    labels: list[int] = []
    subject_ids: list[str] = []
    expected_shape: tuple[int, int] | None = None
    for group, label in GROUP_LABELS.items():
        if group not in payload:
            raise ValueError(f"Missing cohort {group!r} in connectivity artifact")
        for subject_id, subject_methods in payload[group].items():
            if method not in subject_methods:
                raise ValueError(f"{subject_id} has no connectivity method {method!r}")
            matrices = subject_methods[method]
            for band in bands:
                if band not in matrices:
                    raise ValueError(f"{subject_id} has no {band!r} matrix")
                matrix = np.asarray(matrices[band], dtype=float)
                if expected_shape is None:
                    expected_shape = matrix.shape
                if matrix.shape != expected_shape:
                    raise ValueError(
                        f"Matrix shape changed at {subject_id}/{band}: "
                        f"{matrix.shape} != {expected_shape}"
                    )
                features[band].append(vectorize_band_matrix(matrix)[0])
            labels.append(label)
            subject_ids.append(str(subject_id))
    return (
        {band: np.asarray(rows, dtype=float) for band, rows in features.items()},
        np.asarray(labels, dtype=int),
        subject_ids,
    )


def evaluate_band_models(
    connectivity_path: str | Path,
    output_dir: str | Path,
    *,
    method: str = "coh",
    bands: Sequence[str] = DEFAULT_BANDS,
    models: Sequence[str] = DEFAULT_MODELS,
    mode: str = "quick",
    outer_splits: int = 5,
    repeats: int = 1,
    inner_splits: int = 3,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run leakage-safe nested CV separately for every band and model."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_by_band, y, subject_ids = load_connectivity_features(
        connectivity_path, method=method, bands=bands
    )
    summaries: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for band in bands:
        for model_name in models:
            summary, predictions = nested_oof_evaluate(
                X_by_band[band],
                y,
                feature_set=f"{method}_{band}_natural_edges",
                model_name=model_name,
                mode=mode,
                outer_splits=outer_splits,
                repeats=repeats,
                inner_splits=inner_splits,
                n_jobs=n_jobs,
                subject_ids=subject_ids,
            )
            summary["band"] = band
            summary["method"] = method
            summary["tuning_mode"] = mode
            summaries.append(summary)
            predictions["band"] = band
            predictions["method"] = method
            prediction_frames.append(predictions)
    summary_frame = pd.DataFrame(summaries).sort_values(
        ["band", "roc_auc", "brier"], ascending=[True, False, True]
    )
    summary_frame.to_csv(output_dir / "model_comparison.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "oof_predictions_all_models.csv", index=False
    )
    return summary_frame


def select_band_models(summary: pd.DataFrame) -> dict[str, str]:
    """Select a calibrated model without sacrificing meaningful discrimination.

    Families within 0.02 ROC AUC of the band leader are treated as practically
    tied; among them, Brier score and ECE decide because the downstream
    objective uses probability magnitude rather than only rank ordering.
    """

    preference = {
        "logistic_l2": 0,
        "linear_svm_sigmoid": 1,
        "rbf_svm": 2,
        "extra_trees": 3,
        "gcn": 4,
        "brainnetcnn": 5,
    }
    selected: dict[str, str] = {}
    for band, rows in summary.groupby("band", sort=False):
        ranked = rows.copy()
        best_auc = float(ranked["roc_auc"].max())
        ranked = ranked[ranked["roc_auc"] >= best_auc - 0.02].copy()
        ranked["model_preference"] = ranked["model"].map(preference).fillna(99)
        ranked = ranked.sort_values(
            ["brier", "ece_10", "roc_auc", "model_preference"],
            ascending=[True, True, False, True],
        )
        selected[str(band)] = str(ranked.iloc[0]["model"])
    return selected


def _acceptance(
    metrics: Mapping[str, Any],
    *,
    minimum_roc_auc: float = 0.75,
    minimum_balanced_accuracy: float = 0.70,
    maximum_brier: float = 0.20,
) -> tuple[bool, list[str]]:
    """Apply the predeclared minimum evidence gate for optimization use."""

    checks = {
        f"ROC AUC >= {minimum_roc_auc:.3g}": (
            float(metrics["roc_auc"]) >= minimum_roc_auc
        ),
        f"balanced accuracy >= {minimum_balanced_accuracy:.3g}": (
            float(metrics["balanced_accuracy"]) >= minimum_balanced_accuracy
        ),
        f"Brier score <= {maximum_brier:.3g}": (
            float(metrics["brier"]) <= maximum_brier
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return not failed, failed


def train_band_bundles(
    connectivity_path: str | Path,
    channel_metadata_path: str | Path,
    output_dir: str | Path,
    *,
    method: str = "coh",
    bands: Sequence[str] = DEFAULT_BANDS,
    selected_models: Mapping[str, str] | None = None,
    comparison_path: str | Path | None = None,
    mode: str = "full",
    inner_splits: int = 5,
    n_jobs: int = 1,
    minimum_roc_auc: float = 0.75,
    minimum_balanced_accuracy: float = 0.70,
    maximum_brier: float = 0.20,
) -> dict[str, BandConnectivityClassifier]:
    """Fit and save one deployable classifier bundle for every band."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_by_band, y, _ = load_connectivity_features(
        connectivity_path, method=method, bands=bands
    )
    channel_metadata = json.loads(Path(channel_metadata_path).read_text(encoding="utf-8"))
    channel_names = list(channel_metadata.get("channel_names", []))
    if not channel_names:
        raise ValueError("Channel metadata contains no authoritative channel_names")
    expected_features = len(channel_names) * (len(channel_names) - 1) // 2
    comparison = (
        pd.read_csv(comparison_path) if comparison_path is not None else pd.DataFrame()
    )
    if selected_models is None:
        if comparison.empty:
            raise ValueError("Provide selected_models or a model-comparison CSV")
        selected_models = select_band_models(comparison)
    bundles: dict[str, BandConnectivityClassifier] = {}
    manifest: dict[str, Any] = {
        "method": method,
        "label_mapping": GROUP_LABELS,
        "acceptance_thresholds": {
            "minimum_roc_auc": minimum_roc_auc,
            "minimum_balanced_accuracy": minimum_balanced_accuracy,
            "maximum_brier": maximum_brier,
        },
        "bands": {},
    }
    for band in bands:
        X = X_by_band[band]
        if X.shape[1] != expected_features:
            raise ValueError(
                f"{band} has {X.shape[1]} features, expected {expected_features} "
                f"for {len(channel_names)} channels"
            )
        model_name = selected_models[band]
        rows = comparison[
            (comparison["band"] == band) & (comparison["model"] == model_name)
        ]
        if rows.empty:
            raise ValueError(f"No CV evidence for selected {band}/{model_name}")
        cv_metrics = rows.iloc[0].to_dict()
        validated_mode = cv_metrics.get("tuning_mode")
        if isinstance(validated_mode, str) and validated_mode and validated_mode != mode:
            raise ValueError(
                f"Deployment tuning mode {mode!r} differs from validated mode "
                f"{validated_mode!r} for {band}"
            )
        accepted, failed = _acceptance(
            cv_metrics,
            minimum_roc_auc=minimum_roc_auc,
            minimum_balanced_accuracy=minimum_balanced_accuracy,
            maximum_brier=maximum_brier,
        )
        estimator, best_params = fit_tuned_model(
            X,
            y,
            model_name=model_name,
            mode=mode,
            inner_splits=inner_splits,
            n_jobs=n_jobs,
        )
        fitted_probability = estimator.predict_proba(X)[:, 1]
        fitted_auc = float(roc_auc_score(y, fitted_probability))
        healthy_median = float(np.median(fitted_probability[y == 0]))
        patient_median = float(np.median(fitted_probability[y == 1]))
        if fitted_auc < 0.5 or patient_median <= healthy_median:
            raise ValueError(
                f"Fitted {band}/{model_name} violates the probability label contract: "
                f"AUC={fitted_auc:.3f}, Healthy median={healthy_median:.3f}, "
                f"Patient median={patient_median:.3f}"
            )
        mean = np.mean(X, axis=0)
        scale = np.std(X, axis=0, ddof=1)
        scale = np.where(scale > 1e-8, scale, 1.0)
        rms = np.sqrt(np.mean(((X - mean) / scale) ** 2, axis=1))
        reference_z = (X - mean) / scale
        pairwise_rms = np.sqrt(
            np.mean((reference_z[:, None, :] - reference_z[None, :, :]) ** 2, axis=2)
        )
        np.fill_diagonal(pairwise_rms, np.inf)
        nearest_neighbor_rms = np.min(pairwise_rms, axis=1)
        bundle = BandConnectivityClassifier(
            estimator=estimator,
            band=band,
            method=method,
            channel_names=channel_names,
            model_name=model_name,
            n_features=int(X.shape[1]),
            best_params=best_params,
            cv_metrics=cv_metrics,
            accepted_for_optimization=accepted,
            acceptance_reasons=failed,
            feature_mean=mean,
            feature_scale=scale,
            ood_rms_threshold=float(np.quantile(rms, 0.995)),
            training_min=float(np.min(X)),
            training_max=float(np.max(X)),
            training_reference_z=reference_z,
            manifold_rms_threshold=float(np.quantile(nearest_neighbor_rms, 0.995)),
            local_change_rms_threshold=float(np.quantile(nearest_neighbor_rms, 0.95)),
            fit_diagnostics={
                "training_auc": fitted_auc,
                "healthy_probability_median": healthy_median,
                "patient_probability_median": patient_median,
            },
        )
        path = output_dir / f"{band}_classifier.joblib"
        joblib.dump(bundle, path)
        path.with_suffix(".json").write_text(
            json.dumps(bundle.metadata(), indent=2, default=str), encoding="utf-8"
        )
        bundles[band] = bundle
        manifest["bands"][band] = {
            "model_path": path.name,
            "model": model_name,
            "accepted_for_optimization": accepted,
            "failed_acceptance_checks": failed,
            "cv": cv_metrics,
        }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return bundles


def _save_selected_diagnostics(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot held-out probability distributions, ROC, and calibration per band."""

    bands = list(summary["band"])
    fig, axes = plt.subplots(len(bands), 3, figsize=(15, 4.2 * len(bands)), squeeze=False)
    for row, band in enumerate(bands):
        part = predictions[predictions["band"] == band]
        averaged = part.groupby(["subject_id", "y_true"], as_index=False)[
            "patient_probability"
        ].mean()
        for label, name, color in ((0, "Healthy", "#2a9d8f"), (1, "Patient", "#e76f51")):
            values = averaged.loc[averaged["y_true"] == label, "patient_probability"]
            axes[row, 0].hist(values, bins=15, alpha=0.58, label=name, color=color)
        axes[row, 0].axvline(0.5, color="black", linestyle="--", linewidth=1)
        axes[row, 0].set_title(f"{band}: held-out probabilities")
        axes[row, 0].set_xlabel("P(Patient)")
        axes[row, 0].legend()

        RocCurveDisplay.from_predictions(
            averaged["y_true"], averaged["patient_probability"], ax=axes[row, 1]
        )
        axes[row, 1].set_title(f"{band}: ROC")

        observed, predicted = calibration_curve(
            averaged["y_true"], averaged["patient_probability"], n_bins=8, strategy="quantile"
        )
        axes[row, 2].plot(predicted, observed, "o-", label="model")
        axes[row, 2].plot([0, 1], [0, 1], "k--", linewidth=1, label="ideal")
        axes[row, 2].set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted", ylabel="Observed")
        axes[row, 2].set_title(f"{band}: calibration")
        axes[row, 2].legend()
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_band_classifier_pipeline(
    connectivity_path: str | Path,
    channel_metadata_path: str | Path,
    output_dir: str | Path,
    *,
    method: str = "coh",
    bands: Sequence[str] = DEFAULT_BANDS,
    models: Sequence[str] = DEFAULT_MODELS,
    screen_repeats: int = 1,
    validation_repeats: int = 5,
    n_jobs: int = 1,
    minimum_roc_auc: float = 0.75,
    minimum_balanced_accuracy: float = 0.70,
    maximum_brier: float = 0.20,
) -> pd.DataFrame:
    """Compare, validate, fit, and report independent band classifiers."""

    output_dir = Path(output_dir)
    screen_dir = output_dir / "screen"
    screen = evaluate_band_models(
        connectivity_path,
        screen_dir,
        method=method,
        bands=bands,
        models=models,
        mode="quick",
        repeats=screen_repeats,
        n_jobs=n_jobs,
    )
    selected = select_band_models(screen)
    validation_summaries: list[pd.DataFrame] = []
    validation_predictions: list[pd.DataFrame] = []
    for band in bands:
        band_dir = output_dir / "validation" / band
        band_summary = evaluate_band_models(
            connectivity_path,
            band_dir,
            method=method,
            bands=[band],
            models=[selected[band]],
            mode="quick",
            repeats=validation_repeats,
            n_jobs=n_jobs,
        )
        validation_summaries.append(band_summary)
        validation_predictions.append(pd.read_csv(band_dir / "oof_predictions_all_models.csv"))
    validation = pd.concat(validation_summaries, ignore_index=True)
    predictions = pd.concat(validation_predictions, ignore_index=True)
    validation.to_csv(output_dir / "classification_summary_by_band_connectivity.csv", index=False)
    predictions.to_csv(output_dir / "oof_predictions_selected_models.csv", index=False)

    train_band_bundles(
        connectivity_path,
        channel_metadata_path,
        output_dir / "models",
        method=method,
        bands=bands,
        selected_models=selected,
        comparison_path=output_dir / "classification_summary_by_band_connectivity.csv",
        mode="quick",
        n_jobs=n_jobs,
        minimum_roc_auc=minimum_roc_auc,
        minimum_balanced_accuracy=minimum_balanced_accuracy,
        maximum_brier=maximum_brier,
    )
    for band in bands:
        part = predictions[(predictions["band"] == band) & (predictions["y_true"] == 1)]
        ranking = part.groupby("subject_id", as_index=False)["patient_probability"].mean()
        ranking = ranking.sort_values(
            ["patient_probability", "subject_id"], ascending=[True, True]
        ).reset_index(drop=True)
        ranking.insert(1, "rank", np.arange(1, len(ranking) + 1))
        ranking["patient_probability_percentile"] = ranking["patient_probability"].rank(
            pct=True, method="average"
        )
        ranking.to_csv(output_dir / f"classifier_patient_ranking_{band}.csv", index=False)
    _save_selected_diagnostics(
        predictions,
        validation,
        output_dir / "band_classifier_diagnostics.png",
    )
    return validation


def load_band_bundle(path: str | Path) -> BandConnectivityClassifier:
    """Load and type-check a saved band classifier."""

    bundle = joblib.load(path)
    if not isinstance(bundle, BandConnectivityClassifier):
        raise TypeError(f"Unexpected classifier bundle type: {type(bundle).__name__}")
    return bundle


def matrix_ood_rms(bundle: BandConnectivityClassifier, matrix: np.ndarray) -> float:
    """Return standardized RMS distance from the training edge distribution."""

    X = vectorize_band_matrix(matrix, directed=bundle.directed)
    if X.shape[1] != bundle.n_features:
        raise ValueError(f"Expected {bundle.n_features} features, got {X.shape[1]}")
    return float(np.sqrt(np.mean(((X[0] - bundle.feature_mean) / bundle.feature_scale) ** 2)))


def matrix_manifold_rms(bundle: BandConnectivityClassifier, matrix: np.ndarray) -> float:
    """Return distance to the nearest actually observed training subject."""

    X = vectorize_band_matrix(matrix, directed=bundle.directed)
    if bundle.training_reference_z.size == 0:
        return 0.0
    candidate_z = (X[0] - bundle.feature_mean) / bundle.feature_scale
    distances = np.sqrt(np.mean((bundle.training_reference_z - candidate_z) ** 2, axis=1))
    return float(np.min(distances))


def matrix_change_rms(
    bundle: BandConnectivityClassifier,
    original_matrix: np.ndarray,
    candidate_matrix: np.ndarray,
) -> float:
    """Return standardized RMS edge change from this patient's baseline."""

    original = vectorize_band_matrix(original_matrix, directed=bundle.directed)[0]
    candidate = vectorize_band_matrix(candidate_matrix, directed=bundle.directed)[0]
    return float(np.sqrt(np.mean(((candidate - original) / bundle.feature_scale) ** 2)))


def predict_patient_probability(
    bundle: BandConnectivityClassifier,
    matrix: np.ndarray,
    *,
    channel_names: Sequence[str] | None = None,
) -> float:
    """Return P(Patient) for the bundle's single frequency band."""

    if channel_names is not None and list(channel_names) != bundle.channel_names:
        raise ValueError("Channel order does not match the fitted band classifier")
    X = vectorize_band_matrix(matrix, directed=bundle.directed)
    if X.shape[1] != bundle.n_features:
        raise ValueError(f"Expected {bundle.n_features} features, got {X.shape[1]}")
    return float(bundle.estimator.predict_proba(X)[0, 1])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("connectivity", type=Path)
    compare.add_argument("output", type=Path)
    compare.add_argument("--method", default="coh")
    compare.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    compare.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    compare.add_argument("--mode", choices=("quick", "full"), default="quick")
    compare.add_argument("--outer-splits", type=int, default=5)
    compare.add_argument("--repeats", type=int, default=1)
    compare.add_argument("--inner-splits", type=int, default=3)
    compare.add_argument("--n-jobs", type=int, default=1)
    train = sub.add_parser("train")
    train.add_argument("connectivity", type=Path)
    train.add_argument("channel_metadata", type=Path)
    train.add_argument("comparison", type=Path)
    train.add_argument("output", type=Path)
    train.add_argument("--method", default="coh")
    train.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    train.add_argument("--mode", choices=("quick", "full"), default="quick")
    train.add_argument("--inner-splits", type=int, default=5)
    train.add_argument("--n-jobs", type=int, default=1)
    train.add_argument("--minimum-roc-auc", type=float, default=0.75)
    train.add_argument("--minimum-balanced-accuracy", type=float, default=0.70)
    train.add_argument("--maximum-brier", type=float, default=0.20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "compare":
        summary = evaluate_band_models(
            args.connectivity,
            args.output,
            method=args.method,
            bands=args.bands,
            models=args.models,
            mode=args.mode,
            outer_splits=args.outer_splits,
            repeats=args.repeats,
            inner_splits=args.inner_splits,
            n_jobs=args.n_jobs,
        )
        print(summary[["band", "model", "roc_auc", "balanced_accuracy", "brier"]].to_string(index=False))
        return 0
    bundles = train_band_bundles(
        args.connectivity,
        args.channel_metadata,
        args.output,
        method=args.method,
        bands=args.bands,
        comparison_path=args.comparison,
        mode=args.mode,
        inner_splits=args.inner_splits,
        n_jobs=args.n_jobs,
        minimum_roc_auc=args.minimum_roc_auc,
        minimum_balanced_accuracy=args.minimum_balanced_accuracy,
        maximum_brier=args.maximum_brier,
    )
    for band, bundle in bundles.items():
        print(
            f"{band}: {bundle.model_name}, "
            f"accepted_for_optimization={bundle.accepted_for_optimization}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
