"""Regenerate the TD-BRAIN log-gain paper results and figures.

This is a post-analysis workflow. It reads the cached connectivity matrices,
saved out-of-fold predictions, fitted classifier bundles, participant metadata,
and per-subject optimization payloads. It never reruns EEG preprocessing,
connectivity estimation, classifier construction, or optimization.

The primary output directory is the run's ``final-figures`` directory. When
``--paper-figures-dir`` is supplied, the seven manuscript figures are mirrored
there using stable names expected by the LaTeX source.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import mne
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classification_score.band_connectivity_classifier import (
    load_connectivity_features,
    vectorize_band_matrix,
)


BANDS = ("delta", "alpha", "beta")
BAND_LABELS = {"delta": "Delta", "alpha": "Alpha", "beta": "Beta"}
COLORS = {
    "delta": "#3B6FB6",
    "alpha": "#D99000",
    "beta": "#8A5AA5",
    "healthy": "#2A9D8F",
    "patient": "#D55E5E",
    "optimized": "#3767A6",
    "gray": "#6B7280",
}
RANDOM_STATE = 42
MANUSCRIPT_FIGURE_STEMS = (
    "figure1_pipeline",
    "figure2_classifier_performance",
    "figure3_objective_change",
    "figure4_classifier_projection",
    "figure5_target_counts",
    "figure6_target_scalp_maps",
    "figure7_target_concentration",
)


class AgeResidualizer(BaseEstimator, TransformerMixin):
    """Remove training-fold linear age effects from connectivity features.

    Input columns are ``[connectivity edges..., age]``. The fitted training-age
    mean and one slope per edge are reused for held-out rows. The age column is
    removed from the output, so age is a nuisance variable rather than a model
    predictor.
    """

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AgeResidualizer":
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError("AgeResidualizer expects edge columns plus one age column")
        edges, age = values[:, :-1], values[:, -1]
        if not np.all(np.isfinite(edges)) or not np.all(np.isfinite(age)):
            raise ValueError("AgeResidualizer inputs must be finite")
        self.age_mean_ = float(np.mean(age))
        centered_age = age - self.age_mean_
        denominator = float(centered_age @ centered_age)
        if denominator <= np.finfo(float).eps:
            self.slopes_ = np.zeros(edges.shape[1], dtype=float)
        else:
            self.slopes_ = centered_age @ edges / denominator
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("AgeResidualizer input shape differs from fitted shape")
        edges, age = values[:, :-1], values[:, -1]
        return edges - np.outer(age - self.age_mean_, self.slopes_)


@dataclass(frozen=True)
class Inputs:
    results_root: Path
    output_dir: Path
    paper_figures_dir: Path | None
    metadata_csv: Path

    @property
    def connectivity_path(self) -> Path:
        return self.results_root / "data" / "connectivity_matrices.npy"

    @property
    def classifier_dir(self) -> Path:
        return self.results_root / "data" / "connectivity_classifiers"

    @property
    def optimization_dir(self) -> Path:
        return self.results_root / "optimization-log-gain"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.6,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.7,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.5,
            "figure.dpi": 170,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def mirror_manuscript_figures(output_dir: Path, paper_dir: Path | None) -> None:
    if paper_dir is None:
        return
    paper_dir.mkdir(parents=True, exist_ok=True)
    for stem in MANUSCRIPT_FIGURE_STEMS:
        for suffix in (".pdf", ".png"):
            shutil.copy2(output_dir / f"{stem}{suffix}", paper_dir / f"{stem}{suffix}")


def validate_inputs(inputs: Inputs) -> None:
    required = [
        inputs.connectivity_path,
        inputs.classifier_dir / "classification_summary_by_band_connectivity.csv",
        inputs.classifier_dir / "oof_predictions_selected_models.csv",
        inputs.metadata_csv,
    ]
    for band in BANDS:
        required.append(inputs.classifier_dir / "models" / f"{band}_classifier.joblib")
        required.append(inputs.optimization_dir / f"{band}_subject_results")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required cached inputs:\n" + "\n".join(missing))


def aggregate_oof_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"subject_id", "y_true", "patient_probability", "band"}
    if not required.issubset(frame.columns):
        raise ValueError(f"OOF table lacks columns: {sorted(required - set(frame.columns))}")
    aggregate = (
        frame.groupby(["band", "subject_id", "y_true"], as_index=False)[
            "patient_probability"
        ]
        .mean()
        .sort_values(["band", "subject_id"])
    )
    return aggregate


def load_optimized_data(
    optimization_dir: Path,
    subject_ids: Sequence[str],
) -> tuple[dict[str, np.ndarray], pd.DataFrame, list[str]]:
    """Load exact selected candidate matrices and unweighted target labels."""

    patient_ids = list(subject_ids)
    optimized: dict[str, list[np.ndarray]] = {band: [] for band in BANDS}
    target_rows: list[dict[str, object]] = []
    channel_names: list[str] | None = None
    for band in BANDS:
        by_id = {
            path.stem: path
            for path in (optimization_dir / f"{band}_subject_results").glob("*.npy")
        }
        for subject_id in patient_ids:
            if subject_id not in by_id:
                raise FileNotFoundError(f"Missing {band} result for {subject_id}")
            payload = np.load(by_id[subject_id], allow_pickle=True).item()
            solution = payload["best_solution"]
            matrix = np.asarray(solution["updated_connectivity_matrix"], dtype=float)
            names = [str(value) for value in payload["channel_names"]]
            if channel_names is None:
                channel_names = names
            if names != channel_names:
                raise ValueError(f"Channel order changed at {subject_id}/{band}")
            if matrix.shape != (len(names), len(names)) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"Invalid updated matrix at {subject_id}/{band}")
            node = int(solution["node"])
            optimized[band].append(vectorize_band_matrix(matrix, directed=False)[0])
            target_rows.append(
                {
                    "subject_id": subject_id,
                    "band": band,
                    "target_index": node,
                    "target_label": names[node],
                }
            )
    assert channel_names is not None
    return (
        {band: np.asarray(rows, dtype=float) for band, rows in optimized.items()},
        pd.DataFrame(target_rows),
        channel_names,
    )


def load_optimization_scores(
    optimization_dir: Path,
    subject_ids: Sequence[str],
) -> pd.DataFrame:
    """Load baseline and selected-candidate objective values from saved payloads."""

    rows: list[dict[str, object]] = []
    for band in BANDS:
        by_id = {
            path.stem: path
            for path in (optimization_dir / f"{band}_subject_results").glob("*.npy")
        }
        for subject_id in subject_ids:
            if subject_id not in by_id:
                raise FileNotFoundError(f"Missing {band} result for {subject_id}")
            payload = np.load(by_id[subject_id], allow_pickle=True).item()
            initial = np.asarray(payload.get("initial_metrics"), dtype=float).ravel()
            final = np.asarray(payload.get("final_metrics"), dtype=float).ravel()
            if initial.size != 1 or final.size != 1:
                raise ValueError(
                    f"Expected one classifier objective at {subject_id}/{band}"
                )
            rows.append(
                {
                    "subject_id": subject_id,
                    "band": band,
                    "baseline_patient_probability": float(initial[0]),
                    "candidate_patient_probability": float(final[0]),
                }
            )
    frame = pd.DataFrame(rows)
    if not np.isfinite(
        frame[["baseline_patient_probability", "candidate_patient_probability"]]
    ).all().all():
        raise ValueError("Optimization objective values must be finite")
    return frame


def _age_pipeline(n_edges: int) -> tuple[Pipeline, dict[str, list[object]]]:
    k_values: list[int | str] = [value for value in (30, 120) if value < n_edges]
    k_values.append("all")
    pipeline = Pipeline(
        [
            ("age_residualize", AgeResidualizer()),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k="all")),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear", max_iter=3000, random_state=RANDOM_STATE
                ),
            ),
        ]
    )
    return pipeline, {"select__k": k_values, "clf__C": [0.03, 0.3, 3.0]}


def repeated_nested_age_residualized_oof(
    X: np.ndarray,
    age: np.ndarray,
    y: np.ndarray,
    subject_ids: Sequence[str],
    *,
    n_jobs: int,
) -> pd.DataFrame:
    """Return five-repeat nested OOF predictions after fold-wise age removal."""

    augmented = np.column_stack([X, age])
    outer = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=5, random_state=RANDOM_STATE
    )
    rows: list[dict[str, object]] = []
    for split_number, (train, test) in enumerate(outer.split(augmented, y)):
        repeat = split_number // 5
        fold = split_number % 5
        estimator, grid = _age_pipeline(X.shape[1])
        inner = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=RANDOM_STATE + split_number
        )
        search = GridSearchCV(
            estimator,
            grid,
            scoring="roc_auc",
            cv=inner,
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
        )
        search.fit(augmented[train], y[train])
        probability = search.predict_proba(augmented[test])[:, 1]
        for index, value in zip(test, probability):
            rows.append(
                {
                    "subject_id": str(subject_ids[index]),
                    "y_true": int(y[index]),
                    "repeat": repeat,
                    "fold": fold,
                    "patient_probability": float(value),
                    "best_k": str(search.best_params_["select__k"]),
                    "best_C": float(search.best_params_["clf__C"]),
                }
            )
    return pd.DataFrame(rows)


def repeated_age_only_oof(
    age: np.ndarray, y: np.ndarray, subject_ids: Sequence[str]
) -> pd.DataFrame:
    outer = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=5, random_state=RANDOM_STATE
    )
    rows: list[dict[str, object]] = []
    values = np.asarray(age, dtype=float)[:, None]
    for split_number, (train, test) in enumerate(outer.split(values, y)):
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="liblinear", max_iter=3000, random_state=RANDOM_STATE
                    ),
                ),
            ]
        )
        estimator.fit(values[train], y[train])
        probability = estimator.predict_proba(values[test])[:, 1]
        for index, value in zip(test, probability):
            rows.append(
                {
                    "subject_id": str(subject_ids[index]),
                    "y_true": int(y[index]),
                    "repeat": split_number // 5,
                    "fold": split_number % 5,
                    "patient_probability": float(value),
                }
            )
    return pd.DataFrame(rows)


def _subject_average(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["subject_id", "y_true"], as_index=False)["patient_probability"]
        .mean()
        .sort_values("subject_id")
    )


def stratified_auc_difference_ci(
    y: np.ndarray,
    primary: np.ndarray,
    adjusted: np.ndarray,
    *,
    n_bootstrap: int = 10_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    class_indices = [np.flatnonzero(y == label) for label in (0, 1)]
    differences = np.empty(n_bootstrap, dtype=float)
    for iteration in range(n_bootstrap):
        sample = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        differences[iteration] = roc_auc_score(y[sample], adjusted[sample]) - roc_auc_score(
            y[sample], primary[sample]
        )
    return tuple(np.quantile(differences, [0.025, 0.975]).tolist())


def run_age_sensitivity(
    X_by_band: dict[str, np.ndarray],
    y: np.ndarray,
    subject_ids: Sequence[str],
    metadata_csv: Path,
    primary_oof: pd.DataFrame,
    output_dir: Path,
    *,
    n_jobs: int,
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_csv)
    age_by_id = metadata.assign(subject_id=metadata["subject_id"].astype(str)).set_index(
        "subject_id"
    )["age"]
    missing = [subject_id for subject_id in subject_ids if subject_id not in age_by_id.index]
    if missing:
        raise ValueError(f"Age metadata missing for {len(missing)} classifier subjects")
    age = np.asarray([age_by_id.loc[subject_id] for subject_id in subject_ids], dtype=float)
    if not np.all(np.isfinite(age)):
        raise ValueError("Age metadata contains non-finite values")

    age_only = repeated_age_only_oof(age, y, subject_ids)
    age_only_aggregate = _subject_average(age_only)
    age_only_auc = roc_auc_score(
        age_only_aggregate["y_true"], age_only_aggregate["patient_probability"]
    )
    all_adjusted: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for band in BANDS:
        adjusted = repeated_nested_age_residualized_oof(
            X_by_band[band], age, y, subject_ids, n_jobs=n_jobs
        )
        adjusted.insert(0, "band", band)
        all_adjusted.append(adjusted)
        adjusted_aggregate = _subject_average(adjusted)
        primary = primary_oof.loc[primary_oof["band"] == band].copy()
        primary["subject_id"] = primary["subject_id"].astype(str)
        comparison = primary.merge(
            adjusted_aggregate,
            on=["subject_id", "y_true"],
            suffixes=("_primary", "_age_adjusted"),
            validate="one_to_one",
        )
        truth = comparison["y_true"].to_numpy(dtype=int)
        p_primary = comparison["patient_probability_primary"].to_numpy(dtype=float)
        p_adjusted = comparison["patient_probability_age_adjusted"].to_numpy(dtype=float)
        primary_auc = roc_auc_score(truth, p_primary)
        adjusted_auc = roc_auc_score(truth, p_adjusted)
        ci_low, ci_high = stratified_auc_difference_ci(truth, p_primary, p_adjusted)
        summary_rows.append(
            {
                "band": band,
                "primary_auc": primary_auc,
                "age_residualized_auc": adjusted_auc,
                "auc_difference_adjusted_minus_primary": adjusted_auc - primary_auc,
                "auc_difference_ci_low": ci_low,
                "auc_difference_ci_high": ci_high,
                "age_residualized_balanced_accuracy": balanced_accuracy_score(
                    truth, p_adjusted >= 0.5
                ),
                "age_residualized_brier": brier_score_loss(truth, p_adjusted),
                "age_only_auc": age_only_auc,
            }
        )
    adjusted_frame = pd.concat(all_adjusted, ignore_index=True)
    adjusted_frame.to_csv(output_dir / "age_sensitivity_oof_predictions.csv", index=False)
    age_only.to_csv(output_dir / "age_only_oof_predictions.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "age_sensitivity_summary.csv", index=False)
    return summary


def target_concentration_statistics(
    target_frame: pd.DataFrame,
    channel_names: Sequence[str],
    *,
    simulations: int = 100_000,
    bootstraps: int = 10_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare selected-target concentration with a uniform-target null."""

    n_channels = len(channel_names)
    rng = np.random.default_rng(RANDOM_STATE)
    n_subjects = int(target_frame.groupby("band")["subject_id"].nunique().min())
    null_counts = rng.multinomial(
        n_subjects, np.repeat(1.0 / n_channels, n_channels), size=simulations
    )
    null_max = np.max(null_counts, axis=1) / n_subjects
    probabilities = null_counts / n_subjects
    with np.errstate(divide="ignore", invalid="ignore"):
        null_entropy = -np.sum(
            np.where(probabilities > 0, probabilities * np.log(probabilities), 0.0), axis=1
        ) / np.log(n_channels)
    null_max_ci = np.quantile(null_max, [0.005, 0.995])
    null_entropy_ci = np.quantile(null_entropy, [0.005, 0.995])

    rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    for band in BANDS:
        labels = target_frame.loc[target_frame["band"] == band, "target_label"].to_numpy()
        counts = pd.Series(labels).value_counts().reindex(channel_names, fill_value=0)
        for channel, count in counts.items():
            count_rows.append(
                {"band": band, "channel": channel, "count": int(count), "percent": 100 * count / len(labels)}
            )
        observed_top = str(counts.idxmax())
        observed_max = float(counts.max() / len(labels))
        observed_probabilities = counts.to_numpy(dtype=float) / len(labels)
        positive = observed_probabilities > 0
        observed_entropy = float(
            -np.sum(observed_probabilities[positive] * np.log(observed_probabilities[positive]))
            / np.log(n_channels)
        )
        bootstrap_counts = rng.multinomial(
            len(labels), observed_probabilities, size=bootstraps
        )
        bootstrap_top_rate = np.max(bootstrap_counts, axis=1) / len(labels)
        bootstrap_same_top = (
            np.argmax(bootstrap_counts, axis=1) == int(np.argmax(counts.to_numpy()))
        )
        rows.append(
            {
                "band": band,
                "n_subjects": len(labels),
                "top_target": observed_top,
                "top_count": int(counts.max()),
                "top_fraction": observed_max,
                "top_fraction_bootstrap_ci_low": float(np.quantile(bootstrap_top_rate, 0.025)),
                "top_fraction_bootstrap_ci_high": float(np.quantile(bootstrap_top_rate, 0.975)),
                "bootstrap_same_top_probability": float(np.mean(bootstrap_same_top)),
                "normalized_entropy": observed_entropy,
                "uniform_null_max_fraction_ci_low": float(null_max_ci[0]),
                "uniform_null_max_fraction_ci_high": float(null_max_ci[1]),
                "uniform_null_entropy_ci_low": float(null_entropy_ci[0]),
                "uniform_null_entropy_ci_high": float(null_entropy_ci[1]),
                "uniform_null_p_max": float((1 + np.sum(null_max >= observed_max)) / (simulations + 1)),
                "uniform_null_p_entropy": float(
                    (1 + np.sum(null_entropy <= observed_entropy)) / (simulations + 1)
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(count_rows)


def figure_pipeline(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    ax.set_axis_off()
    boxes = [
        (0.03, 0.64, 0.25, 0.23, "Resting EEG", "327 participants\n26 scalp sensors\nDelta, alpha, beta", "#E7F2F8", "#2C7DA0"),
        (0.375, 0.64, 0.25, 0.23, "Coherence networks", "One 26 x 26 matrix\nper participant and band\n325 unique edges", "#E7F2F8", "#2C7DA0"),
        (0.72, 0.64, 0.25, 0.23, "Classifier", "Repeated nested validation\nFull-cohort refit supplies\nthe search coordinate", "#E7F2F8", "#2C7DA0"),
        (0.18, 0.20, 0.27, 0.23, "Per-patient perturbation", "RMS activation baseline\nSelected sensor + one graph hop\nPositive multiplicative ratios", "#FFF0DE", "#C56A1A"),
        (0.55, 0.20, 0.27, 0.23, "Constrained search", "Evaluate candidate network\nKeep activation and network\nfeatures inside set limits", "#FFF0DE", "#C56A1A"),
    ]
    for x, y, width, height, title, body, face, edge in boxes:
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1.2, facecolor=face, edgecolor=edge,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height * 0.72, title, ha="center", va="center", weight="bold", fontsize=8.5, transform=ax.transAxes)
        ax.text(x + width / 2, y + height * 0.37, body, ha="center", va="center", fontsize=7.3, linespacing=1.25, transform=ax.transAxes)
    arrows = [((0.28, 0.755), (0.375, 0.755)), ((0.625, 0.755), (0.72, 0.755)), ((0.81, 0.64), (0.69, 0.43)), ((0.45, 0.315), (0.55, 0.315))]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.15, color="#4A4A4A", transform=ax.transAxes))
    ax.text(0.5, 0.075, "Output: one sensor-target hypothesis for each patient and frequency band", ha="center", va="center", fontsize=9.2, weight="bold", transform=ax.transAxes)
    save_figure(fig, output_dir, "figure1_pipeline")


def figure_classifier_performance(
    summary_path: Path, primary_oof: pd.DataFrame, output_dir: Path
) -> None:
    summary = pd.read_csv(summary_path).set_index("band").loc[list(BANDS)]
    fig, axes = plt.subplots(1, 3, figsize=(7.55, 2.9), gridspec_kw={"width_ratios": [1.25, 1.15, 0.85]})

    ax = axes[0]
    for band in BANDS:
        values = primary_oof.loc[primary_oof["band"] == band]
        fpr, tpr, _ = roc_curve(values["y_true"], values["patient_probability"])
        auc = roc_auc_score(values["y_true"], values["patient_probability"])
        ax.plot(fpr, tpr, color=COLORS[band], linewidth=2, label=f"{BAND_LABELS[band]} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.02))
    ax.set_title("A  Participant-level ROC curves")
    ax.legend(title="AUC", frameon=False, loc="lower right")

    x = np.arange(3)
    width = 0.34
    ax = axes[1]
    ax.bar(
        x - width / 2,
        summary["roc_auc_repeat_mean"],
        width,
        yerr=summary["roc_auc_repeat_sd"],
        capsize=2.5,
        color=[COLORS[band] for band in BANDS],
        alpha=0.9,
        label="ROC AUC",
    )
    ax.bar(x + width / 2, summary["balanced_accuracy"], width, color="#A8B0BA", label="Balanced accuracy")
    ax.scatter(x - width / 2, summary["roc_auc"], marker="D", s=18, facecolor="white", edgecolor="black", zorder=3)
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks(x, [BAND_LABELS[band] for band in BANDS])
    ax.set_ylim(0.45, 0.92)
    ax.set_ylabel("Cross-validated metric")
    ax.set_title("B  Discrimination")

    ax = axes[2]
    bars = ax.bar(x, summary["brier"], color=[COLORS[band] for band in BANDS], width=0.58)
    ax.set_xticks(x, [BAND_LABELS[band] for band in BANDS])
    ax.set_ylim(0, 0.24)
    ax.set_ylabel("Brier score")
    ax.set_title("C  Probability error")
    for bar, value in zip(bars, summary["brier"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=7.3)
    fig.subplots_adjust(wspace=0.38)
    save_figure(fig, output_dir, "figure2_classifier_performance")


def baseline_pca_projection(
    X_baseline: np.ndarray, X_optimized: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project baseline and candidate edges into one baseline-fitted PCA space."""

    scaler = StandardScaler()
    baseline_scaled = scaler.fit_transform(X_baseline)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    baseline = pca.fit_transform(baseline_scaled)
    optimized = pca.transform(scaler.transform(X_optimized))
    return baseline, optimized, 100.0 * pca.explained_variance_ratio_


def figure_classifier_projection(
    X_by_band: dict[str, np.ndarray],
    X_optimized: dict[str, np.ndarray],
    y: np.ndarray,
    output_dir: Path,
) -> None:
    patient_indices = np.flatnonzero(y == 1)
    healthy = y == 0
    patient = y == 1
    fig, axes = plt.subplots(3, 2, figsize=(7.55, 8.0), squeeze=False)
    for row, band in enumerate(BANDS):
        baseline, optimized, explained = baseline_pca_projection(
            X_by_band[band], X_optimized[band]
        )
        left, right = axes[row]
        left.scatter(
            baseline[healthy, 0], baseline[healthy, 1], s=14, alpha=0.64,
            color="#2A9D8F", edgecolor="white", linewidth=0.25, label="Healthy",
        )
        left.scatter(
            baseline[patient, 0], baseline[patient, 1], s=14, alpha=0.60,
            color="#E76F51", edgecolor="white", linewidth=0.25, label="Patient",
        )
        left.set_title(f"{BAND_LABELS[band]}: baseline group separation")

        right.scatter(
            baseline[healthy, 0], baseline[healthy, 1], s=13, alpha=0.25,
            color="#2A9D8F", linewidth=0, label="Healthy baseline",
        )
        for base_index, destination in zip(patient_indices, optimized):
            right.annotate(
                "", xy=destination, xytext=baseline[base_index],
                arrowprops={"arrowstyle": "->", "color": "#6C757D", "alpha": 0.24, "lw": 0.55},
            )
        right.scatter(
            baseline[patient_indices, 0], baseline[patient_indices, 1], s=13,
            alpha=0.58, color="#E76F51", linewidth=0, label="Patients: before",
        )
        right.scatter(
            optimized[:, 0], optimized[:, 1], s=22, marker="x",
            alpha=0.82, color="#264653", linewidth=0.8, label="Patients: after",
        )
        right.set_title(f"{BAND_LABELS[band]}: candidate shifts in the same space")
        for ax in (left, right):
            ax.set_xlabel(f"PC1 ({explained[0]:.1f}% baseline variance)")
            ax.set_ylabel(f"PC2 ({explained[1]:.1f}% baseline variance)")
            ax.axhline(0, color="#D1D5DB", linewidth=0.6, zorder=0)
            ax.axvline(0, color="#D1D5DB", linewidth=0.6, zorder=0)
            ax.grid(alpha=0.13)
    axes[0, 0].legend(frameon=False, loc="best")
    axes[0, 1].legend(frameon=False, loc="best")
    fig.subplots_adjust(hspace=0.42, wspace=0.31)
    save_figure(fig, output_dir, "figure4_classifier_projection")


def figure_objective_change(scores: pd.DataFrame, output_dir: Path) -> None:
    """Plot the retained before/after objective panel without gain summaries."""

    positions: list[float] = []
    values: list[np.ndarray] = []
    colors: list[str] = []
    for index, band in enumerate(BANDS):
        subset = scores.loc[scores["band"] == band]
        positions.extend([index * 3 + 1, index * 3 + 2])
        values.extend(
            [
                subset["baseline_patient_probability"].to_numpy(),
                subset["candidate_patient_probability"].to_numpy(),
            ]
        )
        colors.extend(["#B9B9B9", COLORS[band]])
    fig, ax = plt.subplots(figsize=(6.55, 2.85))
    artists = ax.boxplot(
        values,
        positions=positions,
        widths=0.68,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
    )
    for patch, color in zip(artists["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.88)
    ax.set_xticks([1.5, 4.5, 7.5], [BAND_LABELS[band] for band in BANDS])
    ax.set_ylabel("Fitted model P(patient)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Objective before and after search")
    ax.text(
        0.015, 0.035, "Gray: baseline; color: selected candidate",
        transform=ax.transAxes, fontsize=7.5,
    )
    save_figure(fig, output_dir, "figure3_objective_change")


def _target_bubble_map(
    ax: plt.Axes,
    values: np.ndarray,
    info: mne.Info,
    channel_names: Sequence[str],
    title: str,
) -> None:
    mne.viz.plot_topomap(
        np.zeros_like(values),
        info,
        axes=ax,
        show=False,
        sensors=False,
        contours=0,
        cmap=mpl.colors.ListedColormap(["#FAFAF7"]),
        vlim=(-1, 1),
        outlines="head",
        sphere="auto",
        extrapolate="head",
    )
    from mne.channels.layout import _find_topomap_coords

    coordinates = _find_topomap_coords(info, picks=np.arange(len(channel_names)))
    band_max = max(1.0, float(np.max(values)))
    sizes = 13.0 + 255.0 * np.sqrt(values / band_max)
    image = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=sizes,
        c=values,
        cmap="YlOrRd",
        vmin=0,
        vmax=band_max,
        edgecolor="#555555",
        linewidth=0.35,
        zorder=4,
    )
    # Match the older, easier-to-read target maps: identify the largest target
    # markers in their centers and let radius/color carry magnitude. Exact
    # counts are available in the companion numeric heatmap, so printing them
    # here only competes with the clinically relevant electrode names.
    positive = np.flatnonzero(values > 0)
    prominent = set(
        positive[np.argsort(values[positive], kind="stable")[-min(3, len(positive)):]]
    )
    for index, ((x, y), value, name) in enumerate(
        zip(coordinates, values, channel_names)
    ):
        if index not in prominent:
            continue
        ax.text(
            x,
            y,
            name,
            ha="center",
            va="center",
            fontsize=7.2,
            color="white" if value >= 0.52 * band_max else "#111111",
            weight="semibold",
            zorder=5,
        )
    ax.set_title(title)
    bar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    bar.set_ticks([0, int(round(band_max / 2)), int(band_max)])
    bar.set_label(f"Patients (0--{int(band_max)})")


def figure_target_counts(
    target_frame: pd.DataFrame,
    count_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Restore the numeric target panel using exact, unweighted counts."""

    pivot = count_frame.pivot(index="channel", columns="band", values="count").fillna(0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index, list(BANDS)]
    pivot = pivot.loc[pivot.sum(axis=1) > 0]
    cross_band = target_frame.pivot(index="subject_id", columns="band", values="target_label").loc[:, list(BANDS)]
    all_same = int((cross_band.nunique(axis=1) == 1).sum())
    exactly_two = int((cross_band.nunique(axis=1) == 2).sum())
    all_different = int((cross_band.nunique(axis=1) == 3).sum())

    fig, axes = plt.subplots(1, 2, figsize=(7.55, 4.05), gridspec_kw={"width_ratios": [1.65, 1.0]})
    ax = axes[0]
    vmax = float(pivot.to_numpy().max())
    image = ax.imshow(pivot.to_numpy(), cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(3), [BAND_LABELS[band] for band in BANDS])
    ax.set_yticks(np.arange(len(pivot)), pivot.index)
    ax.set_title("A  Unweighted best-target count by band")
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value = int(pivot.iloc[row, column])
            if value:
                ax.text(
                    column, row, str(value), ha="center", va="center",
                    color="white" if value >= 0.51 * vmax else "#111111", fontsize=7.4,
                )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.042, pad=0.03)
    colorbar.set_label("Patients (absolute count)")

    ax = axes[1]
    labels = ["All 3 same", "Exactly 2 same", "All different"]
    agreement = [all_same, exactly_two, all_different]
    bars = ax.bar(np.arange(3), agreement, color=["#2A9D8F", "#D7A62E", "#A7ADB4"], width=0.62)
    ax.set_xticks(np.arange(3), labels, rotation=24, ha="right")
    ax.set_ylabel("Patients")
    ax.set_ylim(0, max(agreement) * 1.24)
    ax.set_title("B  Within-patient cross-band agreement")
    for bar, count in zip(bars, agreement):
        ax.text(
            bar.get_x() + bar.get_width() / 2, count + 2,
            f"{count}\n({100 * count / len(cross_band):.1f}%)",
            ha="center", va="bottom", fontsize=7.5,
        )
    fig.subplots_adjust(wspace=0.40)
    save_figure(fig, output_dir, "figure5_target_counts")


def figure_target_scalp_maps(
    count_frame: pd.DataFrame,
    channel_names: Sequence[str],
    output_dir: Path,
) -> None:
    info = mne.create_info(list(channel_names), sfreq=500.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="raise")
    values_by_band = {
        band: count_frame.loc[count_frame["band"] == band].set_index("channel").loc[list(channel_names), "count"].to_numpy(dtype=float)
        for band in BANDS
    }
    for band in BANDS:
        fig, ax = plt.subplots(figsize=(4.0, 3.7))
        _target_bubble_map(ax, values_by_band[band], info, channel_names, f"{BAND_LABELS[band]}: unweighted selected-target counts")
        save_figure(fig, output_dir, f"target_topomap_unweighted_{band}")

    fig = plt.figure(figsize=(7.55, 6.15))
    grid = fig.add_gridspec(2, 4, hspace=0.28, wspace=0.48)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[1, 1:3]),
    ]
    for index, (ax, band) in enumerate(zip(axes, BANDS)):
        _target_bubble_map(
            ax, values_by_band[band], info, channel_names,
            f"{chr(65 + index)}  {BAND_LABELS[band]}",
        )
    save_figure(fig, output_dir, "figure6_target_scalp_maps")


def figure_target_concentration(stats: pd.DataFrame, output_dir: Path) -> None:
    frame = stats.set_index("band").loc[list(BANDS)]
    y = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(7.55, 2.70))
    ax = axes[0]
    null_low = float(frame["uniform_null_max_fraction_ci_low"].min())
    null_high = float(frame["uniform_null_max_fraction_ci_high"].max())
    ax.axvspan(null_low, null_high, color="#B8BEC5", alpha=0.42)
    bars = ax.barh(y, frame["top_fraction"], color=[COLORS[band] for band in BANDS], height=0.56)
    ax.set_yticks(y, [BAND_LABELS[band] for band in BANDS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Most frequent target (% of patients)")
    ax.set_title("A  Dominant-target share")
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    for bar, band in zip(bars, BANDS):
        value = float(frame.loc[band, "top_fraction"])
        ax.text(value - 0.025, bar.get_y() + bar.get_height() / 2, f"{frame.loc[band, 'top_target']}  {100 * value:.1f}%", ha="right", va="center", color="white", weight="bold", fontsize=7.2)
    ax.text(
        null_high + 0.01, -0.34, "Uniform-null 99% region",
        ha="left", va="center", fontsize=7.0, color="#626971",
    )

    ax = axes[1]
    observed_concentration = 1.0 - frame["normalized_entropy"]
    null_concentration_low = float((1.0 - frame["uniform_null_entropy_ci_high"]).min())
    null_concentration_high = float((1.0 - frame["uniform_null_entropy_ci_low"]).max())
    ax.axvspan(null_concentration_low, null_concentration_high, color="#B8BEC5", alpha=0.42)
    bars = ax.barh(y, observed_concentration, color=[COLORS[band] for band in BANDS], height=0.56)
    ax.set_yticks(y, [BAND_LABELS[band] for band in BANDS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Entropy-based concentration (1 - entropy)")
    ax.set_title("B  Concentration across 26 sensors")
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    for bar, value in zip(bars, observed_concentration):
        ax.text(float(value) - 0.025, bar.get_y() + bar.get_height() / 2, f"{100 * float(value):.1f}%", ha="right", va="center", color="white", weight="bold", fontsize=7.2)
    fig.subplots_adjust(wspace=0.34)
    save_figure(fig, output_dir, "figure7_target_concentration")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    default_root = Path(
        r"D:\university\projects\worktree\eeg-static-stim-graph-classifiers"
        r"\results-adjact-signed-norej-logistic\TDBRAIN-restEC-coherence"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path(r"D:\university\projects\paper\tables\analyzed_subject_metadata.csv"),
    )
    parser.add_argument("--paper-figures-dir", type=Path, default=None)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--skip-age-sensitivity", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir or (args.results_root / "final-figures")
    inputs = Inputs(args.results_root, output_dir, args.paper_figures_dir, args.metadata_csv)
    validate_inputs(inputs)
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    X_by_band, y, subject_ids = load_connectivity_features(
        inputs.connectivity_path, method="coh", bands=BANDS
    )
    patient_ids = [subject_id for subject_id, label in zip(subject_ids, y) if label == 1]
    feature_cache = inputs.output_dir / "optimized_classifier_features.npz"
    target_cache = inputs.output_dir / "unweighted_selected_targets.csv"
    channel_cache = inputs.output_dir / "channel_names.json"
    if feature_cache.exists() and target_cache.exists() and channel_cache.exists():
        cached = np.load(feature_cache)
        X_optimized = {band: np.asarray(cached[band], dtype=float) for band in BANDS}
        target_frame = pd.read_csv(target_cache, dtype={"subject_id": str})
        channel_names = json.loads(channel_cache.read_text(encoding="utf-8"))
    else:
        X_optimized, target_frame, channel_names = load_optimized_data(
            inputs.optimization_dir, patient_ids
        )
        np.savez_compressed(feature_cache, **X_optimized)
        target_frame.to_csv(target_cache, index=False)
        channel_cache.write_text(json.dumps(channel_names, indent=2), encoding="utf-8")
    primary_oof = aggregate_oof_predictions(
        inputs.classifier_dir / "oof_predictions_selected_models.csv"
    )
    optimization_scores = load_optimization_scores(
        inputs.optimization_dir, patient_ids
    )
    optimization_scores.to_csv(
        inputs.output_dir / "optimization_objective_before_after.csv", index=False
    )

    if not args.skip_age_sensitivity:
        run_age_sensitivity(
            X_by_band,
            y,
            subject_ids,
            inputs.metadata_csv,
            primary_oof,
            inputs.output_dir,
            n_jobs=args.n_jobs,
        )

    stats, counts = target_concentration_statistics(target_frame, channel_names)
    stats.to_csv(inputs.output_dir / "target_concentration_uniform_null.csv", index=False)
    counts.to_csv(inputs.output_dir / "unweighted_target_counts_by_band.csv", index=False)

    figure_pipeline(inputs.output_dir)
    figure_classifier_performance(
        inputs.classifier_dir / "classification_summary_by_band_connectivity.csv",
        primary_oof,
        inputs.output_dir,
    )
    figure_objective_change(optimization_scores, inputs.output_dir)
    figure_classifier_projection(X_by_band, X_optimized, y, inputs.output_dir)
    figure_target_counts(target_frame, counts, inputs.output_dir)
    figure_target_scalp_maps(counts, channel_names, inputs.output_dir)
    figure_target_concentration(stats, inputs.output_dir)
    mirror_manuscript_figures(inputs.output_dir, inputs.paper_figures_dir)

    manifest = {
        "results_root": str(inputs.results_root.resolve()),
        "metadata_csv": str(inputs.metadata_csv.resolve()),
        "n_classifier_subjects": len(subject_ids),
        "n_patient_subjects": len(patient_ids),
        "bands": list(BANDS),
        "random_state": RANDOM_STATE,
        "age_sensitivity_available": (
            inputs.output_dir / "age_sensitivity_summary.csv"
        ).exists(),
        "age_sensitivity_recomputed": not args.skip_age_sensitivity,
        "primary_figure_stems": list(MANUSCRIPT_FIGURE_STEMS),
        "projection": "baseline-fitted standardized PCA",
    }
    (inputs.output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote reproducible final analyses and figures to {inputs.output_dir}")


if __name__ == "__main__":
    main()
