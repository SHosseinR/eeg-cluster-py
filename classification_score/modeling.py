"""Leakage-safe probabilistic model comparison for EEG subject features."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC

from classification_score.neural_graph_models import TorchGraphClassifier


RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelSpec:
    estimator: BaseEstimator
    param_grid: dict[str, list[Any]]


def _k_values(n_features: int, mode: str) -> list[int | str]:
    candidates = [10, 30, 60, 120, 250]
    if mode == "quick":
        candidates = [30, 120]
    values: list[int | str] = [value for value in candidates if value < n_features]
    values.append("all")
    return list(dict.fromkeys(values))


def _linear_pipeline(classifier: BaseEstimator) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k="all")),
            ("clf", classifier),
        ]
    )


def _tree_pipeline(classifier: BaseEstimator) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(score_func=f_classif, k="all")),
            ("clf", classifier),
        ]
    )


def _neural_graph_pipeline(classifier: BaseEstimator) -> Pipeline:
    """Preserve the complete triangular edge layout required by graph models."""

    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("clf", classifier),
        ]
    )


def model_specs(n_features: int, mode: str = "quick") -> dict[str, ModelSpec]:
    """Return all compared probability-capable model families and inner grids."""

    k_values = _k_values(n_features, mode)
    # Shrinkage LDA still forms dense covariance systems.  Keeping it below the
    # sample-scale is both statistically safer and avoids cubic work on fused
    # matrices with thousands of columns.
    lda_k_values = [value for value in k_values if value != "all" and value <= 250]
    if n_features <= 250:
        lda_k_values.append("all")
    if mode == "quick":
        c_values = [0.03, 0.3, 3.0]
        tree_leaves = [2, 6]
        tree_estimators = 150
        neural_epochs = 160
        neural_patience = 20
    else:
        c_values = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
        tree_leaves = [1, 2, 4, 8]
        tree_estimators = 500
        neural_epochs = 300
        neural_patience = 35

    return {
        "dummy_prior": ModelSpec(DummyClassifier(strategy="prior"), {}),
        "logistic_l2": ModelSpec(
            _linear_pipeline(
                LogisticRegression(
                    solver="liblinear", max_iter=3000, random_state=RANDOM_STATE
                )
            ),
            {"select__k": k_values, "clf__C": c_values},
        ),
        "logistic_elasticnet": ModelSpec(
            _linear_pipeline(
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                )
            ),
            {
                "select__k": k_values,
                "clf__C": c_values,
                "clf__l1_ratio": [0.25, 0.75] if mode == "quick" else [0.1, 0.25, 0.5, 0.75, 0.9],
            },
        ),
        "lda_shrinkage": ModelSpec(
            _linear_pipeline(LinearDiscriminantAnalysis(solver="lsqr")),
            {
                "select__k": lda_k_values,
                "clf__shrinkage": ["auto", 0.2, 0.8]
                if mode == "quick"
                else ["auto", 0.05, 0.2, 0.5, 0.8, 0.95],
            },
        ),
        "linear_svm_sigmoid": ModelSpec(
            _linear_pipeline(
                CalibratedClassifierCV(
                    estimator=LinearSVC(dual="auto", max_iter=5000, random_state=RANDOM_STATE),
                    method="sigmoid",
                    cv=3,
                )
            ),
            {"select__k": k_values, "clf__estimator__C": c_values},
        ),
        "rbf_svm": ModelSpec(
            _linear_pipeline(SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            {
                "select__k": k_values,
                "clf__C": [0.3, 3.0] if mode == "quick" else [0.1, 0.3, 1.0, 3.0, 10.0],
                "clf__gamma": ["scale", 0.01] if mode == "quick" else ["scale", 0.001, 0.01, 0.1],
            },
        ),
        "knn": ModelSpec(
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("variance", VarianceThreshold()),
                    ("scale", RobustScaler()),
                    ("select", SelectKBest(score_func=f_classif, k="all")),
                    ("clf", KNeighborsClassifier(weights="distance")),
                ]
            ),
            {
                "select__k": k_values,
                "clf__n_neighbors": [5, 15] if mode == "quick" else [3, 5, 9, 15, 25, 35],
                "clf__p": [2] if mode == "quick" else [1, 2],
            },
        ),
        "gaussian_nb": ModelSpec(
            _linear_pipeline(GaussianNB()),
            {
                "select__k": k_values,
                "clf__var_smoothing": [1e-8, 1e-6]
                if mode == "quick"
                else [1e-10, 1e-8, 1e-6, 1e-4],
            },
        ),
        "random_forest": ModelSpec(
            _tree_pipeline(
                RandomForestClassifier(
                    n_estimators=tree_estimators,
                    class_weight="balanced_subsample",
                    n_jobs=1,
                    random_state=RANDOM_STATE,
                )
            ),
            {
                "select__k": k_values,
                "clf__max_features": ["sqrt", 0.5],
                "clf__min_samples_leaf": tree_leaves,
            },
        ),
        "extra_trees": ModelSpec(
            _tree_pipeline(
                ExtraTreesClassifier(
                    n_estimators=tree_estimators,
                    class_weight="balanced",
                    n_jobs=1,
                    random_state=RANDOM_STATE,
                )
            ),
            {
                "select__k": k_values,
                "clf__max_features": ["sqrt", 0.5],
                "clf__min_samples_leaf": tree_leaves,
            },
        ),
        "hist_gradient_boosting": ModelSpec(
            _tree_pipeline(
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    early_stopping=True,
                    random_state=RANDOM_STATE,
                )
            ),
            {
                "select__k": k_values,
                "clf__max_leaf_nodes": [7, 15] if mode == "quick" else [3, 7, 15, 31],
                "clf__l2_regularization": [0.1, 3.0]
                if mode == "quick"
                else [0.0, 0.1, 1.0, 3.0, 10.0],
            },
        ),
        # Fixed, predeclared neural hyperparameters avoid selecting a large
        # deep-learning search space on this modest subject cohort. Each fit
        # still uses a stratified fold-internal validation split for stopping
        # and probability temperature scaling.
        "gcn": ModelSpec(
            _neural_graph_pipeline(
                TorchGraphClassifier(
                    model_name="gcn",
                    hidden_dim=24,
                    dropout=0.35,
                    learning_rate=1e-3,
                    weight_decay=1e-3,
                    max_epochs=neural_epochs,
                    patience=neural_patience,
                    random_state=RANDOM_STATE,
                )
            ),
            {},
        ),
        "brainnetcnn": ModelSpec(
            _neural_graph_pipeline(
                TorchGraphClassifier(
                    model_name="brainnetcnn",
                    hidden_dim=16,
                    dropout=0.35,
                    learning_rate=7e-4,
                    weight_decay=1e-3,
                    max_epochs=neural_epochs,
                    patience=neural_patience,
                    random_state=RANDOM_STATE,
                )
            ),
            {},
        ),
        "gcn_3band": ModelSpec(
            _neural_graph_pipeline(
                TorchGraphClassifier(
                    model_name="gcn",
                    n_bands=3,
                    hidden_dim=24,
                    dropout=0.35,
                    learning_rate=1e-3,
                    weight_decay=1e-3,
                    max_epochs=neural_epochs,
                    patience=neural_patience,
                    random_state=RANDOM_STATE,
                )
            ),
            {},
        ),
        "brainnetcnn_3band": ModelSpec(
            _neural_graph_pipeline(
                TorchGraphClassifier(
                    model_name="brainnetcnn",
                    n_bands=3,
                    hidden_dim=16,
                    dropout=0.35,
                    learning_rate=7e-4,
                    weight_decay=1e-3,
                    max_epochs=neural_epochs,
                    patience=neural_patience,
                    random_state=RANDOM_STATE,
                )
            ),
            {},
        ),
    }


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    """Equal-width expected calibration error for the Patient probability."""

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (probability < upper)
        if upper == 1.0:
            mask |= probability == 1.0
        if np.any(mask):
            error += np.sum(mask) / total * abs(np.mean(y_true[mask]) - np.mean(probability[mask]))
    return float(error)


def probability_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Compute discrimination, threshold, and calibration metrics."""

    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "ece_10": expected_calibration_error(y_true, probability, bins=10),
    }


def nested_oof_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_set: str,
    model_name: str,
    mode: str = "quick",
    outer_splits: int = 5,
    repeats: int = 1,
    inner_splits: int = 3,
    n_jobs: int = 1,
    subject_ids: Iterable[str] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Nested repeated CV with one prediction per subject in every repeat."""

    specs = model_specs(X.shape[1], mode=mode)
    if model_name not in specs:
        raise KeyError(f"Unknown model {model_name!r}; choose from {sorted(specs)}")
    spec = specs[model_name]
    min_class = int(np.bincount(y).min())
    if outer_splits > min_class:
        raise ValueError(f"outer_splits={outer_splits} exceeds minority class size {min_class}")
    subject_ids = np.asarray(
        list(subject_ids) if subject_ids is not None else [str(i) for i in range(len(y))]
    )
    outer = RepeatedStratifiedKFold(
        n_splits=outer_splits, n_repeats=repeats, random_state=RANDOM_STATE
    )
    prediction_rows: list[dict[str, Any]] = []
    best_params: list[dict[str, Any]] = []
    for outer_index, (train_index, test_index) in enumerate(outer.split(X, y)):
        repeat = outer_index // outer_splits
        fold = outer_index % outer_splits
        train_minority = int(np.bincount(y[train_index]).min())
        actual_inner_splits = min(inner_splits, train_minority)
        if spec.param_grid:
            inner = StratifiedKFold(
                n_splits=actual_inner_splits,
                shuffle=True,
                random_state=RANDOM_STATE + outer_index + 1,
            )
            fitted: BaseEstimator = GridSearchCV(
                spec.estimator,
                spec.param_grid,
                scoring="roc_auc",
                cv=inner,
                n_jobs=n_jobs,
                refit=True,
                error_score="raise",
            )
        else:
            fitted = spec.estimator
        fitted.fit(X[train_index], y[train_index])
        probability = fitted.predict_proba(X[test_index])[:, 1]
        params = fitted.best_params_ if isinstance(fitted, GridSearchCV) else {}
        best_params.append(params)
        for row_index, p in zip(test_index, probability):
            prediction_rows.append(
                {
                    "subject_id": subject_ids[row_index],
                    "y_true": int(y[row_index]),
                    "patient_probability": float(p),
                    "repeat": repeat,
                    "fold": fold,
                    "feature_set": feature_set,
                    "model": model_name,
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    repeat_metrics = [
        probability_metrics(part["y_true"].to_numpy(), part["patient_probability"].to_numpy())
        for _, part in predictions.groupby("repeat", sort=True)
    ]
    averaged_probability = (
        predictions.groupby(["subject_id", "y_true"], as_index=False)["patient_probability"].mean()
    )
    aggregate = probability_metrics(
        averaged_probability["y_true"].to_numpy(),
        averaged_probability["patient_probability"].to_numpy(),
    )
    summary: dict[str, Any] = {
        "feature_set": feature_set,
        "model": model_name,
        "n_subjects": int(len(y)),
        "n_features": int(X.shape[1]),
        "outer_splits": outer_splits,
        "repeats": repeats,
        "inner_splits": inner_splits,
        **aggregate,
    }
    for metric in aggregate:
        values = np.asarray([item[metric] for item in repeat_metrics], dtype=float)
        summary[f"{metric}_repeat_mean"] = float(np.mean(values))
        summary[f"{metric}_repeat_sd"] = float(np.std(values, ddof=1)) if repeats > 1 else 0.0
    param_counts: dict[str, int] = {}
    for params in best_params:
        encoded = json.dumps(params, sort_keys=True, default=str)
        param_counts[encoded] = param_counts.get(encoded, 0) + 1
    summary["inner_selection_counts"] = json.dumps(param_counts, sort_keys=True)
    return summary, predictions


def fit_tuned_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_name: str,
    mode: str = "full",
    inner_splits: int = 5,
    n_jobs: int = 1,
) -> tuple[BaseEstimator, dict[str, Any]]:
    """Tune on all development subjects and return the deployable fitted estimator."""

    spec = model_specs(X.shape[1], mode=mode)[model_name]
    if not spec.param_grid:
        fitted = spec.estimator.fit(X, y)
        return fitted, {}
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        spec.estimator,
        spec.param_grid,
        scoring="roc_auc",
        cv=inner,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_
