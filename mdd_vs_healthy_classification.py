"""
Standalone MDD vs Healthy classification pipeline.

This script loads EEG data (or precomputed network measures), builds
network-measure features across all bands, and evaluates baseline
classifiers commonly used in the literature (linear SVM / logistic
regression) with nested cross-validation.
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score
)

from config import (
    HC_DATA_PATH,
    PATIENT_DATA_PATH,
    OUTPUT_DIR,
    FREQUENCY_BANDS,
    CONNECTIVITY_METHODS,
    SELECTED_METHOD,
    NETWORK_MEASURES,
    N_FOLDS,
    RANDOM_STATE
)
from data_loader import load_group_data, verify_data_consistency
from signal_processing import process_subject_epochs
from connectivity import compute_all_connectivity
from network_measures import compute_network_measures_for_subjects
from statistics_utils import extract_features_for_classification

# ============================================================================
# SCRIPT SETTINGS
# ============================================================================
USE_PRECOMPUTED_NETWORK_MEASURES = True
PRECOMPUTED_NETWORK_MEASURES_PATH = os.path.join(
    OUTPUT_DIR, 'data', 'network_measures.npy'
)
SAVE_NETWORK_MEASURES = True
MAX_SUBJECTS_PER_GROUP = None  # set to an int for quick tests

MODEL_SPECS = {
    'linear_svm': {
        'estimator': LinearSVC(
            class_weight='balanced',
            max_iter=5000,
            random_state=RANDOM_STATE
        ),
        'param_grid': {
            'clf__C': [0.01, 0.1, 1.0, 10.0]
        }
    },
    'logistic': {
        'estimator': LogisticRegression(
            class_weight='balanced',
            solver='liblinear',
            max_iter=2000,
            random_state=RANDOM_STATE
        ),
        'param_grid': {
            'clf__C': [0.01, 0.1, 1.0, 10.0]
        }
    }
}

# ============================================================================
# HELPERS
# ============================================================================

def _ensure_group_order(network_measures):
    """Ensure consistent label ordering: Healthy -> Patient."""
    if 'Healthy' in network_measures and 'Patient' in network_measures:
        return {
            'Healthy': network_measures['Healthy'],
            'Patient': network_measures['Patient']
        }
    return network_measures


def load_or_compute_network_measures():
    """Load precomputed network measures or compute them from raw EEG."""
    if USE_PRECOMPUTED_NETWORK_MEASURES and os.path.exists(PRECOMPUTED_NETWORK_MEASURES_PATH):
        print(f"Loading precomputed network measures: {PRECOMPUTED_NETWORK_MEASURES_PATH}")
        network_measures = np.load(
            PRECOMPUTED_NETWORK_MEASURES_PATH, allow_pickle=True
        ).item()
        return _ensure_group_order(network_measures)

    print("Precomputed network measures not found or disabled. Computing from raw data...")
    healthy_data = load_group_data(HC_DATA_PATH, group_name="Healthy")
    patient_data = load_group_data(PATIENT_DATA_PATH, group_name="Patient")

    if MAX_SUBJECTS_PER_GROUP is not None:
        healthy_data = healthy_data[:MAX_SUBJECTS_PER_GROUP]
        patient_data = patient_data[:MAX_SUBJECTS_PER_GROUP]

    all_data = healthy_data + patient_data
    if not verify_data_consistency(all_data):
        raise ValueError("Data consistency check failed")

    all_subjects_filtered = {}
    for group_data, group_name in [(healthy_data, "Healthy"), (patient_data, "Patient")]:
        all_subjects_filtered[group_name] = {}
        for subject in group_data:
            subject_id = subject['subject_id']
            print(f"Processing {subject_id} ({group_name})...")
            filtered_epochs = process_subject_epochs(subject['data'], subject['fs'])
            all_subjects_filtered[group_name][subject_id] = {
                'filtered_epochs': filtered_epochs,
                'fs': subject['fs'],
                'channels': subject['channels']
            }

    connectivity_matrices = {group: {} for group in all_subjects_filtered.keys()}
    for group_name, subjects_dict in all_subjects_filtered.items():
        for subject_id, subject_data in subjects_dict.items():
            connectivity_matrices[group_name][subject_id] = compute_all_connectivity(
                subject_data['filtered_epochs'],
                subject_data['fs'],
                methods=[SELECTED_METHOD]
            )

    network_measures = compute_network_measures_for_subjects(
        connectivity_matrices,
        list(FREQUENCY_BANDS.keys())
    )
    network_measures = _ensure_group_order(network_measures)

    if SAVE_NETWORK_MEASURES:
        os.makedirs(os.path.dirname(PRECOMPUTED_NETWORK_MEASURES_PATH), exist_ok=True)
        np.save(PRECOMPUTED_NETWORK_MEASURES_PATH, network_measures, allow_pickle=True)
        print(f"Saved network measures: {PRECOMPUTED_NETWORK_MEASURES_PATH}")

    return network_measures


def build_feature_matrix(network_measures):
    """Extract all measures across all bands into a feature matrix."""
    band_names = list(FREQUENCY_BANDS.keys())
    X, y, feature_names, subject_ids = extract_features_for_classification(
        network_measures,
        NETWORK_MEASURES,
        band_names,
        SELECTED_METHOD
    )
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, feature_names, subject_ids


def _get_score_values(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return None


def nested_cv_evaluate(X, y, model_name, model_spec, n_splits=N_FOLDS):
    """Nested CV evaluation with an inner grid search."""
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=max(3, n_splits - 2), shuffle=True, random_state=RANDOM_STATE)

    fold_rows = []
    all_coeffs = []
    best_params = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model_spec['estimator'])
        ])

        grid = GridSearchCV(
            pipeline,
            model_spec['param_grid'],
            scoring='balanced_accuracy',
            cv=inner_cv,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        best_params.append(grid.best_params_)

        y_pred = best_model.predict(X_test)
        scores = _get_score_values(best_model, X_test)

        fold_metrics = {
            'fold': fold_idx,
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        if scores is not None:
            fold_metrics['roc_auc'] = roc_auc_score(y_test, scores)
        else:
            fold_metrics['roc_auc'] = np.nan

        train_pred = best_model.predict(X_train)
        fold_metrics['train_accuracy'] = accuracy_score(y_train, train_pred)
        fold_rows.append(fold_metrics)

        if hasattr(best_model.named_steps['clf'], 'coef_'):
            all_coeffs.append(best_model.named_steps['clf'].coef_[0])

    fold_df = pd.DataFrame(fold_rows)
    mean_metrics = fold_df.mean(numeric_only=True).to_dict()
    std_metrics = fold_df.std(numeric_only=True).to_dict()

    coeffs_mean = np.mean(all_coeffs, axis=0) if all_coeffs else None

    summary = {
        'model': model_name,
        'mean_accuracy': mean_metrics.get('accuracy', np.nan),
        'std_accuracy': std_metrics.get('accuracy', np.nan),
        'mean_balanced_accuracy': mean_metrics.get('balanced_accuracy', np.nan),
        'std_balanced_accuracy': std_metrics.get('balanced_accuracy', np.nan),
        'mean_f1': mean_metrics.get('f1', np.nan),
        'std_f1': std_metrics.get('f1', np.nan),
        'mean_roc_auc': mean_metrics.get('roc_auc', np.nan),
        'std_roc_auc': std_metrics.get('roc_auc', np.nan),
        'mean_train_accuracy': mean_metrics.get('train_accuracy', np.nan),
        'std_train_accuracy': std_metrics.get('train_accuracy', np.nan)
    }

    return summary, fold_df, coeffs_mean, best_params


def fit_final_model(X, y, model_spec):
    """Fit a final model on full data with grid search."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model_spec['estimator'])
    ])

    grid = GridSearchCV(
        pipeline,
        model_spec['param_grid'],
        scoring='balanced_accuracy',
        cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1
    )
    grid.fit(X, y)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X)
    train_accuracy = accuracy_score(y, y_pred)
    coefficients = None
    if hasattr(best_model.named_steps['clf'], 'coef_'):
        coefficients = best_model.named_steps['clf'].coef_[0]
    return best_model, train_accuracy, coefficients, grid.best_params_


# ============================================================================
# MAIN
# ============================================================================

def main():
    output_root = os.path.join(OUTPUT_DIR, 'classification_mdd_vs_hc')
    data_dir = os.path.join(output_root, 'data')
    report_dir = os.path.join(output_root, 'reports')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    network_measures = load_or_compute_network_measures()
    X, y, feature_names, subject_ids = build_feature_matrix(network_measures)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Subjects: {len(subject_ids)}")
    print(f"Class 0 (Healthy): {np.sum(y == 0)}")
    print(f"Class 1 (Patient): {np.sum(y == 1)}")

    summary_rows = []

    for model_name, model_spec in MODEL_SPECS.items():
        print(f"\nEvaluating model: {model_name}")
        summary, fold_df, coeffs_mean, best_params = nested_cv_evaluate(
            X, y, model_name, model_spec
        )
        summary_rows.append(summary)

        fold_df.to_csv(
            os.path.join(data_dir, f'cv_folds_{model_name}.csv'),
            index=False
        )

        if coeffs_mean is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coeffs_mean,
                'abs_coefficient': np.abs(coeffs_mean)
            }).sort_values(by='abs_coefficient', ascending=False)

            importance_df.to_csv(
                os.path.join(data_dir, f'feature_importance_{model_name}.csv'),
                index=False
            )

        best_model, train_acc, final_coeffs, best_params_full = fit_final_model(
            X, y, model_spec
        )

        report_path = os.path.join(report_dir, f'classification_report_{model_name}.txt')
        with open(report_path, 'w') as f:
            f.write("MDD vs Healthy Classification Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Best Params (outer CV mean): {best_params[:3]}...\n")
            f.write(f"Best Params (full data): {best_params_full}\n\n")

            f.write("Nested CV Metrics (mean ± std):\n")
            f.write(f"  Accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}\n")
            f.write(
                f"  Balanced Accuracy: {summary['mean_balanced_accuracy']:.4f} ± "
                f"{summary['std_balanced_accuracy']:.4f}\n"
            )
            f.write(f"  F1: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}\n")
            f.write(f"  ROC AUC: {summary['mean_roc_auc']:.4f} ± {summary['std_roc_auc']:.4f}\n\n")

            f.write(f"Train Accuracy (full data): {train_acc:.4f}\n\n")

            if final_coeffs is not None:
                top_idx = np.argsort(np.abs(final_coeffs))[::-1][:20]
                f.write("Top 20 Features (by |coef|):\n")
                for idx in top_idx:
                    f.write(
                        f"  {feature_names[idx]}: {final_coeffs[idx]:.6f}\n"
                    )

        print(f"Saved report: {report_path}")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by='mean_balanced_accuracy', ascending=False
    )
    summary_df.to_csv(os.path.join(data_dir, 'model_summary.csv'), index=False)
    print(f"Saved summary: {os.path.join(data_dir, 'model_summary.csv')}")


if __name__ == "__main__":
    main()
