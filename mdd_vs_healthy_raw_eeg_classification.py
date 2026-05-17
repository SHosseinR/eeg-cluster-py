"""
MDD vs Healthy classification from raw EEG.

This script loads raw EEG (.set), extracts common spectral features
(band power, relative band power, Hjorth parameters, spectral entropy),
and evaluates linear models with nested cross-validation.
"""

import os
import numpy as np
import pandas as pd
from scipy import signal, integrate

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
    FMIN,
    FMAX,
    EPOCH_DURATION,
    OVERLAP,
    N_FOLDS,
    RANDOM_STATE
)
from data_loader import load_group_data, verify_data_consistency
from signal_processing import create_epochs

# ============================================================================
# SETTINGS
# ============================================================================
USE_FEATURE_CACHE = True
FEATURE_CACHE_PATH = os.path.join(
    OUTPUT_DIR, 'data', 'mdd_vs_hc_raw_eeg_features.npz'
)
MAX_SUBJECTS_PER_GROUP = None  # set to an int for quick tests

MODEL_SPECS = {
    'linear_svm': {
        'estimator': LinearSVC(
            class_weight='balanced',
            max_iter=5000,
            random_state=RANDOM_STATE
        ),
        'param_grid': {'clf__C': [0.01, 0.1, 1.0, 10.0]}
    },
    'logistic': {
        'estimator': LogisticRegression(
            class_weight='balanced',
            solver='liblinear',
            max_iter=2000,
            random_state=RANDOM_STATE
        ),
        'param_grid': {'clf__C': [0.01, 0.1, 1.0, 10.0]}
    }
}

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def _compute_psd_welch(epochs, fs):
    n_samples = epochs.shape[-1]
    nperseg = min(int(fs * 2), n_samples)
    freqs, psd = signal.welch(epochs, fs=fs, nperseg=nperseg, axis=-1)
    return freqs, psd


def _bandpower_from_psd(psd, freqs, band):
    low, high = band
    idx = (freqs >= low) & (freqs <= high)
    if not np.any(idx):
        return np.zeros(psd.shape[:-1])
    if hasattr(np, "trapezoid"):
        return np.trapezoid(psd[..., idx], freqs[idx], axis=-1)
    if hasattr(np, "trapz"):
        return np.trapz(psd[..., idx], freqs[idx], axis=-1)
    return integrate.trapezoid(psd[..., idx], freqs[idx], axis=-1)


def _spectral_entropy(psd, freqs, band):
    low, high = band
    idx = (freqs >= low) & (freqs <= high)
    if not np.any(idx):
        return np.zeros(psd.shape[:-1])
    psd_band = psd[..., idx]
    psd_sum = np.sum(psd_band, axis=-1, keepdims=True)
    prob = np.divide(psd_band, psd_sum, out=np.zeros_like(psd_band), where=psd_sum > 0)
    entropy = -np.sum(prob * np.log(prob + 1e-12), axis=-1)
    return entropy


def _hjorth_parameters(epochs):
    diff1 = np.diff(epochs, axis=-1)
    diff2 = np.diff(diff1, axis=-1)

    var0 = np.var(epochs, axis=-1)
    var1 = np.var(diff1, axis=-1)
    var2 = np.var(diff2, axis=-1)

    activity = var0
    mobility = np.sqrt(np.divide(var1, var0, out=np.zeros_like(var1), where=var0 > 0))
    complexity = np.sqrt(np.divide(var2, var1, out=np.zeros_like(var2), where=var1 > 0))
    complexity = np.divide(complexity, mobility, out=np.zeros_like(complexity), where=mobility > 0)

    return activity, mobility, complexity


def extract_subject_features(data, fs, channel_names):
    epochs = create_epochs(data, fs, epoch_duration=EPOCH_DURATION, overlap=OVERLAP)
    freqs, psd = _compute_psd_welch(epochs, fs)

    total_power = _bandpower_from_psd(psd, freqs, (FMIN, FMAX))

    features = []
    feature_names = []

    for band_name, band in FREQUENCY_BANDS.items():
        band_power = _bandpower_from_psd(psd, freqs, band)
        log_power = np.log10(band_power + 1e-12)
        rel_power = np.divide(band_power, total_power, out=np.zeros_like(band_power), where=total_power > 0)

        mean_log_power = np.mean(log_power, axis=0)
        mean_rel_power = np.mean(rel_power, axis=0)

        for ch_idx, ch_name in enumerate(channel_names):
            features.append(mean_log_power[ch_idx])
            feature_names.append(f"log_power_{band_name}_{ch_name}")
            features.append(mean_rel_power[ch_idx])
            feature_names.append(f"rel_power_{band_name}_{ch_name}")

    activity, mobility, complexity = _hjorth_parameters(epochs)
    activity = np.mean(activity, axis=0)
    mobility = np.mean(mobility, axis=0)
    complexity = np.mean(complexity, axis=0)

    for ch_idx, ch_name in enumerate(channel_names):
        features.append(activity[ch_idx])
        feature_names.append(f"hjorth_activity_{ch_name}")
        features.append(mobility[ch_idx])
        feature_names.append(f"hjorth_mobility_{ch_name}")
        features.append(complexity[ch_idx])
        feature_names.append(f"hjorth_complexity_{ch_name}")

    entropy = _spectral_entropy(psd, freqs, (FMIN, FMAX))
    entropy = np.mean(entropy, axis=0)

    for ch_idx, ch_name in enumerate(channel_names):
        features.append(entropy[ch_idx])
        feature_names.append(f"spectral_entropy_{ch_name}")

    return np.array(features, dtype=float), feature_names


def build_dataset():
    if USE_FEATURE_CACHE and os.path.exists(FEATURE_CACHE_PATH):
        cache = np.load(FEATURE_CACHE_PATH, allow_pickle=True)
        return cache['X'], cache['y'], cache['feature_names'].tolist(), cache['subject_ids'].tolist()

    healthy_data = load_group_data(HC_DATA_PATH, group_name="Healthy")
    patient_data = load_group_data(PATIENT_DATA_PATH, group_name="Patient")

    if MAX_SUBJECTS_PER_GROUP is not None:
        healthy_data = healthy_data[:MAX_SUBJECTS_PER_GROUP]
        patient_data = patient_data[:MAX_SUBJECTS_PER_GROUP]

    all_data = healthy_data + patient_data
    if not verify_data_consistency(all_data):
        raise ValueError("Data consistency check failed")

    X_list = []
    y_list = []
    subject_ids = []
    feature_names_ref = None

    for subject in healthy_data:
        features, feature_names = extract_subject_features(
            subject['data'], subject['fs'], subject['channels']
        )
        if feature_names_ref is None:
            feature_names_ref = feature_names
        X_list.append(features)
        y_list.append(0)
        subject_ids.append(subject['subject_id'])

    for subject in patient_data:
        features, feature_names = extract_subject_features(
            subject['data'], subject['fs'], subject['channels']
        )
        if feature_names_ref is None:
            feature_names_ref = feature_names
        X_list.append(features)
        y_list.append(1)
        subject_ids.append(subject['subject_id'])

    X = np.vstack(X_list)
    y = np.array(y_list)

    os.makedirs(os.path.dirname(FEATURE_CACHE_PATH), exist_ok=True)
    np.savez(
        FEATURE_CACHE_PATH,
        X=X,
        y=y,
        feature_names=np.array(feature_names_ref, dtype=object),
        subject_ids=np.array(subject_ids, dtype=object)
    )

    return X, y, feature_names_ref, subject_ids

# ============================================================================
# MODELING
# ============================================================================

def _get_score_values(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return None


def nested_cv_evaluate(X, y, model_name, model_spec, n_splits=N_FOLDS):
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
    output_root = os.path.join(OUTPUT_DIR, 'classification_mdd_vs_hc_raw_eeg')
    data_dir = os.path.join(output_root, 'data')
    report_dir = os.path.join(output_root, 'reports')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    X, y, feature_names, subject_ids = build_dataset()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

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

        _, train_acc, final_coeffs, best_params_full = fit_final_model(
            X, y, model_spec
        )

        report_path = os.path.join(report_dir, f'classification_report_{model_name}.txt')
        with open(report_path, 'w') as f:
            f.write("MDD vs Healthy Classification Report (Raw EEG)\n")
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
                    f.write(f"  {feature_names[idx]}: {final_coeffs[idx]:.6f}\n")

        print(f"Saved report: {report_path}")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by='mean_balanced_accuracy', ascending=False
    )
    summary_df.to_csv(os.path.join(data_dir, 'model_summary.csv'), index=False)
    print(f"Saved summary: {os.path.join(data_dir, 'model_summary.csv')}")


if __name__ == "__main__":
    main()
