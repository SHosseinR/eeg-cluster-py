"""
Analyze patient separability in SVM-selected 3-metric spaces.

This script discovers all finite scalar network metrics for each active band,
uses SVM-RFE to select exactly 3 metrics per band, trains one linear SVM per
band in that true 3D space, ranks patients by margin distance, and writes
rejection sets for 0, 10, 20, 30, 40, and 50 percent rejection.
"""

import argparse
import math
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import OUTPUT_DIR, FREQUENCY_BANDS, SELECTED_METHOD


REJECTION_PERCENTAGES = [0, 10, 20, 30, 40, 50]


def _safe_subject_value(subject_data, method, band, measure):
    if method not in subject_data:
        return np.nan
    if band not in subject_data[method]:
        return np.nan
    if measure not in subject_data[method][band]:
        return np.nan
    return subject_data[method][band][measure]


def _is_finite_scalar(value) -> bool:
    try:
        array_value = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return array_value.ndim == 0 and bool(np.isfinite(array_value))


def _discover_all_finite_measures(network_measures, method, band_name: str) -> List[str]:
    """Return scalar metrics that are finite for every usable subject in a band."""
    subjects = []
    for group_name in ('Healthy', 'Patient'):
        for subject_data in network_measures.get(group_name, {}).values():
            if method in subject_data and band_name in subject_data[method]:
                subjects.append(subject_data)

    if not subjects:
        return []

    common_measures = set(subjects[0][method][band_name].keys())
    for subject_data in subjects[1:]:
        common_measures.intersection_update(subject_data[method][band_name].keys())

    finite_measures = []
    for measure in sorted(common_measures):
        if all(_is_finite_scalar(_safe_subject_value(subject_data, method, band_name, measure))
               for subject_data in subjects):
            finite_measures.append(measure)

    return finite_measures


def _extract_group_points(network_measures, group_name, method, band, measures):
    points = []
    subject_ids = []
    for subject_id, subject_data in network_measures.get(group_name, {}).items():
        values = [
            _safe_subject_value(subject_data, method, band, measure)
            for measure in measures
        ]
        values = np.asarray(values, dtype=float)
        if np.all(np.isfinite(values)):
            points.append(values)
            subject_ids.append(subject_id)

    if not points:
        return np.empty((0, len(measures)), dtype=float), []
    return np.asarray(points, dtype=float), subject_ids


def _select_three_metrics_svm_rfe(healthy_points, patient_points, candidate_measures, random_state=42):
    X = np.vstack([healthy_points, patient_points])
    y = np.array([0] * len(healthy_points) + [1] * len(patient_points), dtype=int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    estimator = SVC(
        kernel='linear',
        C=1.0,
        class_weight='balanced',
        random_state=random_state
    )
    selector = RFE(
        estimator=estimator,
        n_features_to_select=3,
        step=1
    )
    selector.fit(X_scaled, y)

    selected_indices = np.where(selector.support_)[0]
    selected_indices = sorted(
        selected_indices,
        key=lambda idx: (selector.ranking_[idx], candidate_measures[idx])
    )
    selected_measures = [candidate_measures[idx] for idx in selected_indices]

    return selected_measures, selector.ranking_


def _fit_oriented_svm(healthy_points, patient_points, random_state=42):
    X = np.vstack([healthy_points, patient_points])
    y = np.array([0] * len(healthy_points) + [1] * len(patient_points), dtype=int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if X_scaled.shape[1] < 3:
        raise ValueError(
            f"At least 3 finite metrics are required for 3D plotting; got {X_scaled.shape[1]}"
        )

    clf = SVC(
        kernel='linear',
        C=1.0,
        class_weight='balanced',
        random_state=random_state
    )
    clf.fit(X_scaled, y)

    decision = clf.decision_function(X_scaled)
    if np.mean(decision[y == 1]) < np.mean(decision[y == 0]):
        orientation = -1.0
    else:
        orientation = 1.0

    coef = clf.coef_[0] * orientation
    intercept = float(clf.intercept_[0] * orientation)
    signed_decision = (X_scaled @ coef + intercept).astype(float)
    signed_distance = signed_decision / (np.linalg.norm(coef) + 1e-12)

    return {
        'clf': clf,
        'scaler': scaler,
        'coef': coef,
        'intercept': intercept,
        'signed_decision': signed_decision,
        'signed_distance': signed_distance,
        'X': X,
        'X_scaled': X_scaled,
        'y': y,
    }


def _patient_percentile_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=float)
    if len(values) == 1:
        ranks[order] = 0.0
        return ranks
    ranks[order] = np.arange(len(values), dtype=float) / float(len(values) - 1)
    return ranks


def _evaluate_cv_accuracy(X, y, random_state=42):
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2 or np.min(counts) < 2:
        return np.nan, np.nan, 0

    n_splits = min(5, int(np.min(counts)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs = []
    bal_accs = []

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=random_state)
        clf.fit(X_train, y[train_idx])
        pred = clf.predict(X_test)
        accs.append(accuracy_score(y[test_idx], pred))
        bal_accs.append(balanced_accuracy_score(y[test_idx], pred))

    return float(np.mean(accs)), float(np.mean(bal_accs)), n_splits


def _plane_z_grid(svm_info, x_grid, y_grid, plane_value):
    """Return z values in original metric coordinates for oriented decision=plane_value."""
    scaler = svm_info['scaler']
    coef = svm_info['coef']
    intercept = svm_info['intercept']

    if abs(coef[2]) < 1e-10:
        return None

    x_scaled = (x_grid - scaler.mean_[0]) / scaler.scale_[0]
    y_scaled = (y_grid - scaler.mean_[1]) / scaler.scale_[1]
    z_scaled = -(coef[0] * x_scaled + coef[1] * y_scaled + intercept - plane_value) / coef[2]
    return z_scaled * scaler.scale_[2] + scaler.mean_[2]


def _plot_band_margin_3d(
    band,
    measures,
    healthy_points,
    patient_points,
    healthy_ids,
    patient_ids,
    svm_info,
    rejected_subjects,
    percent,
    accuracy,
    balanced_accuracy,
    n_cv_folds,
    output_path,
):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')

    patient_rejected = np.array([sid in rejected_subjects for sid in patient_ids], dtype=bool)
    patient_retained = ~patient_rejected

    ax.scatter(
        healthy_points[:, 0], healthy_points[:, 1], healthy_points[:, 2],
        c='#2C7FB8', label=f'Healthy (n={len(healthy_ids)})',
        alpha=0.82, s=52, edgecolors='k', linewidths=0.35
    )
    if np.any(patient_retained):
        ax.scatter(
            patient_points[patient_retained, 0],
            patient_points[patient_retained, 1],
            patient_points[patient_retained, 2],
            c='#F2C94C', label=f'Retained patients (n={int(np.sum(patient_retained))})',
            alpha=0.92, s=62, edgecolors='k', linewidths=0.4
        )
    if np.any(patient_rejected):
        ax.scatter(
            patient_points[patient_rejected, 0],
            patient_points[patient_rejected, 1],
            patient_points[patient_rejected, 2],
            c='#D62728', label=f'Rejected patients (n={int(np.sum(patient_rejected))})',
            alpha=0.95, s=76, marker='^', edgecolors='k', linewidths=0.55
        )

    all_points = np.vstack([healthy_points, patient_points])
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    x_pad = (x_max - x_min) * 0.08 or 1.0
    y_pad = (y_max - y_min) * 0.08 or 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min - x_pad, x_max + x_pad, 18),
        np.linspace(y_min - y_pad, y_max + y_pad, 18)
    )

    for plane_value, color, alpha, label in [
        (0.0, '#222222', 0.18, 'SVM plane'),
        (1.0, '#3A7D44', 0.10, 'Patient margin'),
        (-1.0, '#7A8FA6', 0.10, 'Healthy margin'),
    ]:
        zz = _plane_z_grid(svm_info, xx, yy, plane_value)
        if zz is not None:
            ax.plot_surface(xx, yy, zz, color=color, alpha=alpha, linewidth=0, label=label)

    ax.set_xlabel(measures[0])
    ax.set_ylabel(measures[1])
    ax.set_zlabel(measures[2])
    ax.set_title(
        f'{band.upper()} 3-Metric SVM Margin Separation | rejected={percent}%\n'
        f'Accuracy={accuracy:.3f}, Balanced accuracy={balanced_accuracy:.3f}, CV folds={n_cv_folds}',
        fontsize=12,
        fontweight='bold'
    )

    ax.legend(loc='best')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved SVM margin plot: {output_path}')


def _plot_aggregate_ranking(ranking_df, percentages, output_path):
    plot_df = ranking_df.sort_values('aggregate_rank', ascending=True).copy()
    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(max(12, len(plot_df) * 0.38), 6))
    ax.bar(
        x,
        plot_df['aggregate_margin_percentile'],
        color='#F2C94C',
        edgecolor='black',
        linewidth=0.4
    )

    n_patients = len(plot_df)
    for percent in percentages:
        if percent <= 0:
            continue
        reject_n = int(math.ceil(percent / 100.0 * n_patients))
        if reject_n <= 0 or reject_n > n_patients:
            continue
        ax.axvline(reject_n - 0.5, color='#D62728', linestyle='--', linewidth=1.0)
        ax.text(
            reject_n - 0.5,
            1.02,
            f'{percent}%',
            rotation=90,
            ha='right',
            va='bottom',
            color='#D62728',
            fontsize=9
        )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['subject_id'], rotation=75, ha='right', fontsize=8)
    ax.set_ylabel('Aggregate margin percentile (higher = clearer patient-side separation)')
    ax.set_title('Patient Ranking by SVM Margin Separability', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    ax.set_ylim(0, 1.08)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved aggregate ranking plot: {output_path}')


def _plot_band_ranking(band, ranking_df, percentages, output_path):
    plot_df = ranking_df.sort_values('rank', ascending=True).copy()
    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(max(12, len(plot_df) * 0.38), 6))
    ax.bar(
        x,
        plot_df['margin_percentile'],
        color='#F2C94C',
        edgecolor='black',
        linewidth=0.4
    )

    n_patients = len(plot_df)
    for percent in percentages:
        if percent <= 0:
            continue
        reject_n = int(math.ceil(percent / 100.0 * n_patients))
        if reject_n <= 0 or reject_n > n_patients:
            continue
        ax.axvline(reject_n - 0.5, color='#D62728', linestyle='--', linewidth=1.0)
        ax.text(
            reject_n - 0.5,
            1.02,
            f'{percent}%',
            rotation=90,
            ha='right',
            va='bottom',
            color='#D62728',
            fontsize=9
        )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['subject_id'], rotation=75, ha='right', fontsize=8)
    ax.set_ylabel('Margin percentile (higher = clearer patient-side separation)')
    ax.set_title(f'{band.upper()} Patient Ranking by 3-Metric SVM Margin', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    ax.set_ylim(0, 1.08)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {band} ranking plot: {output_path}')


def _plot_band_accuracy_by_rejection(band, accuracy_df, output_path):
    plot_df = (
        accuracy_df[accuracy_df['band'] == band]
        .sort_values('rejection_percent')
        .copy()
    )
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        plot_df['rejection_percent'],
        plot_df['accuracy'],
        marker='o',
        linewidth=2,
        label='Accuracy'
    )
    ax.plot(
        plot_df['rejection_percent'],
        plot_df['balanced_accuracy'],
        marker='s',
        linewidth=2,
        label='Balanced accuracy'
    )
    ax.set_xlabel('Patient rejection (%)')
    ax.set_ylabel('Cross-validated score')
    ax.set_title(f'{band.upper()} 3-Metric SVM Accuracy by Rejection Rate', fontsize=13, fontweight='bold')
    ax.set_xticks(plot_df['rejection_percent'])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {band} accuracy plot: {output_path}')


def analyze_svm_margin_rejection(network_measures, method, output_dir, figures_dir, percentages):
    band_results = {}
    patient_rows_by_band = []
    selected_metric_rows = []
    rejection_rows = []
    accuracy_rows = []
    ranking_paths = {}
    ranking_plot_paths = {}
    accuracy_plot_paths = {}

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    for band in FREQUENCY_BANDS.keys():
        candidate_measures = _discover_all_finite_measures(network_measures, method, band)
        if len(candidate_measures) < 3:
            raise ValueError(
                f"Band '{band}' must have at least 3 finite scalar metrics for SVM-RFE. "
                f"Got {len(candidate_measures)}: {candidate_measures}"
            )

        all_healthy_points, healthy_ids = _extract_group_points(
            network_measures, 'Healthy', method, band, candidate_measures
        )
        all_patient_points, patient_ids = _extract_group_points(
            network_measures, 'Patient', method, band, candidate_measures
        )
        if len(healthy_ids) < 2 or len(patient_ids) < 2:
            raise ValueError(
                f"Band '{band}' needs at least 2 valid subjects per group. "
                f"Healthy={len(healthy_ids)}, Patient={len(patient_ids)}"
            )

        selected_measures, rfe_ranking = _select_three_metrics_svm_rfe(
            all_healthy_points,
            all_patient_points,
            candidate_measures
        )
        print(
            f"Band {band}: selected 3 of {len(candidate_measures)} candidate metrics: "
            f"{selected_measures}"
        )

        healthy_points, healthy_ids = _extract_group_points(
            network_measures, 'Healthy', method, band, selected_measures
        )
        patient_points, patient_ids = _extract_group_points(
            network_measures, 'Patient', method, band, selected_measures
        )

        svm_info = _fit_oriented_svm(healthy_points, patient_points)
        patient_distances = svm_info['signed_distance'][len(healthy_ids):]
        patient_percentiles = _patient_percentile_ranks(patient_distances)

        ranking_df = pd.DataFrame({
            'subject_id': patient_ids,
            'band': band,
            'signed_margin_distance': patient_distances.astype(float),
            'margin_percentile': patient_percentiles.astype(float),
            'selected_metrics': ', '.join(selected_measures),
            'metric_count': len(selected_measures),
            'candidate_metric_count': len(candidate_measures),
        })
        ranking_df = (
            ranking_df
            .sort_values(
                ['margin_percentile', 'signed_margin_distance', 'subject_id'],
                ascending=[True, True, True]
            )
            .reset_index(drop=True)
        )
        ranking_df['rank'] = np.arange(1, len(ranking_df) + 1)

        for row in ranking_df.to_dict('records'):
            patient_rows_by_band.append(row)

        selected_metric_rows.append({
            'band': band,
            'selected_metric_1': selected_measures[0],
            'selected_metric_2': selected_measures[1],
            'selected_metric_3': selected_measures[2],
            'selected_metrics': ', '.join(selected_measures),
            'candidate_metric_count': len(candidate_measures),
            'candidate_metrics': ', '.join(candidate_measures),
            'rfe_ranking': ';'.join(
                f"{measure}:{int(rank)}"
                for measure, rank in zip(candidate_measures, rfe_ranking)
            ),
        })

        band_results[band] = {
            'measures': selected_measures,
            'candidate_measures': candidate_measures,
            'healthy_points': healthy_points,
            'patient_points': patient_points,
            'healthy_ids': healthy_ids,
            'patient_ids': patient_ids,
            'svm_info': svm_info,
            'ranking_df': ranking_df,
        }

    for band, result in band_results.items():
        ranking_df = result['ranking_df'].copy()
        n_patients = len(ranking_df)

        for percent in percentages:
            reject_n = int(math.ceil(float(percent) / 100.0 * n_patients))
            reject_n = min(max(reject_n, 0), n_patients)
            rejected = ranking_df.head(reject_n)['subject_id'].tolist()
            retained = ranking_df.iloc[reject_n:]['subject_id'].tolist()
            ranking_df[f'rejected_at_{int(percent)}'] = ranking_df['subject_id'].isin(rejected)

            retained_mask = np.array([sid not in rejected for sid in result['patient_ids']], dtype=bool)
            X_eval = np.vstack([result['healthy_points'], result['patient_points'][retained_mask]])
            y_eval = np.array(
                [0] * len(result['healthy_ids']) + [1] * int(np.sum(retained_mask)),
                dtype=int
            )
            acc, bal_acc, n_folds = _evaluate_cv_accuracy(X_eval, y_eval)

            rejection_rows.append({
                'band': band,
                'rejection_percent': int(percent),
                'n_patients_total': n_patients,
                'n_rejected': reject_n,
                'n_retained': len(retained),
                'selected_metrics': ', '.join(result['measures']),
                'rejected_subject_ids': ';'.join(rejected),
                'retained_subject_ids': ';'.join(retained),
            })
            accuracy_rows.append({
                'band': band,
                'rejection_percent': int(percent),
                'metrics': ', '.join(result['measures']),
                'metric_count': len(result['measures']),
                'accuracy': acc,
                'balanced_accuracy': bal_acc,
                'n_cv_folds': n_folds,
                'n_healthy': len(result['healthy_ids']),
                'n_patients_retained': int(np.sum(retained_mask)),
                'n_patients_rejected': reject_n,
                'rejected_subject_ids': ';'.join(rejected),
            })

            plot_path = os.path.join(
                figures_dir,
                band,
                f"svm_margin_{band}_reject_{int(percent):02d}.png"
            )
            _plot_band_margin_3d(
                band=band,
                measures=result['measures'],
                healthy_points=result['healthy_points'],
                patient_points=result['patient_points'],
                healthy_ids=result['healthy_ids'],
                patient_ids=result['patient_ids'],
                svm_info=result['svm_info'],
                rejected_subjects=set(rejected),
                percent=int(percent),
                accuracy=acc,
                balanced_accuracy=bal_acc,
                n_cv_folds=n_folds,
                output_path=plot_path,
            )

        ranking_path = os.path.join(output_dir, f'svm_margin_subject_ranking_{band}.csv')
        ranking_plot_path = os.path.join(
            figures_dir,
            band,
            f'svm_margin_{band}_patient_ranking.png'
        )
        ranking_df.to_csv(ranking_path, index=False)
        _plot_band_ranking(band, ranking_df, percentages, ranking_plot_path)
        ranking_paths[band] = ranking_path
        ranking_plot_paths[band] = ranking_plot_path
        band_results[band]['ranking_df'] = ranking_df

    accuracy_df = pd.DataFrame(accuracy_rows)
    rejection_df = pd.DataFrame(rejection_rows)
    selected_metrics_df = pd.DataFrame(selected_metric_rows)
    patient_band_df = pd.DataFrame(patient_rows_by_band)
    aggregate_df = (
        patient_band_df
        .groupby('subject_id')
        .agg(
            aggregate_margin_percentile=('margin_percentile', 'mean'),
            min_signed_margin_distance=('signed_margin_distance', 'min'),
            mean_signed_margin_distance=('signed_margin_distance', 'mean'),
            n_bands=('band', 'count')
        )
        .reset_index()
        .sort_values(
            ['aggregate_margin_percentile', 'min_signed_margin_distance', 'subject_id'],
            ascending=[True, True, True]
        )
        .reset_index(drop=True)
    )
    aggregate_df['aggregate_rank'] = np.arange(1, len(aggregate_df) + 1)

    for band, result in band_results.items():
        for _, row in result['ranking_df'].iterrows():
            aggregate_df.loc[
                aggregate_df['subject_id'] == row['subject_id'],
                f'{band}_rank'
            ] = row['rank']
            aggregate_df.loc[
                aggregate_df['subject_id'] == row['subject_id'],
                f'{band}_margin_percentile'
            ] = row['margin_percentile']
            aggregate_df.loc[
                aggregate_df['subject_id'] == row['subject_id'],
                f'{band}_signed_margin_distance'
            ] = row['signed_margin_distance']

    for band in band_results:
        accuracy_plot_path = os.path.join(
            figures_dir,
            band,
            f'svm_margin_{band}_accuracy_by_rejection.png'
        )
        _plot_band_accuracy_by_rejection(band, accuracy_df, accuracy_plot_path)
        accuracy_plot_paths[band] = accuracy_plot_path

    selected_metrics_path = os.path.join(output_dir, 'svm_margin_selected_metrics.csv')
    aggregate_ranking_path = os.path.join(output_dir, 'svm_margin_subject_ranking.csv')
    aggregate_ranking_reference_path = os.path.join(output_dir, 'svm_margin_subject_ranking_aggregate.csv')
    rejection_path = os.path.join(output_dir, 'svm_margin_rejection_sets_by_band.csv')
    legacy_rejection_path = os.path.join(output_dir, 'svm_margin_rejection_sets.csv')
    accuracy_path = os.path.join(output_dir, 'svm_margin_accuracy_by_rejection.csv')
    aggregate_ranking_plot_path = os.path.join(figures_dir, 'svm_margin_patient_ranking.png')

    selected_metrics_df.to_csv(selected_metrics_path, index=False)
    aggregate_df.to_csv(aggregate_ranking_path, index=False)
    aggregate_df.to_csv(aggregate_ranking_reference_path, index=False)
    rejection_df.to_csv(rejection_path, index=False)
    rejection_df.to_csv(legacy_rejection_path, index=False)
    accuracy_df.to_csv(accuracy_path, index=False)
    _plot_aggregate_ranking(aggregate_df, percentages, aggregate_ranking_plot_path)

    print("\nSaved SVM margin rejection outputs:")
    print(f"  Selected metrics: {selected_metrics_path}")
    print(f"  Per-band rankings: {ranking_paths}")
    print(f"  Aggregate ranking reference: {aggregate_ranking_reference_path}")
    print(f"  Rejection sets by band: {rejection_path}")
    print(f"  Accuracy: {accuracy_path}")
    print(f"  Per-band ranking plots: {ranking_plot_paths}")
    print(f"  Per-band accuracy plots: {accuracy_plot_paths}")

    return {
        'ranking_df': aggregate_df,
        'ranking_paths': ranking_paths,
        'selected_metrics_df': selected_metrics_df,
        'rejection_df': rejection_df,
        'accuracy_df': accuracy_df,
        'selected_metrics_path': selected_metrics_path,
        'ranking_path': aggregate_ranking_path,
        'ranking_reference_path': aggregate_ranking_reference_path,
        'rejection_path': rejection_path,
        'accuracy_path': accuracy_path,
        'ranking_plot_path': aggregate_ranking_plot_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Select 3 metrics with SVM-RFE, train linear SVM margins, and rank patients for rejection.'
    )
    parser.add_argument(
        '--network-measures-path',
        default=os.path.join(OUTPUT_DIR, 'data', 'network_measures.npy'),
        help='Path to network_measures.npy'
    )
    parser.add_argument(
        '--method',
        default=SELECTED_METHOD,
        help='Connectivity method key in network_measures'
    )
    parser.add_argument(
        '--output-dir',
        default=os.path.join(OUTPUT_DIR, 'data'),
        help='Directory for CSV outputs'
    )
    parser.add_argument(
        '--figures-dir',
        default=os.path.join(OUTPUT_DIR, 'figures', 'svm_margin_rejection'),
        help='Directory for SVM margin figures'
    )
    parser.add_argument(
        '--percentages',
        default=','.join(str(p) for p in REJECTION_PERCENTAGES),
        help='Comma-separated rejection percentages'
    )
    args = parser.parse_args()

    percentages = [int(p.strip()) for p in args.percentages.split(',') if p.strip()]
    for percent in percentages:
        if percent < 0 or percent > 100:
            raise ValueError(f'Rejection percent must be between 0 and 100: {percent}')

    if not os.path.exists(args.network_measures_path):
        raise FileNotFoundError(
            f'Network measures file not found: {args.network_measures_path}\n'
            'Run main.py through step 5 first.'
        )

    network_measures = np.load(args.network_measures_path, allow_pickle=True).item()
    result = analyze_svm_margin_rejection(
        network_measures=network_measures,
        method=args.method,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        percentages=percentages,
    )

    print("\nAccuracy by rejection percent:")
    print(result['accuracy_df'].to_string(index=False))


if __name__ == '__main__':
    main()
