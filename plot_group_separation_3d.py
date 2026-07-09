"""
Plot 3D group separation using selected optimization measures.

Loads only precomputed network measures (no raw EEG loading), extracts the
three measures listed in optimization_config.OPTIMIZATION_MEASURES, and plots
Healthy vs Patient subjects in a 3D scatter plot.
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, FREQUENCY_BANDS, SELECTED_METHOD
from optimization_config import OPTIMIZATION_MEASURES


def _safe_subject_value(subject_data, method, band, measure):
    """Safely fetch one measure value for a subject/method/band."""
    if method not in subject_data:
        return np.nan
    if band not in subject_data[method]:
        return np.nan
    if measure not in subject_data[method][band]:
        return np.nan
    return subject_data[method][band][measure]


def _extract_group_points(network_measures, group_name, method, measures, band=None):
    """
    Extract [n_subjects, 3] matrix for one group.

    If band is None, values are averaged across all configured bands.
    """
    subjects = network_measures.get(group_name, {})
    band_names = list(FREQUENCY_BANDS.keys())

    points = []
    subject_ids = []

    for subject_id, subject_data in subjects.items():
        coords = []

        for measure in measures:
            if band is None:
                vals = []
                for band_name in band_names:
                    val = _safe_subject_value(subject_data, method, band_name, measure)
                    if np.isfinite(val):
                        vals.append(float(val))
                coords.append(np.mean(vals) if len(vals) > 0 else np.nan)
            else:
                val = _safe_subject_value(subject_data, method, band, measure)
                coords.append(float(val) if np.isfinite(val) else np.nan)

        if np.all(np.isfinite(coords)):
            points.append(coords)
            subject_ids.append(subject_id)

    if len(points) == 0:
        return np.empty((0, 3)), []

    return np.asarray(points, dtype=float), subject_ids


def plot_group_separation_3d(network_measures, measures, method, output_path, band=None):
    """Create and save a 3D scatter for Healthy vs Patient groups."""
    healthy_points, healthy_ids = _extract_group_points(
        network_measures, 'Healthy', method, measures, band=band
    )
    patient_points, patient_ids = _extract_group_points(
        network_measures, 'Patient', method, measures, band=band
    )

    if healthy_points.shape[0] == 0 and patient_points.shape[0] == 0:
        raise ValueError('No valid points were found for either group.')

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if healthy_points.shape[0] > 0:
        ax.scatter(
            healthy_points[:, 0], healthy_points[:, 1], healthy_points[:, 2],
            c='tab:blue', label=f'Healthy (n={healthy_points.shape[0]})',
            alpha=0.85, s=60, edgecolors='k', linewidths=0.4
        )

    if patient_points.shape[0] > 0:
        ax.scatter(
            patient_points[:, 0], patient_points[:, 1], patient_points[:, 2],
            c='tab:red', label=f'Patient (n={patient_points.shape[0]})',
            alpha=0.85, s=60, edgecolors='k', linewidths=0.4
        )

    ax.set_xlabel(measures[0])
    ax.set_ylabel(measures[1])
    ax.set_zlabel(measures[2])

    band_text = band if band is not None else 'mean across bands'
    ax.set_title(
        f'3D Group Separation | method={method}, band={band_text}\n'
        f'Features: {measures[0]}, {measures[1]}, {measures[2]}'
    )

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved 3D scatter plot to: {output_path}')

    return {
        'healthy_n': healthy_points.shape[0],
        'patient_n': patient_points.shape[0],
        'healthy_subject_ids': healthy_ids,
        'patient_subject_ids': patient_ids,
        'output_path': output_path,
        'method': method,
        'band': band,
        'measures': measures,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Plot 3D Healthy vs Patient separation using selected optimization measures.'
    )
    parser.add_argument(
        '--network-measures-path',
        default=os.path.join(OUTPUT_DIR, 'data', 'network_measures.npy'),
        help='Path to network_measures.npy'
    )
    parser.add_argument(
        '--method',
        default=SELECTED_METHOD,
        help='Connectivity method key in network_measures (default: config.SELECTED_METHOD)'
    )
    parser.add_argument(
        '--band',
        default=None,
        help='Band name (e.g., alpha). If omitted, uses mean across all bands.'
    )
    parser.add_argument(
        '--output',
        default=os.path.join(OUTPUT_DIR, 'figures', 'group_separation_3d.png'),
        help='Output PNG path'
    )
    args = parser.parse_args()

    if len(OPTIMIZATION_MEASURES) < 3:
        raise ValueError(
            'OPTIMIZATION_MEASURES must contain at least 3 measures for a 3D plot.'
        )

    selected_measures = OPTIMIZATION_MEASURES[:3]

    if args.band is not None and args.band not in FREQUENCY_BANDS:
        raise ValueError(
            f"Invalid band '{args.band}'. Valid bands: {list(FREQUENCY_BANDS.keys())}"
        )

    if not os.path.exists(args.network_measures_path):
        raise FileNotFoundError(
            f'Network measures file not found: {args.network_measures_path}\n'
            f'Run main.py step 5 first to generate it.'
        )

    network_measures = np.load(args.network_measures_path, allow_pickle=True).item()

    result = plot_group_separation_3d(
        network_measures=network_measures,
        measures=selected_measures,
        method=args.method,
        output_path=args.output,
        band=args.band,
    )

    print('\nSummary:')
    print(f"  Method: {result['method']}")
    print(f"  Band: {result['band'] if result['band'] is not None else 'mean across bands'}")
    print(f"  Measures: {', '.join(result['measures'])}")
    print(f"  Healthy points: {result['healthy_n']}")
    print(f"  Patient points: {result['patient_n']}")


if __name__ == '__main__':
    main()
