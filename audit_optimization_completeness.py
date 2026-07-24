"""Audit which configured patients have a saved feasible result in each band."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from saved_results_utils import load_dataset_profile, load_npy_dict, ordered_bands


def audit(dataset_config: str, output: str | None = None) -> Path:
    profile = load_dataset_profile(dataset_config)
    connectivity = load_npy_dict(
        profile.data_dir / 'connectivity_matrices.npy', 'connectivity matrices'
    )
    results = load_npy_dict(profile.optimization_results_path, 'optimization results')
    patient_ids = sorted(str(value) for value in connectivity['Patient'])
    metadata = next(iter(results.values()))
    bands = list(metadata.get('band_names') or ordered_bands(results))
    rows = []
    for band in bands:
        by_subject = {
            str(result.get('subject_id')): result
            for result in results.values()
            if isinstance(result, dict) and str(result.get('fixed_band_name')) == band
        }
        for subject_id in patient_ids:
            result = by_subject.get(subject_id)
            solution = (result or {}).get('best_solution') or {}
            success = bool(result is not None and solution and solution.get('feasible', True))
            initial = (result or {}).get('initial_metrics') or [np.nan]
            final = (result or {}).get('final_metrics') or [np.nan]
            rows.append({
                'band': band,
                'subject_id': subject_id,
                'status': 'optimized_feasible' if success else 'no_saved_feasible_solution',
                'initial_patient_probability': float(initial[0]),
                'optimized_patient_probability': float(final[0]),
                'stimulation_model': solution.get(
                    'stimulation_model',
                    (result or {}).get('stimulation_model', 'state_space'),
                ),
                'stimulation_amplitude': float(solution.get('stimulation_amplitude', np.nan)),
                'stimulation_total_change': (
                    float(solution['stimulation_total_change'])
                    if solution.get('stimulation_total_change') is not None
                    else np.nan
                ),
                'constraint_violation': float(solution.get('constraint_violation', np.nan)),
            })
    destination = Path(output) if output else (
        profile.optimization_dir / 'optimization_subject_completeness.csv'
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(destination, index=False)
    summary = table.groupby(['band', 'status']).size().unstack(fill_value=0)
    print(summary.to_string())
    print(f"Saved optimization completeness audit: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-config', required=True)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    audit(args.dataset_config, args.output)


if __name__ == '__main__':
    main()
