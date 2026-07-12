"""Generate standalone within-band and cross-band optimization diagnostics.

This script is intentionally independent from ``run_optimization.py``. It only
reads completed per-band ``.npy`` result files and writes analysis artifacts to
a separate directory.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from band_stability_analysis import run_band_stability_analysis


RESULT_SUFFIX = "_optimization_results.npy"


def _discover_bands(results_dir: str) -> List[str]:
    paths = glob.glob(os.path.join(results_dir, f"*{RESULT_SUFFIX}"))
    bands = []
    for path in paths:
        filename = os.path.basename(path)
        band = filename[:-len(RESULT_SUFFIX)]
        if band:
            bands.append(band)
    return sorted(set(bands))


def load_per_band_results(results_dir: str, bands: List[str] | None = None) -> Dict:
    """Load one completed result dictionary per band."""
    results_dir = os.path.abspath(results_dir)
    bands = bands or _discover_bands(results_dir)
    if not bands:
        raise FileNotFoundError(
            f"No files matching '*{RESULT_SUFFIX}' were found in {results_dir}"
        )

    results_by_band = {}
    missing = []
    for band in bands:
        path = os.path.join(results_dir, f"{band}{RESULT_SUFFIX}")
        if not os.path.exists(path):
            missing.append(path)
            continue
        results = np.load(path, allow_pickle=True).item()
        if not isinstance(results, dict) or not results:
            raise ValueError(f"Result file is empty or invalid: {path}")
        results_by_band[band] = results
    if missing:
        raise FileNotFoundError("Missing requested band result files:\n" + "\n".join(missing))
    if len(results_by_band) < 2:
        raise ValueError("Cross-band analysis requires at least two completed band result files.")
    return results_by_band


def align_subject_cohort(results_by_band: Dict, policy: str = "intersection"):
    """Return matched results and a summary of subjects excluded by alignment."""
    subject_sets = {band: set(results) for band, results in results_by_band.items()}
    common_subjects = set.intersection(*subject_sets.values())
    if not common_subjects:
        raise ValueError("The supplied band result files have no subjects in common.")

    identical = all(subjects == common_subjects for subjects in subject_sets.values())
    if policy == "strict" and not identical:
        counts = {band: len(subjects) for band, subjects in subject_sets.items()}
        raise ValueError(
            "Strict cohort policy requires identical subject IDs in every band. "
            f"Counts={counts}, common={len(common_subjects)}"
        )

    common_subjects = sorted(common_subjects)
    aligned = {
        band: {subject: results[subject] for subject in common_subjects}
        for band, results in results_by_band.items()
    }
    rows = []
    for band, subjects in subject_sets.items():
        excluded = sorted(subjects.difference(common_subjects))
        rows.append({
            "band": band,
            "original_subject_count": len(subjects),
            "common_subject_count": len(common_subjects),
            "excluded_subject_count": len(excluded),
            "excluded_subject_ids": json.dumps(excluded),
            "cohort_policy": policy,
        })
    return aligned, pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create standalone within-band and cross-band optimization plots."
    )
    parser.add_argument(
        "results_dir",
        help="Directory containing <band>_optimization_results.npy files.",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=None,
        help="Band names to analyze. Default: auto-discover completed band files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Separate output directory. Default: <results_dir>/band_stability_analysis.",
    )
    parser.add_argument(
        "--cohort-policy",
        choices=["intersection", "strict"],
        default="intersection",
        help="Use the common subject intersection or require already-identical cohorts.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    output_dir = os.path.abspath(
        args.output_dir or os.path.join(results_dir, "band_stability_analysis")
    )
    figures_dir = os.path.join(output_dir, "figures")
    report_path = os.path.join(output_dir, "band_stability_report.txt")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading per-band results from: {results_dir}")
    results_by_band = load_per_band_results(results_dir, args.bands)
    aligned_results, cohort_summary = align_subject_cohort(
        results_by_band, policy=args.cohort_policy
    )
    cohort_path = os.path.join(output_dir, "cohort_alignment_summary.csv")
    cohort_summary.to_csv(cohort_path, index=False)
    print(cohort_summary.to_string(index=False))

    with open(report_path, "w", encoding="utf-8") as report:
        report.write("STANDALONE BAND STABILITY ANALYSIS\n")
        report.write("=" * 80 + "\n")
        report.write(f"Source results: {results_dir}\n")
        report.write(f"Cohort policy: {args.cohort_policy}\n")
        report.write(
            f"Common paired subjects: {int(cohort_summary['common_subject_count'].iloc[0])}\n"
        )

    output = run_band_stability_analysis(
        results_by_band=aligned_results,
        output_dir=output_dir,
        figures_dir=figures_dir,
        report_path=report_path,
        n_resamples=args.bootstrap_resamples,
        random_seed=args.random_seed,
    )
    print("\nAnalysis complete")
    print(f"Output directory: {output_dir}")
    print(f"Cohort summary: {cohort_path}")
    print(f"Decision: {json.dumps(output['decision'], indent=2)}")


if __name__ == "__main__":
    main()
