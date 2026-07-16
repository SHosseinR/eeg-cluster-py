"""TD-BRAIN age/sex matching sensitivity for connectivity classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .confound_checks import optimal_age_sex_match
    from .connectivity_benchmark import _compute_profile_caches, _load_benchmark_features
    from .connectivity_sensitivity import connectivity_transformations
    from .modeling import nested_oof_evaluate
except ImportError:
    from confound_checks import optimal_age_sex_match
    from connectivity_benchmark import _compute_profile_caches, _load_benchmark_features
    from connectivity_sensitivity import connectivity_transformations
    from modeling import nested_oof_evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="coherence")
    parser.add_argument("--model", default="logistic_l2")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "tdbrain_participants.csv",
    )
    args = parser.parse_args()

    records, data_root = _compute_profile_caches(
        "tdbrain",
        ("fourier", "envelope"),
        n_jobs=args.n_jobs,
        resume=True,
        max_subjects_per_group=None,
    )
    matrices, y, subject_ids, _, _ = _load_benchmark_features(
        records, data_root, [args.method]
    )
    metadata = pd.read_csv(args.metadata, dtype={"subject_id": str}).set_index("subject_id")
    metadata = metadata.loc[subject_ids].reset_index()
    expected_groups = np.where(y == 0, "Healthy", "Patient")
    if not np.array_equal(metadata["group"].to_numpy(), expected_groups):
        raise ValueError("Metadata cohorts do not align with connectivity cache labels")
    metadata["age"] = pd.to_numeric(metadata["age"], errors="raise")
    metadata["gender"] = pd.to_numeric(metadata["gender"], errors="raise").astype(int)

    matches = optimal_age_sex_match(metadata)
    selected = set(matches["healthy_subject_id"]) | set(matches["patient_subject_id"])
    mask = np.asarray([subject_id in selected for subject_id in subject_ids])
    transformations = connectivity_transformations(matrices[args.method])
    output = (
        Path(__file__).resolve().parent
        / "results"
        / "connectivity_benchmark"
        / "tdbrain"
        / "age_sex_matched"
    )
    output.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output / "matches.csv", index=False)
    rows = []
    for transformation in ("natural_edges", "within_subject_centered"):
        feature_set = f"{args.method}__{transformation}__age_sex_matched"
        summary, predictions = nested_oof_evaluate(
            transformations[transformation][mask],
            y[mask],
            feature_set=feature_set,
            model_name=args.model,
            mode="quick",
            outer_splits=5,
            repeats=args.repeats,
            inner_splits=3,
            n_jobs=args.n_jobs,
            subject_ids=subject_ids[mask],
        )
        summary.update(
            {
                "connectivity_method": args.method,
                "transformation": transformation,
                "matched_pairs": len(matches),
                "median_absolute_pair_age_difference": matches[
                    "absolute_age_difference"
                ].median(),
            }
        )
        rows.append(summary)
        predictions.to_csv(output / f"predictions__{transformation}.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    print(
        pd.DataFrame(rows)[
            [
                "feature_set",
                "n_subjects",
                "roc_auc_repeat_mean",
                "roc_auc_repeat_sd",
                "balanced_accuracy_repeat_mean",
                "brier_repeat_mean",
            ]
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
