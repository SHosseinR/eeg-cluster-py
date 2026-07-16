"""TD-BRAIN age/sex sensitivity checks for shortlisted EEG classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

try:
    from .data_features import build_feature_dataset
    from .modeling import nested_oof_evaluate
except ImportError:
    from data_features import build_feature_dataset
    from modeling import nested_oof_evaluate


def _aligned_metadata(dataset, csv_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(csv_path, dtype={"subject_id": str})
    metadata = metadata.set_index("subject_id").loc[dataset.subject_ids].reset_index()
    metadata["age"] = pd.to_numeric(metadata["age"], errors="raise")
    metadata["gender"] = pd.to_numeric(metadata["gender"], errors="raise").astype(int)
    expected = pd.Series(dataset.groups, index=dataset.subject_ids)
    observed = metadata.set_index("subject_id")["group"]
    if not expected.equals(observed.loc[expected.index]):
        raise ValueError("Participant metadata group labels do not match the saved EEG artifacts")
    return metadata


def optimal_age_sex_match(metadata: pd.DataFrame) -> pd.DataFrame:
    """One-to-one nearest-age matching without replacement within sex."""

    rows: list[dict[str, object]] = []
    for gender in sorted(metadata["gender"].unique()):
        healthy = metadata[(metadata["group"] == "Healthy") & (metadata["gender"] == gender)]
        patient = metadata[(metadata["group"] == "Patient") & (metadata["gender"] == gender)]
        cost = np.abs(
            healthy["age"].to_numpy()[:, None] - patient["age"].to_numpy()[None, :]
        )
        healthy_index, patient_index = linear_sum_assignment(cost)
        for h_pos, p_pos in zip(healthy_index, patient_index):
            h = healthy.iloc[h_pos]
            p = patient.iloc[p_pos]
            rows.append(
                {
                    "gender": int(gender),
                    "healthy_subject_id": h["subject_id"],
                    "patient_subject_id": p["subject_id"],
                    "healthy_age": float(h["age"]),
                    "patient_age": float(p["age"]),
                    "absolute_age_difference": float(abs(h["age"] - p["age"])),
                }
            )
    return pd.DataFrame(rows).sort_values(["gender", "patient_subject_id"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "tdbrain_participants.csv",
    )
    parser.add_argument("--feature-set", default="covariance_logcorr")
    parser.add_argument("--model", default="logistic_l2")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    dataset = build_feature_dataset("tdbrain")
    metadata = _aligned_metadata(dataset, args.metadata)
    output = Path(__file__).resolve().parent / "results" / "tdbrain" / "confound_checks"
    output.mkdir(parents=True, exist_ok=True)

    metadata_X = metadata[["age", "gender"]].to_numpy(dtype=float)
    metadata_summary, metadata_predictions = nested_oof_evaluate(
        metadata_X,
        dataset.y,
        feature_set="metadata_age_gender",
        model_name="logistic_l2",
        mode=args.mode,
        repeats=args.repeats,
        inner_splits=3 if args.mode == "quick" else 4,
        n_jobs=args.n_jobs,
        subject_ids=dataset.subject_ids,
    )
    metadata_predictions.to_csv(output / "metadata_age_gender_predictions.csv", index=False)

    matches = optimal_age_sex_match(metadata)
    matches.to_csv(output / "age_sex_matches.csv", index=False)
    selected_ids = set(matches["healthy_subject_id"]) | set(matches["patient_subject_id"])
    mask = np.asarray([subject_id in selected_ids for subject_id in dataset.subject_ids])
    matched_summary, matched_predictions = nested_oof_evaluate(
        dataset.matrices[args.feature_set][mask],
        dataset.y[mask],
        feature_set=f"{args.feature_set}_age_sex_matched",
        model_name=args.model,
        mode=args.mode,
        repeats=args.repeats,
        inner_splits=3 if args.mode == "quick" else 4,
        n_jobs=args.n_jobs,
        subject_ids=dataset.subject_ids[mask],
    )
    matched_predictions.to_csv(output / f"matched_eeg_predictions__{args.model}.csv", index=False)

    matched_metadata = metadata[metadata["subject_id"].isin(selected_ids)]
    audit = {
        "matched_subjects": int(mask.sum()),
        "matched_pairs": int(len(matches)),
        "median_absolute_pair_age_difference": float(matches["absolute_age_difference"].median()),
        "healthy_age_mean": float(
            matched_metadata.loc[matched_metadata["group"] == "Healthy", "age"].mean()
        ),
        "patient_age_mean": float(
            matched_metadata.loc[matched_metadata["group"] == "Patient", "age"].mean()
        ),
    }
    summaries = pd.DataFrame(
        [
            metadata_summary | {"analysis": "metadata_only", **audit},
            matched_summary | {"analysis": "age_sex_matched_eeg", **audit},
        ]
    )
    summaries.to_csv(output / f"summary__{args.model}.csv", index=False)
    print(summaries[["analysis", "n_subjects", "roc_auc", "balanced_accuracy", "brier", "log_loss"]])
    print(audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
