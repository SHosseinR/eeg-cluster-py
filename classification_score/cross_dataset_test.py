"""Train on one dataset and test, without refitting, on the other dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_features import align_feature_matrix, build_feature_dataset
from modeling import fit_tuned_model, probability_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-set", default="covariance_common_logcorr")
    parser.add_argument("--model", default="rbf_svm")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    datasets = {
        profile: build_feature_dataset(profile) for profile in ("first_paper", "tdbrain")
    }
    rows = []
    predictions = []
    for source_name, target_name in (("first_paper", "tdbrain"), ("tdbrain", "first_paper")):
        source = datasets[source_name]
        target = datasets[target_name]
        source_X = source.matrices[args.feature_set]
        target_X = align_feature_matrix(
            target.matrices[args.feature_set],
            target.feature_names[args.feature_set],
            source.feature_names[args.feature_set],
        )
        estimator, params = fit_tuned_model(
            source_X,
            source.y,
            model_name=args.model,
            mode=args.mode,
            inner_splits=5,
            n_jobs=args.n_jobs,
        )
        probability = estimator.predict_proba(target_X)[:, 1]
        metrics = probability_metrics(target.y, probability)
        rows.append(
            {
                "train_dataset": source_name,
                "test_dataset": target_name,
                "feature_set": args.feature_set,
                "model": args.model,
                "train_subjects": len(source.y),
                "test_subjects": len(target.y),
                **metrics,
                "source_cv_best_params": json.dumps(params, sort_keys=True, default=str),
            }
        )
        predictions.extend(
            {
                "train_dataset": source_name,
                "test_dataset": target_name,
                "subject_id": subject_id,
                "y_true": int(label),
                "patient_probability": float(value),
            }
            for subject_id, label, value in zip(target.subject_ids, target.y, probability)
        )

    output = Path(__file__).resolve().parent / "results" / "cross_dataset"
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / f"{args.feature_set}__{args.model}.csv", index=False)
    pd.DataFrame(predictions).to_csv(
        output / f"{args.feature_set}__{args.model}__predictions.csv", index=False
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
