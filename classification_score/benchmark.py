"""CLI for standalone Healthy-vs-Patient EEG classifier comparisons.

Examples
--------
Quick screen on both configured datasets::

    python classification_score/benchmark.py --profiles first_paper tdbrain --mode quick

Repeated nested-CV confirmation of shortlisted combinations::

    python classification_score/benchmark.py --profiles tdbrain --mode full \
        --feature-sets spectral_roi covariance_logcorr --models logistic_l2 lda_shrinkage \
        --repeats 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd

from data_features import build_feature_dataset
from modeling import model_specs, nested_oof_evaluate


DEFAULT_FEATURE_SETS = (
    "graph_global",
    "spectral_roi",
    "spectral_channel",
    "covariance_logcorr",
    "connectivity_edges",
    "connectivity_topology",
    "eeg_fused",
    "connectivity_fused",
    "all_fused",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["first_paper", "tdbrain"],
        choices=["first_paper", "tdbrain"],
    )
    parser.add_argument("--feature-sets", nargs="+", default=list(DEFAULT_FEATURE_SETS))
    parser.add_argument("--models", nargs="+", default=None, help="Default: every registered model")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--inner-splits", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output subdirectory name (default: screen_quick or confirmation_full)",
    )
    parser.add_argument("--resume", action="store_true", help="Skip combinations already in summary.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repeats = args.repeats if args.repeats is not None else (1 if args.mode == "quick" else 5)
    inner_splits = args.inner_splits if args.inner_splits is not None else (3 if args.mode == "quick" else 4)
    run_name = args.run_name or ("screen_quick" if args.mode == "quick" else "confirmation_full")
    output_root = Path(__file__).resolve().parent / "results"

    for profile in args.profiles:
        print(f"\n=== Building/loading features: {profile} ===", flush=True)
        dataset = build_feature_dataset(profile, force=args.force_features)
        profile_output = output_root / profile / run_name
        prediction_output = profile_output / "predictions"
        prediction_output.mkdir(parents=True, exist_ok=True)
        summary_path = profile_output / "summary.csv"
        existing = pd.read_csv(summary_path) if args.resume and summary_path.exists() else pd.DataFrame()
        summaries = existing.to_dict("records")
        completed = (
            set(zip(existing["feature_set"], existing["model"])) if not existing.empty else set()
        )

        requested_feature_sets = args.feature_sets
        unknown_features = sorted(set(requested_feature_sets) - set(dataset.matrices))
        if unknown_features:
            raise KeyError(
                f"Unknown feature sets for {profile}: {unknown_features}; available={sorted(dataset.matrices)}"
            )
        available_models = model_specs(2, mode=args.mode)
        requested_models = args.models or list(available_models)
        unknown_models = sorted(set(requested_models) - set(available_models))
        if unknown_models:
            raise KeyError(f"Unknown models: {unknown_models}; available={sorted(available_models)}")

        total = len(requested_feature_sets) * len(requested_models)
        current = 0
        for feature_set in requested_feature_sets:
            X = dataset.matrices[feature_set]
            for model_name in requested_models:
                current += 1
                key = (feature_set, model_name)
                if key in completed:
                    print(f"[{current}/{total}] skip {feature_set} / {model_name}", flush=True)
                    continue
                print(
                    f"[{current}/{total}] {profile}: {feature_set} ({X.shape[1]} features) / {model_name}",
                    flush=True,
                )
                started = time.perf_counter()
                summary, predictions = nested_oof_evaluate(
                    X,
                    dataset.y,
                    feature_set=feature_set,
                    model_name=model_name,
                    mode=args.mode,
                    outer_splits=args.outer_splits,
                    repeats=repeats,
                    inner_splits=inner_splits,
                    n_jobs=args.n_jobs,
                    subject_ids=dataset.subject_ids,
                )
                summary["profile"] = profile
                summary["elapsed_seconds"] = time.perf_counter() - started
                summaries.append(summary)
                predictions.to_csv(
                    prediction_output / f"{feature_set}__{model_name}.csv", index=False
                )
                summary_frame = pd.DataFrame(summaries).sort_values(
                    ["roc_auc", "brier"], ascending=[False, True]
                )
                summary_frame.to_csv(summary_path, index=False)
                print(
                    f"    AUC={summary['roc_auc']:.3f}  balanced_acc={summary['balanced_accuracy']:.3f} "
                    f"Brier={summary['brier']:.3f}  log_loss={summary['log_loss']:.3f}",
                    flush=True,
                )
        print(f"Saved ranked summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
