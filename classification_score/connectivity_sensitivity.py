"""Test whether connectivity classification uses topology or global coupling."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

try:
    from .connectivity_benchmark import (
        BANDS,
        METHOD_DIRECTION,
        _compute_profile_caches,
        _load_benchmark_features,
    )
    from .modeling import nested_oof_evaluate
except ImportError:
    from connectivity_benchmark import (
        BANDS,
        METHOD_DIRECTION,
        _compute_profile_caches,
        _load_benchmark_features,
    )
    from modeling import nested_oof_evaluate


def connectivity_transformations(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Return natural, global-only, centered, and rank edge representations."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[1] % len(BANDS):
        raise ValueError(f"Feature count is not divisible by {len(BANDS)} bands: {matrix.shape}")
    band_edges = matrix.reshape(matrix.shape[0], len(BANDS), -1)
    band_means = np.mean(band_edges, axis=2)
    centered = band_edges - band_means[:, :, None]
    ranks = np.empty_like(band_edges)
    for subject in range(band_edges.shape[0]):
        for band in range(band_edges.shape[1]):
            ranks[subject, band] = stats.rankdata(
                band_edges[subject, band], method="average"
            ) / band_edges.shape[2]
    return {
        "natural_edges": matrix,
        "band_means_only": band_means,
        "within_subject_centered": centered.reshape(matrix.shape),
        "within_subject_ranks": ranks.reshape(matrix.shape),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", default=["first_paper", "tdbrain"])
    parser.add_argument("--methods", nargs="+", default=["coherence", "plv", "aec"])
    parser.add_argument("--models", nargs="+", default=["logistic_l2"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--run-name", default="topology_sensitivity")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = Path(__file__).resolve().parent / "results" / "connectivity_benchmark"
    for profile in args.profiles:
        records, data_root = _compute_profile_caches(
            profile,
            ("fourier", "envelope"),
            n_jobs=args.n_jobs,
            resume=True,
            max_subjects_per_group=None,
        )
        matrices, y, subject_ids, _, _ = _load_benchmark_features(
            records, data_root, list(args.methods)
        )
        output = output_root / profile / args.run_name
        prediction_output = output / "predictions"
        prediction_output.mkdir(parents=True, exist_ok=True)
        rows = []
        for method in args.methods:
            for transformation, X in connectivity_transformations(matrices[method]).items():
                for model in args.models:
                    feature_set = f"{method}__{transformation}"
                    print(f"{profile}: {feature_set} / {model}", flush=True)
                    summary, predictions = nested_oof_evaluate(
                        X,
                        y,
                        feature_set=feature_set,
                        model_name=model,
                        mode="quick",
                        outer_splits=5,
                        repeats=args.repeats,
                        inner_splits=3,
                        n_jobs=args.n_jobs,
                        subject_ids=subject_ids,
                    )
                    summary.update(
                        {
                            "profile": profile,
                            "connectivity_method": method,
                            "transformation": transformation,
                            "directed": METHOD_DIRECTION[method],
                        }
                    )
                    rows.append(summary)
                    predictions.to_csv(prediction_output / f"{feature_set}__{model}.csv", index=False)
                    pd.DataFrame(rows).sort_values("roc_auc", ascending=False).to_csv(
                        output / "summary.csv", index=False
                    )
                    print(
                        f"  AUC={summary['roc_auc']:.3f}, bal_acc={summary['balanced_accuracy']:.3f}",
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
