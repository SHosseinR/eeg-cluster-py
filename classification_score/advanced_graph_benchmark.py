"""Benchmark graph-neural EEG classifiers on saved connectivity matrices."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classification_score.band_connectivity_classifier import (
    DEFAULT_BANDS,
    evaluate_band_models,
    load_connectivity_features,
)
from classification_score.modeling import nested_oof_evaluate


DEFAULT_BAND_MODELS = ("gcn", "brainnetcnn")
DEFAULT_FUSED_MODELS = ("gcn_3band", "brainnetcnn_3band")


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""

    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def _evaluate_fused_models(
    connectivity_path: Path,
    output_dir: Path,
    *,
    method: str,
    bands: Sequence[str],
    models: Sequence[str],
    mode: str,
    outer_splits: int,
    repeats: int,
    inner_splits: int,
    n_jobs: int,
) -> pd.DataFrame:
    X_by_band, y, subject_ids = load_connectivity_features(
        connectivity_path, method=method, bands=bands
    )
    X = np.concatenate([X_by_band[band] for band in bands], axis=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    predictions = []
    band_label = "+".join(bands)
    for model_name in models:
        summary, model_predictions = nested_oof_evaluate(
            X,
            y,
            feature_set=f"{method}_{band_label}_natural_edges",
            model_name=model_name,
            mode=mode,
            outer_splits=outer_splits,
            repeats=repeats,
            inner_splits=inner_splits,
            n_jobs=n_jobs,
            subject_ids=subject_ids,
        )
        summary.update(
            {
                "band": band_label,
                "method": method,
                "tuning_mode": mode,
                "representation": "three_band_fusion",
            }
        )
        model_predictions["band"] = band_label
        model_predictions["method"] = method
        model_predictions["representation"] = "three_band_fusion"
        summaries.append(summary)
        predictions.append(model_predictions)
    summary_frame = pd.DataFrame(summaries).sort_values(
        ["roc_auc", "brier"], ascending=[False, True]
    )
    summary_frame.to_csv(output_dir / "model_comparison.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        output_dir / "oof_predictions_all_models.csv", index=False
    )
    return summary_frame


def _comparison_figure(summary: pd.DataFrame, output_path: Path) -> None:
    plot_data = summary.copy()
    plot_data["label"] = plot_data["band"].astype(str) + " / " + plot_data["model"].astype(str)
    plot_data = plot_data.sort_values("roc_auc", ascending=True)
    colors = [
        "#4C78A8" if "gcn" in model.lower() else
        "#F58518" if "brainnet" in model.lower() else
        "#9D9D9D"
        for model in plot_data["model"]
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, max(5, 0.38 * len(plot_data))))
    for axis, metric, title, limits in (
        (axes[0], "roc_auc", "ROC AUC (higher is better)", (0.45, 1.0)),
        (axes[1], "balanced_accuracy", "Balanced accuracy", (0.45, 1.0)),
        (axes[2], "brier", "Brier score (lower is better)", (0.0, 0.3)),
    ):
        axis.barh(plot_data["label"], plot_data[metric], color=colors)
        axis.set_xlim(*limits)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    fig.suptitle("TD-BRAIN Healthy vs MDD connectivity classifier comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    connectivity_path: Path,
    elapsed_seconds: float,
) -> None:
    columns = [
        "band", "model", "n_subjects", "roc_auc", "balanced_accuracy",
        "average_precision", "brier", "ece_10",
    ]
    table = summary[columns].copy()
    for column in (
        "roc_auc", "balanced_accuracy", "average_precision", "brier", "ece_10"
    ):
        table[column] = table[column].map(lambda value: f"{float(value):.3f}")
    lines = [
        "# TD-BRAIN advanced connectivity classifier benchmark",
        "",
        f"- Input: `{connectivity_path}`",
        f"- Runtime: {elapsed_seconds / 60.0:.1f} minutes",
        "- Labels: Healthy = 0, Patient/MDD = 1",
        "- Evaluation: repeated subject-level stratified cross-validation; neural "
        "early stopping and temperature scaling use training-fold-only validation data.",
        "",
        _markdown_table(table),
        "",
    ]
    best = summary.sort_values(["roc_auc", "brier"], ascending=[False, True]).iloc[0]
    lines.extend(
        [
            "## Measured conclusion",
            "",
            f"The highest ROC AUC in this run was **{best['roc_auc']:.3f}** from "
            f"`{best['model']}` on `{best['band']}`. Compare calibration using "
            "Brier/ECE as well as discrimination; a higher AUC alone does not make "
            "a model suitable as a stimulation objective.",
            "",
            "These are within-dataset research classification results, not a "
            "diagnostic model and not evidence that simulated stimulation is clinically effective.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(
    connectivity_path: str | Path,
    output_dir: str | Path,
    *,
    method: str = "coh",
    bands: Sequence[str] = DEFAULT_BANDS,
    band_models: Sequence[str] = DEFAULT_BAND_MODELS,
    fused_models: Sequence[str] = DEFAULT_FUSED_MODELS,
    baseline_summaries: Sequence[str | Path] = (),
    mode: str = "quick",
    outer_splits: int = 5,
    repeats: int = 5,
    inner_splits: int = 3,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run bandwise and fused neural comparisons and produce one report."""

    started = time.perf_counter()
    connectivity_path = Path(connectivity_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    if band_models:
        band_summary = evaluate_band_models(
            connectivity_path,
            output_dir / "per_band",
            method=method,
            bands=bands,
            models=band_models,
            mode=mode,
            outer_splits=outer_splits,
            repeats=repeats,
            inner_splits=inner_splits,
            n_jobs=n_jobs,
        )
        band_summary["representation"] = "single_band"
        frames.append(band_summary)
    if fused_models:
        frames.append(
            _evaluate_fused_models(
                connectivity_path,
                output_dir / "fused",
                method=method,
                bands=bands,
                models=fused_models,
                mode=mode,
                outer_splits=outer_splits,
                repeats=repeats,
                inner_splits=inner_splits,
                n_jobs=n_jobs,
            )
        )
    for baseline_path in baseline_summaries:
        baseline = pd.read_csv(baseline_path)
        baseline["representation"] = baseline.get("representation", "single_band_baseline")
        frames.append(baseline)
    if not frames:
        raise ValueError("Choose at least one band or fused model")
    summary = pd.concat(frames, ignore_index=True, sort=False)
    summary = summary.drop_duplicates(["band", "model"], keep="first")
    summary = summary.sort_values(["band", "roc_auc", "brier"], ascending=[True, False, True])
    summary.to_csv(output_dir / "combined_model_comparison.csv", index=False)
    _comparison_figure(summary, output_dir / "advanced_classifier_comparison.png")
    elapsed = time.perf_counter() - started
    _write_report(
        summary,
        output_dir / "ADVANCED_CLASSIFIER_REPORT.md",
        connectivity_path=connectivity_path,
        elapsed_seconds=elapsed,
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("connectivity", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method", default="coh")
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--band-models", nargs="*", default=list(DEFAULT_BAND_MODELS))
    parser.add_argument("--fused-models", nargs="*", default=list(DEFAULT_FUSED_MODELS))
    parser.add_argument("--baseline-summary", action="append", default=[])
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_benchmark(
        args.connectivity,
        args.output,
        method=args.method,
        bands=args.bands,
        band_models=args.band_models,
        fused_models=args.fused_models,
        baseline_summaries=args.baseline_summary,
        mode=args.mode,
        outer_splits=args.outer_splits,
        repeats=args.repeats,
        inner_splits=args.inner_splits,
        n_jobs=args.n_jobs,
    )
    print(
        summary[
            ["band", "model", "roc_auc", "balanced_accuracy", "brier"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
