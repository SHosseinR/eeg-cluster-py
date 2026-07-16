"""
Generate optimization report from a saved results file.

Uses metadata stored in optimization_results.npy and does not recompute GT.
"""
import argparse
import os
from typing import Dict, Optional

import numpy as np

from optimization_config import (
    OPTIMIZATION_OUTPUT_DIR,
    OPTIMIZATION_RESULTS_FILE,
    OPTIMIZATION_FIGURES_DIR
)
from optimization_visualization import (
    create_optimization_report,
    plot_candidate_region_statistics,
    plot_weighted_rank_region_statistics
)
from statistics_utils import (
    compute_candidate_region_selection_stats,
    compute_candidate_region_weighted_rank_stats
)


def _load_pickle_dict(path: str) -> Dict:
    return np.load(path, allow_pickle=True).item()


def _find_metadata(optimization_results: Dict) -> Optional[Dict]:
    for _, results in optimization_results.items():
        if not isinstance(results, dict):
            continue
        if results.get("channel_names") and results.get("band_names"):
            return results
    return None


def _split_results_by_band(optimization_results: Dict) -> Dict[str, Dict]:
    """Split per-band optimization results saved as subject::band keys."""
    results_by_band = {}
    for key, result in optimization_results.items():
        key_text = str(key)
        if "::" not in key_text:
            continue
        subject_id, band_name = key_text.rsplit("::", 1)
        results_by_band.setdefault(band_name, {})[subject_id] = result
    return results_by_band


def _infer_band_name_from_results_path(results_path: str) -> Optional[str]:
    filename = os.path.basename(results_path)
    suffix = f"_{OPTIMIZATION_RESULTS_FILE}"
    if filename.endswith(suffix):
        return filename[:-len(suffix)]
    return None


def _save_candidate_region_stats(
    optimization_results: Dict,
    channel_names,
    output_dir: str,
    figures_dir: str,
    prefix: Optional[str] = None,
    label: Optional[str] = None
) -> None:
    output_label = f" ({label})" if label else ""
    file_stem = f"{prefix}_" if prefix else ""
    hard_figure_prefix = (
        f"{prefix}_hard_best_solution_target_statistics"
        if prefix else
        "hard_best_solution_target_statistics"
    )
    weighted_figure_prefix = (
        f"{prefix}_rank_weighted_target_statistics"
        if prefix else
        "rank_weighted_target_statistics"
    )

    stats_df = compute_candidate_region_selection_stats(
        optimization_results,
        channel_names
    )
    stats_path = os.path.join(
        output_dir,
        f"{file_stem}candidate_region_selection_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved hard best-solution candidate-region statistics{output_label}: {stats_path}")

    weighted_stats_df = compute_candidate_region_weighted_rank_stats(
        optimization_results,
        channel_names
    )
    weighted_stats_path = os.path.join(
        output_dir,
        f"{file_stem}candidate_region_weighted_rank_stats.csv"
    )
    weighted_stats_df.to_csv(weighted_stats_path, index=False)
    print(f"Saved rank-weighted candidate-region statistics{output_label}: {weighted_stats_path}")

    statistics_scope = prefix if prefix else "overall"
    statistics_dir = os.path.join(figures_dir, "target_statistics", statistics_scope)
    figure_paths = plot_candidate_region_statistics(
        stats_df,
        channel_names,
        statistics_dir,
        prefix=hard_figure_prefix
    )
    figure_paths.extend(plot_weighted_rank_region_statistics(
        weighted_stats_df,
        channel_names,
        statistics_dir,
        prefix=weighted_figure_prefix
    ))
    for figure_path in figure_paths:
        print(f"Saved candidate-region statistic figure{output_label}: {figure_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate optimization report from a saved results file."
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Optional override path to optimization_results.npy",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional override path for optimization_report.txt",
    )
    parser.add_argument(
        "--top-k",
        default=None,
        help="Optional override for top-k ranking in report",
    )
    parser.add_argument(
        "--figures-dir",
        default=None,
        help="Optional directory for final-target statistic figures",
    )
    args = parser.parse_args()

    results_path = (
        args.results
        if args.results is not None
        else os.path.join(OPTIMIZATION_OUTPUT_DIR, OPTIMIZATION_RESULTS_FILE)
    )
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Optimization results not found: {results_path}")

    optimization_results = _load_pickle_dict(results_path)
    metadata = _find_metadata(optimization_results)
    if metadata is None:
        raise RuntimeError(
            "Results file is missing metadata (channels/bands). "
            "Re-run optimization with the updated pipeline."
        )

    channel_names = list(metadata["channel_names"])
    band_names = list(metadata["band_names"])
    optimization_measures = list(metadata.get("optimization_measures", []))
    if not optimization_measures:
        raise RuntimeError(
            "Results file is missing optimization measures. "
            "Re-run optimization with the updated pipeline."
        )

    optimization_directions = dict(metadata.get("optimization_directions", {}))

    output_path = (
        args.output
        if args.output is not None
        else os.path.join(OPTIMIZATION_OUTPUT_DIR, "optimization_report.txt")
    )

    top_k = args.top_k
    if top_k is None:
        top_k = metadata.get("top_k")

    create_optimization_report(
        optimization_results=optimization_results,
        channel_names=channel_names,
        band_names=band_names,
        optimization_measures=optimization_measures,
        optimization_directions=optimization_directions,
        output_path=output_path,
        top_k=top_k
    )

    output_dir = os.path.dirname(output_path)
    figures_dir = args.figures_dir if args.figures_dir is not None else OPTIMIZATION_FIGURES_DIR
    _save_candidate_region_stats(
        optimization_results,
        channel_names,
        output_dir,
        figures_dir,
        label="combined across bands"
    )

    results_by_band = _split_results_by_band(optimization_results)
    for band_name, band_results in results_by_band.items():
        _save_candidate_region_stats(
            band_results,
            channel_names,
            output_dir,
            figures_dir,
            prefix=band_name,
            label=f"{band_name} band"
        )

    inferred_band_name = _infer_band_name_from_results_path(results_path)
    if inferred_band_name is not None and not results_by_band:
        _save_candidate_region_stats(
            optimization_results,
            channel_names,
            output_dir,
            figures_dir,
            prefix=inferred_band_name,
            label=f"{inferred_band_name} band"
        )


if __name__ == "__main__":
    main()
