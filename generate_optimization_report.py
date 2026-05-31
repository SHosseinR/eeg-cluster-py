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
    plot_candidate_region_statistics
)
from statistics_utils import compute_candidate_region_selection_stats


def _load_pickle_dict(path: str) -> Dict:
    return np.load(path, allow_pickle=True).item()


def _find_metadata(optimization_results: Dict) -> Optional[Dict]:
    for _, results in optimization_results.items():
        if not isinstance(results, dict):
            continue
        if results.get("channel_names") and results.get("band_names"):
            return results
    return None


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

    stats_df = compute_candidate_region_selection_stats(
        optimization_results,
        channel_names
    )
    stats_path = os.path.join(
        os.path.dirname(output_path),
        "candidate_region_selection_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved candidate-region selection statistics: {stats_path}")

    figures_dir = args.figures_dir if args.figures_dir is not None else OPTIMIZATION_FIGURES_DIR
    figure_paths = plot_candidate_region_statistics(
        stats_df,
        channel_names,
        figures_dir,
        prefix="final_target_statistics"
    )
    for figure_path in figure_paths:
        print(f"Saved final-target statistic figure: {figure_path}")


if __name__ == "__main__":
    main()
