"""Create a cross-dataset top-five target-selection comparison from saved results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from saved_results_utils import SavedDatasetProfile, load_dataset_profile, load_npy_dict


def aggregate_node_selections(results: Mapping, top_k: int = 5) -> dict:
    metadata = next((value for value in results.values() if isinstance(value, Mapping)), None)
    if metadata is None:
        raise ValueError("Optimization results are empty")
    channel_names = list(metadata.get("channel_display_names") or metadata.get("channel_names") or [])
    if not channel_names:
        raise ValueError("Optimization results do not contain channel names")
    hard = np.zeros(len(channel_names), dtype=float)
    weighted = np.zeros(len(channel_names), dtype=float)
    n_units = 0
    for result in results.values():
        if not isinstance(result, Mapping):
            continue
        best = result.get("best_solution") or {}
        try:
            node = int(best["node"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= node < len(channel_names):
            continue
        n_units += 1
        hard[node] += 1.0
        ranked = list(result.get("top_solutions") or [])[:top_k]
        if not ranked:
            ranked = [best]
        for position, solution in enumerate(ranked, start=1):
            try:
                ranked_node = int(solution["node"])
            except (KeyError, TypeError, ValueError):
                continue
            if not 0 <= ranked_node < len(channel_names):
                continue
            rank = solution.get("rank", position)
            try:
                rank = max(float(rank), 1.0)
            except (TypeError, ValueError):
                rank = float(position)
            try:
                strength = float(solution.get("strength", 1.0 / rank))
            except (TypeError, ValueError):
                strength = 1.0 / rank
            weighted[ranked_node] += strength
    if n_units == 0:
        raise ValueError("No valid best solutions were found")
    return {"channel_names": channel_names, "hard": hard, "weighted": weighted, "n_units": n_units}


def _top_rows(profile: SavedDatasetProfile, aggregate: Mapping, method: str, top_n: int = 5):
    scores = np.asarray(aggregate[method], dtype=float)
    order = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))[:top_n]
    return [
        {
            "dataset": profile.label,
            "selection_method": "best_solution" if method == "hard" else "top5_rank_weighted",
            "rank": rank,
            "node_index": node,
            "node_name": aggregate["channel_names"][node],
            "raw_score": float(scores[node]),
            "normalized_rate": float(scores[node] / aggregate["n_units"]),
            "n_optimization_units": int(aggregate["n_units"]),
        }
        for rank, node in enumerate(order, start=1)
    ]


def generate_top_selected_nodes(
    profiles: Sequence[SavedDatasetProfile],
    output_dir: str | Path,
    top_k: int = 5,
) -> dict:
    if not profiles:
        raise ValueError("At least one dataset profile is required")
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures" / "selection"
    data_dir = output_dir / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    aggregates, rows = {}, []
    for profile in profiles:
        results = load_npy_dict(profile.optimization_results_path, "optimization results")
        aggregate = aggregate_node_selections(results, top_k=top_k)
        aggregates[profile.label] = aggregate
        rows.extend(_top_rows(profile, aggregate, "hard"))
        rows.extend(_top_rows(profile, aggregate, "weighted"))
    table = pd.DataFrame(rows)
    csv_path = data_dir / "top5_selected_nodes.csv"
    table.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(len(profiles), 2, figsize=(14, 5.2 * len(profiles)), squeeze=False)
    for row_index, profile in enumerate(profiles):
        for column_index, (selection_method, color, title) in enumerate((
            ("best_solution", "#4472C4", "Rank-1 best-solution frequency"),
            ("top5_rank_weighted", "#ED7D31", "Top-5 rank-weighted score"),
        )):
            ax = axes[row_index, column_index]
            selected = table[
                (table["dataset"] == profile.label) &
                (table["selection_method"] == selection_method)
            ].sort_values("rank", ascending=False)
            values = 100.0 * selected["normalized_rate"].to_numpy()
            bars = ax.barh(selected["node_name"], values, color=color, alpha=0.86)
            ax.set_xlabel(
                "Best-solution frequency (%)" if selection_method == "best_solution"
                else "Rank-weighted score per unit × 100"
            )
            ax.set_title(f"{profile.label}: {title}\n(n={int(selected['n_optimization_units'].iloc[0])} band-subject units)")
            ax.grid(axis="x", alpha=0.25)
            for bar, raw in zip(bars, selected["raw_score"]):
                ax.text(bar.get_width() + max(values.max() * 0.015, 0.05),
                        bar.get_y() + bar.get_height() / 2,
                        f"raw={raw:.2f}", va="center", fontsize=8)
            ax.set_xlim(0, max(values.max() * 1.28, 1.0))
    fig.suptitle("Top five selected stimulation nodes across stored bands", fontsize=16)
    fig.tight_layout()
    png_path = figure_dir / "top5_selected_nodes.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "csv": csv_path, "table": table}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", action="append", required=True)
    parser.add_argument("--output-dir", default="results-comparison")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    profiles = [load_dataset_profile(config) for config in args.dataset_config]
    output = generate_top_selected_nodes(profiles, args.output_dir, args.top_k)
    print(f"Saved: {output['png']}")
    print(f"Saved: {output['csv']}")


if __name__ == "__main__":
    main()
