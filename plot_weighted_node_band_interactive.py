"""
Interactive weighted node x band visualization with detailed hover metadata.

This script loads optimization results from disk (does not re-run optimization)
and writes an HTML report containing a weighted heatmap plus a 3D scatter view
of ranked solutions.
"""
import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from optimization_config import (
    OPTIMIZATION_OUTPUT_DIR,
    OPTIMIZATION_RESULTS_FILE,
    OPTIMIZATION_FIGURES_DIR,
    SIMULATION_CONFIG,
)
from channel_metadata import get_display_channel_names


def _load_pickle_dict(path: str) -> Dict:
    return np.load(path, allow_pickle=True).item()


def _get_result_band_info(results: Dict) -> Tuple[Optional[int], Optional[str]]:
    band_idx = None
    band_name = None

    if results.get("fixed_band_index") is not None:
        try:
            band_idx = int(results["fixed_band_index"])
        except (TypeError, ValueError):
            band_idx = None

    if results.get("fixed_band_name"):
        band_name = str(results["fixed_band_name"])

    best_solution = results.get("best_solution") or {}
    if band_idx is None and "band" in best_solution:
        try:
            band_idx = int(best_solution["band"])
        except (TypeError, ValueError):
            band_idx = None

    if band_name is None and band_idx is not None:
        band_names = results.get("band_names")
        if band_names and 0 <= band_idx < len(band_names):
            band_name = band_names[band_idx]

    return band_idx, band_name


def _parse_band_selector(band_selector: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
    if band_selector is None:
        return None, None
    try:
        return int(band_selector), None
    except (TypeError, ValueError):
        return None, str(band_selector)


def _matches_band_filter(
    results: Dict,
    band_idx_filter: Optional[int],
    band_name_filter: Optional[str],
) -> bool:
    if band_idx_filter is None and band_name_filter is None:
        return True

    band_idx, band_name = _get_result_band_info(results)
    if band_idx_filter is not None:
        return band_idx is not None and band_idx == band_idx_filter
    if band_name_filter is not None:
        return band_name is not None and band_name == band_name_filter
    return True


def _find_metadata(optimization_results: Dict) -> Optional[Dict]:
    for _, results in optimization_results.items():
        if not isinstance(results, dict):
            continue
        if results.get("band_names") and (
            results.get("channel_display_names") or results.get("channel_names")
        ):
            return results
    return None


def _rank_best_front(
    best_front: List[Dict],
    top_k: Optional[int],
    objective_mode: Optional[str] = None,
) -> List[Dict]:
    if not best_front:
        return []

    if top_k is None:
        top_k = len(best_front)
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = len(best_front)

    top_k = max(1, min(top_k, len(best_front)))

    objectives = np.array([sol["objectives"] for sol in best_front])
    if objective_mode == "distance_to_gt":
        ideal_point = np.zeros(objectives.shape[1], dtype=float)
    else:
        ideal_point = objectives.min(axis=0)
    distances = np.linalg.norm(objectives - ideal_point, axis=1)
    order = np.argsort(distances)

    ranked = []
    for rank, idx in enumerate(order[:top_k], start=1):
        sol = best_front[idx]
        ranked.append(
            {
                "node": sol["node"],
                "band": sol["band"],
                "band_name": sol.get("band_name"),
                "stimulation_duration": sol.get("stimulation_duration"),
                "stimulation_amplitude": sol.get("stimulation_amplitude"),
                "objectives": sol["objectives"],
                "distance": float(distances[idx]),
                "rank": rank,
                "strength": 1.0 / float(rank),
            }
        )

    return ranked


def _collect_ranked_solutions(
    optimization_results: Dict,
    top_k: Optional[int],
    band_idx_filter: Optional[int] = None,
    band_name_filter: Optional[str] = None,
) -> List[Dict]:
    collected = []
    for subject_id, results in optimization_results.items():
        if not isinstance(results, dict):
            continue
        if not _matches_band_filter(results, band_idx_filter, band_name_filter):
            continue

        band_idx, band_name = _get_result_band_info(results)
        subject_label = results.get("subject_id", subject_id)
        if band_name and band_name_filter is None and band_idx_filter is None:
            subject_label = f"{subject_label}::{band_name}"

        ranked = []
        if results.get("top_solutions"):
            ranked = results["top_solutions"]
        elif results.get("best_front"):
            ranked = _rank_best_front(
                results["best_front"],
                top_k,
                objective_mode=results.get("objective_mode"),
            )

        if top_k is not None:
            ranked = ranked[:top_k]

        for sol in ranked:
            collected.append({"subject_id": subject_label, **sol})

    return collected


def _format_value(value: Optional[float], precision: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return "n/a"


def _build_hover_details(
    entries: List[Dict],
    leak_value: Optional[float],
    max_details: Optional[int] = None,
) -> str:
    if not entries:
        return "No ranked solutions"

    detail_lines = []
    for idx, sol in enumerate(entries, start=1):
        if max_details is not None and idx > max_details:
            remaining = len(entries) - max_details
            detail_lines.append(f"... and {remaining} more")
            break

        detail_lines.append(
            " | ".join(
                [
                    f"subject={sol.get('subject_id', 'n/a')}",
                    f"rank={sol.get('rank', 'n/a')}",
                    f"strength={_format_value(sol.get('strength'), 3)}",
                    f"closeness={_format_value(sol.get('distance'), 4)}",
                    f"duration={_format_value(sol.get('stimulation_duration'))}",
                    f"amplitude={_format_value(sol.get('stimulation_amplitude'))}",
                    f"leak={_format_value(sol.get('leak', leak_value))}",
                ]
            )
        )

    return "<br>".join(detail_lines)


def _prepare_heatmap_data(
    ranked_solutions: List[Dict],
    n_nodes: int,
    n_bands: int,
    leak_value: Optional[float],
    max_details: Optional[int],
) -> Tuple[np.ndarray, List[List[str]]]:
    node_band_weights = np.zeros((n_nodes, n_bands), dtype=float)
    cell_details = [[[] for _ in range(n_bands)] for _ in range(n_nodes)]

    for sol in ranked_solutions:
        node = int(sol["node"])
        band = int(sol["band"])
        strength = float(sol.get("strength", 1.0))
        node_band_weights[node, band] += strength
        cell_details[node][band].append(sol)

    hover_text = []
    for node in range(n_nodes):
        row = []
        for band in range(n_bands):
            details = _build_hover_details(
                cell_details[node][band], leak_value, max_details=max_details
            )
            row.append(details)
        hover_text.append(row)

    return node_band_weights, hover_text


def _build_heatmap_figure(
    node_band_weights: np.ndarray,
    hover_text: List[List[str]],
    band_names: List[str],
    channel_names: List[str],
) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=node_band_weights,
                x=band_names,
                y=channel_names,
                colorscale="YlGnBu",
                hoverinfo="text",
                text=hover_text,
                colorbar=dict(title="Weighted strength sum"),
            )
        ]
    )

    fig.update_layout(
        title="Weighted node x band heatmap (hover for per-solution details)",
        xaxis_title="Frequency band",
        yaxis_title="Channel/node",
        margin=dict(l=80, r=40, t=80, b=80),
    )
    return fig


def _build_scatter_3d_figure(
    ranked_solutions: List[Dict],
    band_names: List[str],
    channel_names: List[str],
    leak_value: Optional[float],
) -> go.Figure:
    if not ranked_solutions:
        return go.Figure()

    x_vals = [int(sol["band"]) for sol in ranked_solutions]
    y_vals = [int(sol["node"]) for sol in ranked_solutions]
    z_vals = [float(sol.get("strength", 1.0)) for sol in ranked_solutions]

    hover_lines = []
    for sol in ranked_solutions:
        hover_lines.append(
            "<br>".join(
                [
                    f"subject={sol.get('subject_id', 'n/a')}",
                    f"band={band_names[int(sol['band'])]}",
                    f"node={channel_names[int(sol['node'])]}",
                    f"rank={sol.get('rank', 'n/a')}",
                    f"strength={_format_value(sol.get('strength'), 3)}",
                    f"closeness={_format_value(sol.get('distance'), 4)}",
                    f"duration={_format_value(sol.get('stimulation_duration'))}",
                    f"amplitude={_format_value(sol.get('stimulation_amplitude'))}",
                    f"leak={_format_value(sol.get('leak', leak_value))}",
                ]
            )
        )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode="markers",
                marker=dict(
                    size=4,
                    color=z_vals,
                    colorscale="Viridis",
                    opacity=0.8,
                    colorbar=dict(title="Strength"),
                ),
                hoverinfo="text",
                text=hover_lines,
                name="Ranked solutions",
            )
        ]
    )

    fig.update_layout(
        title="Ranked solutions (z = strength)",
        scene=dict(
            xaxis=dict(title="Frequency band", tickmode="array", tickvals=list(range(len(band_names))), ticktext=band_names),
            yaxis=dict(title="Channel/node", tickmode="array", tickvals=list(range(len(channel_names))), ticktext=channel_names),
            zaxis=dict(title="Strength"),
        ),
        margin=dict(l=40, r=40, t=80, b=40),
    )

    return fig


def _build_closeness_bar_figure(
    ranked_solutions: List[Dict],
) -> go.Figure:
    if not ranked_solutions:
        return go.Figure()

    best_by_subject = {}
    for sol in ranked_solutions:
        subject_id = sol.get("subject_id")
        distance = sol.get("distance")
        if subject_id is None or distance is None:
            continue
        try:
            distance = float(distance)
        except (TypeError, ValueError):
            continue
        if subject_id not in best_by_subject or distance < best_by_subject[subject_id]:
            best_by_subject[subject_id] = distance

    if not best_by_subject:
        return go.Figure()

    subjects = sorted(best_by_subject.keys())
    values = [best_by_subject[s] for s in subjects]
    avg_value = float(np.mean(values)) if values else float("nan")

    fig = go.Figure(
        data=[
            go.Bar(
                x=subjects,
                y=values,
                marker=dict(color="#1f77b4"),
                hovertemplate="subject=%{x}<br>closeness=%{y:.4f}<extra></extra>",
                name="Best closeness",
            )
        ]
    )

    annotation_text = f"Avg closeness: {avg_value:.4f}" if np.isfinite(avg_value) else "Avg closeness: n/a"
    fig.update_layout(
        title="Best closeness per subject (lower is better)",
        xaxis_title="Subject",
        yaxis_title="Closeness (distance to ideal)",
        margin=dict(l=60, r=40, t=80, b=120),
        annotations=[
            dict(
                text=annotation_text,
                xref="paper",
                yref="paper",
                x=0.98,
                y=0.98,
                showarrow=False,
                xanchor="right",
                yanchor="top",
                font=dict(size=12),
            )
        ],
    )

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive weighted node x band heatmap with hover metadata."
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Optional override path to optimization_results.npy",
    )
    parser.add_argument(
        "--html-output",
        default=None,
        help="Optional HTML output path",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional override for top-k ranked solutions",
    )
    parser.add_argument(
        "--max-details",
        type=int,
        default=None,
        help="Optional limit of per-cell hover details",
    )
    parser.add_argument(
        "--band",
        default=None,
        help="Optional band name or index to select from combined results",
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
    # print(f'{optimization_results=}')
    metadata = _find_metadata(optimization_results)
    if metadata is None:
        raise RuntimeError("Results file missing band_names/channel_names metadata")

    band_names = list(metadata["band_names"])
    exact_channel_names = metadata.get("channel_names")
    n_nodes = len(exact_channel_names) if exact_channel_names else int(metadata.get("n_nodes", 0))
    channel_names = get_display_channel_names(metadata, n_nodes=n_nodes)
    if not channel_names:
        raise RuntimeError("Results file missing channel label metadata")

    leak_value = SIMULATION_CONFIG.get("leak")

    band_idx_filter, band_name_filter = _parse_band_selector(args.band)
    ranked_solutions = _collect_ranked_solutions(
        optimization_results,
        top_k=args.top_k,
        band_idx_filter=band_idx_filter,
        band_name_filter=band_name_filter,
    )
    if not ranked_solutions:
        raise RuntimeError("No ranked solutions found in results file")

    node_band_weights, hover_text = _prepare_heatmap_data(
        ranked_solutions,
        n_nodes=len(channel_names),
        n_bands=len(band_names),
        leak_value=leak_value,
        max_details=args.max_details,
    )

    heatmap_fig = _build_heatmap_figure(
        node_band_weights,
        hover_text,
        band_names,
        channel_names,
    )
    scatter_fig = _build_scatter_3d_figure(
        ranked_solutions,
        band_names,
        channel_names,
        leak_value,
    )

    closeness_fig = _build_closeness_bar_figure(ranked_solutions)

    html_output = (
        args.html_output
        if args.html_output is not None
        else os.path.join(OPTIMIZATION_FIGURES_DIR, "targets", "weighted_node_band_interactive.html")
    )
    os.makedirs(os.path.dirname(html_output), exist_ok=True)

    heatmap_html = pio.to_html(heatmap_fig, full_html=False, include_plotlyjs="inline")
    scatter_html = pio.to_html(scatter_fig, full_html=False, include_plotlyjs=False)
    closeness_html = pio.to_html(closeness_fig, full_html=False, include_plotlyjs=False)

    full_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Weighted node x band visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h2 {{ margin-top: 40px; }}
    </style>
</head>
<body>
    <h1>Weighted node x band visualization</h1>
    <p>Heatmap uses weighted strength sum. Hover a cell to see per-solution details.</p>
    {heatmap_div}
    <h2>3D view of ranked solutions</h2>
    <p>Each point represents a ranked solution. z-axis shows strength.</p>
    {scatter_div}
    <h2>Best closeness per subject</h2>
    <p>Closeness equals distance to the ideal point (same ranking criterion).</p>
    {closeness_div}
</body>
</html>
""".format(heatmap_div=heatmap_html, scatter_div=scatter_html, closeness_div=closeness_html)

    with open(html_output, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Saved interactive visualization to: {html_output}")


if __name__ == "__main__":
    main()
