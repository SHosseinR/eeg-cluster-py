"""
Plot 3D metric shifts (initial -> final) per subject vs healthy mean.

This script reads precomputed metrics from optimization results and does not
recalculate healthy baselines or final states.
"""
import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from optimization_config import (
    OPTIMIZATION_OUTPUT_DIR,
    OPTIMIZATION_RESULTS_FILE,
    OPTIMIZATION_FIGURES_DIR,
)


def _load_pickle_dict(path: str) -> Dict:
    return np.load(path, allow_pickle=True).item()


def _find_metadata(optimization_results: Dict) -> Optional[Dict]:
    for _, results in optimization_results.items():
        if not isinstance(results, dict):
            continue
        if results.get("healthy_measure_baselines") and results.get("optimization_measures"):
            return results
    return None


def _extract_points(
    optimization_results: Dict,
    measures: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    initial_points = []
    final_points = []
    skipped = []

    for subject_id, results in optimization_results.items():
        if not isinstance(results, dict):
            continue

        initial_metrics = results.get("initial_metrics")
        final_metrics = results.get("final_metrics")

        if initial_metrics is None:
            skipped.append((subject_id, "missing initial_metrics"))
            continue
        if final_metrics is None:
            skipped.append((subject_id, "missing final_metrics"))
            continue

        initial_vals = np.array(initial_metrics, dtype=float)
        final_vals = np.array(final_metrics, dtype=float)

        if initial_vals.size != len(measures):
            skipped.append((subject_id, "initial_metrics size mismatch"))
            continue
        if final_vals.size != len(measures):
            skipped.append((subject_id, "final_metrics size mismatch"))
            continue
        if not np.all(np.isfinite(initial_vals)):
            skipped.append((subject_id, "initial_metrics not finite"))
            continue
        if not np.all(np.isfinite(final_vals)):
            skipped.append((subject_id, "final_metrics not finite"))
            continue

        initial_points.append(initial_vals)
        final_points.append(final_vals)

    if not initial_points:
        return np.empty((0, len(measures))), np.empty((0, len(measures))), skipped

    return np.vstack(initial_points), np.vstack(final_points), skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot 3D metric shifts (initial -> final) vs healthy mean."
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Optional override path to optimization_results.npy",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG path for the figure",
    )
    parser.add_argument(
        "--html-output",
        default=None,
        help="Optional output HTML path for the interactive figure",
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
            "Results file is missing optimization metadata. "
            "Re-run optimization with the updated pipeline to store baselines."
        )

    measures = list(metadata["optimization_measures"])
    if len(measures) != 3:
        raise ValueError(
            "This plot requires exactly 3 optimization measures. "
            f"Got {len(measures)}: {measures}"
        )

    baselines = metadata["healthy_measure_baselines"]
    gt_point = np.array([baselines[m] for m in measures], dtype=float)

    initial_points, final_points, skipped = _extract_points(optimization_results, measures)
    if initial_points.size == 0:
        raise RuntimeError(
            "No subjects with valid initial/final metrics. "
            "Re-run optimization with the updated pipeline to store metrics."
        )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        initial_points[:, 0],
        initial_points[:, 1],
        initial_points[:, 2],
        s=30,
        alpha=0.65,
        color="#1f77b4",
        label="Initial",
    )
    ax.scatter(
        final_points[:, 0],
        final_points[:, 1],
        final_points[:, 2],
        s=30,
        alpha=0.75,
        color="#d62728",
        label="Final",
    )

    for idx in range(initial_points.shape[0]):
        ax.plot(
            [initial_points[idx, 0], final_points[idx, 0]],
            [initial_points[idx, 1], final_points[idx, 1]],
            [initial_points[idx, 2], final_points[idx, 2]],
            color="#444444",
            alpha=0.55,
            linewidth=0.8,
        )

    ax.scatter(
        [gt_point[0]],
        [gt_point[1]],
        [gt_point[2]],
        s=180,
        color="black",
        marker="*",
        label="Healthy mean",
    )

    ax.set_xlabel(measures[0].replace("_", " "))
    ax.set_ylabel(measures[1].replace("_", " "))
    ax.set_zlabel(measures[2].replace("_", " "))
    ax.set_title("Metric Shift: Initial -> Final vs Healthy Mean")
    ax.legend(loc="best")

    output_path = (
        args.output
        if args.output is not None
        else os.path.join(OPTIMIZATION_FIGURES_DIR, "metric_shift_3d.png")
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved 3D metric shift plot to: {output_path}")

    html_output = (
        args.html_output
        if args.html_output is not None
        else os.path.join(OPTIMIZATION_FIGURES_DIR, "metric_shift_3d.html")
    )

    line_x = []
    line_y = []
    line_z = []
    for idx in range(initial_points.shape[0]):
        line_x.extend([initial_points[idx, 0], final_points[idx, 0], None])
        line_y.extend([initial_points[idx, 1], final_points[idx, 1], None])
        line_z.extend([initial_points[idx, 2], final_points[idx, 2], None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=initial_points[:, 0],
            y=initial_points[:, 1],
            z=initial_points[:, 2],
            mode="markers",
            name="Initial",
            marker=dict(size=4, color="#1f77b4", opacity=0.65),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=final_points[:, 0],
            y=final_points[:, 1],
            z=final_points[:, 2],
            mode="markers",
            name="Final",
            marker=dict(size=4, color="#d62728", opacity=0.75),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode="lines",
            name="Shift",
            line=dict(color="#444444", width=3),
            opacity=0.55,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[gt_point[0]],
            y=[gt_point[1]],
            z=[gt_point[2]],
            mode="markers",
            name="Healthy mean",
            marker=dict(size=8, color="black", symbol="diamond"),
        )
    )

    fig.update_layout(
        title="Metric Shift: Initial -> Final vs Healthy Mean",
        scene=dict(
            xaxis_title=measures[0].replace("_", " "),
            yaxis_title=measures[1].replace("_", " "),
            zaxis_title=measures[2].replace("_", " "),
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    os.makedirs(os.path.dirname(html_output), exist_ok=True)
    fig.write_html(html_output)
    print(f"Saved interactive 3D plot to: {html_output}")

    if skipped:
        print("Skipped subjects:")
        for subject_id, reason in skipped:
            print(f"  - {subject_id}: {reason}")


if __name__ == "__main__":
    main()
