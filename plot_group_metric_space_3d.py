"""Generate static and interactive Healthy/Patient metric-space plots from saved data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from figure_paths import ensure_figure_tree, optimization_figure_dir
from saved_results_utils import (
    SavedDatasetProfile, first_band_metadata, load_analysis_metadata,
    load_dataset_profile, load_npy_dict, ordered_bands, results_for_band,
)


GROUP_COLORS = {"Healthy": "#1f77b4", "Patient": "#d62728"}
GROUP_MARKERS_MPL = {"Healthy": "o", "Patient": "^"}
# Plotly Scatter3d supports only a small symbol set (no triangles).
GROUP_MARKERS_PLOTLY = {"Healthy": "circle", "Patient": "diamond"}
CLUSTER_COLORS = (
    "#0072B2", "#56B4E9", "#009E73", "#F0E442", "#E69F00", "#D55E00",
    "#CC79A7", "#6A3D9A", "#1B9E77", "#A6761D", "#666666", "#E7298A",
)


def healthy_relative(points: np.ndarray, healthy_target: Sequence[float]) -> np.ndarray:
    target = np.asarray(healthy_target, dtype=float)
    denominator = np.where(np.abs(target) > 1e-10, np.abs(target), 1.0)
    return (np.asarray(points, dtype=float) - target) / denominator


def extract_group_points(
    network_measures: Mapping,
    group: str,
    method: str,
    band: str,
    measures: Sequence[str],
    healthy_target: Sequence[float],
    subject_filter: set[str] | None = None,
) -> tuple[np.ndarray, list[str], int]:
    points, ids, skipped = [], [], 0
    for subject_id, subject in network_measures.get(group, {}).items():
        if subject_filter is not None and str(subject_id) not in subject_filter:
            continue
        try:
            values = [float(subject[method][band][measure]) for measure in measures]
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        if not np.all(np.isfinite(values)):
            skipped += 1
            continue
        points.append(values)
        ids.append(str(subject_id))
    if not points:
        return np.empty((0, len(measures))), ids, skipped
    return healthy_relative(np.asarray(points), healthy_target), ids, skipped


def select_kmeans(points: np.ndarray, max_k: int = 6, random_seed: int = 42):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return np.zeros(len(points), dtype=int), 1, pd.DataFrame([
            {"k": 1, "silhouette_score": np.nan, "selected": True}
        ])
    scaled = StandardScaler().fit_transform(points)
    rows, candidates = [], []
    for k in range(2, min(max_k, len(points) - 1) + 1):
        model = KMeans(n_clusters=k, random_state=random_seed, n_init=50)
        labels = model.fit_predict(scaled)
        score = float(silhouette_score(scaled, labels))
        rows.append({"k": k, "silhouette_score": score})
        candidates.append((score, -k, labels, k))
    _, _, labels, selected_k = max(candidates, key=lambda item: (item[0], item[1]))
    table = pd.DataFrame(rows)
    table["selected"] = table["k"] == selected_k
    return labels.astype(int), selected_k, table


def _axis_labels(measures: Sequence[str]) -> list[str]:
    return [f"{measure.replace('_', ' ')}\n(relative difference)" for measure in measures]


def _limits(group_points: Mapping[str, np.ndarray]) -> list[tuple[float, float]]:
    combined = np.vstack([points for points in group_points.values() if points.size])
    result = []
    for axis in range(3):
        values = np.append(combined[:, axis], 0.0)
        low, high = float(np.min(values)), float(np.max(values))
        span = high - low
        pad = 0.07 * span if span > 1e-12 else 0.08
        result.append((low - pad, high + pad))
    return result


def _plot_static(
    group_points: Mapping[str, np.ndarray], group_ids: Mapping[str, list[str]],
    measures: Sequence[str], title: str, output: Path,
    clusters: Mapping[str, np.ndarray] | None = None,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    palette_index = 0
    for group in ("Healthy", "Patient"):
        points = group_points[group]
        if clusters is None:
            ax.scatter(*points.T, s=38, alpha=0.72, color=GROUP_COLORS[group],
                       marker=GROUP_MARKERS_MPL[group], edgecolor="black", linewidth=0.25,
                       label=f"{group} (n={len(points)})")
        else:
            labels = clusters[group]
            for cluster in sorted(np.unique(labels)):
                selected = points[labels == cluster]
                color = CLUSTER_COLORS[palette_index % len(CLUSTER_COLORS)]
                palette_index += 1
                ax.scatter(*selected.T, s=43, alpha=0.78, color=color,
                           marker=GROUP_MARKERS_MPL[group], edgecolor="black", linewidth=0.3,
                           label=f"{group} C{cluster + 1} (n={len(selected)})")
    labels = _axis_labels(measures)
    ax.set_xlabel(labels[0], labelpad=10)
    ax.set_ylabel(labels[1], labelpad=10)
    ax.set_zlabel(labels[2], labelpad=10)
    for setter, limit in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), _limits(group_points)):
        setter(*limit)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title, fontsize=11, pad=14)
    ax.legend(loc="best", fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_interactive(
    group_points: Mapping[str, np.ndarray], group_ids: Mapping[str, list[str]],
    measures: Sequence[str], title: str, output: Path,
    clusters: Mapping[str, np.ndarray] | None = None,
) -> None:
    fig = go.Figure()
    palette_index = 0
    for group in ("Healthy", "Patient"):
        points, ids = group_points[group], np.asarray(group_ids[group])
        if clusters is None:
            trace_specs = [(np.ones(len(points), dtype=bool), GROUP_COLORS[group], f"{group} (n={len(points)})")]
        else:
            trace_specs = []
            for cluster in sorted(np.unique(clusters[group])):
                mask = clusters[group] == cluster
                color = CLUSTER_COLORS[palette_index % len(CLUSTER_COLORS)]
                palette_index += 1
                trace_specs.append((mask, color, f"{group} C{cluster + 1} (n={int(mask.sum())})"))
        for mask, color, name in trace_specs:
            selected = points[mask]
            hover = [f"subject={sid}<br>group={group}" for sid in ids[mask]]
            fig.add_trace(go.Scatter3d(
                x=selected[:, 0], y=selected[:, 1], z=selected[:, 2], mode="markers",
                name=name, text=hover, hovertemplate="%{text}<extra></extra>",
                marker=dict(size=4.5, opacity=0.75, color=color,
                            symbol=GROUP_MARKERS_PLOTLY[group], line=dict(color="black", width=0.5)),
            ))
    labels = _axis_labels(measures)
    limits = _limits(group_points)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title=labels[0], range=list(limits[0]), zeroline=True),
            yaxis=dict(title=labels[1], range=list(limits[1]), zeroline=True),
            zaxis=dict(title=labels[2], range=list(limits[2]), zeroline=True),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=55),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs=True)


def generate_group_metric_figures(profile: SavedDatasetProfile) -> list[dict]:
    ensure_figure_tree(profile)
    network = load_npy_dict(profile.data_dir / "network_measures.npy", "network measures")
    results = load_npy_dict(profile.optimization_results_path, "optimization results")
    analysis = load_analysis_metadata(profile)
    method = str(analysis.get("selected_method") or analysis.get("connectivity_methods", [None])[0])
    outputs = []
    for band in ordered_bands(results):
        metadata = first_band_metadata(results, band)
        measures = list(metadata.get("optimization_measures") or [])
        if len(measures) != 3:
            raise ValueError(f"Band {band} requires exactly three stored optimization measures")
        healthy_target = [metadata["healthy_measure_baselines"][measure] for measure in measures]
        optimized_ids = {
            str(result.get("subject_id", key))
            for key, result in results_for_band(results, band).items()
        }
        for cohort, patient_filter in (("all", None), ("optimized", optimized_ids)):
            group_points, group_ids, skipped = {}, {}, {}
            for group in ("Healthy", "Patient"):
                filter_ids = patient_filter if group == "Patient" else None
                points, ids, n_skipped = extract_group_points(
                    network, group, method, band, measures, healthy_target, filter_ids,
                )
                if points.size == 0:
                    raise ValueError(f"No valid {group} points for {profile.label}, {band}, {cohort}")
                group_points[group], group_ids[group], skipped[group] = points, ids, n_skipped
            output_dir = optimization_figure_dir(profile, "metric_space", f"group_{cohort}")
            cohort_label = "full dataset" if cohort == "all" else "optimization-selected Patients"
            subtitle = (
                f"{profile.label} | {band.capitalize()} | {cohort_label}\n"
                f"Valid H={len(group_ids['Healthy'])}, P={len(group_ids['Patient'])}"
            )
            scatter_png = output_dir / f"group_scatter_3d_{band}.png"
            scatter_html = output_dir / f"group_scatter_3d_{band}.html"
            _plot_static(group_points, group_ids, measures, subtitle, scatter_png)
            _plot_interactive(group_points, group_ids, measures, subtitle, scatter_html)

            cluster_labels, score_tables, selected_ks = {}, [], {}
            assignment_rows = []
            for group in ("Healthy", "Patient"):
                labels, selected_k, scores = select_kmeans(group_points[group])
                cluster_labels[group], selected_ks[group] = labels, selected_k
                scores.insert(0, "group", group)
                score_tables.append(scores)
                for subject_id, cluster, point in zip(group_ids[group], labels, group_points[group]):
                    assignment_rows.append({
                        "subject_id": subject_id, "group": group, "cluster": int(cluster + 1),
                        **{measure: float(value) for measure, value in zip(measures, point)},
                    })
            cluster_title = subtitle + f"; k(H)={selected_ks['Healthy']}, k(P)={selected_ks['Patient']}"
            cluster_png = output_dir / f"group_clusters_3d_{band}.png"
            cluster_html = output_dir / f"group_clusters_3d_{band}.html"
            _plot_static(group_points, group_ids, measures, cluster_title, cluster_png, cluster_labels)
            _plot_interactive(group_points, group_ids, measures, cluster_title, cluster_html, cluster_labels)
            assignments = output_dir / f"group_clusters_3d_{band}_assignments.csv"
            scores_path = output_dir / f"group_clusters_3d_{band}_silhouette_scores.csv"
            pd.DataFrame(assignment_rows).to_csv(assignments, index=False)
            pd.concat(score_tables, ignore_index=True).to_csv(scores_path, index=False)
            outputs.append({
                "band": band, "cohort": cohort, "scatter_png": scatter_png,
                "scatter_html": scatter_html, "cluster_png": cluster_png,
                "cluster_html": cluster_html, "assignments": assignments,
                "scores": scores_path, "skipped": skipped,
            })
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True)
    args = parser.parse_args()
    profile = load_dataset_profile(args.dataset_config)
    for output in generate_group_metric_figures(profile):
        print(f"Saved {output['band']} {output['cohort']}: {output['scatter_png']}")


if __name__ == "__main__":
    main()
