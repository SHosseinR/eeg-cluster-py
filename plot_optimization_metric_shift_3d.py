"""Plot per-band 3D metric shifts from stored optimization results.

By default, each axis is the signed relative difference from the healthy mean,
``(value - healthy_mean) / abs(healthy_mean)``.  This is the signed counterpart
of the distance-to-GT objectives and makes differently-scaled measures directly
comparable.  Raw metric coordinates remain available with ``--scale raw``.
"""
import argparse
import os
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from optimization_config import (
    OPTIMIZATION_FIGURES_DIR,
    OPTIMIZATION_OUTPUT_DIR,
    OPTIMIZATION_RESULTS_FILE,
)


def _load_pickle_dict(path: str) -> Dict:
    return np.load(path, allow_pickle=True).item()


def _get_result_band_info(results: Dict) -> Tuple[Optional[int], Optional[str]]:
    band_idx = results.get("fixed_band_index")
    try:
        band_idx = int(band_idx) if band_idx is not None else None
    except (TypeError, ValueError):
        band_idx = None

    band_name = results.get("fixed_band_name")
    band_name = str(band_name) if band_name else None
    best_solution = results.get("best_solution") or {}
    if band_idx is None and "band" in best_solution:
        try:
            band_idx = int(best_solution["band"])
        except (TypeError, ValueError):
            pass
    if band_name is None and band_idx is not None:
        band_names = results.get("band_names") or []
        if 0 <= band_idx < len(band_names):
            band_name = str(band_names[band_idx])
    return band_idx, band_name


def _available_bands(optimization_results: Dict) -> List[Tuple[Optional[int], str]]:
    """Return bands in stored metadata order, without silently mixing measures."""
    found = {}
    preferred_order = []
    for result in optimization_results.values():
        if not isinstance(result, dict):
            continue
        if not preferred_order and result.get("band_names"):
            preferred_order = [str(name) for name in result["band_names"]]
        band_idx, band_name = _get_result_band_info(result)
        if band_name is not None:
            found[band_name] = band_idx

    ordered = [name for name in preferred_order if name in found]
    ordered.extend(name for name in found if name not in ordered)
    return [(found[name], name) for name in ordered]


def _find_metadata(optimization_results: Dict, band_name: str) -> Optional[Dict]:
    for result in optimization_results.values():
        if not isinstance(result, dict):
            continue
        if _get_result_band_info(result)[1] != band_name:
            continue
        if result.get("healthy_measure_baselines") and result.get("optimization_measures"):
            return result
    return None


def _extract_points(
    optimization_results: Dict,
    measures: List[str],
    band_name: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple[str, str]]]:
    initial_points, final_points, labels, skipped = [], [], [], []
    for key, result in optimization_results.items():
        if not isinstance(result, dict) or _get_result_band_info(result)[1] != band_name:
            continue
        label = str(result.get("subject_id", key))
        initial, final = result.get("initial_metrics"), result.get("final_metrics")
        if initial is None or final is None:
            skipped.append((label, "missing initial_metrics or final_metrics"))
            continue
        initial = np.asarray(initial, dtype=float)
        final = np.asarray(final, dtype=float)
        if initial.size != len(measures) or final.size != len(measures):
            skipped.append((label, "metric count does not match this band's measures"))
            continue
        if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(final)):
            skipped.append((label, "metrics contain non-finite values"))
            continue
        initial_points.append(initial)
        final_points.append(final)
        labels.append(label)

    shape = (0, len(measures))
    if not initial_points:
        return np.empty(shape), np.empty(shape), labels, skipped
    return np.vstack(initial_points), np.vstack(final_points), labels, skipped


def _healthy_relative(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    denominator = np.where(np.abs(target) > 1e-10, np.abs(target), 1.0)
    return (points - target) / denominator


def _padded_limits(values: Iterable[float], include: float = 0.0) -> Tuple[float, float]:
    values = np.append(np.asarray(list(values), dtype=float), include)
    low, high = float(np.min(values)), float(np.max(values))
    span = high - low
    padding = 0.08 * span if span > 1e-12 else max(abs(low) * 0.08, 0.08)
    return low - padding, high + padding


def _output_path(base: Optional[str], default_dir: str, stem: str, extension: str) -> str:
    if base is None:
        return os.path.join(default_dir, f"{stem}.{extension}")
    if "{band}" in base:
        return base
    root, ext = os.path.splitext(base)
    if ext:
        return f"{root}_{stem.rsplit('_', 1)[-1]}{ext}"
    return os.path.join(base, f"{stem}.{extension}")


def _diagnose_shift_geometry(initial: np.ndarray, final: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative_delta = _healthy_relative(final, target) - _healthy_relative(initial, target)
    centered = relative_delta - np.mean(relative_delta, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    variance = singular_values ** 2
    return variance / variance.sum() if variance.sum() > 0 else np.zeros(relative_delta.shape[1])


def _plot_band(
    optimization_results: Dict,
    band_name: str,
    scale: str,
    output: Optional[str],
    html_output: Optional[str],
) -> Tuple[str, str, np.ndarray, List[Tuple[str, str]]]:
    metadata = _find_metadata(optimization_results, band_name)
    if metadata is None:
        raise RuntimeError(f"No optimization metadata found for band {band_name!r}")
    measures = list(metadata["optimization_measures"])
    if len(measures) != 3:
        raise ValueError(f"Band {band_name!r} requires exactly 3 measures; got {measures}")
    target_raw = np.array(
        [metadata["healthy_measure_baselines"][measure] for measure in measures], dtype=float
    )
    initial_raw, final_raw, subject_labels, skipped = _extract_points(
        optimization_results, measures, band_name
    )
    if initial_raw.size == 0:
        raise RuntimeError(f"No valid initial/final metric pairs for band {band_name!r}")

    explained_variance = _diagnose_shift_geometry(initial_raw, final_raw, target_raw)
    if scale == "healthy-relative":
        initial = _healthy_relative(initial_raw, target_raw)
        final = _healthy_relative(final_raw, target_raw)
        target = np.zeros(3)
        axis_suffix = " (relative difference from healthy)"
    else:
        initial, final, target = initial_raw, final_raw, target_raw
        axis_suffix = " (raw)"

    limits = [
        _padded_limits(np.concatenate([initial[:, axis], final[:, axis]]), target[axis])
        for axis in range(3)
    ]
    title = f"{band_name.capitalize()} metric shift: initial -> final vs healthy mean"
    stem = f"metric_shift_3d_{band_name}"
    png_path = _output_path(output, OPTIMIZATION_FIGURES_DIR, stem, "png").replace(
        "{band}", band_name
    )
    html_path = _output_path(html_output, OPTIMIZATION_FIGURES_DIR, stem, "html").replace(
        "{band}", band_name
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.scatter(*initial.T, s=30, alpha=0.60, color="#1f77b4", label="Initial", zorder=2)
    ax.scatter(*final.T, s=34, alpha=0.78, color="#d62728", label="Final", zorder=3)
    for start, end in zip(initial, final):
        ax.plot(*np.vstack([start, end]).T, color="#444444", alpha=0.45, linewidth=0.8, zorder=1)
    # Draw last with fixed z-order, high contrast, and an outline so it remains visible.
    ax.scatter(
        *target.reshape(3, 1), s=330, color="#ffdd00", edgecolor="black", linewidth=1.5,
        marker="*", depthshade=False, label="Healthy mean target", zorder=20,
    )
    ax.text(*(target + np.array([0.0, 0.0, 0.025 * (limits[2][1] - limits[2][0])])),
            " healthy target", color="black", fontsize=9, zorder=21)
    for axis, measure in enumerate(measures):
        label = measure.replace("_", " ") + axis_suffix
        (ax.set_xlabel, ax.set_ylabel, ax.set_zlabel)[axis](label, labelpad=10)
        (ax.set_xlim, ax.set_ylim, ax.set_zlim)[axis](*limits[axis])
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title)
    ax.legend(loc="upper left")
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plt.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    line_coordinates = [[], [], []]
    for start, end in zip(initial, final):
        for axis in range(3):
            line_coordinates[axis].extend([start[axis], end[axis], None])
    hover_initial = [f"{label}<br>state=initial" for label in subject_labels]
    hover_final = [f"{label}<br>state=final" for label in subject_labels]
    interactive = go.Figure()
    interactive.add_trace(go.Scatter3d(
        x=initial[:, 0], y=initial[:, 1], z=initial[:, 2], mode="markers", name="Initial",
        text=hover_initial, hovertemplate="%{text}<extra></extra>",
        marker=dict(size=4, color="#1f77b4", opacity=0.65),
    ))
    interactive.add_trace(go.Scatter3d(
        x=final[:, 0], y=final[:, 1], z=final[:, 2], mode="markers", name="Final",
        text=hover_final, hovertemplate="%{text}<extra></extra>",
        marker=dict(size=4, color="#d62728", opacity=0.78),
    ))
    interactive.add_trace(go.Scatter3d(
        x=line_coordinates[0], y=line_coordinates[1], z=line_coordinates[2], mode="lines",
        line=dict(color="#555555", width=3), opacity=0.45, showlegend=False, hoverinfo="skip",
    ))
    interactive.add_trace(go.Scatter3d(
        x=[target[0]], y=[target[1]], z=[target[2]], mode="markers+text",
        name="Healthy mean target", text=["Healthy target"], textposition="top center",
        marker=dict(size=11, color="#ffdd00", line=dict(color="black", width=4), symbol="diamond"),
        hovertemplate="Healthy mean target<extra></extra>",
    ))
    scene_axes = []
    for axis, measure in enumerate(measures):
        scene_axes.append(dict(
            title=measure.replace("_", " ") + axis_suffix,
            range=list(limits[axis]), zeroline=True, zerolinewidth=3, zerolinecolor="#999999",
        ))
    interactive.update_layout(
        title=title,
        scene=dict(xaxis=scene_axes[0], yaxis=scene_axes[1], zaxis=scene_axes[2], aspectmode="cube"),
        legend=dict(x=0.01, y=0.99), margin=dict(l=0, r=0, b=0, t=45),
    )
    os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)
    interactive.write_html(html_path)
    return png_path, html_path, explained_variance, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-band 3D metric shifts vs healthy mean")
    parser.add_argument("--results", default=None, help="Override optimization_results.npy path")
    parser.add_argument("--output", default=None, help="PNG directory/path; supports {band}")
    parser.add_argument("--html-output", default=None, help="HTML directory/path; supports {band}")
    parser.add_argument("--band", default=None, help="Only plot this band (default: every stored band)")
    parser.add_argument(
        "--scale", choices=("healthy-relative", "raw"), default="healthy-relative",
        help="Axis scale; healthy-relative puts the healthy target at (0,0,0)",
    )
    args = parser.parse_args()
    results_path = args.results or os.path.join(OPTIMIZATION_OUTPUT_DIR, OPTIMIZATION_RESULTS_FILE)
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Optimization results not found: {results_path}")
    optimization_results = _load_pickle_dict(results_path)
    available = _available_bands(optimization_results)
    bands = [name for _, name in available]
    if args.band is not None:
        try:
            requested_idx = int(args.band)
            bands = [name for idx, name in available if idx == requested_idx]
        except ValueError:
            bands = [name for name in bands if name.casefold() == args.band.casefold()]
        if not bands:
            raise ValueError(f"Band {args.band!r} not found; available bands: {[name for _, name in available]}")
    if not bands:
        raise RuntimeError("No per-band metadata found in optimization results")

    for band_name in bands:
        png_path, html_path, explained, skipped = _plot_band(
            optimization_results, band_name, args.scale, args.output, args.html_output
        )
        print(f"Saved {band_name} metric-shift PNG: {png_path}")
        print(f"Saved {band_name} metric-shift HTML: {html_path}")
        print(
            f"  normalized-shift PCA explained variance: "
            f"PC1={explained[0]:.1%}, PC2={explained[1]:.1%}, PC3={explained[2]:.1%}"
        )
        for subject_id, reason in skipped:
            print(f"  skipped {subject_id}: {reason}")


if __name__ == "__main__":
    main()
