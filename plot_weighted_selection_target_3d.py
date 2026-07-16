"""Plot rank-weighted stimulation-selection targets on a 2D EEG head map.

The input is the already-computed optimization results. Each subject's stored
top solutions contribute their stored ``strength`` (normally 1/rank) to the
selected electrode. No optimization data is modified or recomputed.

The historical filename is retained so existing commands keep working, but the
figure is deliberately a top-down 2D 10-20-style layout rather than a rendered
3D scalp viewed through an oblique camera.
"""
import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Polygon
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import CHANNEL_ALIAS_MONTAGE
from optimization_config import (
    OPTIMIZATION_FIGURES_DIR,
    OPTIMIZATION_OUTPUT_DIR,
    OPTIMIZATION_RESULTS_FILE,
    OPTIMIZATION_TOP_K,
)
from saved_results_utils import load_dataset_profile


def _load_results(path: str) -> Dict:
    return np.load(path, allow_pickle=True).item()


def _band_name(result: Dict) -> Optional[str]:
    name = result.get("fixed_band_name")
    if name:
        return str(name)
    solution = result.get("best_solution") or {}
    band_idx = result.get("fixed_band_index", solution.get("band"))
    try:
        band_idx = int(band_idx)
    except (TypeError, ValueError):
        return None
    names = result.get("band_names") or []
    return str(names[band_idx]) if 0 <= band_idx < len(names) else f"band{band_idx}"


def _metadata(results: Dict) -> Dict:
    for result in results.values():
        if isinstance(result, dict) and result.get("channel_names"):
            return result
    raise RuntimeError("Optimization results do not contain channel metadata")


def _ordered_bands(results: Dict, metadata: Dict) -> List[str]:
    present = {_band_name(result) for result in results.values() if isinstance(result, dict)}
    present.discard(None)
    order = [str(name) for name in metadata.get("band_names", []) if str(name) in present]
    order.extend(sorted(name for name in present if name not in order))
    return order


def _channel_positions(channel_names: List[str], metadata: Dict) -> np.ndarray:
    """Resolve exact labels first, with fallbacks for legacy referenced datasets."""
    import mne
    montage = mne.channels.make_standard_montage(CHANNEL_ALIAS_MONTAGE)
    montage_positions = montage.get_positions()["ch_pos"]
    casefold_positions = {name.casefold(): value for name, value in montage_positions.items()}
    aliases = (metadata.get("channel_metadata") or {}).get("channel_aliases", {})

    def montage_position(label: str) -> Optional[np.ndarray]:
        return casefold_positions.get(str(label).strip().casefold())

    def referenced_position(label: str) -> Optional[np.ndarray]:
        """Interpret unresolved legacy reference labels without changing exact labels."""
        label = str(label).strip()
        for suffix in ("-LE", "-REF", "-AVG"):
            if label.upper().endswith(suffix):
                position = montage_position(label[:-len(suffix)])
                if position is not None:
                    return position
        if label.count("-") == 1:
            first, second = (part.strip() for part in label.split("-", 1))
            first_position = montage_position(first)
            second_position = montage_position(second)
            if first_position is not None and second_position is not None:
                return (first_position + second_position) / 2.0
        return None

    positions, missing = [], []
    for name in channel_names:
        candidates = [str(name), str(aliases.get(name, ""))]
        position = None
        for candidate in candidates:
            if candidate:
                position = montage_position(candidate)
            if position is not None:
                break
        if position is None:
            position = referenced_position(name)
        if position is None:
            missing.append(name)
            positions.append([np.nan, np.nan, np.nan])
        else:
            positions.append(position)
    if missing:
        raise ValueError(
            f"Channels are absent from montage {CHANNEL_ALIAS_MONTAGE!r}: {missing}. "
            "Store standard channel aliases in channel_metadata before plotting; "
            "legacy -LE and bipolar reference labels are handled automatically."
        )
    return np.asarray(positions, dtype=float)


def _project_topdown(positions: np.ndarray) -> np.ndarray:
    """Project montage coordinates to a top-down azimuthal scalp layout.

    Azimuth is preserved and radius is proportional to angular distance from
    the vertex. This is a true coordinate projection, not a flattened 3D image.
    """
    positions = np.asarray(positions, dtype=float)
    norms = np.linalg.norm(positions, axis=1)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("Cannot project an electrode at the montage origin")
    unit = positions / norms[:, None]
    planar = np.linalg.norm(unit[:, :2], axis=1)
    theta = np.arctan2(planar, unit[:, 2])
    radius = theta / (np.pi / 2.0)
    projected = np.zeros((len(positions), 2), dtype=float)
    non_vertex = planar > np.finfo(float).eps
    projected[non_vertex] = (
        radius[non_vertex, None]
        * unit[non_vertex, :2]
        / planar[non_vertex, None]
    )
    return projected


def _channel_positions_2d(channel_names: List[str], metadata: Dict) -> np.ndarray:
    """Resolve channels directly for the 2D head map.

    A stored bipolar label such as A2-A1 represents A2 referenced to A1. Its
    3D midpoint is useful as a neutral fallback elsewhere, but has no scalp
    location; on this placement map it is therefore shown at the active (first)
    electrode, A2.
    """
    # Stable 10-20/10-10 top-down coordinates for the channels used by the
    # configured datasets. This avoids importing the full MNE stack for a
    # saved-results plotting task. Unknown labels retain the exact MNE fallback.
    standard = {
        "fp1": (-0.48, 0.92), "fp2": (0.48, 0.92),
        "f7": (-0.93, 0.46), "f3": (-0.48, 0.52), "fz": (0.0, 0.57),
        "f4": (0.48, 0.52), "f8": (0.93, 0.46),
        "fc3": (-0.47, 0.27), "fcz": (0.0, 0.29), "fc4": (0.47, 0.27),
        "t7": (-1.0, 0.0), "t3": (-1.0, 0.0), "c3": (-0.52, 0.0),
        "cz": (0.0, 0.0), "c4": (0.52, 0.0), "t8": (1.0, 0.0), "t4": (1.0, 0.0),
        "cp3": (-0.47, -0.27), "cpz": (0.0, -0.29), "cp4": (0.47, -0.27),
        "p7": (-0.91, -0.48), "t5": (-0.91, -0.48), "p3": (-0.47, -0.53),
        "pz": (0.0, -0.58), "p4": (0.47, -0.53), "p8": (0.91, -0.48),
        "t6": (0.91, -0.48), "o1": (-0.36, -0.91), "oz": (0.0, -0.95),
        "o2": (0.36, -0.91), "a1": (-1.25, -0.08), "a2": (1.25, -0.08),
    }
    projected, unresolved = [], []
    suffixes = ("-LE", "-REF", "-AVG")
    for index, name in enumerate(channel_names):
        label = str(name).strip()
        for suffix in suffixes:
            if label.upper().endswith(suffix):
                label = label[:-len(suffix)]
                break
        if label.count("-") == 1:
            label = label.split("-", 1)[0].strip()
        position = standard.get(label.casefold())
        if position is None:
            projected.append((np.nan, np.nan))
            unresolved.append(index)
        else:
            projected.append(position)
    projected = np.asarray(projected, dtype=float)
    if unresolved:
        fallback = _project_topdown(_channel_positions(channel_names, metadata))
        projected[unresolved] = fallback[unresolved]
    return projected


def _display_labels(labels: List[str]) -> List[str]:
    """Remove legacy reference suffixes that add clutter to the scalp map."""
    cleaned = []
    for label in labels:
        text = str(label).strip()
        for suffix in ("-LE", "-REF", "-AVG"):
            if text.upper().endswith(suffix):
                text = text[:-len(suffix)]
                break
        cleaned.append(text)
    return cleaned


def _weighted_scores(results: Dict, band: str, n_channels: int, top_k: int) -> Tuple[np.ndarray, int]:
    scores = np.zeros(n_channels, dtype=float)
    n_subjects = 0
    for result in results.values():
        if not isinstance(result, dict) or _band_name(result) != band:
            continue
        ranked = list(result.get("top_solutions") or [])[:top_k]
        if not ranked and result.get("best_solution"):
            ranked = [result["best_solution"]]
        contributed = False
        for position, solution in enumerate(ranked, start=1):
            if solution is None or solution.get("node") is None:
                continue
            node = int(solution["node"])
            if not 0 <= node < n_channels:
                continue
            rank = solution.get("rank", position)
            try:
                rank = max(float(rank), 1.0)
            except (TypeError, ValueError):
                rank = float(position)
            strength = solution.get("strength", 1.0 / rank)
            scores[node] += float(strength)
            contributed = True
        n_subjects += int(contributed)
    return scores, n_subjects


def _validity_score(median_relative_improvement: float) -> float:
    return float(np.clip(float(median_relative_improvement), 0.0, 1.0))


def _marker_radius(median_relative_improvement: float) -> float:
    """Marker radius in points; linear in clipped median improvement."""
    return 9.0 + 12.0 * _validity_score(median_relative_improvement)


def _marker_sizes(n_nodes: int, median_relative_improvement: float) -> np.ndarray:
    radius = _marker_radius(median_relative_improvement)
    return np.full(n_nodes, np.pi * radius ** 2, dtype=float)


def aggregate_validity_weighted_scores(
    band_scores: Dict[str, np.ndarray],
    validity: Dict[str, Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Return raw and band-validity-weighted sums across all stored bands."""
    if not band_scores:
        raise ValueError("At least one band's node scores are required")
    shape = next(iter(band_scores.values())).shape
    raw_sum = np.zeros(shape, dtype=float)
    weighted_sum = np.zeros(shape, dtype=float)
    factors = {}
    for band, scores in band_scores.items():
        if band not in validity:
            raise KeyError(f"Band {band!r} is absent from the stability summary")
        values = np.asarray(scores, dtype=float)
        if values.shape != shape:
            raise ValueError("All bands must use the same channel ordering")
        factor = _validity_score(validity[band]["median"])
        raw_sum += values
        weighted_sum += factor * values
        factors[band] = factor
    return raw_sum, weighted_sum, factors


def _combined_marker_radii(weighted_scores: np.ndarray) -> np.ndarray:
    """Radius is linear in the combined validity-weighted node score."""
    scores = np.asarray(weighted_scores, dtype=float)
    maximum = float(np.max(scores)) if scores.size else 0.0
    normalized = scores / maximum if maximum > 0 else np.zeros_like(scores)
    return 7.0 + 17.0 * normalized


def _load_band_validity(summary_path: str) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Band stability summary not found: {summary_path}. "
            "Run plot_band_stability_analysis.py on the saved per-band results first."
        )
    summary = pd.read_csv(summary_path)
    required = {"band", "median_relative_improvement", "n_subjects"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Band stability summary is missing columns: {sorted(missing)}")
    return {
        str(row["band"]): {
            "median": float(row["median_relative_improvement"]),
            "n_subjects": int(row["n_subjects"]),
        }
        for _, row in summary.iterrows()
    }


def _head_geometry() -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return shared 2D outline geometry for Matplotlib and Plotly."""
    head_radius = 1.10
    angle = np.linspace(0.0, 2.0 * np.pi, 361)
    head = np.column_stack((head_radius * np.cos(angle), head_radius * np.sin(angle)))
    guide = np.column_stack((0.78 * np.cos(angle), 0.78 * np.sin(angle)))
    nose = np.array([[-0.18, head_radius], [0.0, 1.31], [0.18, head_radius]])
    return head_radius, head, guide, nose


def _plot_static(
    positions: np.ndarray,
    labels: List[str],
    scores: np.ndarray,
    band: str,
    n_subjects: int,
    top_k: int,
    median_improvement: float,
    validity_n: int,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 9))
    head_radius, _, _, nose = _head_geometry()
    outline_color = "#4b5563"
    guide_color = "#b8bec7"
    ax.add_patch(Circle((0, 0), head_radius, facecolor="#f7f8fa",
                        edgecolor=outline_color, linewidth=1.8, zorder=0))
    ax.add_patch(Circle((0, 0), 0.78, fill=False, edgecolor=guide_color,
                        linewidth=1.0, linestyle=(0, (2, 3)), zorder=1))
    ax.plot([0, 0], [-head_radius, head_radius], color=guide_color,
            linewidth=1.0, linestyle=(0, (2, 3)), zorder=1)
    ax.plot([-head_radius, head_radius], [0, 0], color=guide_color,
            linewidth=1.0, linestyle=(0, (2, 3)), zorder=1)
    ax.add_patch(Polygon(nose, closed=False, fill=False, edgecolor=outline_color,
                         linewidth=1.8, joinstyle="round", zorder=1))
    ax.add_patch(Arc((-head_radius, 0), 0.29, 0.58, theta1=75, theta2=285,
                     color=outline_color, linewidth=1.8, zorder=1))
    ax.add_patch(Arc((head_radius, 0), 0.29, 0.58, theta1=-105, theta2=105,
                     color=outline_color, linewidth=1.8, zorder=1))
    scatter = ax.scatter(
        positions[:, 0], positions[:, 1], c=scores,
        s=_marker_sizes(len(scores), median_improvement),
        cmap="viridis", edgecolor="black", linewidth=0.7, zorder=3,
    )
    maximum = float(np.max(scores))
    for label, point, score in zip(labels, positions, scores):
        font_size = 9 if len(labels) <= 24 else 7.5
        if len(label) > 4:
            font_size = min(font_size, 7)
        ax.text(point[0], point[1], label, ha="center", va="center",
                fontsize=font_size, fontweight="semibold",
                color="white" if maximum > 0 and score > 0.58 * maximum else "black",
                zorder=4)
    top = int(np.argmax(scores))
    marker_sizes = _marker_sizes(len(scores), median_improvement)
    ax.scatter(positions[top, 0], positions[top, 1], s=marker_sizes[top] + 280,
               facecolors="none", edgecolors="#dc267f", linewidths=3, zorder=5)
    ax.set_title(
        f"{band.capitalize()} rank-weighted selection targets\n"
        f"{n_subjects} subjects, top-{top_k}, weight = 1/rank; "
        f"top = {labels[top]} ({scores[top]:.2f})\n"
        f"Radius validity={_validity_score(median_improvement):.3f} "
        f"(median improvement={median_improvement:.3f}, stability n={validity_n})"
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.58, 1.58)
    ax.set_ylim(-1.25, 1.43)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.02)
    colorbar.set_label("rank-weighted selection score")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_interactive(
    positions: np.ndarray,
    labels: List[str],
    scores: np.ndarray,
    band: str,
    n_subjects: int,
    top_k: int,
    median_improvement: float,
    validity_n: int,
    output_path: str,
) -> None:
    top = int(np.argmax(scores))
    hover = [
        f"channel={label}<br>weighted score={score:.4f}<br>"
        f"score/subject={score / max(n_subjects, 1):.4f}"
        for label, score in zip(labels, scores)
    ]
    fig = go.Figure()
    head_radius, head, guide, nose = _head_geometry()
    outline_style = dict(color="#4b5563", width=3)
    guide_style = dict(color="#b8bec7", width=1, dash="dot")
    for points, style in ((head, outline_style), (guide, guide_style), (nose, outline_style)):
        fig.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1], mode="lines", line=style,
            hoverinfo="skip", showlegend=False,
        ))
    fig.add_shape(type="line", x0=0, x1=0, y0=-head_radius, y1=head_radius,
                  line=guide_style)
    fig.add_shape(type="line", x0=-head_radius, x1=head_radius, y0=0, y1=0,
                  line=guide_style)
    fig.add_shape(type="circle", x0=-1.24, x1=-0.99, y0=-0.30, y1=0.30,
                  line=outline_style, fillcolor="rgba(0,0,0,0)")
    fig.add_shape(type="circle", x0=0.99, x1=1.24, y0=-0.30, y1=0.30,
                  line=outline_style, fillcolor="rgba(0,0,0,0)")
    maximum = float(np.max(scores))
    marker_diameter = 2.0 * _marker_radius(median_improvement)
    fig.add_trace(go.Scatter(
        x=positions[:, 0], y=positions[:, 1], mode="markers+text",
        text=labels, textposition="middle center", hovertext=hover,
        hovertemplate="%{hovertext}<extra></extra>",
        marker=dict(
            size=np.full(len(scores), marker_diameter), color=scores,
            colorscale="Viridis", showscale=True,
            colorbar=dict(title="Weighted score"), line=dict(color="black", width=1),
        ), name="Electrodes",
    ))
    fig.add_trace(go.Scatter(
        x=[positions[top, 0]], y=[positions[top, 1]], mode="markers",
        marker=dict(size=49, color="rgba(0,0,0,0)", line=dict(color="#dc267f", width=5)),
        name=f"Top target: {labels[top]}",
        hovertemplate=f"Top target: {labels[top]}<extra></extra>",
    ))
    fig.update_layout(
        title=(f"{band.capitalize()} rank-weighted selection targets — "
               f"{n_subjects} subjects, top-{top_k}, weight = 1/rank; "
               f"radius validity={_validity_score(median_improvement):.3f} "
               f"(median improvement={median_improvement:.3f}, stability n={validity_n})"),
        xaxis=dict(visible=False, range=[-1.58, 1.58]),
        yaxis=dict(visible=False, range=[-1.25, 1.43], scaleanchor="x", scaleratio=1),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10, r=10, b=10, t=60),
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.write_html(output_path)


def _plot_combined_static(
    positions: np.ndarray,
    labels: List[str],
    raw_scores: np.ndarray,
    weighted_scores: np.ndarray,
    factors: Dict[str, float],
    n_units: int,
    top_k: int,
    output_path: str,
) -> None:
    """Plot raw all-band scores as color and validity-weighted sums as radius."""
    fig, ax = plt.subplots(figsize=(10, 9))
    head_radius, _, _, nose = _head_geometry()
    outline_color = "#4b5563"
    guide_color = "#b8bec7"
    ax.add_patch(Circle((0, 0), head_radius, facecolor="#f7f8fa",
                        edgecolor=outline_color, linewidth=1.8, zorder=0))
    ax.add_patch(Circle((0, 0), 0.78, fill=False, edgecolor=guide_color,
                        linewidth=1.0, linestyle=(0, (2, 3)), zorder=1))
    ax.plot([0, 0], [-head_radius, head_radius], color=guide_color,
            linewidth=1.0, linestyle=(0, (2, 3)), zorder=1)
    ax.plot([-head_radius, head_radius], [0, 0], color=guide_color,
            linewidth=1.0, linestyle=(0, (2, 3)), zorder=1)
    ax.add_patch(Polygon(nose, closed=False, fill=False, edgecolor=outline_color,
                         linewidth=1.8, joinstyle="round", zorder=1))
    ax.add_patch(Arc((-head_radius, 0), 0.29, 0.58, theta1=75, theta2=285,
                     color=outline_color, linewidth=1.8, zorder=1))
    ax.add_patch(Arc((head_radius, 0), 0.29, 0.58, theta1=-105, theta2=105,
                     color=outline_color, linewidth=1.8, zorder=1))

    radii = _combined_marker_radii(weighted_scores)
    sizes = np.pi * radii ** 2
    scatter = ax.scatter(
        positions[:, 0], positions[:, 1], c=raw_scores, s=sizes,
        cmap="viridis", edgecolor="black", linewidth=0.7, zorder=3,
    )
    color_maximum = float(np.max(raw_scores))
    for label, point, raw_score in zip(labels, positions, raw_scores):
        font_size = 9 if len(labels) <= 24 else 7.5
        if len(label) > 4:
            font_size = min(font_size, 7)
        ax.text(
            point[0], point[1], label, ha="center", va="center",
            fontsize=font_size, fontweight="semibold",
            color="white" if color_maximum > 0 and raw_score > 0.58 * color_maximum else "black",
            zorder=4,
        )

    top = int(np.argmax(weighted_scores))
    ax.scatter(
        positions[top, 0], positions[top, 1], s=sizes[top] + 280,
        facecolors="none", edgecolors="#dc267f", linewidths=3, zorder=5,
    )
    factor_text = ", ".join(f"{band}={factor:.3f}" for band, factor in factors.items())
    ax.set_title(
        "All-band rank-weighted selection targets\n"
        f"Color = raw band sum; radius = sum(score x band validity); "
        f"top = {labels[top]} ({weighted_scores[top]:.2f})\n"
        f"Validity factors: {factor_text}; {n_units} band-subject units, top-{top_k}"
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.58, 1.58)
    ax.set_ylim(-1.25, 1.43)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.02)
    colorbar.set_label("raw all-band rank-weighted selection sum")

    weighted_maximum = float(np.max(weighted_scores))
    if weighted_maximum > 0:
        legend_values = weighted_maximum * np.array([0.25, 0.50, 1.0])
        legend_radii = 7.0 + 17.0 * legend_values / weighted_maximum
        handles = [
            ax.scatter([], [], s=np.pi * radius ** 2, facecolor="#d1d5db",
                       edgecolor="black", linewidth=0.7)
            for radius in legend_radii
        ]
        ax.legend(
            handles, [f"{value:.2f}" for value in legend_values],
            title="Validity-weighted sum\n(node radius)", loc="lower left",
            frameon=True, fontsize=8, title_fontsize=8, scatterpoints=1,
            markerscale=0.55, labelspacing=1.35, handletextpad=1.2,
            borderpad=0.9,
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_interactive(
    positions: np.ndarray,
    labels: List[str],
    raw_scores: np.ndarray,
    weighted_scores: np.ndarray,
    band_scores: Dict[str, np.ndarray],
    factors: Dict[str, float],
    n_units: int,
    top_k: int,
    output_path: str,
) -> None:
    top = int(np.argmax(weighted_scores))
    hover = []
    for node, label in enumerate(labels):
        detail = "".join(
            f"<br>{band}: raw={scores[node]:.4f}, factor={factors[band]:.4f}, "
            f"contribution={scores[node] * factors[band]:.4f}"
            for band, scores in band_scores.items()
        )
        hover.append(
            f"channel={label}<br>raw all-band sum={raw_scores[node]:.4f}"
            f"<br>validity-weighted sum={weighted_scores[node]:.4f}{detail}"
        )

    fig = go.Figure()
    head_radius, head, guide, nose = _head_geometry()
    outline_style = dict(color="#4b5563", width=3)
    guide_style = dict(color="#b8bec7", width=1, dash="dot")
    for points, style in ((head, outline_style), (guide, guide_style), (nose, outline_style)):
        fig.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1], mode="lines", line=style,
            hoverinfo="skip", showlegend=False,
        ))
    fig.add_shape(type="line", x0=0, x1=0, y0=-head_radius, y1=head_radius,
                  line=guide_style)
    fig.add_shape(type="line", x0=-head_radius, x1=head_radius, y0=0, y1=0,
                  line=guide_style)
    fig.add_shape(type="circle", x0=-1.24, x1=-0.99, y0=-0.30, y1=0.30,
                  line=outline_style, fillcolor="rgba(0,0,0,0)")
    fig.add_shape(type="circle", x0=0.99, x1=1.24, y0=-0.30, y1=0.30,
                  line=outline_style, fillcolor="rgba(0,0,0,0)")

    diameters = 2.0 * _combined_marker_radii(weighted_scores)
    fig.add_trace(go.Scatter(
        x=positions[:, 0], y=positions[:, 1], mode="markers+text",
        text=labels, textposition="middle center", hovertext=hover,
        hovertemplate="%{hovertext}<extra></extra>",
        marker=dict(
            size=diameters, color=raw_scores, colorscale="Viridis", showscale=True,
            colorbar=dict(title="Raw all-band sum"), line=dict(color="black", width=1),
        ), name="Electrodes",
    ))
    fig.add_trace(go.Scatter(
        x=[positions[top, 0]], y=[positions[top, 1]], mode="markers",
        marker=dict(
            size=float(diameters[top] + 12), color="rgba(0,0,0,0)",
            line=dict(color="#dc267f", width=5),
        ),
        name=f"Top validity-weighted target: {labels[top]}",
        hovertemplate=f"Top validity-weighted target: {labels[top]}<extra></extra>",
    ))
    factor_text = ", ".join(f"{band}={factor:.3f}" for band, factor in factors.items())
    fig.update_layout(
        title=("All-band rank-weighted selection targets - color = raw sum; "
               "radius = validity-weighted sum<br>"
               f"Validity factors: {factor_text}; {n_units} band-subject units, top-{top_k}"),
        xaxis=dict(visible=False, range=[-1.58, 1.58]),
        yaxis=dict(visible=False, range=[-1.25, 1.43], scaleanchor="x", scaleratio=1),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10, r=10, b=10, t=80),
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.write_html(output_path)


def _write_combined_scores(
    channel_names: List[str],
    display_names: List[str],
    band_scores: Dict[str, np.ndarray],
    validity: Dict[str, Dict[str, float]],
    band_units: Dict[str, int],
    raw_scores: np.ndarray,
    weighted_scores: np.ndarray,
    output_path: str,
) -> None:
    rows = []
    weighted_maximum = float(np.max(weighted_scores))
    for node, (channel, display) in enumerate(zip(channel_names, display_names)):
        for band, scores in band_scores.items():
            factor = _validity_score(validity[band]["median"])
            rows.append({
                "node_index": node,
                "electrode": channel,
                "display_electrode": display,
                "band": band,
                "band_rank_weighted_score": float(scores[node]),
                "band_median_relative_improvement": float(validity[band]["median"]),
                "clipped_band_validity": factor,
                "validity_weighted_contribution": float(scores[node] * factor),
                "band_optimization_unit_count": int(band_units[band]),
                "band_stability_sample_size": int(validity[band]["n_subjects"]),
                "combined_raw_score": float(raw_scores[node]),
                "combined_validity_weighted_score": float(weighted_scores[node]),
                "combined_validity_weighted_rate": (
                    float(weighted_scores[node] / weighted_maximum)
                    if weighted_maximum > 0 else 0.0
                ),
            })
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _path(base: Optional[str], band: str, extension: str) -> str:
    filename = f"weighted_selection_target_2d_{band}.{extension}"
    if base is None:
        return os.path.join(OPTIMIZATION_FIGURES_DIR, "targets", filename)
    if "{band}" in base:
        return base.replace("{band}", band)
    root, ext = os.path.splitext(base)
    if ext:
        return f"{root}_{band}{ext}"
    return os.path.join(base, filename)


def _combined_path(base: str, extension: str) -> str:
    stem = "weighted_selection_target_2d_all_bands_validity_weighted"
    if "{band}" in base:
        candidate = base.replace("{band}", "all_bands_validity_weighted")
        return os.path.splitext(candidate)[0] + f".{extension}"
    root, ext = os.path.splitext(base)
    if ext:
        return f"{root}_all_bands_validity_weighted.{extension}"
    return os.path.join(base, f"{stem}.{extension}")


def generate_weighted_target_figures(
    results_path: str,
    stability_summary_path: str,
    output_base: str,
    html_output_base: Optional[str] = None,
    top_k: int = OPTIMIZATION_TOP_K,
    band_filter: Optional[str] = None,
) -> List[Dict]:
    results = _load_results(results_path)
    metadata = _metadata(results)
    channel_names = list(metadata["channel_names"])
    display_names = _display_labels(list(metadata.get("channel_display_names") or channel_names))
    positions = _channel_positions_2d(channel_names, metadata)
    bands = _ordered_bands(results, metadata)
    if band_filter is not None:
        bands = [band for band in bands if band.casefold() == band_filter.casefold()]
        if not bands:
            raise ValueError(f"Band {band_filter!r} not found in stored results")
    validity = _load_band_validity(stability_summary_path)
    outputs = []
    band_scores = {}
    band_units = {}
    for band in bands:
        if band not in validity:
            raise KeyError(f"Band {band!r} is absent from {stability_summary_path}")
        scores, n_subjects = _weighted_scores(results, band, len(channel_names), top_k)
        if n_subjects == 0:
            continue
        band_scores[band] = scores
        band_units[band] = n_subjects
        median = validity[band]["median"]
        validity_n = validity[band]["n_subjects"]
        png_path = _path(output_base, band, "png")
        html_path = _path(html_output_base or output_base, band, "html")
        _plot_static(
            positions, display_names, scores, band, n_subjects, top_k,
            median, validity_n, png_path,
        )
        _plot_interactive(
            positions, display_names, scores, band, n_subjects, top_k,
            median, validity_n, html_path,
        )
        outputs.append({
            "band": band, "png": png_path, "html": html_path,
            "median_improvement": median, "validity_n": validity_n,
            "scores": scores,
        })
    if band_filter is None and len(band_scores) > 1:
        raw_scores, weighted_scores, factors = aggregate_validity_weighted_scores(
            band_scores, validity,
        )
        combined_png = _combined_path(output_base, "png")
        combined_html = _combined_path(html_output_base or output_base, "html")
        combined_csv = _combined_path(output_base, "csv")
        n_units = int(sum(band_units.values()))
        _plot_combined_static(
            positions, display_names, raw_scores, weighted_scores, factors,
            n_units, top_k, combined_png,
        )
        _plot_combined_interactive(
            positions, display_names, raw_scores, weighted_scores, band_scores,
            factors, n_units, top_k, combined_html,
        )
        _write_combined_scores(
            channel_names, display_names, band_scores, validity, band_units,
            raw_scores, weighted_scores, combined_csv,
        )
        outputs.append({
            "band": "all_bands_validity_weighted",
            "png": combined_png,
            "html": combined_html,
            "csv": combined_csv,
            "raw_scores": raw_scores,
            "validity_weighted_scores": weighted_scores,
            "validity_factors": factors,
            "n_units": n_units,
        })
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-band rank-weighted targets on a 2D EEG head map"
    )
    parser.add_argument("--results", default=None, help="Override optimization_results.npy path")
    parser.add_argument("--dataset-config", default=None, help="Dataset TOML profile name/path")
    parser.add_argument("--stability-summary", default=None, help="Override band_comparison_summary.csv")
    parser.add_argument("--output", default=None, help="PNG directory/path; supports {band}")
    parser.add_argument("--html-output", default=None, help="HTML directory/path; supports {band}")
    parser.add_argument("--band", default=None, help="Only plot this band (default: every stored band)")
    parser.add_argument("--top-k", type=int, default=OPTIMIZATION_TOP_K)
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    profile = load_dataset_profile(args.dataset_config) if args.dataset_config else None
    results_path = args.results or (
        str(profile.optimization_results_path) if profile
        else os.path.join(OPTIMIZATION_OUTPUT_DIR, OPTIMIZATION_RESULTS_FILE)
    )
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Optimization results not found: {results_path}")
    summary_path = args.stability_summary or (
        str(profile.band_summary_path) if profile else
        os.path.join(OPTIMIZATION_OUTPUT_DIR, "band_stability_analysis", "band_comparison_summary.csv")
    )
    structured_default = str(profile.optimization_figures_dir / "targets") if profile else None
    png_base = args.output or structured_default or os.path.join(OPTIMIZATION_FIGURES_DIR, "targets")
    html_base = args.html_output or structured_default or os.path.join(OPTIMIZATION_FIGURES_DIR, "targets")
    for output in generate_weighted_target_figures(
        results_path, summary_path, png_base, html_base, args.top_k, args.band,
    ):
        print(f"Saved {output['band']} weighted-target PNG: {output['png']}")
        print(f"Saved {output['band']} weighted-target HTML: {output['html']}")
        if output.get("csv"):
            print(f"Saved {output['band']} weighted-target CSV: {output['csv']}")


if __name__ == "__main__":
    main()
