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
import mne
import numpy as np
import plotly.graph_objects as go

from config import CHANNEL_ALIAS_MONTAGE
from optimization_config import (
    OPTIMIZATION_FIGURES_DIR,
    OPTIMIZATION_OUTPUT_DIR,
    OPTIMIZATION_RESULTS_FILE,
    OPTIMIZATION_TOP_K,
)


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
    projected = _project_topdown(_channel_positions(channel_names, metadata))
    montage_positions = mne.channels.make_standard_montage(
        CHANNEL_ALIAS_MONTAGE
    ).get_positions()["ch_pos"]
    casefold_positions = {name.casefold(): value for name, value in montage_positions.items()}
    reference_suffixes = ("-LE", "-REF", "-AVG")
    for index, name in enumerate(channel_names):
        label = str(name).strip()
        if label.upper().endswith(reference_suffixes) or label.count("-") != 1:
            continue
        active, reference = (part.strip() for part in label.split("-", 1))
        active_position = casefold_positions.get(active.casefold())
        reference_position = casefold_positions.get(reference.casefold())
        if active_position is not None and reference_position is not None:
            projected[index] = _project_topdown(np.asarray([active_position]))[0]
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


def _marker_sizes(scores: np.ndarray) -> np.ndarray:
    if np.max(scores) <= 0:
        return np.full_like(scores, 300.0)
    return 280.0 + 650.0 * np.sqrt(scores / np.max(scores))


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
        positions[:, 0], positions[:, 1], c=scores, s=_marker_sizes(scores),
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
    ax.scatter(positions[top, 0], positions[top, 1], s=_marker_sizes(scores)[top] + 280,
               facecolors="none", edgecolors="#dc267f", linewidths=3, zorder=5)
    ax.set_title(
        f"{band.capitalize()} rank-weighted selection targets\n"
        f"{n_subjects} subjects, top-{top_k}, weight = 1/rank; "
        f"top = {labels[top]} ({scores[top]:.2f})"
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
    normalized = scores / maximum if maximum > 0 else np.zeros_like(scores)
    fig.add_trace(go.Scatter(
        x=positions[:, 0], y=positions[:, 1], mode="markers+text",
        text=labels, textposition="middle center", hovertext=hover,
        hovertemplate="%{hovertext}<extra></extra>",
        marker=dict(
            size=19 + 20 * np.sqrt(normalized), color=scores,
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
               f"{n_subjects} subjects, top-{top_k}, weight = 1/rank"),
        xaxis=dict(visible=False, range=[-1.58, 1.58]),
        yaxis=dict(visible=False, range=[-1.25, 1.43], scaleanchor="x", scaleratio=1),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10, r=10, b=10, t=60),
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.write_html(output_path)


def _path(base: Optional[str], band: str, extension: str) -> str:
    filename = f"weighted_selection_target_2d_{band}.{extension}"
    if base is None:
        return os.path.join(OPTIMIZATION_FIGURES_DIR, filename)
    if "{band}" in base:
        return base.replace("{band}", band)
    root, ext = os.path.splitext(base)
    if ext:
        return f"{root}_{band}{ext}"
    return os.path.join(base, filename)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-band rank-weighted targets on a 2D EEG head map"
    )
    parser.add_argument("--results", default=None, help="Override optimization_results.npy path")
    parser.add_argument("--output", default=None, help="PNG directory/path; supports {band}")
    parser.add_argument("--html-output", default=None, help="HTML directory/path; supports {band}")
    parser.add_argument("--band", default=None, help="Only plot this band (default: every stored band)")
    parser.add_argument("--top-k", type=int, default=OPTIMIZATION_TOP_K)
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    results_path = args.results or os.path.join(OPTIMIZATION_OUTPUT_DIR, OPTIMIZATION_RESULTS_FILE)
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Optimization results not found: {results_path}")
    results = _load_results(results_path)
    metadata = _metadata(results)
    channel_names = list(metadata["channel_names"])
    display_names = _display_labels(list(metadata.get("channel_display_names") or channel_names))
    positions = _channel_positions_2d(channel_names, metadata)
    bands = _ordered_bands(results, metadata)
    if args.band is not None:
        bands = [band for band in bands if band.casefold() == args.band.casefold()]
        if not bands:
            raise ValueError(f"Band {args.band!r} not found in stored results")

    for band in bands:
        scores, n_subjects = _weighted_scores(results, band, len(channel_names), args.top_k)
        if n_subjects == 0:
            print(f"Skipping {band}: no ranked solutions")
            continue
        png_path = _path(args.output, band, "png")
        html_path = _path(args.html_output, band, "html")
        _plot_static(
            positions, display_names, scores, band, n_subjects, args.top_k, png_path,
        )
        _plot_interactive(
            positions, display_names, scores, band, n_subjects, args.top_k, html_path,
        )
        top = int(np.argmax(scores))
        print(f"Saved {band} weighted-target PNG: {png_path}")
        print(f"Saved {band} weighted-target HTML: {html_path}")
        print(f"  top target: {display_names[top]} (weighted score={scores[top]:.4f})")


if __name__ == "__main__":
    main()
