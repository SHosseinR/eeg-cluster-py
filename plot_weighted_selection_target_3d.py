"""Plot rank-weighted stimulation-selection targets on a 3D scalp, per band.

The input is the already-computed optimization results.  Each subject's stored
top solutions contribute their stored ``strength`` (normally 1/rank) to the
selected electrode.  No optimization data is modified or recomputed.
"""
import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    """Resolve stored exact/alias labels against the configured standard montage."""
    montage = mne.channels.make_standard_montage(CHANNEL_ALIAS_MONTAGE)
    montage_positions = montage.get_positions()["ch_pos"]
    casefold_positions = {name.casefold(): value for name, value in montage_positions.items()}
    aliases = (metadata.get("channel_metadata") or {}).get("channel_aliases", {})
    positions, missing = [], []
    for name in channel_names:
        candidates = [str(name), str(aliases.get(name, ""))]
        position = None
        for candidate in candidates:
            if candidate and candidate.casefold() in casefold_positions:
                position = casefold_positions[candidate.casefold()]
                break
        if position is None:
            missing.append(name)
            positions.append([np.nan, np.nan, np.nan])
        else:
            positions.append(position)
    if missing:
        raise ValueError(
            f"Channels are absent from montage {CHANNEL_ALIAS_MONTAGE!r}: {missing}. "
            "Store standard channel aliases in channel_metadata before plotting."
        )
    return np.asarray(positions, dtype=float)


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


def _scalp_surface(positions: np.ndarray, resolution: int = 60):
    """Construct a fitted translucent scalp ellipsoid around montage coordinates."""
    center = np.mean(positions, axis=0)
    centered = positions - center
    radii = np.max(np.abs(centered), axis=0) * np.array([1.12, 1.10, 1.12])
    radii = np.maximum(radii, np.array([0.075, 0.090, 0.075]))
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution // 2)
    x = center[0] + radii[0] * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radii[1] * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radii[2] * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _marker_sizes(scores: np.ndarray) -> np.ndarray:
    if np.max(scores) <= 0:
        return np.full_like(scores, 24.0)
    return 28.0 + 260.0 * np.sqrt(scores / np.max(scores))


def _plot_static(
    positions: np.ndarray,
    labels: List[str],
    scores: np.ndarray,
    band: str,
    n_subjects: int,
    top_k: int,
    output_path: str,
) -> None:
    x, y, z = _scalp_surface(positions)
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, color="#f2c9a5", alpha=0.13, linewidth=0, shade=True)
    scatter = ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=scores, s=_marker_sizes(scores), cmap="viridis", edgecolor="black", linewidth=0.7,
        depthshade=False,
    )
    for label, point, score in zip(labels, positions, scores):
        ax.text(*point, f" {label}\n {score:.2f}", fontsize=7)
    top = int(np.argmax(scores))
    ax.scatter(*positions[top].reshape(3, 1), s=_marker_sizes(scores)[top] + 180,
               facecolors="none", edgecolors="#dc267f", linewidths=3, depthshade=False)
    ax.set_title(
        f"{band.capitalize()} rank-weighted selection targets\n"
        f"{n_subjects} subjects, top-{top_k}, weight = 1/rank; top = {labels[top]} ({scores[top]:.2f})"
    )
    ax.set_xlabel("left-right")
    ax.set_ylabel("posterior-anterior")
    ax.set_zlabel("inferior-superior")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=130)
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.68, pad=0.08)
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
    x, y, z = _scalp_surface(positions)
    top = int(np.argmax(scores))
    hover = [
        f"channel={label}<br>weighted score={score:.4f}<br>score/subject={score/max(n_subjects, 1):.4f}"
        for label, score in zip(labels, scores)
    ]
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x, y=y, z=z, surfacecolor=np.zeros_like(x), colorscale=[[0, "#f2c9a5"], [1, "#f2c9a5"]],
        opacity=0.16, showscale=False, hoverinfo="skip", name="Scalp",
    ))
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2], mode="markers+text",
        text=labels, textposition="top center", hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
        marker=dict(
            size=5 + 19 * np.sqrt(scores / np.max(scores)) if np.max(scores) > 0 else 5,
            color=scores, colorscale="Viridis", showscale=True,
            colorbar=dict(title="Weighted score"), line=dict(color="black", width=1),
        ), name="Electrodes",
    ))
    fig.add_trace(go.Scatter3d(
        x=[positions[top, 0]], y=[positions[top, 1]], z=[positions[top, 2]], mode="markers",
        marker=dict(size=28, color="rgba(0,0,0,0)", line=dict(color="#dc267f", width=8)),
        name=f"Top target: {labels[top]}", hovertemplate=f"Top target: {labels[top]}<extra></extra>",
    ))
    fig.update_layout(
        title=(f"{band.capitalize()} rank-weighted selection targets — {n_subjects} subjects, "
               f"top-{top_k}, weight = 1/rank"),
        scene=dict(
            xaxis_title="left-right", yaxis_title="posterior-anterior", zaxis_title="inferior-superior",
            aspectmode="data", camera=dict(eye=dict(x=1.35, y=-1.55, z=1.15)),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.write_html(output_path)


def _path(base: Optional[str], band: str, extension: str) -> str:
    filename = f"weighted_selection_target_3d_{band}.{extension}"
    if base is None:
        return os.path.join(OPTIMIZATION_FIGURES_DIR, filename)
    if "{band}" in base:
        return base.replace("{band}", band)
    root, ext = os.path.splitext(base)
    if ext:
        return f"{root}_{band}{ext}"
    return os.path.join(base, filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-band rank-weighted targets on a 3D scalp")
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
    display_names = list(metadata.get("channel_display_names") or channel_names)
    positions = _channel_positions(channel_names, metadata)
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
        _plot_static(positions, display_names, scores, band, n_subjects, args.top_k, png_path)
        _plot_interactive(positions, display_names, scores, band, n_subjects, args.top_k, html_path)
        top = int(np.argmax(scores))
        print(f"Saved {band} weighted-target PNG: {png_path}")
        print(f"Saved {band} weighted-target HTML: {html_path}")
        print(f"  top target: {display_names[top]} (weighted score={scores[top]:.4f})")


if __name__ == "__main__":
    main()
