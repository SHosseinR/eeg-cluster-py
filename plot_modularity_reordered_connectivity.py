"""Plot Healthy and Patient mean connectivity using a shared modularity order."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import bct
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_paths import ensure_figure_tree, main_figure_dir
from saved_results_utils import (
    SavedDatasetProfile, load_analysis_metadata, load_channel_metadata,
    load_dataset_profile, load_npy_dict,
)


def _prepare_detection_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    matrix[matrix < 0] = 0.0
    np.fill_diagonal(matrix, 0.0)
    return (matrix + matrix.T) / 2.0


def best_louvain_partition(matrix: np.ndarray, n_restarts: int = 100) -> tuple[np.ndarray, float]:
    prepared = _prepare_detection_matrix(matrix)
    if prepared.shape[0] == 0:
        raise ValueError("Cannot partition an empty matrix")
    if float(prepared.sum()) <= np.finfo(float).eps:
        return np.ones(prepared.shape[0], dtype=int), 0.0
    best_ci, best_q = None, -np.inf
    for seed in range(n_restarts):
        ci, q = bct.community_louvain(prepared, seed=seed)
        ci = np.asarray(ci, dtype=int).reshape(-1)
        q = float(q)
        signature = tuple(ci.tolist())
        best_signature = tuple(best_ci.tolist()) if best_ci is not None else None
        if q > best_q + 1e-12 or (abs(q - best_q) <= 1e-12 and signature < best_signature):
            best_ci, best_q = ci, q
    if best_q <= 0:
        return np.ones(prepared.shape[0], dtype=int), 0.0
    return best_ci, best_q


def modularity_node_order(matrix: np.ndarray, communities: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    communities = np.asarray(communities, dtype=int)
    if matrix.shape[0] != communities.size:
        raise ValueError("Community vector length does not match matrix")
    strengths = np.nansum(matrix, axis=0) + np.nansum(matrix, axis=1)
    module_members = []
    for module in np.unique(communities):
        members = np.flatnonzero(communities == module)
        module_members.append((int(np.min(members)), members))
    module_members.sort(key=lambda item: item[0])
    order, normalized = [], np.zeros_like(communities)
    for new_module, (_, members) in enumerate(module_members, start=1):
        ranked = sorted(members.tolist(), key=lambda idx: (-float(strengths[idx]), idx))
        order.extend(ranked)
        normalized[members] = new_module
    return np.asarray(order, dtype=int), normalized


def _group_mean(connectivity: Mapping, group: str, method: str, band: str) -> np.ndarray:
    matrices = []
    for subject in connectivity.get(group, {}).values():
        try:
            matrix = np.asarray(subject[method][band], dtype=float)
        except KeyError:
            continue
        if matrix.ndim == 2:
            matrices.append(matrix)
    if not matrices:
        raise ValueError(f"No {group} matrices found for method={method}, band={band}")
    return np.nanmean(np.stack(matrices), axis=0)


def _module_boundaries(ordered_communities: np.ndarray) -> list[int]:
    return (np.flatnonzero(np.diff(ordered_communities)) + 1).tolist()


def plot_modularity_band(
    connectivity: Mapping,
    method: str,
    band: str,
    channel_labels: Sequence[str],
    output_png: Path,
    output_csv: Path,
    n_restarts: int = 100,
) -> dict:
    healthy = _group_mean(connectivity, "Healthy", method, band)
    patient = _group_mean(connectivity, "Patient", method, band)
    if healthy.shape != patient.shape or healthy.shape[0] != len(channel_labels):
        raise ValueError("Connectivity dimensions do not match channel labels")
    pooled = (healthy + patient) / 2.0
    communities, q = best_louvain_partition(pooled, n_restarts=n_restarts)
    order, normalized = modularity_node_order(pooled, communities)
    ordered_communities = normalized[order]
    ordered_labels = [str(channel_labels[index]) for index in order]

    table = pd.DataFrame({
        "electrode": list(channel_labels),
        "original_index": np.arange(len(channel_labels)),
        "reordered_index": np.argsort(order),
        "module": normalized,
    }).sort_values("reordered_index")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)

    reordered = [healthy[np.ix_(order, order)], patient[np.ix_(order, order)]]
    vmin = float(min(np.nanmin(matrix) for matrix in reordered))
    vmax = float(max(np.nanmax(matrix) for matrix in reordered))
    if vmax <= vmin:
        vmax = vmin + 1.0
    size = max(10.0, len(channel_labels) * 0.48)
    fig, axes = plt.subplots(1, 2, figsize=(size * 1.65, size * 0.82), constrained_layout=True)
    images = []
    boundaries = _module_boundaries(ordered_communities)
    for ax, matrix, group in zip(axes, reordered, ("Healthy", "Patient")):
        image = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        images.append(image)
        ax.set_title(f"{group} mean (n={len(connectivity.get(group, {}))})")
        ticks = np.arange(len(ordered_labels))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ordered_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(ordered_labels, fontsize=7)
        ax.set_xlabel("Target electrode")
        ax.set_ylabel("Source electrode")
        for boundary in boundaries:
            ax.axhline(boundary - 0.5, color="white", linewidth=1.4)
            ax.axvline(boundary - 0.5, color="white", linewidth=1.4)
        for index, module in enumerate(ordered_communities):
            ax.add_patch(plt.Rectangle((-1.35, index - 0.5), 0.55, 1.0,
                                       color=plt.cm.tab20((module - 1) % 20), clip_on=False))
    fig.colorbar(images[0], ax=axes, shrink=0.78, label="Connectivity strength")
    fig.suptitle(
        f"{band.capitalize()} {method.upper()} connectivity, modularity-reordered "
        f"(Q={q:.3f}, modules={len(np.unique(normalized))})",
        fontsize=14,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"png": output_png, "csv": output_csv, "q": q, "order": order, "modules": normalized}


def generate_modularity_figures(profile: SavedDatasetProfile, n_restarts: int = 100) -> list[dict]:
    ensure_figure_tree(profile)
    connectivity = load_npy_dict(profile.data_dir / "connectivity_matrices.npy", "connectivity matrices")
    analysis = load_analysis_metadata(profile)
    channels = load_channel_metadata(profile)
    method = str(analysis.get("selected_method") or analysis.get("connectivity_methods", [None])[0])
    bands = list((analysis.get("frequency_bands") or {}).keys())
    labels = channels.get("channel_display_names") or channels.get("channel_names")
    if not method or not bands or not labels:
        raise ValueError(f"Incomplete saved metadata for {profile.label}")
    output_dir = main_figure_dir(profile, "connectivity")
    data_dir = profile.data_dir / "modularity_orders"
    outputs = []
    for band in bands:
        outputs.append(plot_modularity_band(
            connectivity, method, band, labels,
            output_dir / f"modularity_reordered_connectivity_{band}.png",
            data_dir / f"modularity_order_{band}.csv",
            n_restarts=n_restarts,
        ))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--restarts", type=int, default=100)
    args = parser.parse_args()
    profile = load_dataset_profile(args.dataset_config)
    for output in generate_modularity_figures(profile, args.restarts):
        print(f"Saved: {output['png']} (Q={output['q']:.4f})")


if __name__ == "__main__":
    main()
