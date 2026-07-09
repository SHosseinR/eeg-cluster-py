"""
Plot activation change (time series) and adjacency before/after for one subject.

Uses optimization results and stored connectivity matrices. Does not re-run
optimization.
"""
import argparse
import os
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, SELECTED_METHOD
from optimization_config import (
    OPTIMIZATION_OUTPUT_DIR,
    OPTIMIZATION_RESULTS_FILE,
    OPTIMIZATION_FIGURES_DIR,
    SIMULATION_CONFIG,
    PLASTICITY_CONFIG,
    OPTIMIZATION_DEBUG_SUBJECT,
)
from state_space_simulation import run_full_simulation
from plasticity import compute_plasticity_effect
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


def _select_result_key(optimization_results: Dict, subject_id: str, band_selector: Optional[str]) -> str:
    if subject_id in optimization_results:
        return subject_id

    band_idx_filter, band_name_filter = _parse_band_selector(band_selector)

    candidates = []
    for key, results in optimization_results.items():
        if not isinstance(results, dict):
            continue

        stored_subject = results.get("subject_id")
        if stored_subject != subject_id and not str(key).startswith(f"{subject_id}::"):
            continue

        band_idx, band_name = _get_result_band_info(results)
        if band_idx_filter is not None and band_idx != band_idx_filter:
            continue
        if band_name_filter is not None and band_name != band_name_filter:
            continue

        candidates.append((key, band_name))

    if not candidates:
        raise KeyError(f"Subject not found in results: {subject_id}")

    if len(candidates) > 1:
        available_bands = sorted({name for _, name in candidates if name})
        hint = " Use --band to select a specific band."
        if available_bands:
            hint = f" Use --band to select a specific band. Available: {', '.join(available_bands)}"
        raise ValueError(f"Multiple results found for subject '{subject_id}'.{hint}")

    return candidates[0][0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot activation change and adjacency before/after for one subject."
    )
    parser.add_argument(
        "--subject",
        default=OPTIMIZATION_DEBUG_SUBJECT,
        help="Subject ID to plot (default from config).",
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Optional override path to optimization_results.npy",
    )
    parser.add_argument(
        "--band",
        default=None,
        help="Optional band name or index when results contain multiple bands.",
    )
    parser.add_argument(
        "--output-dir",
        default=OPTIMIZATION_FIGURES_DIR,
        help="Optional output directory for figures.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Disable legend for activation plot.",
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
    subject_id = args.subject
    result_key = _select_result_key(optimization_results, subject_id, args.band)
    results = optimization_results[result_key]
    subject_id = results.get("subject_id", subject_id)
    best_solution = results.get("best_solution")
    if best_solution is None:
        raise RuntimeError(f"No best_solution for subject: {subject_id}")

    baseline_activation = results.get("baseline_activation")
    print(f'{baseline_activation=}')
    # min_nonzero = baseline_activation[baseline_activation > 0].min()
    # baseline_activation[baseline_activation == 0] = min_nonzero
    # print(f'{baseline_activation=}')

    if baseline_activation is None:
        raise RuntimeError(f"Missing baseline_activation for subject: {subject_id}")

    band_names = results.get("band_names")
    exact_channel_names = results.get("channel_names")
    n_nodes = len(exact_channel_names) if exact_channel_names is not None else len(baseline_activation)
    channel_names = get_display_channel_names(results, n_nodes=n_nodes)
    if band_names is None:
        raise RuntimeError("Missing band_names in results.")

    band_idx = int(best_solution["band"])
    band_name = best_solution.get("band_name") or band_names[band_idx]

    duration = best_solution.get("stimulation_duration")
    amplitude = best_solution.get("stimulation_amplitude")
    if duration is None:
        duration = float(SIMULATION_CONFIG["stimulation_duration"])
    if amplitude is None:
        amplitude = float(SIMULATION_CONFIG["stimulation_amplitude"])

    connectivity_path = os.path.join(OUTPUT_DIR, "data", "connectivity_matrices.npy")
    if not os.path.exists(connectivity_path):
        raise FileNotFoundError(f"Connectivity matrices not found: {connectivity_path}")

    connectivity_matrices = _load_pickle_dict(connectivity_path)
    try:
        original_matrix = connectivity_matrices["Patient"][subject_id][SELECTED_METHOD][band_name]
    except KeyError as exc:
        raise KeyError(
            f"Connectivity not found for {subject_id}, {SELECTED_METHOD}, {band_name}"
        ) from exc

    sim_results = run_full_simulation(
        adjacency_matrix=original_matrix,
        baseline_activation=np.array(baseline_activation, dtype=float),
        stimulation_node=int(best_solution["node"]),
        stimulation_duration=float(duration),
        stimulation_amplitude=float(amplitude),
        dt=float(SIMULATION_CONFIG["dt"]),
        stability_constant=float(SIMULATION_CONFIG["stability_constant"]),
        leak=float(SIMULATION_CONFIG.get("leak", 0.0)),
    )
    node_idx = int(best_solution["node"])
    node_label = channel_names[node_idx] if node_idx < len(channel_names) else f"Node {node_idx}"
    print(f'{best_solution["node"]=} ({node_label})')

    if PLASTICITY_CONFIG.get("plasticity_enabled", True):
        updated_matrix = compute_plasticity_effect(
            adjacency_matrix=original_matrix,
            activation_ratios=sim_results["activation_ratios"],
            normalize=False,
            scaling=float(PLASTICITY_CONFIG.get("plasticity_scaling", 1.0)),
        )
        # print(f'{original_matrix=}')
        # print(f'{updated_matrix=}')
    else:
        updated_matrix = original_matrix

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Activation change plot (time series)
    trajectory = sim_results["trajectory"]
    baseline = np.array(baseline_activation, dtype=float).reshape(-1, 1)
    delta = trajectory - baseline
    time = np.arange(trajectory.shape[1]) * float(SIMULATION_CONFIG["dt"])

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx in range(delta.shape[0]):
        label = channel_names[idx] if idx < len(channel_names) else f"Ch{idx}"
        ax.plot(time, delta[idx, :], linewidth=1.0, alpha=0.8, label=label)

    ax.set_title(
        f"Activation Change Over Time - {subject_id} ({band_name}, node {node_label})"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Activation change (x - baseline)")
    ax.grid(alpha=0.3)

    if not args.no_legend and delta.shape[0] <= 25:
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    file_tag = f"{subject_id}_{band_name}" if band_name else subject_id
    activation_path = os.path.join(output_dir, f"{file_tag}_activation_change.png")
    fig.tight_layout()
    fig.savefig(activation_path, dpi=300, bbox_inches="tight")
    print(f"Saved activation change plot to: {activation_path}")

    # Activation before/after heatmap
    final_state = np.array(sim_results["final_state"], dtype=float)
    baseline_vec = np.array(baseline_activation, dtype=float)
    activation_matrix = np.vstack([baseline_vec, final_state])
    act_vmin = float(np.min(activation_matrix))
    act_vmax = float(np.max(activation_matrix))

    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(activation_matrix, cmap="viridis", vmin=act_vmin, vmax=act_vmax, aspect="auto")
    ax.set_title(f"Electrode Activations (Before/After) - {subject_id}")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Before", "After"])
    ax.set_xticks(np.arange(len(channel_names)))
    ax.set_xticklabels(channel_names, rotation=45, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()

    activation_heatmap_path = os.path.join(
        output_dir, f"{file_tag}_activation_before_after_heatmap.png"
    )
    fig.savefig(activation_heatmap_path, dpi=300, bbox_inches="tight")
    print(f"Saved activation heatmap to: {activation_heatmap_path}")

    # Adjacency before/after plot
    vmin = float(np.min([original_matrix.min(), updated_matrix.min()]))
    vmax = float(np.max([original_matrix.max(), updated_matrix.max()]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(original_matrix, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Adjacency (Before)")
    axes[0].set_xlabel("Node")
    axes[0].set_ylabel("Node")
    axes[0].set_xticks(np.arange(len(channel_names)))
    axes[0].set_yticks(np.arange(len(channel_names)))
    axes[0].set_xticklabels(channel_names, rotation=45, ha="right", fontsize=7)
    axes[0].set_yticklabels(channel_names, fontsize=7)

    im1 = axes[1].imshow(updated_matrix, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Adjacency (After)")
    axes[1].set_xlabel("Node")
    axes[1].set_ylabel("Node")
    axes[1].set_xticks(np.arange(len(channel_names)))
    axes[1].set_yticks(np.arange(len(channel_names)))
    axes[1].set_xticklabels(channel_names, rotation=45, ha="right", fontsize=7)
    axes[1].set_yticklabels(channel_names, fontsize=7)

    fig.colorbar(im1, ax=axes, shrink=0.8, pad=-0.3, fraction=0.07)
    fig.tight_layout()

    adjacency_path = os.path.join(output_dir, f"{file_tag}_adjacency_before_after.png")
    fig.savefig(adjacency_path, dpi=300, bbox_inches="tight")
    print(f"Saved adjacency comparison to: {adjacency_path}")


if __name__ == "__main__":
    main()
