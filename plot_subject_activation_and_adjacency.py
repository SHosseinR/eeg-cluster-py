"""Plot one subject's configured stimulation effect and adjacency change.

State-space results produce activation plots. Dynamics-free static results
produce selected-edge change plots instead. Optimization is never rerun.
"""
import argparse
import os
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SELECTED_METHOD
from optimization_config import (
    OPTIMIZATION_OUTPUT_DIR,
    OPTIMIZATION_RESULTS_FILE,
    OPTIMIZATION_FIGURES_DIR,
    OPTIMIZATION_ANALYSIS_INPUT_DIR,
    SIMULATION_CONFIG,
    PLASTICITY_CONFIG,
    OPTIMIZATION_DEBUG_SUBJECT,
    STATIC_STIMULATION_EDGE_SCOPE,
)
from state_space_simulation import run_full_simulation
from plasticity import compute_plasticity_effect
from channel_metadata import get_display_channel_names
from stimulation_models import apply_static_adjacency_stimulation


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
        "--connectivity",
        default=None,
        help="Optional override path to connectivity_matrices.npy",
    )
    parser.add_argument(
        "--band",
        default=None,
        help="Optional band name or index when results contain multiple bands.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
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
    if subject_id == "__first__":
        first_item = next(iter(optimization_results.items()), None)
        if first_item is None:
            raise RuntimeError("No subject result is available for automatic selection")
        result_key, first_result = first_item
        if not isinstance(first_result, dict) or not first_result.get("subject_id"):
            raise RuntimeError("No subject result is available for automatic selection")
        subject_id = str(first_result["subject_id"])
    else:
        result_key = _select_result_key(optimization_results, subject_id, args.band)
    results = optimization_results[result_key]
    subject_id = results.get("subject_id", subject_id)
    best_solution = results.get("best_solution")
    if best_solution is None:
        raise RuntimeError(f"No best_solution for subject: {subject_id}")

    band_names = results.get("band_names")
    if band_names is None:
        raise RuntimeError("Missing band_names in results.")

    band_idx = int(best_solution["band"])
    band_name = best_solution.get("band_name") or band_names[band_idx]
    connectivity_path = args.connectivity or os.path.join(
        OPTIMIZATION_ANALYSIS_INPUT_DIR, "data", "connectivity_matrices.npy"
    )
    if not os.path.exists(connectivity_path):
        raise FileNotFoundError(f"Connectivity matrices not found: {connectivity_path}")

    connectivity_matrices = _load_pickle_dict(connectivity_path)
    try:
        original_matrix = connectivity_matrices["Patient"][subject_id][SELECTED_METHOD][band_name]
    except KeyError as exc:
        raise KeyError(
            f"Connectivity not found for {subject_id}, {SELECTED_METHOD}, {band_name}"
        ) from exc

    n_nodes = int(original_matrix.shape[0])
    channel_names = get_display_channel_names(results, n_nodes=n_nodes)
    node_idx = int(best_solution["node"])
    node_label = channel_names[node_idx] if node_idx < len(channel_names) else f"Node {node_idx}"
    print(f'{best_solution["node"]=} ({node_label})')
    stimulation_model = str(
        best_solution.get(
            "stimulation_model",
            results.get("stimulation_model", "state_space"),
        )
    ).strip().lower()
    is_static = stimulation_model == "static_adjacency"

    if is_static:
        total_change = best_solution.get(
            "stimulation_total_change",
            best_solution.get("stimulation_amplitude"),
        )
        if total_change is None:
            raise RuntimeError("Static solution is missing stimulation_total_change")
        edge_scope = str(
            results.get("static_edge_scope") or STATIC_STIMULATION_EDGE_SCOPE
        )
        updated_matrix, static_details = apply_static_adjacency_stimulation(
            original_matrix,
            node_idx,
            float(total_change),
            edge_scope=edge_scope,
        )
        print(
            "Static adjacency change: "
            f"requested={float(total_change):.6g}, "
            f"realized_l1={static_details['realized_total_change_l1']:.6g}, "
            f"scope={edge_scope}"
        )
        sim_results = None
    else:
        baseline_activation = results.get("baseline_activation")
        if baseline_activation is None:
            raise RuntimeError(f"Missing baseline_activation for subject: {subject_id}")
        duration = best_solution.get("stimulation_duration")
        amplitude = best_solution.get("stimulation_amplitude")
        leak = best_solution.get("leak")
        if duration is None:
            duration = float(SIMULATION_CONFIG["stimulation_duration"])
        if amplitude is None:
            amplitude = float(SIMULATION_CONFIG["stimulation_amplitude"])
        if leak is None:
            leak = float(SIMULATION_CONFIG.get("leak", 0.0))
        sim_results = run_full_simulation(
            adjacency_matrix=original_matrix,
            baseline_activation=np.array(baseline_activation, dtype=float),
            stimulation_node=node_idx,
            stimulation_duration=float(duration),
            stimulation_amplitude=float(amplitude),
            dt=float(SIMULATION_CONFIG["dt"]),
            stability_constant=float(SIMULATION_CONFIG["stability_constant"]),
            leak=float(leak),
        )
        if PLASTICITY_CONFIG.get("plasticity_enabled", True):
            updated_matrix = compute_plasticity_effect(
                adjacency_matrix=original_matrix,
                activation_ratios=sim_results["activation_ratios"],
                normalize=False,
                scaling=float(PLASTICITY_CONFIG.get("plasticity_scaling", 1.0)),
            )
        else:
            updated_matrix = original_matrix

    safe_subject = "".join(
        character if character.isalnum() or character in ".-_" else "_"
        for character in str(subject_id)
    ).strip("_") or "unknown-subject"
    output_dir = args.output_dir or os.path.join(
        OPTIMIZATION_FIGURES_DIR, "subjects", safe_subject, str(band_name)
    )
    os.makedirs(output_dir, exist_ok=True)

    file_tag = f"{subject_id}_{band_name}" if band_name else subject_id
    if is_static:
        matrix_delta = updated_matrix - original_matrix
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(n_nodes)
        outgoing = matrix_delta[node_idx, :]
        incoming = matrix_delta[:, node_idx]
        if edge_scope == "outgoing":
            ax.bar(x, outgoing, width=0.7, label="Outgoing")
        elif edge_scope == "incoming":
            ax.bar(x, incoming, width=0.7, label="Incoming")
        elif np.allclose(incoming, outgoing):
            ax.bar(x, outgoing, width=0.7, label="Incident (symmetric)")
        else:
            ax.bar(x - 0.18, outgoing, width=0.36, label="Outgoing")
            ax.bar(x + 0.18, incoming, width=0.36, label="Incoming")
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(
            f"Static Edge Changes - {subject_id} ({band_name}, node {node_label})"
        )
        ax.set_xlabel("Connected electrode")
        ax.set_ylabel("Adjacency-weight change")
        ax.set_xticks(x)
        ax.set_xticklabels(channel_names, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        edge_change_path = os.path.join(output_dir, f"{file_tag}_edge_change.png")
        fig.tight_layout()
        fig.savefig(edge_change_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved static edge-change plot to: {edge_change_path}")

        if edge_scope == "incoming":
            before_values = original_matrix[:, node_idx]
            after_values = updated_matrix[:, node_idx]
            change_values = incoming
            profile_labels = channel_names
        elif edge_scope == "outgoing" or np.allclose(incoming, outgoing):
            before_values = original_matrix[node_idx, :]
            after_values = updated_matrix[node_idx, :]
            change_values = outgoing
            profile_labels = channel_names
        else:
            before_values = np.concatenate(
                [original_matrix[node_idx, :], original_matrix[:, node_idx]]
            )
            after_values = np.concatenate(
                [updated_matrix[node_idx, :], updated_matrix[:, node_idx]]
            )
            change_values = np.concatenate([outgoing, incoming])
            profile_labels = [
                *[f"out:{label}" for label in channel_names],
                *[f"in:{label}" for label in channel_names],
            ]
        before_after = np.vstack([before_values, after_values])
        change_profile = change_values.reshape(1, -1)
        n_profile_edges = len(profile_labels)
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 4.8),
            height_ratios=(2, 1),
            constrained_layout=True,
        )
        profile_min = float(np.min(before_after))
        profile_max = float(np.max(before_after))
        profile_image = axes[0].imshow(
            before_after,
            cmap="viridis",
            vmin=profile_min,
            vmax=profile_max,
            aspect="auto",
        )
        axes[0].set_title(f"Stimulated-Node Edge Profile - {subject_id}")
        axes[0].set_yticks([0, 1])
        axes[0].set_yticklabels(["Before", "After"])
        axes[0].set_xticks([])
        fig.colorbar(profile_image, ax=axes[0], shrink=0.9, label="Adjacency weight")

        change_limit = max(
            float(np.max(np.abs(change_profile))),
            np.finfo(float).eps,
        )
        change_image = axes[1].imshow(
            change_profile,
            cmap="coolwarm",
            vmin=-change_limit,
            vmax=change_limit,
            aspect="auto",
        )
        axes[1].set_yticks([0])
        axes[1].set_yticklabels(["Change"])
        axes[1].set_xticks(np.arange(n_profile_edges))
        axes[1].set_xticklabels(
            profile_labels, rotation=45, ha="right", fontsize=8
        )
        fig.colorbar(
            change_image,
            ax=axes[1],
            shrink=0.9,
            label="Weight change",
        )
        edge_heatmap_path = os.path.join(
            output_dir, f"{file_tag}_edge_profile_heatmap.png"
        )
        fig.savefig(edge_heatmap_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved static edge-profile heatmap to: {edge_heatmap_path}")
    else:
        trajectory = sim_results["trajectory"]
        baseline = np.array(baseline_activation, dtype=float).reshape(-1, 1)
        delta = trajectory - baseline
        time = np.arange(trajectory.shape[1]) * float(SIMULATION_CONFIG["dt"])

        fig, ax = plt.subplots(figsize=(12, 6))
        for idx in range(delta.shape[0]):
            label = channel_names[idx] if idx < len(channel_names) else f"Ch{idx}"
            ax.plot(time, delta[idx, :], linewidth=1.0, alpha=0.8, label=label)
        ax.set_title(
            f"Activation Change Over Time - {subject_id} "
            f"({band_name}, node {node_label})"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Activation change (x - baseline)")
        ax.grid(alpha=0.3)
        if not args.no_legend and delta.shape[0] <= 25:
            ax.legend(loc="upper right", fontsize=8, ncol=2)
        activation_path = os.path.join(
            output_dir, f"{file_tag}_activation_change.png"
        )
        fig.tight_layout()
        fig.savefig(activation_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved activation change plot to: {activation_path}")

        final_state = np.array(sim_results["final_state"], dtype=float)
        baseline_vec = np.array(baseline_activation, dtype=float)
        activation_matrix = np.vstack([baseline_vec, final_state])
        act_vmin = float(np.min(activation_matrix))
        act_vmax = float(np.max(activation_matrix))
        fig, ax = plt.subplots(figsize=(12, 3))
        im = ax.imshow(
            activation_matrix,
            cmap="viridis",
            vmin=act_vmin,
            vmax=act_vmax,
            aspect="auto",
        )
        ax.set_title(f"Electrode Activations (Before/After) - {subject_id}")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Before", "After"])
        ax.set_xticks(np.arange(len(channel_names)))
        ax.set_xticklabels(channel_names, rotation=45, ha="right", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.9)
        fig.tight_layout()
        activation_heatmap_path = os.path.join(
            output_dir, f"{file_tag}_act_heatmap.png"
        )
        fig.savefig(activation_heatmap_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved activation heatmap to: {activation_heatmap_path}")

    # Adjacency before/after plot
    vmin = float(np.min([original_matrix.min(), updated_matrix.min()]))
    vmax = float(np.max([original_matrix.max(), updated_matrix.max()]))

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True
    )
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

    fig.colorbar(
        im1,
        ax=axes,
        shrink=0.8,
        pad=0.02,
        fraction=0.035,
        label="Adjacency weight",
    )
    adjacency_path = os.path.join(output_dir, f"{file_tag}_adj_compare.png")
    fig.savefig(adjacency_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved adjacency comparison to: {adjacency_path}")


if __name__ == "__main__":
    main()
