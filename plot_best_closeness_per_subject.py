"""
Plot best closeness (distance to ideal) per subject as a PNG.

Loads optimization results from disk and does not re-run optimization.
"""
import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimization_config import OPTIMIZATION_OUTPUT_DIR, OPTIMIZATION_RESULTS_FILE


def _load_pickle_dict(path: str) -> Dict:
    return np.load(path, allow_pickle=True).item()


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
) -> List[Dict]:
    collected = []
    for subject_id, results in optimization_results.items():
        if not isinstance(results, dict):
            continue

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
            collected.append({"subject_id": subject_id, **sol})

    return collected


def _best_closeness_by_subject(ranked_solutions: List[Dict]) -> Dict[str, float]:
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

    return best_by_subject


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot best closeness (distance to ideal) per subject."
    )
    parser.add_argument(
        "--output-dir",
        default=OPTIMIZATION_OUTPUT_DIR,
        help="Optimization output directory (default from config)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional override for top-k ranked solutions",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG path",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    results_path = os.path.join(output_dir, OPTIMIZATION_RESULTS_FILE)
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Optimization results not found: {results_path}")

    optimization_results = _load_pickle_dict(results_path)
    ranked_solutions = _collect_ranked_solutions(optimization_results, top_k=args.top_k)
    if not ranked_solutions:
        raise RuntimeError("No ranked solutions found in results file")

    best_by_subject = _best_closeness_by_subject(ranked_solutions)
    if not best_by_subject:
        raise RuntimeError("No closeness values found for any subject")

    subjects = sorted(best_by_subject.keys())
    values = [best_by_subject[s] for s in subjects]
    avg_value = float(np.mean(values)) if values else float("nan")

    fig, ax = plt.subplots(figsize=(max(10, len(subjects) * 0.4), 6))
    ax.bar(subjects, values, color="#1f77b4")
    ax.set_title("Best closeness per subject (lower is better)")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Closeness (distance to ideal)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    avg_text = f"Avg closeness: {avg_value:.4f}" if np.isfinite(avg_value) else "Avg closeness: n/a"
    ax.text(
        0.98,
        0.98,
        avg_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    fig.tight_layout()

    figures_dir = os.path.join(output_dir, "optimization", "figures")
    output_path = args.output if args.output is not None else os.path.join(
        figures_dir, "best_closeness_per_subject.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved closeness bar plot to: {output_path}")


if __name__ == "__main__":
    main()
