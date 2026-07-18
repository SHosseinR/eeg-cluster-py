"""Visualize baseline group separation and optimized shifts in one fixed PCA space.

For each band, scaling and PCA are fitted once to the original Healthy and
Patient natural coherence edges. Optimized matrices are transformed with that
same fitted transform; the projection is never refitted after optimization.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from classification_score.band_connectivity_classifier import vectorize_band_matrix
from saved_results_utils import load_dataset_profile, load_npy_dict, ordered_bands


GROUP_STYLE = {
    "Healthy": {"color": "#2a9d8f", "marker": "o"},
    "Patient": {"color": "#e76f51", "marker": "o"},
}


def _baseline_rows(
    connectivity: Mapping,
    band: str,
    method: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    features, groups, subjects = [], [], []
    for group in ("Healthy", "Patient"):
        for subject_id, methods in connectivity[group].items():
            matrix = np.asarray(methods[method][band], dtype=float)
            features.append(vectorize_band_matrix(matrix)[0])
            groups.append(group)
            subjects.append(str(subject_id))
    return np.asarray(features), groups, subjects


def _band_results(results: Mapping, band: str) -> dict[str, Mapping]:
    selected = {}
    for result in results.values():
        if not isinstance(result, Mapping) or str(result.get("fixed_band_name")) != band:
            continue
        subject_id = str(result.get("subject_id"))
        if not subject_id or subject_id == "None":
            raise ValueError(f"An optimization result for {band} has no subject_id")
        selected[subject_id] = result
    return selected


def _projection_for_band(
    connectivity: Mapping,
    results: Mapping,
    band: str,
    method: str,
) -> tuple[pd.DataFrame, float, float]:
    X, groups, subjects = _baseline_rows(connectivity, band, method)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    baseline_xy = pca.fit_transform(standardized)

    rows = [
        {
            "band": band,
            "subject_id": subject,
            "group": group,
            "state": "baseline",
            "pc1": float(point[0]),
            "pc2": float(point[1]),
            "initial_patient_probability": np.nan,
            "optimized_patient_probability": np.nan,
        }
        for subject, group, point in zip(subjects, groups, baseline_xy)
    ]
    baseline_lookup = {
        subject: point for subject, group, point in zip(subjects, groups, baseline_xy)
        if group == "Patient"
    }

    for subject_id, result in _band_results(results, band).items():
        if subject_id not in baseline_lookup:
            raise KeyError(f"Optimized patient {subject_id!r} is absent from baseline connectivity")
        solution = result.get("best_solution") or {}
        updated = solution.get("updated_connectivity_matrix")
        if updated is None:
            raise KeyError(
                f"{subject_id}/{band} has no saved updated_connectivity_matrix. "
                "Rerun run_optimization.py with the current code before plotting."
            )
        channel_names = list(result.get("channel_names") or [])
        matrix = np.asarray(updated, dtype=float)
        if channel_names and matrix.shape != (len(channel_names), len(channel_names)):
            raise ValueError(
                f"{subject_id}/{band} optimized matrix shape {matrix.shape} does not "
                f"match {len(channel_names)} saved channels"
            )
        optimized_xy = pca.transform(
            scaler.transform(vectorize_band_matrix(matrix))
        )[0]
        initial = result.get("initial_metrics") or [np.nan]
        final = result.get("final_metrics") or [np.nan]
        rows.append({
            "band": band,
            "subject_id": subject_id,
            "group": "Patient",
            "state": "optimized",
            "pc1": float(optimized_xy[0]),
            "pc2": float(optimized_xy[1]),
            "initial_patient_probability": float(initial[0]),
            "optimized_patient_probability": float(final[0]),
        })
    explained = pca.explained_variance_ratio_ * 100.0
    return pd.DataFrame(rows), float(explained[0]), float(explained[1])


def _plot_band(
    coordinates: pd.DataFrame,
    band: str,
    pc1_variance: float,
    pc2_variance: float,
    axes=None,
):
    own_figure = axes is None
    if own_figure:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    else:
        fig = axes[0].figure
    baseline = coordinates[coordinates["state"] == "baseline"]
    optimized = coordinates[coordinates["state"] == "optimized"]

    for group in ("Healthy", "Patient"):
        part = baseline[baseline["group"] == group]
        axes[0].scatter(
            part["pc1"], part["pc2"], s=32, alpha=0.66,
            color=GROUP_STYLE[group]["color"], marker=GROUP_STYLE[group]["marker"],
            edgecolor="white", linewidth=0.35, label=group,
        )
    axes[0].set_title(f"{band}: original group separation")
    axes[0].legend(frameon=False)

    healthy = baseline[baseline["group"] == "Healthy"]
    patients = baseline[baseline["group"] == "Patient"]
    axes[1].scatter(
        healthy["pc1"], healthy["pc2"], s=26, alpha=0.24,
        color=GROUP_STYLE["Healthy"]["color"], label="Healthy baseline",
    )
    axes[1].scatter(
        patients["pc1"], patients["pc2"], s=24, alpha=0.15,
        color=GROUP_STYLE["Patient"]["color"], label="All Patient baseline",
    )
    patient_lookup = patients.set_index("subject_id")
    for _, row in optimized.iterrows():
        before = patient_lookup.loc[row["subject_id"]]
        axes[1].annotate(
            "", xy=(row["pc1"], row["pc2"]),
            xytext=(before["pc1"], before["pc2"]),
            arrowprops=dict(arrowstyle="->", color="#6c757d", alpha=0.28, lw=0.7),
        )
    selected_ids = set(optimized["subject_id"])
    selected_before = patients[patients["subject_id"].isin(selected_ids)]
    axes[1].scatter(
        selected_before["pc1"], selected_before["pc2"], s=24,
        color="#e76f51", alpha=0.60, label="Optimized patients: before",
    )
    axes[1].scatter(
        optimized["pc1"], optimized["pc2"], s=38, marker="x",
        color="#264653", alpha=0.85, label="Optimized patients: after",
    )
    axes[1].set_title(f"{band}: shifts in the same baseline-fitted projection")
    axes[1].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.set_xlabel(f"PC1 ({pc1_variance:.1f}% baseline variance)")
        ax.set_ylabel(f"PC2 ({pc2_variance:.1f}% baseline variance)")
        ax.axhline(0, color="#d1d5db", lw=0.6, zorder=0)
        ax.axvline(0, color="#d1d5db", lw=0.6, zorder=0)
        ax.grid(alpha=0.15)
    if own_figure:
        fig.tight_layout()
    return fig


def generate_projection_figures(dataset_config: str, output_dir: str | None = None) -> list[Path]:
    profile = load_dataset_profile(dataset_config)
    connectivity = load_npy_dict(
        profile.data_dir / "connectivity_matrices.npy", "connectivity matrices"
    )
    results = load_npy_dict(profile.optimization_results_path, "optimization results")
    # The connectivity dictionary is authoritative. These dedicated profiles
    # contain exactly one method, preventing a silent mismatch with a model.
    methods = set.intersection(*[
        set(subject_methods.keys())
        for group in connectivity.values()
        for subject_methods in group.values()
    ])
    if len(methods) != 1:
        raise ValueError(f"Expected one connectivity method, found {sorted(methods)}")
    method = next(iter(methods))

    bands = ordered_bands(results)
    destination = Path(output_dir) if output_dir else (
        profile.optimization_figures_dir / "classifier_projection"
    )
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    tables, projections = [], {}
    for band in bands:
        table, pc1, pc2 = _projection_for_band(connectivity, results, band, method)
        tables.append(table)
        projections[band] = (table, pc1, pc2)
        fig = _plot_band(table, band, pc1, pc2)
        path = destination / f"{band}_baseline_pca_and_optimized_shifts.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)

    fig, axes = plt.subplots(len(bands), 2, figsize=(13, 5.1 * len(bands)), squeeze=False)
    for row, band in enumerate(bands):
        table, pc1, pc2 = projections[band]
        _plot_band(table, band, pc1, pc2, axes=axes[row])
    fig.tight_layout()
    combined = destination / "all_bands_baseline_pca_and_optimized_shifts.png"
    fig.savefig(combined, dpi=300, bbox_inches="tight")
    plt.close(fig)
    outputs.append(combined)
    pd.concat(tables, ignore_index=True).to_csv(
        destination / "baseline_pca_and_optimized_coordinates.csv", index=False
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot original cohorts and optimized patients in fixed per-band PCA spaces"
    )
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    for path in generate_projection_figures(args.dataset_config, args.output_dir):
        print(f"Saved classifier projection: {path}")


if __name__ == "__main__":
    main()
