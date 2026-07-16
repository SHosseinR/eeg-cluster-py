"""Generate structured figures from saved pipeline/optimization artifacts only."""

from __future__ import annotations

import argparse

from figure_paths import (
    ensure_figure_tree, organize_profile_figures, write_figure_manifest,
)
from plot_group_metric_space_3d import generate_group_metric_figures
from plot_modularity_reordered_connectivity import generate_modularity_figures
from plot_top_selected_nodes import generate_top_selected_nodes
from plot_weighted_selection_target_3d import generate_weighted_target_figures
from saved_results_utils import (
    ensure_band_stability_summary, load_dataset_profile, load_npy_dict,
    ordered_bands, validate_saved_inputs,
)


def generate_profile(profile, organize_existing: bool = False) -> dict:
    validate_saved_inputs(profile)
    results = load_npy_dict(profile.optimization_results_path, "optimization results")
    bands = ordered_bands(results)
    if len(bands) < 2:
        raise ValueError(f"At least two stored bands are required for {profile.label}")
    ensure_figure_tree(profile)
    stability_path = ensure_band_stability_summary(profile, bands)
    organization = None
    if organize_existing:
        organization = organize_profile_figures(profile, bands)
    modularity = generate_modularity_figures(profile)
    metric_space = generate_group_metric_figures(profile)
    targets = generate_weighted_target_figures(
        str(profile.optimization_results_path), str(stability_path),
        str(profile.optimization_figures_dir / "targets"),
        str(profile.optimization_figures_dir / "targets"),
    )
    write_figure_manifest(profile.main_figures_dir)
    write_figure_manifest(profile.optimization_figures_dir)
    return {
        "profile": profile,
        "organization": organization,
        "modularity": modularity,
        "metric_space": metric_space,
        "targets": targets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", action="append", required=True)
    parser.add_argument("--organize-existing", action="store_true")
    parser.add_argument("--comparison-output-dir", default="results-comparison")
    args = parser.parse_args()
    profiles = [load_dataset_profile(config) for config in args.dataset_config]
    outputs = []
    for profile in profiles:
        print(f"\nGenerating saved-result figures for {profile.label}")
        output = generate_profile(profile, organize_existing=args.organize_existing)
        outputs.append(output)
        print(f"  modularity figures: {len(output['modularity'])}")
        print(f"  metric-space figure sets: {len(output['metric_space'])}")
        print(f"  weighted-target figure sets: {len(output['targets'])}")
        if output["organization"] is not None:
            print(f"  organized existing figures: {output['organization']}")
    if len(profiles) > 1:
        comparison = generate_top_selected_nodes(profiles, args.comparison_output_dir)
        print(f"\nSaved cross-dataset comparison: {comparison['png']}")
        print(f"Saved cross-dataset table: {comparison['csv']}")


if __name__ == "__main__":
    main()
