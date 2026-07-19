"""Regenerate optimization overview and candidate-statistics figures from NPY results."""

from __future__ import annotations

import argparse

from optimization_visualization import plot_optimization_summary
from run_optimization import save_candidate_region_selection_stats
from saved_results_utils import (
    load_dataset_profile,
    load_npy_dict,
    ordered_bands,
    results_for_band,
)


def regenerate(dataset_config: str) -> None:
    profile = load_dataset_profile(dataset_config)
    results = load_npy_dict(profile.optimization_results_path, 'optimization results')
    metadata = next(iter(results.values()))
    channel_names = list(
        metadata.get('channel_display_names') or metadata.get('channel_names') or []
    )
    bands = ordered_bands(results)
    configured_bands = list(metadata.get('band_names') or bands)
    measures = list(metadata.get('optimization_measures') or ['patient_probability'])
    top_k = int(metadata.get('top_k') or 5)
    plot_optimization_summary(
        optimization_results=results,
        channel_names=channel_names,
        # Preserve the original configured indices. If a classifier is gated
        # out, remaining solutions may still use indices 1 and 2; passing only
        # the two present names would make histogram/heatmap shapes disagree.
        band_names=configured_bands,
        optimization_measures=measures,
        output_dir=str(profile.optimization_figures_dir),
        top_k=top_k,
    )
    for band in bands:
        save_candidate_region_selection_stats(
            results_for_band(results, band),
            channel_names,
            str(profile.optimization_dir),
            figures_dir=str(profile.optimization_figures_dir),
            file_prefix=band,
            figure_prefix=band,
            label=f'{band} band',
        )
    save_candidate_region_selection_stats(
        results,
        channel_names,
        str(profile.optimization_dir),
        figures_dir=str(profile.optimization_figures_dir),
        label='combined across bands',
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-config', required=True)
    args = parser.parse_args()
    regenerate(args.dataset_config)


if __name__ == '__main__':
    main()
