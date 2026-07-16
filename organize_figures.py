"""Move existing flat figures into the categorized saved-results layout."""

from __future__ import annotations

import argparse

from figure_paths import organize_profile_figures
from saved_results_utils import load_dataset_profile, load_npy_dict, ordered_bands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", action="append", required=True)
    args = parser.parse_args()
    for config in args.dataset_config:
        profile = load_dataset_profile(config)
        results = load_npy_dict(profile.optimization_results_path, "optimization results")
        counts = organize_profile_figures(profile, ordered_bands(results))
        print(f"{profile.label}: {counts}")


if __name__ == "__main__":
    main()
