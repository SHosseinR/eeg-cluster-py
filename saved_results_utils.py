"""Shared loaders for figures generated from already-saved analysis results."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tomllib
from typing import Dict, Iterable, Mapping, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SavedDatasetProfile:
    config_path: Path
    config_name: str
    label: str
    output_dir: Path
    optimization_dir: Path

    @property
    def data_dir(self) -> Path:
        return self.output_dir / "data"

    @property
    def main_figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def optimization_results_path(self) -> Path:
        return self.optimization_dir / "optimization_results.npy"

    @property
    def optimization_figures_dir(self) -> Path:
        return self.optimization_dir / "optimization" / "figures"

    @property
    def band_stability_dir(self) -> Path:
        return self.optimization_dir / "band_stability_analysis"

    @property
    def band_summary_path(self) -> Path:
        return self.band_stability_dir / "band_comparison_summary.csv"


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_dataset_profile(config_name_or_path: str | Path) -> SavedDatasetProfile:
    path = Path(config_name_or_path)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / "dataset_configs" / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    for key in ("output_directory", "optimization_output_subdirectory"):
        if key not in config:
            raise KeyError(f"Dataset config {path} is missing {key!r}")
    labels = {"tdbrain": "TD-BRAIN", "first_paper": "First-paper"}
    output_dir = _resolve_project_path(str(config["output_directory"]))
    return SavedDatasetProfile(
        config_path=path,
        config_name=path.name,
        label=labels.get(path.stem, path.stem.replace("_", " ").title()),
        output_dir=output_dir,
        optimization_dir=output_dir / str(config["optimization_output_subdirectory"]),
    )


def load_npy_dict(path: str | Path, purpose: str = "saved result") -> Dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")
    loaded = np.load(path, allow_pickle=True).item()
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a dictionary in {path}")
    return loaded


def validate_saved_inputs(profile: SavedDatasetProfile) -> None:
    requirements = {
        profile.data_dir / "connectivity_matrices.npy": "main.py connectivity stage",
        profile.data_dir / "network_measures.npy": "main.py network-measures stage",
        profile.data_dir / "analysis_metadata.npy": "main.py metadata stage",
        profile.data_dir / "channel_metadata.json": "main.py channel-metadata stage",
        profile.optimization_results_path: "run_optimization.py",
    }
    missing = [(path, stage) for path, stage in requirements.items() if not path.exists()]
    if missing:
        detail = "\n".join(f"- {path} (generate via {stage})" for path, stage in missing)
        raise FileNotFoundError("Required saved artifacts are missing:\n" + detail)


def load_analysis_metadata(profile: SavedDatasetProfile) -> Dict:
    return load_npy_dict(profile.data_dir / "analysis_metadata.npy", "analysis metadata")


def load_channel_metadata(profile: SavedDatasetProfile) -> Dict:
    path = profile.data_dir / "channel_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing channel metadata: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def band_name(result: Mapping) -> Optional[str]:
    value = result.get("fixed_band_name")
    if value:
        return str(value)
    solution = result.get("best_solution") or {}
    idx = result.get("fixed_band_index", solution.get("band"))
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return None
    names = result.get("band_names") or []
    return str(names[idx]) if 0 <= idx < len(names) else None


def ordered_bands(results: Mapping) -> list[str]:
    metadata = next((r for r in results.values() if isinstance(r, Mapping)), {})
    present = {band_name(r) for r in results.values() if isinstance(r, Mapping)}
    present.discard(None)
    order = [str(name) for name in metadata.get("band_names", []) if str(name) in present]
    order.extend(sorted(name for name in present if name not in order))
    return order


def results_for_band(results: Mapping, band: str) -> Dict:
    return {
        str(key): value for key, value in results.items()
        if isinstance(value, Mapping) and band_name(value) == band
    }


def first_band_metadata(results: Mapping, band: str) -> Mapping:
    selected = results_for_band(results, band)
    if not selected:
        raise KeyError(f"No optimization metadata found for band {band!r}")
    return next(iter(selected.values()))


def ensure_band_stability_summary(profile: SavedDatasetProfile, bands: Iterable[str]) -> Path:
    """Generate the existing standalone stability outputs if the summary is absent."""
    if profile.band_summary_path.exists():
        return profile.band_summary_path
    from plot_band_stability_analysis import align_subject_cohort, load_per_band_results
    from band_stability_analysis import run_band_stability_analysis

    output_dir = profile.band_stability_dir
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_band = load_per_band_results(str(profile.optimization_dir), list(bands))
    aligned, cohort = align_subject_cohort(per_band, policy="intersection")
    cohort.to_csv(output_dir / "cohort_alignment_summary.csv", index=False)
    run_band_stability_analysis(
        aligned, str(output_dir), str(figures_dir),
        report_path=str(output_dir / "band_stability_report.txt"),
    )
    return profile.band_summary_path
