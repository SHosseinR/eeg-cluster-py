"""Categorized figure paths and migration helpers."""

from __future__ import annotations

import csv
import filecmp
from pathlib import Path
import re
from typing import Iterable

from saved_results_utils import SavedDatasetProfile


FIGURE_EXTENSIONS = {".png", ".html", ".svg", ".pdf"}
MAIN_CATEGORIES = ("connectivity", "network_statistics", "classification", "misc")
OPTIMIZATION_CATEGORIES = (
    "overview", "metric_space", "targets", "target_statistics", "subjects", "misc"
)


def ensure_figure_tree(profile: SavedDatasetProfile) -> None:
    for category in MAIN_CATEGORIES:
        (profile.main_figures_dir / category).mkdir(parents=True, exist_ok=True)
    opt = profile.optimization_figures_dir
    for path in (
        opt / "overview",
        opt / "metric_space" / "shifts",
        opt / "metric_space" / "group_all",
        opt / "metric_space" / "group_optimized",
        opt / "targets",
        opt / "target_statistics" / "overall",
        opt / "subjects",
        opt / "misc",
    ):
        path.mkdir(parents=True, exist_ok=True)


def main_figure_dir(profile: SavedDatasetProfile, category: str) -> Path:
    if category not in MAIN_CATEGORIES:
        raise ValueError(f"Unknown main figure category: {category}")
    return profile.main_figures_dir / category


def optimization_figure_dir(profile: SavedDatasetProfile, category: str, *parts: str) -> Path:
    if category not in OPTIMIZATION_CATEGORIES:
        raise ValueError(f"Unknown optimization figure category: {category}")
    return profile.optimization_figures_dir / category / Path(*parts)


def _main_destination(root: Path, filename: str) -> Path:
    lower = filename.casefold()
    if lower.startswith(("viz1_", "viz2_", "viz3_")) or "connectivity" in lower:
        category = "connectivity"
    elif lower.startswith(("viz4_", "viz7_", "viz8_")):
        category = "network_statistics"
    elif lower.startswith(("viz5_", "viz6_")) or "classification" in lower:
        category = "classification"
    else:
        category = "misc"
    return root / category / filename


def _optimization_destination(root: Path, filename: str, bands: Iterable[str]) -> Path:
    lower = filename.casefold()
    if lower.startswith("metric_shift_3d_"):
        return root / "metric_space" / "shifts" / filename
    if lower.startswith(("weighted_selection_", "weighted_node_band_interactive")):
        return root / "targets" / filename
    if "target_statistics" in lower:
        band = next((b for b in bands if lower.startswith(f"{b.casefold()}_")), "overall")
        return root / "target_statistics" / band / filename
    if lower.endswith(("_activation_change.png", "_activation_before_after_heatmap.png", "_adjacency_before_after.png")):
        band = next((b for b in bands if f"_{b.casefold()}_" in lower), "unknown-band")
        subject = filename.rsplit(f"_{band}_", 1)[0] if band != "unknown-band" else "unknown-subject"
        subject = re.sub(r"[^A-Za-z0-9._-]+", "_", subject).strip("_") or "unknown-subject"
        return root / "subjects" / subject / band / filename
    overview_prefixes = (
        "optimal_", "weighted_nodes_", "weighted_bands_", "node_band_",
        "weighted_node_band_", "pareto_", "best_closeness_",
    )
    if lower.startswith(overview_prefixes):
        return root / "overview" / filename
    return root / "misc" / filename


def _move_collision_safe(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return "unchanged"
    if destination.exists():
        if filecmp.cmp(source, destination, shallow=False):
            source.unlink()
            return "deduplicated"
        raise FileExistsError(f"Figure migration collision: {source} -> {destination}")
    source.replace(destination)
    return "moved"


def write_figure_manifest(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "figure_manifest.csv"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.casefold() in FIGURE_EXTENSIONS:
            relative = path.relative_to(root)
            rows.append({
                "category": relative.parts[0] if len(relative.parts) > 1 else "root",
                "relative_path": relative.as_posix(),
                "format": path.suffix.lstrip(".").lower(),
                "size_bytes": path.stat().st_size,
            })
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("category", "relative_path", "format", "size_bytes"))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def organize_profile_figures(profile: SavedDatasetProfile, bands: Iterable[str]) -> dict[str, int]:
    ensure_figure_tree(profile)
    counts = {"moved": 0, "deduplicated": 0, "unchanged": 0}
    if profile.main_figures_dir.exists():
        for source in list(profile.main_figures_dir.iterdir()):
            if source.is_file() and source.suffix.casefold() in FIGURE_EXTENSIONS:
                action = _move_collision_safe(source, _main_destination(profile.main_figures_dir, source.name))
                counts[action] += 1
    if profile.optimization_figures_dir.exists():
        for source in list(profile.optimization_figures_dir.iterdir()):
            if source.is_file() and source.suffix.casefold() in FIGURE_EXTENSIONS:
                destination = _optimization_destination(profile.optimization_figures_dir, source.name, bands)
                action = _move_collision_safe(source, destination)
                counts[action] += 1
    write_figure_manifest(profile.main_figures_dir)
    write_figure_manifest(profile.optimization_figures_dir)
    return counts
