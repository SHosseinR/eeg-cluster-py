"""
Shared EEG channel metadata helpers.

Exact channel names are used for indexing and computation. Display names are
only for plots and reports.
"""

import json
import os
from typing import Dict, Iterable, List, Optional

import numpy as np

try:
    import mne
except Exception:  # pragma: no cover - MNE is expected in the project env.
    mne = None


def _as_list(channel_names: Iterable[str]) -> List[str]:
    return [str(name) for name in channel_names]


def _montage_positions(montage_name: str) -> Dict[str, np.ndarray]:
    if mne is None:
        return {}
    montage = mne.channels.make_standard_montage(montage_name)
    return {
        str(name): np.asarray(pos, dtype=float)
        for name, pos in montage.get_positions()["ch_pos"].items()
        if np.all(np.isfinite(pos)) and np.linalg.norm(pos) > 0
    }


def _raw_positions(raw) -> Dict[str, np.ndarray]:
    if raw is None:
        return {}

    ch_pos = {}
    montage = raw.get_montage() if hasattr(raw, "get_montage") else None
    if montage is not None:
        for name, pos in montage.get_positions()["ch_pos"].items():
            pos = np.asarray(pos, dtype=float)
            if np.all(np.isfinite(pos)) and np.linalg.norm(pos) > 0:
                ch_pos[str(name)] = pos

    if ch_pos:
        return ch_pos

    for ch in raw.info.get("chs", []):
        name = str(ch.get("ch_name"))
        loc = np.asarray(ch.get("loc", [])[:3], dtype=float)
        if loc.shape == (3,) and np.all(np.isfinite(loc)) and np.linalg.norm(loc) > 0:
            ch_pos[name] = loc

    return ch_pos


def _source_positions(channel_names: List[str], raw=None) -> Dict[str, np.ndarray]:
    """Prefer the known HydroCel 128 montage for EGI E1..E128 channel labels."""
    source_montage = _montage_positions("GSN-HydroCel-128")
    if source_montage and all(name in source_montage for name in channel_names):
        return {name: source_montage[name] for name in channel_names}

    raw_pos = _raw_positions(raw)
    if raw_pos:
        return {name: raw_pos[name] for name in channel_names if name in raw_pos}

    return {}


def _unit_vector(vector: np.ndarray) -> Optional[np.ndarray]:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm <= 0 or not np.isfinite(norm):
        return None
    return vector / norm


def _nearest_aliases(
    channel_names: List[str],
    source_pos: Dict[str, np.ndarray],
    alias_pos: Dict[str, np.ndarray],
    max_distance_m: float,
) -> Dict[str, Dict[str, float]]:
    if not source_pos or not alias_pos:
        return {}

    alias_names = list(alias_pos.keys())
    alias_vectors = []
    valid_alias_names = []
    for alias_name in alias_names:
        unit = _unit_vector(alias_pos[alias_name])
        if unit is not None:
            valid_alias_names.append(alias_name)
            alias_vectors.append(unit)

    if not alias_vectors:
        return {}

    alias_matrix = np.vstack(alias_vectors)
    alias_radius = float(np.median([np.linalg.norm(alias_pos[name]) for name in valid_alias_names]))
    if not np.isfinite(alias_radius) or alias_radius <= 0:
        alias_radius = 0.095

    matches = {}
    for channel_name in channel_names:
        source_unit = _unit_vector(source_pos.get(channel_name))
        if source_unit is None:
            continue

        chord_distances = np.linalg.norm(alias_matrix - source_unit.reshape(1, 3), axis=1)
        nearest_idx = int(np.argmin(chord_distances))
        distance_m = float(chord_distances[nearest_idx] * alias_radius)
        if distance_m <= max_distance_m:
            matches[channel_name] = {
                "alias": valid_alias_names[nearest_idx],
                "distance_m": distance_m,
            }

    return matches


def select_nearest_channels(
    channel_names: Iterable[str],
    target_names: Iterable[str],
    raw=None,
    source_montage: str = "GSN-HydroCel-128",
    target_montage: str = "standard_1005",
) -> Dict:
    """
    Select dataset channels nearest to a target scalp montage list.

    Returns selected exact labels in target-name order plus an audit mapping.
    """
    exact_names = _as_list(channel_names)
    target_names = _as_list(target_names)

    source_pos = _montage_positions(source_montage)
    if source_pos and all(name in source_pos for name in exact_names):
        source_pos = {name: source_pos[name] for name in exact_names}
    else:
        raw_pos = _raw_positions(raw)
        source_pos = {name: raw_pos[name] for name in exact_names if name in raw_pos}

    target_pos = _montage_positions(target_montage)
    if not source_pos:
        raise ValueError("No source channel positions available for channel selection.")
    if not target_pos:
        raise ValueError(f"Target montage positions not available: {target_montage}")

    source_vectors = []
    source_names = []
    for source_name in exact_names:
        unit = _unit_vector(source_pos.get(source_name))
        if unit is not None:
            source_names.append(source_name)
            source_vectors.append(unit)

    if not source_vectors:
        raise ValueError("No valid source channel positions available for channel selection.")

    source_matrix = np.vstack(source_vectors)
    target_radius = float(np.median([
        np.linalg.norm(pos)
        for pos in target_pos.values()
        if np.all(np.isfinite(pos)) and np.linalg.norm(pos) > 0
    ]))
    if not np.isfinite(target_radius) or target_radius <= 0:
        target_radius = 0.095

    selected_channels = []
    selected_targets = []
    channel_aliases = {}
    channel_alias_distances_m = {}
    target_to_channel = {}
    used_channels = set()

    for target_name in target_names:
        if target_name not in target_pos:
            raise ValueError(f"Target channel not found in {target_montage}: {target_name}")

        target_unit = _unit_vector(target_pos[target_name])
        if target_unit is None:
            continue

        distances = np.linalg.norm(source_matrix - target_unit.reshape(1, 3), axis=1)
        order = np.argsort(distances)
        chosen_idx = None
        for candidate_idx in order:
            candidate_name = source_names[int(candidate_idx)]
            if candidate_name not in used_channels:
                chosen_idx = int(candidate_idx)
                break

        if chosen_idx is None:
            continue

        channel_name = source_names[chosen_idx]
        distance_m = float(distances[chosen_idx] * target_radius)
        used_channels.add(channel_name)
        selected_channels.append(channel_name)
        selected_targets.append(target_name)
        channel_aliases[channel_name] = target_name
        channel_alias_distances_m[channel_name] = distance_m
        target_to_channel[target_name] = channel_name

    return {
        "selected_channels": selected_channels,
        "selected_targets": selected_targets,
        "target_to_channel": target_to_channel,
        "channel_aliases": channel_aliases,
        "channel_alias_distances_m": channel_alias_distances_m,
        "source_montage": source_montage,
        "target_montage": target_montage,
    }


def build_channel_metadata(
    channel_names: Iterable[str],
    raw=None,
    label_style: str = "e_alias",
    alias_montage: str = "standard_1005",
    max_distance_m: float = 0.02,
    alias_overrides: Optional[Dict[str, str]] = None,
    alias_distance_overrides: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Build exact and display labels for a channel list.

    Exact names are unchanged. Display labels add nearest scalp aliases only when
    a standard-montage alias is close enough.
    """
    exact_names = _as_list(channel_names)
    channel_aliases = {}
    channel_alias_distances_m = {}

    if label_style == "e_alias" and mne is not None:
        matches = _nearest_aliases(
            exact_names,
            _source_positions(exact_names, raw=raw),
            _montage_positions(alias_montage),
            float(max_distance_m),
        )
        channel_aliases = {
            channel_name: match["alias"]
            for channel_name, match in matches.items()
        }
        channel_alias_distances_m = {
            channel_name: match["distance_m"]
            for channel_name, match in matches.items()
        }

    if alias_overrides:
        channel_aliases.update({
            str(channel_name): str(alias)
            for channel_name, alias in alias_overrides.items()
            if str(channel_name) in exact_names
        })
    if alias_distance_overrides:
        channel_alias_distances_m.update({
            str(channel_name): float(distance)
            for channel_name, distance in alias_distance_overrides.items()
            if str(channel_name) in exact_names
        })

    display_names = [
        f"{name}/{channel_aliases[name]}" if name in channel_aliases else name
        for name in exact_names
    ]

    return {
        "channel_names": exact_names,
        "channel_display_names": display_names,
        "channel_aliases": channel_aliases,
        "channel_alias_distances_m": channel_alias_distances_m,
        "channel_label_style": label_style,
        "channel_alias_montage": alias_montage,
        "channel_alias_max_distance_m": float(max_distance_m),
    }


def validate_channel_metadata(metadata: Dict, n_channels: Optional[int] = None) -> None:
    channel_names = list(metadata.get("channel_names") or [])
    display_names = list(metadata.get("channel_display_names") or [])

    if n_channels is not None and len(channel_names) != int(n_channels):
        raise ValueError(
            f"Channel metadata has {len(channel_names)} names; expected {n_channels}."
        )
    if len(channel_names) != len(display_names):
        raise ValueError("channel_names and channel_display_names length mismatch.")
    if len(set(channel_names)) != len(channel_names):
        raise ValueError("Duplicate exact channel names found in channel metadata.")


def save_channel_metadata(metadata: Dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def load_channel_metadata(input_path: str) -> Dict:
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_display_channel_names(metadata_or_result: Optional[Dict], n_nodes: Optional[int] = None) -> List[str]:
    """
    Prefer display labels, then exact channel names, then Node <idx> fallback.
    """
    if isinstance(metadata_or_result, dict):
        labels = metadata_or_result.get("channel_display_names")
        if labels:
            labels = list(labels)
            if n_nodes is None or len(labels) == int(n_nodes):
                return labels

        labels = metadata_or_result.get("channel_names") or metadata_or_result.get("channels")
        if labels:
            labels = list(labels)
            if n_nodes is None or len(labels) == int(n_nodes):
                return labels

    if n_nodes is None:
        return []
    return [f"Node {idx}" for idx in range(int(n_nodes))]
