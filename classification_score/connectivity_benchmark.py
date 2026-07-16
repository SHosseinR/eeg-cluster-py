"""Reliability and leakage-safe classification benchmark for EEG connectivity.

This script consumes the saved band-filtered epochs produced by ``main.py``.
It never overwrites production connectivity artifacts. Each estimator is kept
on its natural scale; feature scaling/selection occurs inside CV training folds.

Examples
--------
Fast estimators on both datasets::

    python classification_score/connectivity_benchmark.py --profiles first_paper tdbrain

Include the slower conditional VAR family::

    python classification_score/connectivity_benchmark.py --profiles tdbrain \
        --families legacy fourier envelope var --n-jobs 4 --resume

Pairwise orthogonalized AEC is deliberately opt-in because it is much slower::

    python classification_score/connectivity_benchmark.py --profiles tdbrain \
        --families orthogonalized --n-jobs 4 --resume
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import sys
import time
import tomllib
from typing import Any

import numpy as np
import pandas as pd

try:  # Support both direct CLI execution and ``python -m``/test imports.
    from .connectivity_methods import (
        DIRECTED_PHASE_METHODS,
        DIRECTED_VAR_METHODS,
        UNDIRECTED_METHODS,
        compute_conditional_var_connectivity,
        compute_envelope_connectivity,
        compute_fourier_connectivity,
        edge_vector,
        split_half_reliability,
    )
    from .modeling import nested_oof_evaluate
except ImportError:
    from connectivity_methods import (
        DIRECTED_PHASE_METHODS,
        DIRECTED_VAR_METHODS,
        UNDIRECTED_METHODS,
        compute_conditional_var_connectivity,
        compute_envelope_connectivity,
        compute_fourier_connectivity,
        edge_vector,
        split_half_reliability,
    )
    from modeling import nested_oof_evaluate


REPO_ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("Healthy", "Patient")
BANDS = {"delta": (1.0, 4.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
FOURIER_METHODS = tuple(UNDIRECTED_METHODS[:7]) + tuple(DIRECTED_PHASE_METHODS)
METHOD_DIRECTION = {
    **{method: False for method in UNDIRECTED_METHODS},
    **{method: True for method in DIRECTED_PHASE_METHODS + DIRECTED_VAR_METHODS},
    "legacy_gc": True,
}
CACHE_VERSION = 2
FUSION_DEFINITIONS = {
    "coherence_aec": ("coherence", "aec"),
    "coherence_aec_imcoh": ("coherence", "aec", "imaginary_coherence"),
    "plv_aec": ("plv", "aec"),
    "lag_resistant": ("imaginary_coherence", "ciplv", "pli", "wpli2_debiased"),
}


def _safe_name(value: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return compact or "subject"


def _profile_results_root(profile: str) -> Path:
    config_path = REPO_ROOT / "dataset_configs" / f"{profile.removesuffix('.toml')}.toml"
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    output = Path(config["output_directory"])
    return output if output.is_absolute() else (REPO_ROOT / output).resolve()


def _load_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required upstream artifact is missing: {path}")
    value = np.load(path, allow_pickle=True).item()
    if not isinstance(value, dict):
        raise TypeError(f"Expected dictionary artifact at {path}")
    return value


def _subject_cache_path(cache_root: Path, group: str, subject_id: str) -> Path:
    return cache_root / _safe_name(group) / f"{_safe_name(subject_id)}.npz"


def _compute_subject_cache(task: tuple[str, str, str, str, tuple[str, ...]]) -> dict[str, Any]:
    """Windows-safe worker that computes full and odd/even split matrices."""

    group, subject_id, epoch_path_text, cache_path_text, families = task
    epoch_path = Path(epoch_path_text)
    cache_path = Path(cache_path_text)
    payload = _load_dict(epoch_path)
    filtered = payload["filtered_epochs"]
    fs = float(payload["fs"])
    arrays: dict[str, np.ndarray] = {}
    var_diagnostics: dict[str, dict[str, Any]] = {}
    cached_families: set[str] = set()
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as existing:
                existing_metadata = json.loads(str(existing["__metadata_json"].item()))
                if existing_metadata.get("cache_version") == CACHE_VERSION:
                    arrays.update(
                        {
                            key: np.asarray(existing[key])
                            for key in existing.files
                            if key != "__metadata_json"
                        }
                    )
                    cached_families.update(existing_metadata.get("families", []))
                    var_diagnostics.update(existing_metadata.get("var_diagnostics", {}))
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            arrays.clear()
            cached_families.clear()
            var_diagnostics.clear()

    for band_name, (fmin, fmax) in BANDS.items():
        epochs = np.asarray(filtered[band_name], dtype=float)
        if epochs.shape[0] < 4:
            raise ValueError(f"{group}/{subject_id}/{band_name} has only {epochs.shape[0]} epochs")
        splits = {"full": epochs, "odd": epochs[::2], "even": epochs[1::2]}
        for split_name, split_epochs in splits.items():
            matrices: dict[str, np.ndarray] = {}
            if "fourier" in families:
                matrices.update(
                    compute_fourier_connectivity(
                        split_epochs, fs, fmin, fmax, methods=FOURIER_METHODS
                    )
                )
            if "envelope" in families:
                matrices.update(
                    compute_envelope_connectivity(
                        split_epochs, include_orthogonalized=False
                    )
                )
            if "orthogonalized" in families:
                matrices["orthogonalized_aec"] = compute_envelope_connectivity(
                    split_epochs, include_orthogonalized=True
                )["orthogonalized_aec"]
            if "var" in families:
                var_matrices, diagnostics = compute_conditional_var_connectivity(
                    split_epochs,
                    fs,
                    fmin,
                    fmax,
                    target_sfreq=100.0,
                    lag_ms=100.0,
                    ridge_alpha=10.0,
                )
                matrices.update(var_matrices)
                if split_name == "full":
                    var_diagnostics[band_name] = diagnostics.__dict__
            for method, matrix in matrices.items():
                if matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
                    raise ValueError(
                        f"Invalid {method} matrix for {group}/{subject_id}/{band_name}: {matrix.shape}"
                    )
                arrays[f"{method}__{band_name}__{split_name}"] = matrix.astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    arrays["__metadata_json"] = np.asarray(
        json.dumps(
            {
                "cache_version": CACHE_VERSION,
                "group": group,
                "subject_id": subject_id,
                "fs": fs,
                "families": sorted(cached_families | set(families)),
                "var_diagnostics": var_diagnostics,
            },
            sort_keys=True,
        )
    )
    np.savez_compressed(cache_path, **arrays)
    return {"group": group, "subject_id": subject_id, "cache_path": str(cache_path)}


def _cache_is_compatible(path: Path, families: tuple[str, ...]) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["__metadata_json"].item()))
            cached = set(metadata.get("families", []))
            return metadata.get("cache_version") == CACHE_VERSION and set(families) <= cached
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def _compute_profile_caches(
    profile: str,
    families: tuple[str, ...],
    *,
    n_jobs: int,
    resume: bool,
    max_subjects_per_group: int | None,
) -> tuple[list[tuple[str, str, Path]], Path]:
    results_root = _profile_results_root(profile)
    data_root = results_root / "data"
    index = _load_dict(data_root / "filtered_epochs_index.npy")
    cache_root = Path(__file__).resolve().parent / "cache" / f"connectivity_v{CACHE_VERSION}" / profile
    records: list[tuple[str, str, Path]] = []
    tasks: list[tuple[str, str, str, str, tuple[str, ...]]] = []
    compute_families = tuple(item for item in families if item != "legacy")

    for group in GROUPS:
        subject_ids = sorted(index.get(group, {}))
        if max_subjects_per_group is not None:
            subject_ids = subject_ids[:max_subjects_per_group]
        if not subject_ids:
            raise ValueError(f"No filtered subjects found for {profile}/{group}")
        for subject_id in subject_ids:
            cache_path = _subject_cache_path(cache_root, group, subject_id)
            records.append((group, subject_id, cache_path))
            if not compute_families:
                continue
            if resume and _cache_is_compatible(cache_path, compute_families):
                continue
            epoch_path = data_root / "filtered_epochs" / group / f"{subject_id}.npy"
            tasks.append((group, subject_id, str(epoch_path), str(cache_path), compute_families))

    if tasks:
        print(
            f"{profile}: computing {compute_families} for {len(tasks)} subjects "
            f"with {n_jobs} worker(s)",
            flush=True,
        )
        if n_jobs == 1:
            for number, task in enumerate(tasks, 1):
                _compute_subject_cache(task)
                if number == 1 or number % 10 == 0 or number == len(tasks):
                    print(f"  connectivity {number}/{len(tasks)}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = [executor.submit(_compute_subject_cache, task) for task in tasks]
                for number, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if number == 1 or number % 10 == 0 or number == len(tasks):
                        print(f"  connectivity {number}/{len(tasks)}", flush=True)
    return records, data_root


def _methods_for_families(families: tuple[str, ...]) -> list[str]:
    methods: list[str] = []
    if "legacy" in families:
        methods.append("legacy_gc")
    if "fourier" in families:
        methods.extend(FOURIER_METHODS)
    if "envelope" in families:
        methods.append("aec")
    if "orthogonalized" in families:
        methods.append("orthogonalized_aec")
    if "var" in families:
        methods.extend(DIRECTED_VAR_METHODS)
    return methods


def _load_benchmark_features(
    records: list[tuple[str, str, Path]],
    data_root: Path,
    methods: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    legacy = _load_dict(data_root / "connectivity_matrices.npy") if "legacy_gc" in methods else {}
    features: dict[str, list[np.ndarray]] = {method: [] for method in methods}
    labels: list[int] = []
    subject_ids: list[str] = []
    reliability_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []

    for group, subject_id, cache_path in records:
        labels.append(0 if group == "Healthy" else 1)
        subject_ids.append(subject_id)
        cache_context = np.load(cache_path, allow_pickle=False) if cache_path.exists() else None
        try:
            metadata = (
                json.loads(str(cache_context["__metadata_json"].item()))
                if cache_context is not None
                else {}
            )
            for band, values in metadata.get("var_diagnostics", {}).items():
                diagnostics_rows.append(
                    {"group": group, "subject_id": subject_id, "band": band, **values}
                )
            for method in methods:
                band_vectors = []
                for band in BANDS:
                    if method == "legacy_gc":
                        matrix = np.asarray(legacy[group][subject_id]["gc"][band], dtype=float)
                    else:
                        if cache_context is None:
                            raise FileNotFoundError(f"Missing connectivity cache: {cache_path}")
                        matrix = np.asarray(cache_context[f"{method}__{band}__full"], dtype=float)
                        odd = np.asarray(cache_context[f"{method}__{band}__odd"], dtype=float)
                        even = np.asarray(cache_context[f"{method}__{band}__even"], dtype=float)
                        reliability_rows.append(
                            {
                                "group": group,
                                "subject_id": subject_id,
                                "method": method,
                                "band": band,
                                **split_half_reliability(
                                    odd, even, directed=METHOD_DIRECTION[method]
                                ),
                            }
                        )
                    band_vectors.append(edge_vector(matrix, METHOD_DIRECTION[method]))
                features[method].append(np.concatenate(band_vectors))
        finally:
            if cache_context is not None:
                cache_context.close()

    matrices = {method: np.vstack(rows) for method, rows in features.items()}
    return (
        matrices,
        np.asarray(labels, dtype=int),
        np.asarray(subject_ids, dtype=object),
        pd.DataFrame(reliability_rows),
        pd.DataFrame(diagnostics_rows),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", default=["first_paper", "tdbrain"])
    parser.add_argument(
        "--families",
        nargs="+",
        choices=["legacy", "fourier", "envelope", "orthogonalized", "var"],
        default=["legacy", "fourier", "envelope"],
    )
    parser.add_argument(
        "--models", nargs="+", default=["logistic_l2", "rbf_svm", "extra_trees"]
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional estimator subset after resolving --families.",
    )
    parser.add_argument(
        "--fusions",
        nargs="+",
        choices=sorted(FUSION_DEFINITIONS),
        default=None,
        help="Concatenate selected connectivity representations before fold-local selection.",
    )
    parser.add_argument(
        "--fusion-only",
        action="store_true",
        help="Evaluate requested fusions but not their component methods.",
    )
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--max-subjects-per-group", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name", default="screen")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    families = tuple(dict.fromkeys(args.families))
    methods = _methods_for_families(families)
    if args.methods is not None:
        unknown = sorted(set(args.methods) - set(methods))
        if unknown:
            raise ValueError(f"Requested methods are not enabled by --families: {unknown}")
        methods = list(dict.fromkeys(args.methods))
    base_methods = list(methods)
    output_root = Path(__file__).resolve().parent / "results" / "connectivity_benchmark"

    for profile in args.profiles:
        started = time.perf_counter()
        records, data_root = _compute_profile_caches(
            profile,
            families,
            n_jobs=args.n_jobs,
            resume=args.resume,
            max_subjects_per_group=args.max_subjects_per_group,
        )
        matrices, y, subject_ids, reliability, diagnostics = _load_benchmark_features(
            records, data_root, base_methods
        )
        component_methods = list(base_methods)
        fusion_names = list(dict.fromkeys(args.fusions or []))
        for fusion_name in fusion_names:
            components = FUSION_DEFINITIONS[fusion_name]
            missing = [item for item in components if item not in matrices]
            if missing:
                raise ValueError(
                    f"Fusion {fusion_name!r} requires methods not loaded by --families/--methods: {missing}"
                )
            matrices[fusion_name] = np.hstack([matrices[item] for item in components])
        profile_methods = fusion_names if args.fusion_only else component_methods + fusion_names
        profile_output = output_root / profile / args.run_name
        predictions_output = profile_output / "predictions"
        predictions_output.mkdir(parents=True, exist_ok=True)
        if not reliability.empty:
            reliability.to_csv(profile_output / "split_half_subjects.csv", index=False)
            reliability.groupby(["method", "band"], as_index=False).agg(
                edge_spearman_median=("edge_spearman", "median"),
                edge_spearman_mean=("edge_spearman", "mean"),
                normalized_edge_error_median=("normalized_edge_error", "median"),
                top10pct_jaccard_median=("top10pct_jaccard", "median"),
                n_subjects=("subject_id", "count"),
            ).to_csv(profile_output / "split_half_summary.csv", index=False)
        if not diagnostics.empty:
            diagnostics.to_csv(profile_output / "var_diagnostics.csv", index=False)

        summaries: list[dict[str, Any]] = []
        total = len(profile_methods) * len(args.models)
        number = 0
        for method in profile_methods:
            X = matrices[method]
            for model in args.models:
                number += 1
                print(
                    f"[{number}/{total}] {profile}: {method} ({X.shape[1]} edges) / {model}",
                    flush=True,
                )
                result, predictions = nested_oof_evaluate(
                    X,
                    y,
                    feature_set=method,
                    model_name=model,
                    mode=args.mode,
                    outer_splits=args.outer_splits,
                    repeats=args.repeats,
                    inner_splits=args.inner_splits,
                    n_jobs=1,
                    subject_ids=subject_ids,
                )
                result.update(
                    {
                        "profile": profile,
                        "directed": METHOD_DIRECTION.get(method, "mixed"),
                        "natural_scale": method != "legacy_gc",
                    }
                )
                summaries.append(result)
                predictions.to_csv(predictions_output / f"{method}__{model}.csv", index=False)
                pd.DataFrame(summaries).sort_values(
                    ["roc_auc", "brier"], ascending=[False, True]
                ).to_csv(profile_output / "classification_summary.csv", index=False)
                print(
                    f"  AUC={result['roc_auc']:.3f}, bal_acc={result['balanced_accuracy']:.3f}, "
                    f"Brier={result['brier']:.3f}",
                    flush=True,
                )
        metadata = {
            "profile": profile,
            "families": families,
            "methods": profile_methods,
            "models": args.models,
            "n_subjects": int(y.size),
            "n_healthy": int(np.sum(y == 0)),
            "n_patient": int(np.sum(y == 1)),
            "bands": BANDS,
            "var_target_sfreq": 100.0,
            "var_lag_ms": 100.0,
            "var_ridge_alpha": 10.0,
            "elapsed_seconds": time.perf_counter() - started,
        }
        (profile_output / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(f"Saved {profile} benchmark to {profile_output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
