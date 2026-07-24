"""Broadband PDC/DTF reliability, validity, and classification benchmark.

Raw EEGLAB data are loaded subject-by-subject in the authoritative saved
channel order. VAR order and ridge strength are selected without cohort labels
using held-out epochs from that subject. The selected parameters are then used
for full, odd/even, and time-reversed broadband fits.
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

import mne
import numpy as np
import pandas as pd
from scipy import stats

try:
    from .broadband_var import BANDS, fit_subject_connectivity
    from .connectivity_methods import edge_vector, split_half_reliability
    from .modeling import nested_oof_evaluate
except ImportError:
    from broadband_var import BANDS, fit_subject_connectivity
    from connectivity_methods import edge_vector, split_half_reliability
    from modeling import nested_oof_evaluate


REPO_ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("Healthy", "Patient")
CACHE_VERSION = 3
METHODS = ("pdc", "dtf")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "subject"


def _profile_config(profile: str) -> dict[str, Any]:
    path = REPO_ROOT / "dataset_configs" / f"{profile.removesuffix('.toml')}.toml"
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _results_root(config: dict[str, Any]) -> Path:
    path = Path(config["output_directory"])
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    value = np.load(path, allow_pickle=True).item()
    if not isinstance(value, dict):
        raise TypeError(f"Expected dictionary artifact: {path}")
    return value


def _epoch_payload(path: Path) -> dict[str, Any]:
    return _load_dict(path)


def _load_raw_broadband_epochs(
    epoch_payload_path: Path,
    raw_subject_folder: Path,
    *,
    epoch_duration: float = 10.0,
) -> tuple[np.ndarray, float, list[str], str]:
    payload = _epoch_payload(epoch_payload_path)
    channels = list(payload.get("channel_names", payload.get("channels", [])))
    if not channels:
        raise ValueError(f"No authoritative channel order in {epoch_payload_path}")
    saved_fs = float(payload["fs"])
    if "broadband_epochs" in payload:
        broadband = np.asarray(payload["broadband_epochs"], dtype=float)
        if broadband.shape[1] != len(channels):
            raise ValueError("Saved broadband epochs do not match channel metadata")
        return broadband, saved_fs, channels, "saved_broadband_epochs"

    set_files = sorted(raw_subject_folder.glob("*.set"))
    if not set_files:
        raise FileNotFoundError(f"No EEGLAB .set files in {raw_subject_folder}")
    recordings = []
    for set_path in set_files:
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
        missing = [channel for channel in channels if channel not in raw.ch_names]
        if missing:
            raise ValueError(f"{set_path} lacks saved channels: {missing}")
        raw.pick(channels)
        if list(raw.ch_names) != channels:
            raise ValueError(f"Channel order mismatch after selecting {set_path}")
        if not np.isclose(float(raw.info["sfreq"]), saved_fs):
            raise ValueError(f"Sampling frequency mismatch in {set_path}")
        recordings.append(raw.get_data())
    continuous = np.concatenate(recordings, axis=1)
    samples_per_epoch = int(round(epoch_duration * saved_fs))
    n_epochs = continuous.shape[1] // samples_per_epoch
    if n_epochs < 4:
        raise ValueError(f"Only {n_epochs} complete broadband epochs in {raw_subject_folder}")
    truncated = continuous[:, : n_epochs * samples_per_epoch]
    epochs = truncated.reshape(len(channels), n_epochs, samples_per_epoch).transpose(1, 0, 2)
    return epochs, saved_fs, channels, "raw_eeglab"


def _cache_parameters(
    target_sfreq: float,
    broadband_fmin: float,
    broadband_fmax: float,
    lag_ms_candidates: tuple[float, ...],
    ridge_candidates: tuple[float, ...],
) -> dict[str, Any]:
    return {
        "target_sfreq": float(target_sfreq),
        "broadband_fmin": float(broadband_fmin),
        "broadband_fmax": float(broadband_fmax),
        "lag_ms_candidates": list(lag_ms_candidates),
        "ridge_candidates": list(ridge_candidates),
    }


def _cache_compatible(path: Path, parameters: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["__metadata_json"].item()))
        return (
            metadata.get("cache_version") == CACHE_VERSION
            and metadata.get("parameters") == parameters
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def _compute_subject(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        group,
        subject_id,
        epoch_path_text,
        raw_folder_text,
        cache_path_text,
        target_sfreq,
        broadband_fmin,
        broadband_fmax,
        lag_ms_candidates,
        ridge_candidates,
    ) = task
    started = time.perf_counter()
    epochs, fs, channels, source = _load_raw_broadband_epochs(
        Path(epoch_path_text), Path(raw_folder_text)
    )
    connectivity, diagnostics = fit_subject_connectivity(
        epochs,
        fs,
        target_sfreq=float(target_sfreq),
        broadband_fmin=float(broadband_fmin),
        broadband_fmax=float(broadband_fmax),
        lag_ms_candidates=tuple(lag_ms_candidates),
        ridge_candidates=tuple(ridge_candidates),
    )
    arrays: dict[str, np.ndarray] = {}
    for split, method_data in connectivity.items():
        for method, band_data in method_data.items():
            for band, matrix in band_data.items():
                if matrix.shape != (len(channels), len(channels)):
                    raise ValueError(f"Invalid {method}/{band} shape: {matrix.shape}")
                if not np.all(np.isfinite(matrix)):
                    raise ValueError(f"Non-finite {method}/{band} values")
                arrays[f"{method}__{band}__{split}"] = matrix.astype(np.float32)
    parameters = _cache_parameters(
        float(target_sfreq),
        float(broadband_fmin),
        float(broadband_fmax),
        tuple(lag_ms_candidates),
        tuple(ridge_candidates),
    )
    metadata = {
        "cache_version": CACHE_VERSION,
        "group": group,
        "subject_id": subject_id,
        "channels": channels,
        "original_sfreq": fs,
        "broadband_source": source,
        "parameters": parameters,
        "diagnostics": diagnostics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    arrays["__metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    cache_path = Path(cache_path_text)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    return {
        "group": group,
        "subject_id": subject_id,
        "cache_path": str(cache_path),
        "elapsed_seconds": metadata["elapsed_seconds"],
    }


def _records_and_tasks(
    profile: str,
    *,
    cache_root: Path,
    parameters: dict[str, Any],
    resume: bool,
    max_subjects_per_group: int | None,
) -> tuple[list[tuple[str, str, Path]], list[tuple[Any, ...]]]:
    config = _profile_config(profile)
    results_data = _results_root(config) / "data"
    index = _load_dict(results_data / "filtered_epochs_index.npy")
    raw_root = Path(config["dataset_root"])
    raw_subdirectories = {
        "Healthy": config["healthy_subdirectory"],
        "Patient": config["patient_subdirectory"],
    }
    records = []
    tasks = []
    lag_candidates = tuple(parameters["lag_ms_candidates"])
    ridge_candidates = tuple(parameters["ridge_candidates"])
    for group in GROUPS:
        subject_ids = sorted(index.get(group, {}))
        if max_subjects_per_group is not None:
            subject_ids = subject_ids[:max_subjects_per_group]
        for subject_id in subject_ids:
            cache_path = cache_root / group / f"{_safe_name(subject_id)}.npz"
            records.append((group, subject_id, cache_path))
            if resume and _cache_compatible(cache_path, parameters):
                continue
            epoch_path = results_data / "filtered_epochs" / group / f"{subject_id}.npy"
            raw_folder = raw_root / raw_subdirectories[group] / subject_id
            tasks.append(
                (
                    group,
                    subject_id,
                    str(epoch_path),
                    str(raw_folder),
                    str(cache_path),
                    float(parameters["target_sfreq"]),
                    float(parameters["broadband_fmin"]),
                    float(parameters["broadband_fmax"]),
                    lag_candidates,
                    ridge_candidates,
                )
            )
    return records, tasks


def _load_features(
    records: list[tuple[str, str, Path]],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    feature_rows = {method: [] for method in METHODS}
    y = []
    subject_ids = []
    reliability_rows = []
    diagnostic_rows = []
    reference_channels = None
    for group, subject_id, cache_path in records:
        with np.load(cache_path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["__metadata_json"].item()))
            channels = metadata["channels"]
            if reference_channels is None:
                reference_channels = channels
            elif channels != reference_channels:
                raise ValueError(f"Channel order changed at {group}/{subject_id}")
            diagnostics = dict(metadata["diagnostics"])
            diagnostics.pop("selection_grid", None)
            diagnostic_rows.append(
                {
                    "group": group,
                    "subject_id": subject_id,
                    "broadband_source": metadata["broadband_source"],
                    "elapsed_seconds": metadata["elapsed_seconds"],
                    **diagnostics,
                }
            )
            for method in METHODS:
                full_vectors = []
                for band in BANDS:
                    full = np.asarray(cache[f"{method}__{band}__full"], dtype=float)
                    odd = np.asarray(cache[f"{method}__{band}__odd"], dtype=float)
                    even = np.asarray(cache[f"{method}__{band}__even"], dtype=float)
                    full_vectors.append(edge_vector(full, directed=True))
                    reliability_rows.append(
                        {
                            "group": group,
                            "subject_id": subject_id,
                            "method": method,
                            "band": band,
                            **split_half_reliability(odd, even, directed=True),
                        }
                    )
                feature_rows[method].append(np.concatenate(full_vectors))
        y.append(0 if group == "Healthy" else 1)
        subject_ids.append(subject_id)
    return (
        {method: np.vstack(rows) for method, rows in feature_rows.items()},
        np.asarray(y, dtype=int),
        np.asarray(subject_ids, dtype=object),
        pd.DataFrame(reliability_rows),
        pd.DataFrame(diagnostic_rows),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", default=["first_paper", "tdbrain"])
    parser.add_argument("--target-sfreq", type=float, default=100.0)
    parser.add_argument("--broadband-fmin", type=float, default=1.0)
    parser.add_argument("--broadband-fmax", type=float, default=45.0)
    parser.add_argument("--lag-ms-candidates", nargs="+", type=float, default=[50, 100, 200])
    parser.add_argument(
        "--ridge-candidates", nargs="+", type=float, default=[1, 10, 100, 1000]
    )
    parser.add_argument("--models", nargs="+", default=["logistic_l2"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--max-subjects-per-group", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument(
        "--fail-on-subject-error",
        action="store_true",
        help="Stop after recording any subject that cannot produce a valid stable VAR.",
    )
    parser.add_argument("--run-name", default="broadband_var")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    parameters = _cache_parameters(
        args.target_sfreq,
        args.broadband_fmin,
        args.broadband_fmax,
        tuple(args.lag_ms_candidates),
        tuple(args.ridge_candidates),
    )
    parameter_tag = (
        f"fs{args.target_sfreq:g}_broad{args.broadband_fmin:g}-{args.broadband_fmax:g}_"
        f"lags{'-'.join(f'{x:g}' for x in args.lag_ms_candidates)}_"
        f"ridge{'-'.join(f'{x:g}' for x in args.ridge_candidates)}"
    )
    base = Path(__file__).resolve().parent
    for profile in args.profiles:
        output = base / "results" / "connectivity_benchmark" / profile / args.run_name
        output.mkdir(parents=True, exist_ok=True)
        cache_root = base / "cache" / "broadband_var_v3" / parameter_tag / profile
        records, tasks = _records_and_tasks(
            profile,
            cache_root=cache_root,
            parameters=parameters,
            resume=args.resume,
            max_subjects_per_group=args.max_subjects_per_group,
        )
        print(
            f"{profile}: {len(records)} aligned subjects, {len(tasks)} to compute, "
            f"workers={args.n_jobs}",
            flush=True,
        )
        failures: list[dict[str, str]] = []
        if tasks and args.n_jobs == 1:
            for number, task in enumerate(tasks, 1):
                try:
                    result = _compute_subject(task)
                except Exception as exc:
                    failures.append(
                        {
                            "group": str(task[0]),
                            "subject_id": str(task[1]),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    print(
                        f"  REJECT {task[0]}/{task[1]}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue
                print(
                    f"  {number}/{len(tasks)} {result['group']}/{result['subject_id']} "
                    f"{result['elapsed_seconds']:.1f}s",
                    flush=True,
                )
        elif tasks:
            with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
                futures = {
                    executor.submit(_compute_subject, task): task for task in tasks
                }
                for number, future in enumerate(as_completed(futures), 1):
                    task = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        failures.append(
                            {
                                "group": str(task[0]),
                                "subject_id": str(task[1]),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                        print(
                            f"  REJECT {task[0]}/{task[1]}: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        continue
                    if number == 1 or number % 10 == 0 or number == len(tasks):
                        print(
                            f"  {number}/{len(tasks)} latest={result['group']}/"
                            f"{result['subject_id']} {result['elapsed_seconds']:.1f}s",
                            flush=True,
                        )
        usable_records = [
            record for record in records if _cache_compatible(record[2], parameters)
        ]
        failed_keys = {(row["group"], row["subject_id"]) for row in failures}
        for group, subject_id, _ in records:
            if (group, subject_id) not in failed_keys and not any(
                item[0] == group and item[1] == subject_id for item in usable_records
            ):
                failures.append(
                    {
                        "group": group,
                        "subject_id": subject_id,
                        "error_type": "MissingValidCache",
                        "error": "No compatible broadband VAR cache was produced",
                    }
                )
        failure_frame = pd.DataFrame(failures)
        failure_frame.to_csv(output / "subject_failures.csv", index=False)
        if failures and args.fail_on_subject_error:
            raise RuntimeError(
                f"{profile}: {len(failures)} subject VAR failures; see "
                f"{output / 'subject_failures.csv'}"
            )
        if not usable_records:
            raise RuntimeError(f"{profile}: no subjects produced valid broadband VAR caches")
        features, y, subject_ids, reliability, diagnostics = _load_features(
            usable_records
        )
        reliability.to_csv(output / "split_half_subjects.csv", index=False)
        reliability.groupby(["method", "band"], as_index=False).agg(
            edge_spearman_median=("edge_spearman", "median"),
            edge_spearman_mean=("edge_spearman", "mean"),
            normalized_edge_error_median=("normalized_edge_error", "median"),
            top10pct_jaccard_median=("top10pct_jaccard", "median"),
            n_subjects=("subject_id", "count"),
        ).to_csv(output / "split_half_summary.csv", index=False)
        diagnostics.to_csv(output / "var_diagnostics.csv", index=False)
        attempted_by_group = {
            group: sum(record[0] == group for record in records) for group in GROUPS
        }
        rejected_by_group = {
            group: sum(row["group"] == group for row in failures) for group in GROUPS
        }
        rejection_table = [
            [
                rejected_by_group["Healthy"],
                attempted_by_group["Healthy"] - rejected_by_group["Healthy"],
            ],
            [
                rejected_by_group["Patient"],
                attempted_by_group["Patient"] - rejected_by_group["Patient"],
            ],
        ]
        rejection_odds_ratio, rejection_fisher_p = stats.fisher_exact(
            rejection_table
        )
        validity_failures = []
        total_rejection_fraction = len(failures) / len(records)
        if total_rejection_fraction > 0.05:
            validity_failures.append("more_than_5_percent_of_subjects_rejected")
        if rejection_fisher_p < 0.05:
            validity_failures.append("VAR_rejection_is_associated_with_cohort")
        if float(diagnostics["residual_whiteness_pass"].mean()) < 0.90:
            validity_failures.append("residual_whiteness_pass_below_90_percent")
        if min(
            float(diagnostics["odd_stable"].mean()),
            float(diagnostics["even_stable"].mean()),
            float(diagnostics["reversed_stable"].mean()),
        ) < 0.95:
            validity_failures.append("split_or_reversed_stability_below_95_percent")
        if min(
            float(diagnostics["pdc_time_reversal_spearman"].median()),
            float(diagnostics["dtf_time_reversal_spearman"].median()),
        ) <= 0:
            validity_failures.append("directed_asymmetry_does_not_reverse_in_time")
        diagnostic_summary = {
            "profile": profile,
            "n_attempted_subjects": int(len(records)),
            "n_subjects": int(len(diagnostics)),
            "n_rejected_subjects": int(len(failures)),
            "attempted_healthy": int(attempted_by_group["Healthy"]),
            "attempted_patient": int(attempted_by_group["Patient"]),
            "rejected_healthy": int(rejected_by_group["Healthy"]),
            "rejected_patient": int(rejected_by_group["Patient"]),
            "rejection_fraction": float(total_rejection_fraction),
            "rejection_odds_ratio_healthy_vs_patient": (
                float(rejection_odds_ratio)
                if np.isfinite(rejection_odds_ratio)
                else None
            ),
            "rejection_fisher_exact_p": float(rejection_fisher_p),
            "stable_fraction": float(diagnostics["stable"].mean()),
            "odd_stable_fraction": float(diagnostics["odd_stable"].mean()),
            "even_stable_fraction": float(diagnostics["even_stable"].mean()),
            "reversed_stable_fraction": float(diagnostics["reversed_stable"].mean()),
            "residual_whiteness_pass_fraction": float(
                diagnostics["residual_whiteness_pass"].mean()
            ),
            "residual_lag1_abs_mean_median": float(
                diagnostics["residual_lag1_abs_mean"].median()
            ),
            "residual_autocorrelation_abs_mean_median": float(
                diagnostics["residual_autocorrelation_abs_mean"].median()
            ),
            "pdc_time_reversal_spearman_median": float(
                diagnostics["pdc_time_reversal_spearman"].median()
            ),
            "dtf_time_reversal_spearman_median": float(
                diagnostics["dtf_time_reversal_spearman"].median()
            ),
            "classification_valid": not validity_failures,
            "validity_failures": validity_failures,
            "parameters": parameters,
        }
        (output / "diagnostic_summary.json").write_text(
            json.dumps(diagnostic_summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(json.dumps(diagnostic_summary, indent=2), flush=True)
        if args.skip_classification:
            continue
        summaries = []
        prediction_output = output / "predictions"
        prediction_output.mkdir(exist_ok=True)
        for method in METHODS:
            for model in args.models:
                summary, predictions = nested_oof_evaluate(
                    features[method],
                    y,
                    feature_set=f"broadband_{method}",
                    model_name=model,
                    mode="quick",
                    outer_splits=args.outer_splits,
                    repeats=args.repeats,
                    inner_splits=args.inner_splits,
                    n_jobs=1,
                    subject_ids=subject_ids,
                )
                summary.update({"profile": profile, "broadband_var": True})
                summary["n_rejected_subjects"] = int(len(failures))
                summary["classification_valid"] = not validity_failures
                summary["validity_failures"] = ";".join(validity_failures)
                summaries.append(summary)
                predictions.to_csv(
                    prediction_output / f"{method}__{model}.csv", index=False
                )
                pd.DataFrame(summaries).sort_values("roc_auc", ascending=False).to_csv(
                    output / "classification_summary.csv", index=False
                )
                print(
                    f"{profile} {method}/{model}: AUC={summary['roc_auc']:.3f}, "
                    f"bal_acc={summary['balanced_accuracy']:.3f}, Brier={summary['brier']:.3f}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
