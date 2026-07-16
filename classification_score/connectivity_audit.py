"""Audit production GC assumptions on synthetic truth and saved artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mne
import numpy as np
import pandas as pd
from mne_connectivity import spectral_connectivity_epochs

try:
    from .connectivity_benchmark import GROUPS, _load_dict, _profile_results_root
except ImportError:
    from connectivity_benchmark import GROUPS, _load_dict, _profile_results_root


def synthetic_var_chain(seed: int = 42) -> tuple[np.ndarray, float]:
    """Return epochs with planted 0→1→2 lagged influence."""

    rng = np.random.default_rng(seed)
    epochs, channels, samples, fs = 14, 3, 500, 100.0
    data = np.zeros((epochs, channels, samples))
    for epoch in range(epochs):
        noise = rng.normal(scale=0.6, size=(channels, samples))
        for sample in range(2, samples):
            data[epoch, 0, sample] = 0.72 * data[epoch, 0, sample - 1] + noise[0, sample]
            data[epoch, 1, sample] = (
                0.55 * data[epoch, 1, sample - 1]
                + 0.48 * data[epoch, 0, sample - 1]
                + noise[1, sample]
            )
            data[epoch, 2, sample] = (
                0.50 * data[epoch, 2, sample - 1]
                + 0.42 * data[epoch, 1, sample - 1]
                + noise[2, sample]
            )
    return data, fs


def pairwise_mne_gc(
    epochs: np.ndarray, fs: float, *, n_lags: int, method: str
) -> np.ndarray:
    n_channels = epochs.shape[1]
    sources, targets = np.where(~np.eye(n_channels, dtype=bool))
    indices = ([[int(i)] for i in sources], [[int(j)] for j in targets])
    info = mne.create_info([f"Ch{i}" for i in range(n_channels)], fs, "eeg")
    result = spectral_connectivity_epochs(
        mne.EpochsArray(epochs, info, verbose=False),
        method=method,
        mode="multitaper",
        indices=indices,
        sfreq=fs,
        fmin=1.0,
        fmax=30.0,
        faverage=True,
        gc_n_lags=n_lags,
        verbose="ERROR",
    ).get_data()
    values = result[:, 0] if result.ndim == 2 else result
    matrix = np.zeros((n_channels, n_channels), dtype=float)
    matrix[sources, targets] = values
    return matrix


def synthetic_gc_lag_audit() -> pd.DataFrame:
    data, fs = synthetic_var_chain()
    rows = []
    for n_lags in (2, 5, 10, 20, 40):
        forward = pairwise_mne_gc(data, fs, n_lags=n_lags, method="gc")
        reversed_gc = pairwise_mne_gc(data, fs, n_lags=n_lags, method="gc_tr")
        true_edges = np.mean([forward[0, 1], forward[1, 2]])
        reverse_edges = np.mean([forward[1, 0], forward[2, 1]])
        forward_net = (forward[0, 1] - forward[1, 0]) + (
            forward[1, 2] - forward[2, 1]
        )
        reversed_net = (reversed_gc[0, 1] - reversed_gc[1, 0]) + (
            reversed_gc[1, 2] - reversed_gc[2, 1]
        )
        rows.append(
            {
                "gc_n_lags": n_lags,
                "lag_duration_ms": 1000.0 * n_lags / fs,
                "true_edge_mean": true_edges,
                "reverse_edge_mean": reverse_edges,
                "direction_ratio": true_edges / max(reverse_edges, np.finfo(float).eps),
                "indirect_0_to_2": forward[0, 2],
                "trgc_net_contrast": forward_net - reversed_net,
            }
        )
    return pd.DataFrame(rows)


def saved_gc_qc(profile: str) -> dict[str, float | int | str]:
    data_root = _profile_results_root(profile) / "data"
    connectivity = _load_dict(data_root / "connectivity_matrices.npy")
    index = _load_dict(data_root / "filtered_epochs_index.npy")
    matrices = []
    sampling_rates = []
    for group in GROUPS:
        for subject_id in sorted(connectivity[group]):
            sampling_rates.append(float(index[group][subject_id]["fs"]))
            for band_matrix in connectivity[group][subject_id]["gc"].values():
                matrices.append(np.asarray(band_matrix, dtype=float))
    array = np.stack(matrices)
    off_diagonal = ~np.eye(array.shape[1], dtype=bool)
    edges = array[:, off_diagonal]
    return {
        "profile": profile,
        "n_subject_band_matrices": int(array.shape[0]),
        "n_channels": int(array.shape[1]),
        "sampling_rate_hz": float(np.median(sampling_rates)),
        "default_40_lag_duration_ms": float(40000.0 / np.median(sampling_rates)),
        "nonfinite_values": int(np.sum(~np.isfinite(array))),
        "all_zero_matrices": int(np.sum(np.all(array == 0, axis=(1, 2)))),
        "edge_fraction_exact_zero": float(np.mean(edges == 0)),
        "edge_fraction_exact_one": float(np.mean(edges == 1)),
        "edge_mean": float(np.mean(edges)),
        "edge_sd": float(np.std(edges)),
        "median_directional_asymmetry": float(
            np.median(np.mean(np.abs(array - array.transpose(0, 2, 1)), axis=(1, 2)))
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", default=["first_paper", "tdbrain"])
    args = parser.parse_args(argv)
    output = Path(__file__).resolve().parent / "results" / "connectivity_audit"
    output.mkdir(parents=True, exist_ok=True)
    lag_results = synthetic_gc_lag_audit()
    lag_results.to_csv(output / "synthetic_gc_lag_sensitivity.csv", index=False)
    qc = [saved_gc_qc(profile) for profile in args.profiles]
    pd.DataFrame(qc).to_csv(output / "saved_gc_qc.csv", index=False)
    information_loss = {
        "within_subject_minmax_scale_invariance": bool(
            np.allclose(
                (np.array([0.0, 0.2, 0.5]) - 0.0) / 0.5,
                (np.array([0.0, 2.0, 5.0]) - 0.0) / 5.0,
            )
        ),
        "interpretation": (
            "Per-subject min-max normalization makes matrices that differ only in absolute "
            "strength identical and forces every nonconstant subject-band matrix to contain 1."
        ),
    }
    (output / "audit_summary.json").write_text(
        json.dumps(
            {
                "synthetic_best_direction_ratio": lag_results.loc[
                    lag_results["direction_ratio"].idxmax()
                ].to_dict(),
                "saved_gc_qc": qc,
                "normalization": information_loss,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(lag_results.to_string(index=False))
    print(pd.DataFrame(qc).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
