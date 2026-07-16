"""Train, save, load, and apply the selected dataset-specific classifiers.

The fitted model consumes features extracted from all configured band epochs.
For an optimization experiment that changes only one band, supply the changed
band together with the subject's unchanged other bands; do not mix channel
orders or dataset profiles.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import joblib
import numpy as np
import sklearn

from data_features import (
    _subject_epoch_features,
    align_feature_matrix,
    build_feature_dataset,
)
from modeling import fit_tuned_model


SELECTED = {
    "tdbrain": {"feature_set": "covariance_logcorr", "model": "rbf_svm"},
    "first_paper": {"feature_set": "eeg_portable_fused", "model": "rbf_svm"},
}


@dataclass(frozen=True)
class ScoringModel:
    """Stable in-memory interface to a saved Patient-probability model."""

    estimator: Any
    profile: str
    feature_set: str
    feature_names: list[str]
    model_name: str

    def predict_features(self, matrix: np.ndarray, names: list[str]) -> np.ndarray:
        """Return P(Patient) after strict feature-name alignment."""

        aligned = align_feature_matrix(
            np.atleast_2d(np.asarray(matrix, dtype=float)), names, self.feature_names
        )
        return self.estimator.predict_proba(aligned)[:, 1]

    def predict_band_epochs(
        self, band_epochs: dict[str, np.ndarray], channels: list[str]
    ) -> float:
        """Return P(Patient) for one subject's band-filtered EEG epochs."""

        values, names = _subject_epoch_features(band_epochs, channels)
        if self.feature_set == "eeg_portable_fused":
            values[self.feature_set] = np.concatenate(
                [values["spectral_roi"], values["covariance_common_logcorr"]]
            )
            names[self.feature_set] = (
                names["spectral_roi"] + names["covariance_common_logcorr"]
            )
        if self.feature_set not in values:
            raise ValueError(
                f"Saved feature set {self.feature_set!r} cannot be extracted from band epochs"
            )
        return float(self.predict_features(values[self.feature_set], names[self.feature_set])[0])


def load_scoring_model(path: str | Path) -> ScoringModel:
    """Load a model artifact created by this module."""

    payload = joblib.load(path)
    required = {"estimator", "profile", "feature_set", "feature_names", "model_name"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Invalid scoring artifact; missing keys: {sorted(missing)}")
    return ScoringModel(**{key: payload[key] for key in required})


def train_selected(
    profile: str,
    *,
    mode: str = "quick",
    n_jobs: int = 1,
    output: Path | None = None,
) -> Path:
    """Tune the selected family on all development subjects and save it."""

    choice = SELECTED[profile]
    dataset = build_feature_dataset(profile)
    feature_set = choice["feature_set"]
    estimator, best_params = fit_tuned_model(
        dataset.matrices[feature_set],
        dataset.y,
        model_name=choice["model"],
        mode=mode,
        inner_splits=5,
        n_jobs=n_jobs,
    )
    output = output or (
        Path(__file__).resolve().parent
        / "models"
        / f"{profile}__{feature_set}__{choice['model']}.joblib"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "estimator": estimator,
        "profile": profile,
        "feature_set": feature_set,
        "feature_names": dataset.feature_names[feature_set],
        "model_name": choice["model"],
        "classes": {"Healthy": 0, "Patient": 1},
        "best_params": best_params,
        "training_subject_ids_sha256": hashlib.sha256(
            "\n".join(map(str, dataset.subject_ids)).encode("utf-8")
        ).hexdigest(),
        "training_class_counts": {
            "Healthy": int(np.sum(dataset.y == 0)),
            "Patient": int(np.sum(dataset.y == 1)),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "warning": (
            "Research classifier only. A probability is not a diagnosis or treatment target. "
            "Do not apply across dataset profiles without external calibration."
        ),
    }
    joblib.dump(payload, output)
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps({key: value for key, value in payload.items() if key != "estimator"}, indent=2),
        encoding="utf-8",
    )
    return output


def score_saved_epoch_file(model_path: Path, epoch_path: Path) -> float:
    """Score a main-pipeline filtered-epoch NPY file."""

    payload = np.load(epoch_path, allow_pickle=True).item()
    model = load_scoring_model(model_path)
    channels = list(payload.get("channel_names", payload.get("channels", [])))
    return model.predict_band_epochs(payload["filtered_epochs"], channels)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Fit and save the selected model")
    train.add_argument("--profile", choices=sorted(SELECTED), required=True)
    train.add_argument("--mode", choices=["quick", "full"], default="quick")
    train.add_argument("--n-jobs", type=int, default=1)
    train.add_argument("--output", type=Path, default=None)
    score = subparsers.add_parser("score-epochs", help="Score one filtered-epoch NPY")
    score.add_argument("--model", type=Path, required=True)
    score.add_argument("--epochs", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "train":
        path = train_selected(args.profile, mode=args.mode, n_jobs=args.n_jobs, output=args.output)
        print(path)
        return 0
    probability = score_saved_epoch_file(args.model, args.epochs)
    print(json.dumps({"patient_probability": probability, "healthy_probability": 1 - probability}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
