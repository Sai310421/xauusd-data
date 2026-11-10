"""Time-series model fitting, calibration, and walk-forward evaluation."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .features import FEATURES


@dataclass(frozen=True)
class FoldDefinition:
    """One immutable, calendar-bounded walk-forward split."""

    name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str


FOLDS = (
    FoldDefinition("fold_1", "2019-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    FoldDefinition("fold_2", "2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    FoldDefinition("fold_3", "2021-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
)


@dataclass
class CalibrationArtifact:
    """Serializable calibration model and its chosen method."""

    method: str
    estimator: IsotonicRegression | LogisticRegression

    def transform(self, probability: np.ndarray) -> np.ndarray:
        """Apply a bounded calibration mapping."""
        values = np.asarray(probability, dtype=float)
        if self.method == "isotonic":
            return np.asarray(self.estimator.predict(values), dtype=float)
        return np.asarray(
            self.estimator.predict_proba(values.reshape(-1, 1))[:, 1],
            dtype=float,
        )


@dataclass
class FittedTarget:
    """A LightGBM model plus its time-series calibration artifact."""

    model: LGBMClassifier
    calibrator: CalibrationArtifact
    validation_rows: int
    validation_positive_count: int


def model_params(n_estimators: int = 1000) -> dict[str, Any]:
    """Return the production LightGBM parameter contract."""
    return {
        "objective": "binary",
        "n_estimators": n_estimators,
        "learning_rate": 0.03,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }


def _timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def split_fold(
    frame: pd.DataFrame,
    fold: FoldDefinition,
    embargo_bars: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one fold and purge train rows whose labels could enter test."""
    indexed = frame.copy()
    indexed["datetime_utc"] = pd.to_datetime(indexed["datetime_utc"], utc=True)
    train_end = _timestamp(fold.train_end) - timedelta(hours=embargo_bars)
    train = indexed.loc[
        (indexed["datetime_utc"] >= _timestamp(fold.train_start))
        & (indexed["datetime_utc"] <= train_end)
    ].copy()
    test = indexed.loc[
        (indexed["datetime_utc"] >= _timestamp(fold.test_start))
        & (indexed["datetime_utc"] <= _timestamp(fold.test_end) + timedelta(days=1))
    ].copy()
    if not train.empty and not test.empty:
        assert train["datetime_utc"].max() < test["datetime_utc"].min()
    return train, test


def _validation_split(frame: pd.DataFrame, fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronological training data without shuffling."""
    ordered = frame.sort_values("datetime_utc").reset_index(drop=True)
    boundary = max(1, int(len(ordered) * (1.0 - fraction)))
    return ordered.iloc[:boundary].copy(), ordered.iloc[boundary:].copy()


def _calibrator(
    probability: np.ndarray,
    target: np.ndarray,
    isotonic_min_rows: int,
    isotonic_min_positive: int,
) -> CalibrationArtifact:
    """Fit isotonic only when the time-series validation sample is sufficient."""
    positive = int(np.sum(target))
    if len(target) >= isotonic_min_rows and positive >= isotonic_min_positive:
        estimator = IsotonicRegression(out_of_bounds="clip")
        estimator.fit(probability, target)
        return CalibrationArtifact("isotonic", estimator)
    estimator = LogisticRegression(random_state=42, solver="lbfgs")
    estimator.fit(probability.reshape(-1, 1), target)
    return CalibrationArtifact("sigmoid", estimator)


def fit_target(
    train: pd.DataFrame,
    target: str,
    n_estimators: int = 1000,
    isotonic_min_rows: int = 1000,
    isotonic_min_positive: int = 100,
) -> FittedTarget:
    """Fit one target with a strictly later validation slice for calibration."""
    if train[target].nunique() < 2:
        raise ValueError(f"{target} has one class")
    fit_rows, validation = _validation_split(train)
    if fit_rows[target].nunique() < 2 or validation[target].nunique() < 2:
        raise ValueError(f"{target} has insufficient chronological class coverage")
    model = LGBMClassifier(**model_params(n_estimators))
    model.fit(fit_rows[FEATURES], fit_rows[target].astype(int))
    raw = model.predict_proba(validation[FEATURES])[:, 1]
    calibrator = _calibrator(
        raw,
        validation[target].to_numpy(dtype=int),
        isotonic_min_rows,
        isotonic_min_positive,
    )
    return FittedTarget(
        model=model,
        calibrator=calibrator,
        validation_rows=len(validation),
        validation_positive_count=int(validation[target].sum()),
    )


def predict_probability(fitted: FittedTarget, frame: pd.DataFrame) -> np.ndarray:
    """Predict calibrated probability for a fixed feature schema."""
    raw = fitted.model.predict_proba(frame[FEATURES])[:, 1]
    return np.clip(fitted.calibrator.transform(raw), 0.0, 1.0)


def binary_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Compute non-optimized classification metrics at a fixed 0.5 threshold."""
    label = np.asarray(target, dtype=int)
    prob = np.asarray(probability, dtype=float)
    prediction = (prob >= 0.5).astype(int)
    return {
        "auc_roc": float(roc_auc_score(label, prob)),
        "pr_auc": float(average_precision_score(label, prob)),
        "precision": float(precision_score(label, prediction, zero_division=0)),
        "recall": float(recall_score(label, prediction, zero_division=0)),
        "f1": float(f1_score(label, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(label, prediction)),
        "brier_score": float(brier_score_loss(label, prob)),
        "positive_count": float(label.sum()),
    }


def walk_forward(
    frame: pd.DataFrame,
    targets: Iterable[str] = ("target_05", "target_10", "target_20"),
    folds: Iterable[FoldDefinition] = FOLDS,
    min_positive_samples: int = 50,
    n_estimators: int = 1000,
) -> pd.DataFrame:
    """Run only chronological fold evaluations; never use train_test_split."""
    records: list[dict[str, Any]] = []
    for fold in folds:
        train, test = split_fold(frame, fold)
        for target in targets:
            record: dict[str, Any] = {
                "fold": fold.name,
                "target": target,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "train_rows": len(train),
                "test_rows": len(test),
                "status": "PASS",
            }
            if train.empty or test.empty:
                record["status"] = "BLOCKED_NO_ROWS"
            elif int(train[target].sum()) < min_positive_samples:
                record["status"] = "BLOCKED_POSITIVE_COUNT"
            elif train[target].nunique() < 2 or test[target].nunique() < 2:
                record["status"] = "BLOCKED_CLASS_COVERAGE"
            if record["status"] == "PASS":
                try:
                    fitted = fit_target(train, target, n_estimators=n_estimators)
                    metrics = binary_metrics(test[target], predict_probability(fitted, test))
                    record.update(metrics)
                    record["calibration_method"] = fitted.calibrator.method
                except ValueError as error:
                    record["status"] = "BLOCKED_CALIBRATION"
                    record["reason"] = str(error)
            records.append(record)
    return pd.DataFrame(records)


def fold_metadata() -> list[dict[str, str]]:
    """Expose fixed fold definitions for reports and manifests."""
    return [asdict(fold) for fold in FOLDS]
