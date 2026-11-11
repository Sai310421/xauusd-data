"""Feature-schema-specific LightGBM fitting for Gold and US-stock targets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .modeling import CalibrationArtifact, model_params


@dataclass
class FittedAssetTarget:
    model: LGBMClassifier
    calibrator: CalibrationArtifact
    feature_order: list[str]


def fit_asset_target(frame: pd.DataFrame, target: str, features: list[str], n_estimators: int = 1000) -> FittedAssetTarget:
    """Fit chronological train/validation slices without random splitting."""
    ordered = frame.sort_values("datetime_utc").reset_index(drop=True)
    if ordered[target].nunique() != 2:
        raise ValueError(f"{target} has insufficient class coverage")
    boundary = max(1, int(len(ordered) * 0.8))
    train, validation = ordered.iloc[:boundary], ordered.iloc[boundary:]
    if train[target].nunique() != 2 or validation[target].nunique() != 2:
        raise ValueError(f"{target} has insufficient chronological validation coverage")
    model = LGBMClassifier(**model_params(n_estimators))
    model.fit(train[features], train[target].astype(int))
    probability = model.predict_proba(validation[features])[:, 1]
    if len(validation) >= 1000 and int(validation[target].sum()) >= 100:
        estimator = IsotonicRegression(out_of_bounds="clip").fit(probability, validation[target])
        calibrator = CalibrationArtifact("isotonic", estimator)
    else:
        estimator = LogisticRegression(random_state=42, solver="lbfgs").fit(probability.reshape(-1, 1), validation[target])
        calibrator = CalibrationArtifact("sigmoid", estimator)
    return FittedAssetTarget(model=model, calibrator=calibrator, feature_order=features)


def predict_asset_probability(fitted: FittedAssetTarget, frame: pd.DataFrame) -> np.ndarray:
    """Return calibrated bounded asset probabilities in manifest feature order."""
    raw = fitted.model.predict_proba(frame[fitted.feature_order])[:, 1]
    return np.clip(fitted.calibrator.transform(raw), 0.0, 1.0)
