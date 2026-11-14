"""Portfolio Hedge Engine model-connection contract tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.asset_features import FEATURES_GOLD, FEATURES_STOCK
from src.asset_modeling import FittedAssetTarget
from src.portfolio_engine import (
    PortfolioAllocation,
    PortfolioEngineError,
    portfolio_advisory_from_models,
)


class _Model:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.tile([1.0 - self.probability, self.probability], (len(frame), 1))


class _Calibrator:
    def transform(self, values: np.ndarray) -> np.ndarray:
        return values


def _fitted(features: list[str], probability: float) -> FittedAssetTarget:
    return FittedAssetTarget(model=_Model(probability), calibrator=_Calibrator(), feature_order=features)  # type: ignore[arg-type]


def _row(features: list[str], when: str) -> pd.DataFrame:
    row = {feature: 0.1 for feature in features}
    row.update({"datetime_utc": pd.Timestamp(when, tz="UTC"), "smallcap_stress": 0.8, "breadth_div": -0.5})
    return pd.DataFrame([row])


def test_models_produce_advisory_only_portfolio_contract() -> None:
    gold = _fitted(FEATURES_GOLD, 0.74)
    stock = _fitted(FEATURES_STOCK, 0.68)
    output = portfolio_advisory_from_models(
        gold,
        stock,
        _row(FEATURES_GOLD, "2025-01-01T00:00:00"),
        _row(FEATURES_STOCK, "2025-01-01T00:00:00"),
        PortfolioAllocation(0.70, 0.30, 0.00),
        {"risk_score": 0.08, "flow_vx": 0.12},
        "gold-v001",
        "stock-v001",
    )
    assert output.gold_prob_up2 == 0.74
    assert output.stock_crash_prob == 0.68
    assert output.hedge_action == "Reduce Equity Increase Gold"
    assert output.portfolio_risk >= 0


def test_critical_psi_stops_advisory_before_hedge_decision() -> None:
    with pytest.raises(PortfolioEngineError, match="MODEL_DRIFT_CRITICAL"):
        portfolio_advisory_from_models(
            _fitted(FEATURES_GOLD, 0.74),
            _fitted(FEATURES_STOCK, 0.68),
            _row(FEATURES_GOLD, "2025-01-01T00:00:00"),
            _row(FEATURES_STOCK, "2025-01-01T00:00:00"),
            PortfolioAllocation(0.70, 0.30, 0.00),
            {"risk_score": 0.26},
            "gold-v001",
            "stock-v001",
        )


def test_asset_timestamp_mismatch_stops_advisory() -> None:
    with pytest.raises(PortfolioEngineError, match="ASSET_TIMESTAMP_MISMATCH"):
        portfolio_advisory_from_models(
            _fitted(FEATURES_GOLD, 0.74),
            _fitted(FEATURES_STOCK, 0.68),
            _row(FEATURES_GOLD, "2025-01-01T00:00:00"),
            _row(FEATURES_STOCK, "2025-01-01T01:00:00"),
            PortfolioAllocation(0.70, 0.30, 0.00),
            {"risk_score": 0.08},
            "gold-v001",
            "stock-v001",
        )
