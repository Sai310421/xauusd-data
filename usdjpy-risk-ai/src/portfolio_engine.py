"""Advisory-only bridge from asset-model probabilities to portfolio hedge risk."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite

import pandas as pd

from .asset_modeling import FittedAssetTarget, predict_asset_probability
from .multiasset import portfolio_decision


class PortfolioEngineError(ValueError):
    """Raised when a portfolio advisory cannot be safely calculated."""


@dataclass(frozen=True)
class PortfolioAllocation:
    """Long-only portfolio weights used only for advisory risk estimation."""

    gold_weight: float
    stock_weight: float
    cash_weight: float


@dataclass(frozen=True)
class PortfolioAdvisory:
    """A non-executable portfolio-risk result with model provenance."""

    as_of: str
    gold_prob_up2: float
    stock_crash_prob: float
    smallcap_stress: float
    breadth_div: float
    portfolio_risk: int
    portfolio_regime: str
    hedge_action: str
    gold_model_version: str
    stock_model_version: str

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-ready contract."""
        return asdict(self)


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise PortfolioEngineError(f"{name}_INVALID")
    return value


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value):
        raise PortfolioEngineError(f"{name}_INVALID")
    return value


def _psi_is_critical(psi: Mapping[str, float]) -> bool:
    """Validate PSI inputs and identify the production stop condition."""
    if not psi:
        raise PortfolioEngineError("PSI_MISSING")
    values = [_finite(value, f"PSI_{feature}") for feature, value in psi.items()]
    return any(value > 0.25 for value in values)


def _single_probability(model: FittedAssetTarget, row: pd.DataFrame, name: str) -> float:
    if len(row) != 1:
        raise PortfolioEngineError(f"{name}_ROW_COUNT_INVALID")
    output = predict_asset_probability(model, row)
    if len(output) != 1:
        raise PortfolioEngineError(f"{name}_PREDICTION_INVALID")
    return _probability(float(output[0]), name)


def portfolio_advisory_from_models(
    gold_model: FittedAssetTarget,
    stock_model: FittedAssetTarget,
    gold_row: pd.DataFrame,
    stock_row: pd.DataFrame,
    allocation: PortfolioAllocation,
    psi: Mapping[str, float],
    gold_model_version: str,
    stock_model_version: str,
) -> PortfolioAdvisory:
    """Connect calibrated Gold/Stock models to a fail-closed hedge advisory.

    This function never places orders, sizes positions, or accesses broker APIs.
    """
    if _psi_is_critical(psi):
        raise PortfolioEngineError("MODEL_DRIFT_CRITICAL")
    gold_as_of = pd.Timestamp(gold_row.iloc[0]["datetime_utc"])
    stock_as_of = pd.Timestamp(stock_row.iloc[0]["datetime_utc"])
    if gold_as_of != stock_as_of:
        raise PortfolioEngineError("ASSET_TIMESTAMP_MISMATCH")
    gold_probability = _single_probability(gold_model, gold_row, "GOLD_PROBABILITY")
    stock_probability = _single_probability(stock_model, stock_row, "STOCK_PROBABILITY")
    stress = _finite(float(stock_row.iloc[0]["smallcap_stress"]), "SMALLCAP_STRESS")
    breadth = _finite(float(stock_row.iloc[0]["breadth_div"]), "BREADTH_DIVERGENCE")
    decision = portfolio_decision(
        gold_probability,
        stock_probability,
        stress,
        breadth,
        False,
        allocation.gold_weight,
        allocation.stock_weight,
        allocation.cash_weight,
    )
    return PortfolioAdvisory(
        as_of=gold_as_of.isoformat(),
        gold_prob_up2=gold_probability,
        stock_crash_prob=stock_probability,
        smallcap_stress=stress,
        breadth_div=breadth,
        portfolio_risk=decision.portfolio_risk,
        portfolio_regime=decision.portfolio_regime,
        hedge_action=decision.hedge_action,
        gold_model_version=gold_model_version,
        stock_model_version=stock_model_version,
    )
