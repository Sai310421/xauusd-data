"""Gold, equity, small-cap, and portfolio hedge decision calculations only."""
from __future__ import annotations

from dataclasses import dataclass


def smallcap_metrics(z_rty: float, z_es: float, z_nq: float) -> tuple[float, float, bool]:
    """Return stress, breadth divergence, and deterministic warning state."""
    stress = -z_rty
    breadth = z_rty - 0.5 * z_es - 0.5 * z_nq
    return stress, breadth, breadth < -2 or stress > 2


def gold_score(flow_gc: float, dxy_z: float, flow_zn: float, flow_vx: float, us10y_z: float, smallcap_stress: float) -> float:
    """Compute the specified Gold AI causal risk score."""
    return 0.30 * flow_gc - 0.20 * dxy_z + 0.15 * flow_zn + 0.15 * flow_vx - 0.10 * us10y_z + 0.10 * smallcap_stress


@dataclass(frozen=True)
class PortfolioDecision:
    """Non-execution hedge decision output for a supplied portfolio allocation."""

    portfolio_risk: int
    portfolio_regime: str
    hedge_action: str


def portfolio_decision(gold_prob_up2: float, stock_crash_prob: float, smallcap_stress: float, breadth_div: float, psi_critical: bool, gold_weight: float, stock_weight: float, cash_weight: float) -> PortfolioDecision:
    """Convert monitoring probabilities to a bounded advisory-only hedge action."""
    if min(gold_weight, stock_weight, cash_weight) < 0 or abs(gold_weight + stock_weight + cash_weight - 1.0) > 1e-9:
        raise ValueError("PORTFOLIO_WEIGHT_INVALID")
    risk = 100 * (0.55 * stock_crash_prob * stock_weight + 0.20 * max(smallcap_stress, 0) / 3 + 0.15 * max(-breadth_div, 0) / 3 + 0.10 * float(psi_critical))
    risk = round(min(100, max(0, risk)))
    regime = "LOW" if risk <= 30 else "MEDIUM" if risk <= 60 else "HIGH" if risk <= 80 else "CRITICAL"
    stock_bear, gold_bull = stock_crash_prob >= 0.55, gold_prob_up2 >= 0.55
    action = "Reduce Equity Increase Gold" if stock_bear and gold_bull else "Increase Cash" if stock_bear else "Maintain Current Allocation"
    return PortfolioDecision(risk, regime, action)
