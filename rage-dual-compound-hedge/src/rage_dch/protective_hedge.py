"""ProtectiveHedge formulas.

v1-original is the baseline. It is not modified in place.
v2-budget-split is a separate candidate. It is not a proven improvement.

Vol unit for both versions: annualized fractional volatility.
0.16 means 16 percent. Passing 16.0 is a different, incorrect input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

FORMULA_V1 = "v1-original"
FORMULA_V2 = "v2-budget-split"


@dataclass(frozen=True)
class HedgeParams:
    kappa1: float = 2.0
    kappa2: float = 4.0
    lambda1: float = 1.0
    lambda2: float = 0.5
    lambda3: float = 0.25
    delta: float = 0.0
    rho: float = 1.0
    h_max: float = 1.0
    tau: float = 1.0 / 12.0


@dataclass(frozen=True)
class HedgeBreakdown:
    version: str
    w: float
    r: float
    h_raw: float
    h: float
    d: float


def _clip_logit(z: float) -> float:
    return max(-60.0, min(60.0, z))


def drawdown_proxy(k: float, s: float) -> float:
    """Original long-only drawdown proxy.

    Non-positive spot returns 0, matching the first committed skeleton.
    That hides a bad tick. Callers that need a hard failure use
    drawdown_proxy_v2.
    """
    if s <= 0:
        return 0.0
    return max((k - s) / s, 0.0)


def drawdown_proxy_v2(k: float, s: float, side: str) -> float:
    """Side-aware adverse move. Rejects non-positive prices."""
    if s <= 0 or k <= 0:
        raise ValueError("price and reference must be positive")
    if side == "long":
        return max((k - s) / s, 0.0)
    if side == "short":
        return max((s - k) / s, 0.0)
    raise ValueError("side must be 'long' or 'short'")


def _parts(
    sigma_t: float,
    sigma_ref: float,
    d_t: float,
    weight_logit: float,
    params: HedgeParams,
    dt: float,
    denom_risk: float,
) -> tuple[float, float, float, float]:
    if sigma_ref <= 0:
        raise ValueError("sigma_ref must be positive")
    if sigma_t < 0 or d_t < 0 or denom_risk < 0 or dt < 0 or params.tau < 0:
        raise ValueError("volatilities, drawdown, risk, and time must be non-negative")
    if params.h_max < 0:
        raise ValueError("h_max must be non-negative")
    w = 1.0 / (1.0 + math.exp(_clip_logit(weight_logit)))
    r = (
        params.lambda1 * d_t
        + params.lambda2 * max((sigma_t - sigma_ref) / sigma_ref, 0.0)
        + params.lambda3 * sigma_t * math.sqrt(params.tau)
    )
    decay = math.exp(-params.delta * dt)
    h_raw = (w * r * decay) / (1.0 + denom_risk)
    h = min(params.h_max, max(0.0, h_raw))
    return w, r, h_raw, h


def protective_hedge_v1(
    sigma_t: float,
    sigma_ref: float,
    d_t: float,
    risk_t: float,
    dt: float,
    params: HedgeParams | None = None,
) -> HedgeBreakdown:
    """Original formula. risk_t shrinks the hedge, including in a crisis."""
    if sigma_ref <= 0:
        raise ValueError("sigma_ref must be positive")
    p = params or HedgeParams()
    logit = -p.kappa1 * (sigma_t / sigma_ref - 1.0) - p.kappa2 * d_t
    w, r, h_raw, h = _parts(sigma_t, sigma_ref, d_t, logit, p, dt, p.rho * risk_t)
    return HedgeBreakdown(FORMULA_V1, w, r, h_raw, h, d_t)


def protective_hedge_v2(
    sigma_t: float,
    sigma_ref: float,
    d_t: float,
    budget_risk: float,
    market_stress: float,
    dt: float,
    params: HedgeParams | None = None,
    kappa_market: float = 1.0,
) -> HedgeBreakdown:
    """Candidate only. Budget stress shrinks size. Market stress does not.

    The original puts every 'Risk_t' in the denominator, so a volatility
    spike can cut the hedge while the book needs it. v2 keeps that shrink
    only for financing/margin stress and lets market_stress increase weight.
    """
    p = params or HedgeParams()
    if sigma_ref <= 0:
        raise ValueError("sigma_ref must be positive")
    if market_stress < 0:
        raise ValueError("market_stress must be non-negative")
    logit = (
        -p.kappa1 * (sigma_t / sigma_ref - 1.0)
        - p.kappa2 * d_t
        - kappa_market * market_stress
    )
    w, r, h_raw, h = _parts(sigma_t, sigma_ref, d_t, logit, p, dt, p.rho * budget_risk)
    return HedgeBreakdown(FORMULA_V2, w, r, h_raw, h, d_t)


def compute_hedge_intensity(
    sigma_t: float,
    sigma_ref: float,
    d_t: float,
    risk_t: float,
    delta: float,
    dt: float,
    kappa1: float,
    kappa2: float,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    tau: float,
    rho: float,
    h_max: float,
) -> float:
    """Signature kept from the skeleton commit. Returns clipped H only."""
    params = HedgeParams(
        kappa1=kappa1,
        kappa2=kappa2,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        delta=delta,
        rho=rho,
        h_max=h_max,
        tau=tau,
    )
    return protective_hedge_v1(sigma_t, sigma_ref, d_t, risk_t, dt, params).h


def quantity_from_intensity(
    h: float,
    protected_notional: float,
    contract_notional: float,
    spread_fraction: float,
) -> float:
    """Map dimensionless H into contracts.

    This conversion is not part of the original formula. H alone is not a
    position. Spread widens the notional that one contract must cover.
    """
    if h < 0 or protected_notional < 0 or spread_fraction < 0:
        raise ValueError("intensity, notional, and spread must be non-negative")
    if contract_notional <= 0:
        raise ValueError("contract_notional must be positive")
    return h * protected_notional / (contract_notional * (1.0 + spread_fraction))
