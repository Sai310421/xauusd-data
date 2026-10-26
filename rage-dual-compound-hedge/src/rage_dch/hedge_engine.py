"""Protective sleeve on top of v1.

Policy, not a formula change: if the main book is flat, do not open a hedge.
v1 still emits a positive intensity from the lambda3 term when drawdown is 0.
Paying that intensity with nothing to protect is a known failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass

from rage_dch.main_engine import OrderIntent, Quote
from rage_dch.protective_hedge import HedgeParams, protective_hedge_v1, quantity_from_intensity


@dataclass(frozen=True)
class HedgeDecision:
    intensity: float
    target_oz: float
    orders: tuple[OrderIntent, ...]
    note: str


def decide_hedge(
    quote: Quote,
    main_position_oz: float,
    reference_price: float,
    sigma_t: float,
    sigma_ref: float,
    risk_t: float,
    dt: float,
    contract_notional: float,
    spread_fraction: float,
    current_hedge_oz: float,
    params: HedgeParams | None = None,
) -> HedgeDecision:
    if quote.ask < quote.bid:
        raise ValueError("invalid quote")
    mid = (quote.bid + quote.ask) / 2.0
    if main_position_oz == 0:
        close = _delta_orders(current_hedge_oz, 0.0)
        return HedgeDecision(0.0, 0.0, close, "flat main: protective sleeve stays off")
    side_d = 0.0
    if main_position_oz > 0:
        side_d = max((reference_price - mid) / mid, 0.0) if mid > 0 else 0.0
    else:
        side_d = max((mid - reference_price) / mid, 0.0) if mid > 0 else 0.0
    intensity = protective_hedge_v1(sigma_t, sigma_ref, side_d, risk_t, dt, params).h
    protected = abs(main_position_oz) * reference_price
    contracts = quantity_from_intensity(intensity, protected, contract_notional, spread_fraction)
    oz = contracts * (contract_notional / reference_price) if reference_price > 0 else 0.0
    target = -math_copysign(oz, main_position_oz)
    orders = _delta_orders(current_hedge_oz, target)
    return HedgeDecision(intensity, target, orders, "v1 intensity, opposite the main exposure")


def math_copysign(magnitude: float, sign_source: float) -> float:
    return magnitude if sign_source >= 0 else -magnitude


def _delta_orders(current: float, target: float) -> tuple[OrderIntent, ...]:
    delta = target - current
    if abs(delta) < 1e-12:
        return ()
    side = "buy" if delta > 0 else "sell"
    return (
        OrderIntent("XAUUSD", side, abs(delta), "protective-sleeve", "hedge"),
    )
