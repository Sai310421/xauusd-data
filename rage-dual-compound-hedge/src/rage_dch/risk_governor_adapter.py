"""Hard stop in front of both engines. Research-only. Not a broker connection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GovernorDecision:
    quantity: float
    status: str


def govern(
    proposed_qty: float,
    equity: float,
    current_drawdown: float,
    max_drawdown: float,
    max_gross_qty: float,
) -> GovernorDecision:
    if proposed_qty < 0 or max_gross_qty < 0 or max_drawdown < 0:
        raise ValueError("quantities and drawdown limits must be non-negative")
    if equity <= 0:
        return GovernorDecision(0.0, "halt-no-equity")
    if current_drawdown >= max_drawdown:
        return GovernorDecision(0.0, "halt-drawdown")
    if proposed_qty > max_gross_qty:
        return GovernorDecision(max_gross_qty, "capped")
    return GovernorDecision(proposed_qty, "ok")
