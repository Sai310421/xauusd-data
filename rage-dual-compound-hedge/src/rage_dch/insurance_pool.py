"""Insurance payoff, kept apart from the hedge book's trading pnl."""

from __future__ import annotations


def insurance_payoff(main_pnl: float, coverage: float, deductible: float) -> float:
    if coverage < 0 or deductible < 0:
        raise ValueError("coverage and deductible must be non-negative")
    if main_pnl >= 0:
        return 0.0
    return min(coverage, max(0.0, -main_pnl - deductible))


def total_pnl(p_main: float, p_hedge_trading: float, p_insurance: float, cost: float) -> float:
    return p_main + p_hedge_trading + p_insurance - cost
