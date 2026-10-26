"""Period returns, geometric growth, drawdown, simultaneous loss.

A 10% mean is not a 10% geometric rate, and neither is a promise that
every month clears 10%.
"""

from __future__ import annotations

import math


def period_return(start_equity: float, end_equity: float) -> float:
    if start_equity <= 0:
        raise ValueError("start equity must be positive")
    return end_equity / start_equity - 1.0


def geometric_growth(simple_returns: list[float]) -> float:
    if not simple_returns:
        raise ValueError("need at least one return")
    acc = 0.0
    for r in simple_returns:
        if r <= -1.0:
            return -1.0
        acc += math.log1p(r)
    return math.exp(acc / len(simple_returns)) - 1.0


def max_floating_drawdown(equity_path: list[float]) -> float:
    if not equity_path:
        raise ValueError("empty path")
    peak = equity_path[0]
    worst = 0.0
    for equity in equity_path:
        if equity < 0:
            raise ValueError("negative equity is outside this drawdown definition")
        peak = max(peak, equity)
        if peak > 0:
            worst = max(worst, (peak - equity) / peak)
    return worst


def simultaneous_loss_rate(
    main_pnls: list[float],
    hedge_pnls: list[float],
    eps: float = 0.0,
) -> float:
    if len(main_pnls) != len(hedge_pnls) or not main_pnls:
        raise ValueError("aligned non-empty pnl series required")
    both = sum(1 for a, b in zip(main_pnls, hedge_pnls) if a < -eps and b < -eps)
    return both / len(main_pnls)


def months_below(simple_returns: list[float], target: float) -> int:
    return sum(1 for r in simple_returns if r < target)
