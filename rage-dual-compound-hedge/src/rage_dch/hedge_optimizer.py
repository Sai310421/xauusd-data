"""Two-stage optimizer.

Stage 1 minimizes hedge cost subject to a loss bound on a named scenario set.
Stage 2 may use only the capital that stage 1 did not lock, and only sleeves
the caller actually supplies. A missed 10% target is reported as missed.
The bound is not relaxed to manufacture a feasible point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    main_pnl: float
    hedge_pnl_per_unit: float
    cost_per_unit: float


@dataclass(frozen=True)
class StageOne:
    feasible: bool
    quantity: float
    cost: float
    reason: str
    lower: float
    upper: float


@dataclass(frozen=True)
class StageTwo:
    sleeve: str
    fraction_of_free_capital: float
    mean_log_growth: float
    geometric_monthly: float
    meets_target: bool


def min_protection_quantity(
    scenarios: list[Scenario],
    loss_bound: float,
    q_max: float,
    unit_cost_price: float,
) -> StageOne:
    """Smallest q in [0, q_max] such that every scenario stays inside -loss_bound.

    loss_bound is a positive number of currency units that may be lost.
    """
    if loss_bound < 0 or q_max < 0 or unit_cost_price < 0:
        raise ValueError("bound, q_max, and unit cost must be non-negative")
    if not scenarios:
        raise ValueError("scenario set is empty")
    lower = 0.0
    upper = q_max
    for scenario in scenarios:
        unit = scenario.hedge_pnl_per_unit - scenario.cost_per_unit
        need = -loss_bound - scenario.main_pnl
        if unit > 1e-12:
            if need > 0:
                lower = max(lower, need / unit)
        elif unit < -1e-12:
            if need > 1e-9:
                return StageOne(False, 0.0, 0.0, f"{scenario.name}: hedge worsens a breach", lower, upper)
            upper = min(upper, need / unit)
        elif need > 1e-9:
            return StageOne(False, 0.0, 0.0, f"{scenario.name}: zero hedge pnl cannot cover", lower, upper)
    if lower > upper + 1e-9:
        return StageOne(False, 0.0, 0.0, "lower bound exceeds upper bound", lower, upper)
    quantity = lower
    return StageOne(True, quantity, quantity * unit_cost_price, "min cost inside the bound", lower, upper)


def required_monthly_edge(locked_fraction: float, cost_rate: float, target: float = 0.10) -> float | None:
    """Edge the unlocked sleeve must earn so the whole book still compounds at `target`.

    Locked capital earns (1 - cost_rate) over the month, i.e. it pays cost_rate.
    Returns None when the lock is total: no finite sleeve edge can hit the target
    if the locked slice itself does not.
    """
    if not 0.0 <= locked_fraction < 1.0:
        return None
    if cost_rate < 0 or target <= -1.0:
        raise ValueError("invalid cost or target")
    return (1.0 + target - locked_fraction * (1.0 - cost_rate)) / (1.0 - locked_fraction) - 1.0


def _mean_log(ratios: list[float]) -> float:
    if any(r <= 0 for r in ratios):
        return float("-inf")
    return sum(math.log(r) for r in ratios) / len(ratios)


def maximize_log_growth(
    sleeves: dict[str, list[float]],
    locked_fraction: float,
    locked_cost_rate: float,
    target: float = 0.10,
) -> StageTwo:
    """Grid search over caller-supplied sleeves. Flat cash is always a candidate.

    sleeves map a name to simple monthly returns of the alpha sleeve.
    Wealth ratio of the whole book:
        free * (1 + pi * r) + locked * (1 - cost)
    """
    if not 0.0 <= locked_fraction < 1.0:
        raise ValueError("locked fraction must be in [0, 1)")
    free = 1.0 - locked_fraction
    candidates: list[tuple[str, float, float]] = []
    search = {"cash": [0.0]}
    search.update(sleeves)
    grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    for name, returns in search.items():
        if not returns:
            raise ValueError(f"sleeve {name} has no returns")
        for pi in grid if name != "cash" else (0.0,):
            ratios = [
                free * (1.0 + pi * r) + locked_fraction * (1.0 - locked_cost_rate) for r in returns
            ]
            candidates.append((name, pi, _mean_log(ratios)))
    name, pi, score = max(candidates, key=lambda item: item[2])
    geom = math.exp(score) - 1.0 if math.isfinite(score) else -1.0
    return StageTwo(name, pi, score, geom, geom + 1e-12 >= target)
