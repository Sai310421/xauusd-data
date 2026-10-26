"""Structural overlay model. Not a market backtest and not XAUUSD.

Monthly main return: mu + sigma * z, z standard normal.
Pure hedge sleeve: -beta * main_return - cost.
A second study uses v1 intensity as the hedge ratio. Neither is a tick result.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from rage_dch.compound_accounting import geometric_growth, simultaneous_loss_rate
from rage_dch.protective_hedge import protective_hedge_v1


@dataclass(frozen=True)
class OverlaySummary:
    label: str
    paths: int
    months: int
    seed: int
    mu_main: float
    sigma: float
    beta: float | None
    cost: float
    mean_geom_main: float
    mean_geom_hedge: float
    paths_main_ge_target: int
    paths_hedge_ge_target: int
    simultaneous_loss_rate: float
    mean_shortfall_when_main_below_bound: float
    loss_bound: float


def _randn(rng: random.Random) -> float:
    return rng.gauss(0.0, 1.0)


def _finish(label, rows, paths, months, seed, mu_main, sigma, beta, cost, target, loss_bound):
    geoms_m, geoms_h, hit_m, hit_h, sim_rates, shortfalls = rows
    return OverlaySummary(
        label=label,
        paths=paths,
        months=months,
        seed=seed,
        mu_main=mu_main,
        sigma=sigma,
        beta=beta,
        cost=cost,
        mean_geom_main=sum(geoms_m) / paths,
        mean_geom_hedge=sum(geoms_h) / paths,
        paths_main_ge_target=hit_m,
        paths_hedge_ge_target=hit_h,
        simultaneous_loss_rate=sum(sim_rates) / paths,
        mean_shortfall_when_main_below_bound=(sum(shortfalls) / len(shortfalls) if shortfalls else 0.0),
        loss_bound=loss_bound,
    )


def _accumulate(paths, months, seed, shock):
    rng = random.Random(seed)
    geoms_m = []
    geoms_h = []
    hit_m = 0
    hit_h = 0
    sim_rates = []
    shortfalls = []
    target = 0.10
    loss_bound = None
    for _ in range(paths):
        main_r = []
        hedge_r = []
        for _m in range(months):
            r_m, r_h, bound = shock(rng)
            loss_bound = bound
            main_r.append(r_m)
            hedge_r.append(r_h)
            if r_m < -bound:
                total = r_m + r_h
                shortfalls.append(max(0.0, -bound - total))
        gm = geometric_growth(main_r)
        gh = geometric_growth(hedge_r)
        geoms_m.append(gm)
        geoms_h.append(gh)
        hit_m += int(gm >= target)
        hit_h += int(gh >= target)
        sim_rates.append(simultaneous_loss_rate(main_r, hedge_r))
    return geoms_m, geoms_h, hit_m, hit_h, sim_rates, shortfalls, loss_bound


def run_overlay(
    mu_main: float = 0.0,
    sigma: float = 0.04,
    beta: float = 1.0,
    cost: float = 0.002,
    loss_bound: float = 0.03,
    months: int = 24,
    paths: int = 400,
    seed: int = 7,
    target: float = 0.10,
) -> OverlaySummary:
    def shock(rng: random.Random):
        r_m = mu_main + sigma * _randn(rng)
        return r_m, -beta * r_m - cost, loss_bound

    rows = _accumulate(paths, months, seed, shock)
    body = rows[:-1]
    return _finish(
        "SYNTHETIC_STRUCTURAL_NOT_MARKET",
        body,
        paths,
        months,
        seed,
        mu_main,
        sigma,
        beta,
        cost,
        target,
        loss_bound,
    )


def run_v1_intensity_overlay(
    mu_main: float = 0.0,
    sigma_month: float = 0.04,
    sigma_annual: float = 0.16,
    sigma_ref: float = 0.16,
    risk_t: float = 0.2,
    cost: float = 0.002,
    loss_bound: float = 0.03,
    months: int = 24,
    paths: int = 400,
    seed: int = 7,
    target: float = 0.10,
) -> OverlaySummary:
    """Hedge ratio is v1 intensity, not a perfect beta of 1.

    Drawdown input is the month's loss fraction. That is a coarse proxy for
    (K-S)/S, not a tick path.
    """

    def shock(rng: random.Random):
        r_m = mu_main + sigma_month * _randn(rng)
        d_t = max(-r_m, 0.0)
        h = protective_hedge_v1(sigma_annual, sigma_ref, d_t, risk_t, 1.0 / 12.0).h
        return r_m, -h * r_m - cost, loss_bound

    rows = _accumulate(paths, months, seed, shock)
    return _finish(
        "SYNTHETIC_V1_INTENSITY_NOT_MARKET",
        rows[:-1],
        paths,
        months,
        seed,
        mu_main,
        sigma_month,
        None,
        cost,
        target,
        loss_bound,
    )


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return 0.0
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / math.sqrt(va * vb)
