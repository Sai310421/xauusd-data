from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Cell:
    dd_limit: float
    edge_bps: float
    opportunities_per_day: int
    risk_per_entry_pct: float
    days: int
    seed: int


def simulate(c: Cell) -> dict:
    """Mechanical controller-capacity test, NOT a tradable-alpha backtest.

    We inject a stationary positive-EV opportunity stream and let the DD controller
    throttle entry count/size as DD approaches the user-selected wall.  This isolates
    the risk OS from alpha quality.  Results are an upper/mechanical capacity frontier,
    not evidence of achievable trading returns.
    """
    rng = np.random.default_rng(c.seed)
    equity = 1000.0
    peak = equity
    maxdd = 0.0
    realized = 0.0
    entries = wins = losses = 0
    gw = gl = 0.0
    daily = []
    overshoot = 0.0

    # Positive-EV synthetic outcome model.  edge_bps is expected return per accepted
    # opportunity in basis points of equity before DD throttling.  Noise is deliberately
    # large relative to edge to create losing streaks and stress the controller.
    edge = c.edge_bps / 10000.0
    sigma = max(edge * 6.0, 0.00015)

    for _day in range(c.days):
        day_start = equity
        for _ in range(c.opportunities_per_day):
            dd = max(0.0, (peak - equity) / max(peak, 1e-12) * 100.0)
            maxdd = max(maxdd, dd)
            if dd >= c.dd_limit:
                overshoot = max(overshoot, dd - c.dd_limit)
                break

            # Continuous DD-headroom controller; no binary step-down before the wall.
            h = max(0.0, 1.0 - dd / max(c.dd_limit, 1e-12))
            budget = h * h
            # Opportunity count is reduced by Bernoulli admission; per-entry risk is
            # also scaled continuously.  This models 'more/fewer entries' rather than
            # leverage-only control.
            if rng.random() > budget:
                continue
            risk_pct = c.risk_per_entry_pct * max(0.05, budget)

            r = rng.normal(edge, sigma)
            pnl = equity * risk_pct / 100.0 * (r / sigma)
            equity += pnl
            realized += pnl
            entries += 1
            if pnl >= 0:
                wins += 1; gw += pnl
            else:
                losses += 1; gl += -pnl
            peak = max(peak, equity)
            dd2 = max(0.0, (peak - equity) / max(peak, 1e-12) * 100.0)
            maxdd = max(maxdd, dd2)
            overshoot = max(overshoot, max(0.0, dd2 - c.dd_limit))
            if dd2 >= c.dd_limit:
                break
        daily.append((equity / day_start - 1.0) * 100.0)

    pf = gw / gl if gl > 0 else (math.inf if gw > 0 else 0.0)
    mean_daily = float(np.mean(daily)) if daily else 0.0
    med_daily = float(np.median(daily)) if daily else 0.0
    compounded_monthly_21 = (math.prod((1.0 + x / 100.0) for x in daily[:21]) - 1.0) * 100.0 if daily else 0.0
    return {
        "dd_limit_pct": c.dd_limit,
        "edge_bps_per_opportunity": c.edge_bps,
        "opportunities_per_day": c.opportunities_per_day,
        "risk_per_entry_pct": c.risk_per_entry_pct,
        "days": c.days,
        "entries": entries,
        "WR_pct": 100.0 * wins / max(1, entries),
        "PF": pf,
        "return_pct": (equity / 1000.0 - 1.0) * 100.0,
        "mean_daily_pct": mean_daily,
        "median_daily_pct": med_daily,
        "monthly_21d_compound_pct": compounded_monthly_21,
        "max_DD_pct": maxdd,
        "boundary_overshoot_pctpt": overshoot,
        "mechanical_only": True,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--seed", type=int, default=260910)
    a = p.parse_args()

    dd_grid = [4.5, 10.0, 15.0, 20.0, 30.0]
    edge_grid = [0.5, 1.0, 2.0, 4.0]   # bps/opportunity synthetic EV
    n_grid = [50, 100, 250, 500, 1000, 2000]
    risk_grid = [0.02, 0.05, 0.10, 0.20]

    rows = []
    k = 0
    for dd in dd_grid:
        for edge in edge_grid:
            for n in n_grid:
                for risk in risk_grid:
                    k += 1
                    rows.append(simulate(Cell(dd, edge, n, risk, a.days, a.seed + k)))

    # Frontier: maximum mean daily return while respecting boundary within 0.10 pct-pt.
    frontier = []
    for dd in dd_grid:
        feasible = [r for r in rows if r["dd_limit_pct"] == dd and r["max_DD_pct"] <= dd + 0.10]
        if feasible:
            best = max(feasible, key=lambda r: r["mean_daily_pct"])
            frontier.append(best)

    result = {
        "test_type": "AE_CONTROLLER_MECHANICAL_MAX_OUTPUT_FRONTIER_V1",
        "warning": "Synthetic positive-EV mechanical stress test only. Not a tradable-alpha or expected-return claim.",
        "days": a.days,
        "grid_cells": len(rows),
        "frontier": frontier,
        "all_cells": rows,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"grid_cells": len(rows), "frontier": frontier}, indent=2))


if __name__ == "__main__":
    main()
