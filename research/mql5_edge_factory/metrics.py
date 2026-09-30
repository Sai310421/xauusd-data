from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable, Optional


@dataclass
class EdgeMetrics:
    n: int
    wins: int
    losses: int
    wr_pct: float
    net: float
    expectancy: float
    pf: float
    max_dd: float
    avg_win: float
    avg_loss: float
    avg_mfe: Optional[float] = None
    avg_mae: Optional[float] = None

    def to_dict(self):
        return asdict(self)


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def metrics(pnls: Iterable[float], mfes: Optional[Iterable[float]] = None, maes: Optional[Iterable[float]] = None) -> EdgeMetrics:
    p = [float(x) for x in pnls]
    wins = [x for x in p if x > 0]
    losses = [x for x in p if x < 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    equity = peak = dd = 0.0
    for x in p:
        equity += x
        peak = max(peak, equity)
        dd = max(dd, peak - equity)
    return EdgeMetrics(
        n=len(p),
        wins=len(wins),
        losses=len(losses),
        wr_pct=(100.0 * len(wins) / len(p)) if p else 0.0,
        net=sum(p),
        expectancy=(sum(p) / len(p)) if p else 0.0,
        pf=(gross_profit / gross_loss) if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0),
        max_dd=dd,
        avg_win=_mean(wins) or 0.0,
        avg_loss=_mean(losses) or 0.0,
        avg_mfe=_mean([float(x) for x in mfes]) if mfes is not None else None,
        avg_mae=_mean([float(x) for x in maes]) if maes is not None else None,
    )


def matched_delta(edge_pnls: Iterable[float], control_pnls: Iterable[float]) -> dict:
    e = metrics(edge_pnls)
    c = metrics(control_pnls)
    return {
        "edge": e.to_dict(),
        "control": c.to_dict(),
        "delta_expectancy": e.expectancy - c.expectancy,
        "delta_pf": e.pf - c.pf if math.isfinite(e.pf) and math.isfinite(c.pf) else None,
        "delta_max_dd": e.max_dd - c.max_dd,
    }
