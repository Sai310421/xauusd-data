from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return default if abs(b) < 1e-12 else a / b


def _f(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    agent: str
    strategy: str
    symbol: str
    timeframe: str
    initial_equity: float
    final_equity: float
    return_pct: float
    max_dd_pct: float
    profit_factor: float
    win_rate_pct: float
    trades: int
    ruin_probability: float
    oos_retention: float
    cost_retention: float
    regime_pass_rate: float
    discovery_efficiency: float
    source_file: str

    @property
    def net_profit(self) -> float:
        return self.final_equity - self.initial_equity

    @classmethod
    def from_dict(cls, d: dict[str, Any], source_file: str = "") -> "Candidate":
        initial = _f(d, "initial_equity", 1000.0)
        final = _f(d, "final_equity", initial)
        ret = d.get("return_pct")
        return_pct = _f(d, "return_pct", _safe_div(final - initial, initial) * 100.0) if ret is not None else _safe_div(final - initial, initial) * 100.0
        return cls(
            candidate_id=str(d.get("candidate_id") or d.get("experiment_id") or Path(source_file).stem),
            agent=str(d.get("agent", "unknown")),
            strategy=str(d.get("strategy") or d.get("edge") or "unknown"),
            symbol=str(d.get("symbol", "UNKNOWN")),
            timeframe=str(d.get("timeframe", "UNKNOWN")),
            initial_equity=initial,
            final_equity=final,
            return_pct=return_pct,
            max_dd_pct=max(0.0, _f(d, "max_dd_pct")),
            profit_factor=max(0.0, _f(d, "profit_factor")),
            win_rate_pct=_f(d, "win_rate_pct"),
            trades=max(0, int(_f(d, "trades"))),
            ruin_probability=_clamp(_f(d, "ruin_probability")),
            oos_retention=_clamp(_f(d, "oos_retention")),
            cost_retention=_clamp(_f(d, "cost_retention")),
            regime_pass_rate=_clamp(_f(d, "regime_pass_rate")),
            discovery_efficiency=_clamp(_f(d, "discovery_efficiency")),
            source_file=source_file,
        )


@dataclass(frozen=True)
class ArenaScore:
    candidate_id: str
    agent: str
    strategy: str
    arena_a_edge_score: float
    arena_b_absolute_profit: float
    arena_b_risk_adjusted_score: float
    prop_eligible: bool
    broker_eligible: bool
    max_dd_pct: float
    profit_factor: float
    return_pct: float
    final_equity: float
    trades: int
    source_file: str


# Arena A: reusable EDGE quality. Profit alone cannot dominate this score.
def edge_score(c: Candidate) -> float:
    pf_score = _clamp((c.profit_factor - 1.0) / 2.0)
    dd_score = 1.0 - _clamp(c.max_dd_pct / 20.0)
    ruin_score = 1.0 - c.ruin_probability
    durability = (c.oos_retention + c.cost_retention + c.regime_pass_rate) / 3.0
    trade_confidence = _clamp(math.log10(max(c.trades, 1)) / 4.0)
    return 100.0 * (
        0.30 * durability
        + 0.15 * pf_score
        + 0.15 * dd_score
        + 0.10 * ruin_score
        + 0.10 * c.cost_retention
        + 0.10 * c.regime_pass_rate
        + 0.05 * c.discovery_efficiency
        + 0.05 * trade_confidence
    )


# Arena B risk-adjusted league: rewards money made but penalizes DD and ruin.
def risk_adjusted_profit_score(c: Candidate) -> float:
    return_on_dd = _safe_div(max(c.return_pct, 0.0), max(c.max_dd_pct, 0.25))
    pf_bonus = min(c.profit_factor, 4.0) / 4.0
    survival = 1.0 - c.ruin_probability
    return return_on_dd * (0.70 + 0.15 * pf_bonus + 0.15 * survival)


def score_candidate(c: Candidate) -> ArenaScore:
    return ArenaScore(
        candidate_id=c.candidate_id,
        agent=c.agent,
        strategy=c.strategy,
        arena_a_edge_score=round(edge_score(c), 6),
        arena_b_absolute_profit=round(c.net_profit, 6),
        arena_b_risk_adjusted_score=round(risk_adjusted_profit_score(c), 6),
        prop_eligible=(c.max_dd_pct <= 5.0 and c.ruin_probability <= 0.01),
        broker_eligible=(c.max_dd_pct <= 20.0 and c.ruin_probability <= 0.05),
        max_dd_pct=c.max_dd_pct,
        profit_factor=c.profit_factor,
        return_pct=c.return_pct,
        final_equity=c.final_equity,
        trades=c.trades,
        source_file=c.source_file,
    )


def load_candidates(paths: Iterable[Path]) -> list[Candidate]:
    out: list[Candidate] = []
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = raw if isinstance(raw, list) else raw.get("candidates") if isinstance(raw, dict) and isinstance(raw.get("candidates"), list) else [raw]
        for row in rows:
            if not isinstance(row, dict):
                continue
            # Do not rank smoke manifests that do not contain performance metrics.
            metric_keys = {"final_equity", "return_pct", "profit_factor", "max_dd_pct", "trades"}
            if not metric_keys.intersection(row):
                continue
            out.append(Candidate.from_dict(row, str(path)))
    return out


def rank(scores: list[ArenaScore]) -> dict[str, list[dict[str, Any]]]:
    a = sorted(scores, key=lambda x: (x.arena_a_edge_score, x.return_pct), reverse=True)
    absolute = sorted(scores, key=lambda x: (x.arena_b_absolute_profit, -x.max_dd_pct), reverse=True)
    risk = sorted(scores, key=lambda x: (x.arena_b_risk_adjusted_score, x.return_pct), reverse=True)
    prop = sorted((x for x in scores if x.prop_eligible), key=lambda x: (x.arena_b_absolute_profit, x.arena_b_risk_adjusted_score), reverse=True)
    broker = sorted((x for x in scores if x.broker_eligible), key=lambda x: (x.arena_b_absolute_profit, x.arena_b_risk_adjusted_score), reverse=True)
    return {
        "arena_a_edge_discovery": [asdict(x) for x in a],
        "arena_b_absolute_profit": [asdict(x) for x in absolute],
        "arena_b_risk_adjusted": [asdict(x) for x in risk],
        "arena_b_prop_dd5": [asdict(x) for x in prop],
        "arena_b_broker_dd20": [asdict(x) for x in broker],
    }


def feedback_queue(scores: list[ArenaScore], top_n: int = 5) -> list[dict[str, Any]]:
    # Winners are sent back to Math EDGE Diagnostic for component extraction.
    winners: dict[str, ArenaScore] = {}
    for league in rank(scores).values():
        for row in league[:top_n]:
            s = next(x for x in scores if x.candidate_id == row["candidate_id"])
            winners[s.candidate_id] = s
    return [
        {
            "candidate_id": s.candidate_id,
            "agent": s.agent,
            "strategy": s.strategy,
            "action": "MATH_EDGE_DIAGNOSTIC",
            "decompose": ["direction", "timing", "exit", "cost", "regime", "tail", "sizing", "recovery"],
            "source_file": s.source_file,
        }
        for s in winners.values()
    ]
