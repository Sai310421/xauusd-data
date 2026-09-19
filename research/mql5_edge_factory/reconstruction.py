from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class MatchScore:
    n_reference: int
    direction: float
    entry_timing: float
    exit_timing: float
    add_count: float
    composite: float

    def to_dict(self):
        return asdict(self)


def _near(a: float, b: float, tolerance: float) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def score(reference: list[dict], model: list[dict], entry_tolerance_s: float = 5.0, exit_tolerance_s: float = 10.0) -> MatchScore:
    n = min(len(reference), len(model))
    if n == 0:
        return MatchScore(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    d = et = xt = adds = 0
    for r, m in zip(reference[:n], model[:n]):
        d += str(r.get("side", "")).lower() == str(m.get("side", "")).lower()
        et += _near(r.get("entry_ts", 0), m.get("entry_ts", 0), entry_tolerance_s)
        xt += _near(r.get("exit_ts", 0), m.get("exit_ts", 0), exit_tolerance_s)
        adds += int(r.get("add_count", 0)) == int(m.get("add_count", 0))
    vals = [d/n, et/n, xt/n, adds/n]
    return MatchScore(n, vals[0], vals[1], vals[2], vals[3], sum(vals)/len(vals))
