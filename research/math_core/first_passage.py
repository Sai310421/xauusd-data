from __future__ import annotations

from dataclasses import dataclass
from math import exp

EPS = 1e-12


def brownian_upper_hit_probability(x: float, lower: float, upper: float, mu: float, sigma: float) -> float:
    if not lower < x < upper:
        return 1.0 if x >= upper else 0.0
    span = upper - lower
    if sigma <= EPS:
        return 1.0 if mu > 0 else 0.0 if mu < 0 else (x - lower) / span
    if abs(mu) <= EPS:
        return (x - lower) / span
    a = -2.0 * mu / (sigma * sigma)
    num = 1.0 - exp(a * (x - lower))
    den = 1.0 - exp(a * span)
    if abs(den) <= EPS:
        return (x - lower) / span
    return min(1.0, max(0.0, num / den))


@dataclass(frozen=True)
class FirstPassageState:
    p_up: float
    p_down: float
    edge: float
    upper: float
    lower: float

    @classmethod
    def from_state(cls, x: float, up_dist: float, down_dist: float, mu: float, sigma: float) -> "FirstPassageState":
        upper = x + max(up_dist, EPS)
        lower = x - max(down_dist, EPS)
        p_up = brownian_upper_hit_probability(x, lower, upper, mu, sigma)
        return cls(p_up=p_up, p_down=1.0 - p_up, edge=2.0 * p_up - 1.0, upper=upper, lower=lower)
