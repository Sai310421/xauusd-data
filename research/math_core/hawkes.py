from __future__ import annotations

from dataclasses import dataclass
from math import exp


@dataclass
class BivariateExpHawkes:
    mu_up: float = 0.05
    mu_down: float = 0.05
    alpha_self: float = 0.30
    alpha_cross: float = 0.10
    beta: float = 1.50

    def __post_init__(self):
        self.l_up = self.mu_up
        self.l_down = self.mu_down
        self.last_ts = None

    def update(self, ts_seconds: float, event_side: int) -> dict:
        if self.last_ts is not None:
            decay = exp(-self.beta * max(0.0, ts_seconds - self.last_ts))
            self.l_up = self.mu_up + (self.l_up - self.mu_up) * decay
            self.l_down = self.mu_down + (self.l_down - self.mu_down) * decay
        self.last_ts = ts_seconds
        if event_side > 0:
            self.l_up += self.alpha_self
            self.l_down += self.alpha_cross
        elif event_side < 0:
            self.l_down += self.alpha_self
            self.l_up += self.alpha_cross
        total = self.l_up + self.l_down
        imbalance = (self.l_up - self.l_down) / max(total, 1e-12)
        branching = (self.alpha_self + self.alpha_cross) / max(self.beta, 1e-12)
        return {"lambda_up": self.l_up, "lambda_down": self.l_down, "imbalance": imbalance, "branching_ratio": branching}
