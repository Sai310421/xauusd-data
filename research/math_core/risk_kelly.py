from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RiskConstrainedKelly:
    fraction: float = 0.25
    max_fraction: float = 0.03
    dd_limit: float = 0.05

    def binary_kelly(self, win_prob: float, payoff_b: float) -> float:
        if payoff_b <= 0:
            return 0.0
        q = 1.0 - win_prob
        f = (payoff_b * win_prob - q) / payoff_b
        return max(0.0, f)

    def drawdown_gate(self, current_dd: float) -> float:
        if self.dd_limit <= 0:
            return 0.0
        return float(np.clip(1.0 - current_dd / self.dd_limit, 0.0, 1.0))

    def size_fraction(self, win_prob: float, payoff_b: float, current_dd: float, regime_gate: float = 1.0, tail_gate: float = 1.0, liquidity_gate: float = 1.0) -> float:
        raw = self.binary_kelly(win_prob, payoff_b) * self.fraction
        gated = raw * self.drawdown_gate(current_dd) * regime_gate * tail_gate * liquidity_gate
        return float(np.clip(gated, 0.0, self.max_fraction))
