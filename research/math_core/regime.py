from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class OnlineRegimeState:
    window: int = 64

    def __post_init__(self):
        self.returns = deque(maxlen=self.window)
        self.prev_price = None

    def update(self, price: float) -> dict:
        if self.prev_price is not None and self.prev_price > 0:
            self.returns.append(float(np.log(price / self.prev_price)))
        self.prev_price = price
        if len(self.returns) < max(12, self.window // 4):
            return {"trend_prob": 0.25, "meanrev_prob": 0.25, "breakout_prob": 0.25, "chaos_prob": 0.25, "change_prob": 0.0}
        r = np.asarray(self.returns, dtype=float)
        vol = float(np.std(r)) + 1e-12
        mean = float(np.mean(r))
        ac1 = float(np.corrcoef(r[1:], r[:-1])[0,1]) if len(r) > 3 and np.std(r[1:]) > 0 and np.std(r[:-1]) > 0 else 0.0
        z = abs(mean) / vol
        tail = float(np.mean(np.abs(r) > 2.5 * vol))
        recent = float(np.std(r[-max(8, len(r)//4):]))
        old = float(np.std(r[:-max(8, len(r)//4)])) if len(r) > 16 else recent
        cp = min(1.0, abs(recent - old) / (old + 1e-12))
        trend = min(1.0, 0.5 * z + max(ac1, 0.0))
        meanrev = min(1.0, max(-ac1, 0.0) + 0.25 * (1.0 - min(z, 1.0)))
        breakout = min(1.0, 2.0 * tail + cp)
        chaos = min(1.0, cp + tail)
        raw = np.array([trend, meanrev, breakout, chaos], dtype=float) + 1e-6
        raw /= raw.sum()
        return {"trend_prob": float(raw[0]), "meanrev_prob": float(raw[1]), "breakout_prob": float(raw[2]), "chaos_prob": float(raw[3]), "change_prob": cp}
