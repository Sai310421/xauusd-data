from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class RoughnessState:
    window: int = 256

    def __post_init__(self):
        self.returns = deque(maxlen=self.window)
        self.prev_price = None

    def update(self, price: float) -> dict:
        if self.prev_price is not None and self.prev_price > 0:
            self.returns.append(float(np.log(price / self.prev_price)))
        self.prev_price = price
        if len(self.returns) < 64:
            return {"hurst_proxy": 0.5, "roughness": 0.0, "vol": 0.0}
        r = np.asarray(self.returns, dtype=float)
        vol = float(np.std(r)) + 1e-12
        scales = np.array([1, 2, 4, 8, 16], dtype=int)
        vars_ = []
        good_scales = []
        for s in scales:
            n = len(r) // s
            if n < 4:
                continue
            agg = r[: n * s].reshape(n, s).sum(axis=1)
            v = float(np.var(agg))
            if v > 0:
                vars_.append(v)
                good_scales.append(s)
        if len(vars_) < 2:
            h = 0.5
        else:
            slope = float(np.polyfit(np.log(good_scales), np.log(vars_), 1)[0])
            h = min(0.99, max(0.01, slope / 2.0))
        return {"hurst_proxy": h, "roughness": max(0.0, 0.5 - h), "vol": vol}
