from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class JumpTailState:
    window: int = 256
    jump_z: float = 3.0
    cvar_alpha: float = 0.95

    def __post_init__(self):
        self.returns = deque(maxlen=self.window)
        self.prev_price = None

    def update(self, price: float) -> dict:
        if self.prev_price is not None and self.prev_price > 0:
            self.returns.append(float(np.log(price / self.prev_price)))
        self.prev_price = price
        if len(self.returns) < 32:
            return {"jump_prob": 0.0, "jump_ratio": 0.0, "cvar": 0.0, "tail_index_proxy": 0.0}
        r = np.asarray(self.returns, dtype=float)
        sigma = float(np.std(r)) + 1e-12
        jump_mask = np.abs(r) > self.jump_z * sigma
        jump_prob = float(np.mean(jump_mask))
        rv = float(np.sum(r * r))
        bv = float((np.pi / 2.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))
        jv = max(rv - bv, 0.0)
        losses = -r[r < 0]
        if len(losses):
            q = float(np.quantile(losses, self.cvar_alpha))
            tail = losses[losses >= q]
            cvar = float(np.mean(tail)) if len(tail) else q
        else:
            cvar = 0.0
        absr = np.sort(np.abs(r))
        k = max(3, min(len(absr) // 10, 25))
        top = absr[-k:]
        base = max(absr[-k - 1], 1e-12) if len(absr) > k else max(top[0], 1e-12)
        hill = float(np.mean(np.log(np.maximum(top, 1e-12) / base))) if len(top) else 0.0
        return {"jump_prob": jump_prob, "jump_ratio": jv / max(rv, 1e-12), "cvar": cvar, "tail_index_proxy": hill}
