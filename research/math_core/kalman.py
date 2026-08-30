from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Kalman1DDrift:
    dt: float = 1.0
    q_price: float = 1e-4
    q_velocity: float = 1e-5
    r_obs: float = 1e-3

    def __post_init__(self):
        self.x = np.zeros(2, dtype=float)
        self.P = np.eye(2, dtype=float)
        self.initialized = False

    def update(self, price: float) -> dict:
        if not self.initialized:
            self.x[:] = (price, 0.0)
            self.initialized = True
            return {"price": price, "velocity": 0.0, "innovation": 0.0, "innovation_z": 0.0, "confidence": 0.5}
        F = np.array([[1.0, self.dt], [0.0, 1.0]], dtype=float)
        H = np.array([[1.0, 0.0]], dtype=float)
        Q = np.diag([self.q_price, self.q_velocity])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        y = float(price - (H @ self.x)[0])
        S = float((H @ self.P @ H.T)[0, 0] + self.r_obs)
        K = (self.P @ H.T) / S
        self.x = self.x + K[:, 0] * y
        self.P = (np.eye(2) - K @ H) @ self.P
        conf = 1.0 / (1.0 + float(np.trace(self.P)))
        return {"price": float(self.x[0]), "velocity": float(self.x[1]), "innovation": y, "innovation_z": y / max(S ** 0.5, 1e-12), "confidence": conf}
