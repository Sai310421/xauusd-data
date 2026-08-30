from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import numpy as np

EPS = 1e-12


@dataclass(slots=True)
class MarketStateVector:
    timestamp_ns: int = 0
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0
    spread: float = 0.0
    spread_z: float = 0.0

    # First Passage
    p_up_fp: float = 0.5
    p_down_fp: float = 0.5
    fp_edge: float = 0.0
    tau_up_exp: float = 0.0
    tau_down_exp: float = 0.0

    # Quote-event Hawkes. These are not labelled trade aggressor buys/sells.
    hawkes_lambda_up: float = 0.0
    hawkes_lambda_down: float = 0.0
    hawkes_imbalance: float = 0.0
    hawkes_branching_ratio: float = 0.0

    # Kalman
    kalman_price: float = 0.0
    kalman_velocity: float = 0.0
    kalman_acceleration: float = 0.0
    kalman_innovation_z: float = 0.0

    # Regime / change point
    regime_trend_prob: float = 0.5
    regime_meanrev_prob: float = 0.5
    regime_breakout_prob: float = 0.0
    p_change_point: float = 0.0

    # Vol / tail / information
    realized_vol: float = 0.0
    rough_h: float = 0.5
    jump_prob: float = 0.0
    cvar_95: float = 0.0
    entropy_norm: float = 0.0

    # VGRSI branch
    vgrsi_er: float = 50.0
    vgrsi_vg: float = 50.0
    vgrsi_mtf_bias: float = 0.0

    # Portfolio / recovery
    inventory: float = 0.0
    current_dd: float = 0.0
    recovery_debt: float = 0.0
    margin_level: float = 0.0

    # Unified policy outputs
    action_score_long: float = 0.0
    action_score_short: float = 0.0
    action_score_wait: float = 0.0
    action_score_reduce: float = 0.0
    action_score_reverse: float = 0.0
    action_score_hedge: float = 0.0


class FastKalmanCV:
    """Causal constant-velocity Kalman filter with dt-scaled process noise."""

    def __init__(self, accel_var: float = 1e-4, measurement_var: float = 1e-3):
        self.accel_var = max(float(accel_var), EPS)
        self.r = max(float(measurement_var), EPS)
        self.x = np.zeros(2, dtype=float)
        self.P = np.eye(2, dtype=float)
        self.initialized = False
        self.prev_v = 0.0

    def update(self, z: float, dt: float) -> tuple[float, float, float, float]:
        dt = max(float(dt), 1e-6)
        if not self.initialized:
            self.x[0] = float(z)
            self.initialized = True

        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
        g = np.array([0.5 * dt * dt, dt], dtype=float)
        Q = self.accel_var * np.outer(g, g)
        H = np.array([1.0, 0.0], dtype=float)

        xp = F @ self.x
        Pp = F @ self.P @ F.T + Q
        y = float(z) - float(H @ xp)
        S = float(H @ Pp @ H.T + self.r)
        K = (Pp @ H.T) / max(S, EPS)
        self.x = xp + K * y
        self.P = (np.eye(2) - np.outer(K, H)) @ Pp

        v = float(self.x[1])
        a = (v - self.prev_v) / dt
        self.prev_v = v
        innov_z = y / math.sqrt(max(S, EPS))
        return float(self.x[0]), v, float(a), float(innov_z)


class LocalFirstPassage:
    """Two-sided hitting probability for locally constant drift/volatility."""

    @staticmethod
    def p_up(x: float, lower: float, upper: float, mu: float, sigma: float) -> float:
        if upper <= lower:
            return 0.5
        if x <= lower:
            return 0.0
        if x >= upper:
            return 1.0

        width = upper - lower
        pos = x - lower
        var = max(float(sigma) ** 2, 1e-12)
        gamma = 2.0 * float(mu) / var

        if abs(gamma * width) < 1e-6:
            return float(np.clip(pos / width, 0.0, 1.0))

        # Stable evaluation of (1-exp(-gamma*pos))/(1-exp(-gamma*width)).
        if gamma > 0:
            num = -math.expm1(-min(gamma * pos, 700.0))
            den = -math.expm1(-min(gamma * width, 700.0))
            return float(np.clip(num / max(den, EPS), 0.0, 1.0))

        g = -gamma
        # Algebraically equivalent form that avoids exp(+large).
        a = min(g * (width - pos), 700.0)
        b = min(g * width, 700.0)
        num = math.exp(a) * (-math.expm1(-min(g * pos, 700.0)))
        den = -math.expm1(-b)
        return float(np.clip(num / max(den, EPS), 0.0, 1.0))


class ExponentialQuoteHawkes:
    """Lightweight quote-direction event intensity proxy for Raw Bid/Ask streams."""

    def __init__(self, baseline: float = 0.1, alpha: float = 0.25, beta: float = 1.0):
        self.baseline = max(float(baseline), 0.0)
        self.alpha = max(float(alpha), 0.0)
        self.beta = max(float(beta), EPS)
        self.lambda_up = self.baseline
        self.lambda_down = self.baseline
        self.last_t: float | None = None

    @property
    def branching_ratio(self) -> float:
        return self.alpha / self.beta

    def step(self, t_seconds: float, up_event: bool, down_event: bool) -> tuple[float, float, float]:
        t = float(t_seconds)
        dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
        self.last_t = t
        decay = math.exp(-self.beta * dt)
        self.lambda_up = self.baseline + (self.lambda_up - self.baseline) * decay
        self.lambda_down = self.baseline + (self.lambda_down - self.baseline) * decay
        if up_event:
            self.lambda_up += self.alpha
        if down_event:
            self.lambda_down += self.alpha
        den = self.lambda_up + self.lambda_down + EPS
        imbalance = (self.lambda_up - self.lambda_down) / den
        return self.lambda_up, self.lambda_down, float(imbalance)


class ParallelStateAssembler:
    """Merge independent branch outputs without coupling their calculations."""

    def __init__(self):
        self.state = MarketStateVector()

    def market(self, timestamp_ns: int, bid: float, ask: float, spread_z: float = 0.0) -> MarketStateVector:
        self.state.timestamp_ns = int(timestamp_ns)
        self.state.bid = float(bid)
        self.state.ask = float(ask)
        self.state.mid = 0.5 * (float(bid) + float(ask))
        self.state.spread = max(0.0, float(ask) - float(bid))
        self.state.spread_z = float(spread_z)
        return self.state

    def merge(self, branch: Dict[str, float]) -> MarketStateVector:
        for key, value in branch.items():
            if hasattr(self.state, key):
                setattr(self.state, key, float(value))
        self.state.p_down_fp = 1.0 - self.state.p_up_fp
        self.state.fp_edge = 2.0 * self.state.p_up_fp - 1.0
        return self.state
