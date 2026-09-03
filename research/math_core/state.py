from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class UnifiedMathState:
    fp_edge: float = 0.0
    fp_p_up: float = 0.5
    kalman_velocity: float = 0.0
    kalman_innovation_z: float = 0.0
    regime_trend_prob: float = 0.25
    regime_meanrev_prob: float = 0.25
    regime_breakout_prob: float = 0.25
    regime_chaos_prob: float = 0.25
    change_prob: float = 0.0
    hawkes_imbalance: float = 0.0
    hawkes_branching: float = 0.0
    jump_prob: float = 0.0
    jump_ratio: float = 0.0
    cvar: float = 0.0
    hurst_proxy: float = 0.5
    roughness: float = 0.0
    spread: float = 0.0
    drawdown: float = 0.0
    recovery_debt: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)
