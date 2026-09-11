from __future__ import annotations

"""AE disturbance-affine DR-MPC recovery proxy (DERIVED).

This is a low-dimensional, auditable precursor to a full Wasserstein min-max MPC.
It maps recovery state and realized disturbance estimates to bounded control actions.
It does not create directional trading edge and must be A/B tested against the
existing reflected recovery controller.
"""

from dataclasses import dataclass
from typing import Iterable


def _clip(x: float, lo: float, hi: float) -> float:
    return min(max(float(x), lo), hi)


@dataclass(frozen=True)
class RecoveryState:
    debt_ratio: float
    debt_drift_ratio: float
    mae_ratio: float
    recovery_age_ratio: float
    tail_probability: float
    spread_stress: float
    volatility_stress: float
    dd_ratio: float

    def vector(self) -> tuple[float, ...]:
        return tuple(_clip(v, 0.0, 1.5) for v in (
            self.debt_ratio, self.debt_drift_ratio, self.mae_ratio,
            self.recovery_age_ratio, self.tail_probability, self.spread_stress,
            self.volatility_stress, self.dd_ratio,
        ))


@dataclass(frozen=True)
class Disturbance:
    price_shock: float
    spread_shock: float
    volatility_shock: float
    recovery_forecast_error: float

    def vector(self) -> tuple[float, ...]:
        return tuple(_clip(v, -2.0, 2.0) for v in (
            self.price_shock, self.spread_shock,
            self.volatility_shock, self.recovery_forecast_error,
        ))


@dataclass(frozen=True)
class RecoveryControl:
    add_scale: float
    reduce_fraction: float
    hedge_scale: float
    recovery_scale: float
    robust_score: float


@dataclass(frozen=True)
class DRMPConfig:
    ambiguity_radius: float = 0.15
    shock_penalty: float = 0.35
    state_penalty: float = 0.55
    add_floor: float = 0.0
    hedge_cap: float = 1.0
    reduce_cap: float = 0.75
    recovery_cap: float = 1.0


def _mean(xs: Iterable[float]) -> float:
    vals = tuple(float(x) for x in xs)
    return sum(vals) / max(len(vals), 1)


def robust_state_score(state: RecoveryState, disturbance: Disturbance, cfg: DRMPConfig | None = None) -> float:
    """AE DERIVED distributionally-robust proxy score in [0,1].

    The ambiguity radius inflates observed disturbance magnitude rather than solving
    the full Wasserstein dual problem. This is deliberately labeled PROXY mathematics.
    """
    cfg = cfg or DRMPConfig()
    x = state.vector()
    w = disturbance.vector()
    state_cost = _mean(x)
    shock_mag = _mean(abs(v) for v in w)
    robust_shock = shock_mag + max(cfg.ambiguity_radius, 0.0)
    score = cfg.state_penalty * state_cost + cfg.shock_penalty * robust_shock
    return _clip(score, 0.0, 1.0)


def disturbance_affine_control(state: RecoveryState, disturbance: Disturbance, cfg: DRMPConfig | None = None) -> RecoveryControl:
    """Causal affine-style recovery action proxy.

    Positive shock / stress suppresses adds and increases reduction / hedge intensity.
    Positive recovery forecast error means realized recovery is worse than expected and
    raises intervention intensity.
    """
    cfg = cfg or DRMPConfig()
    score = robust_state_score(state, disturbance, cfg)
    w = disturbance.vector()
    shock = _clip(0.35*w[0] + 0.25*w[1] + 0.25*w[2] + 0.15*w[3], -1.0, 1.0)

    add_scale = _clip(1.0 - score - 0.25*max(shock, 0.0), cfg.add_floor, 1.0)
    reduce_fraction = _clip((score - 0.30) / 0.70 + 0.15*max(shock, 0.0), 0.0, cfg.reduce_cap)
    hedge_scale = _clip((score - 0.50) / 0.50 + 0.20*max(shock, 0.0), 0.0, cfg.hedge_cap)
    recovery_scale = _clip(0.25 + 0.75*score + 0.10*max(w[3], 0.0), 0.0, cfg.recovery_cap)

    return RecoveryControl(
        add_scale=add_scale,
        reduce_fraction=reduce_fraction,
        hedge_scale=hedge_scale,
        recovery_scale=recovery_scale,
        robust_score=score,
    )
