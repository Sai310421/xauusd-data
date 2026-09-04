from __future__ import annotations

"""AE high-dimensional reflected/singular recovery controller (DERIVED).

This module converts the Math EDGE formulation into deterministic BOT logic:

    state -> hazard/recovery value -> reflected boundary -> action

IMPORTANT
---------
- This is an AE-derived implementation, not a verbatim source-paper algorithm.
- It is a controller candidate for A/B testing, not evidence of trading edge.
- Inputs should come from Raw Bid/Ask QuoteTick-derived state in canonical tests.
"""

from dataclasses import dataclass
from enum import Enum
import math


class AEAction(str, Enum):
    WAIT = "WAIT"
    STOP_ADD = "STOP_ADD"
    REDUCE = "REDUCE"
    HEDGE_LOCK = "HEDGE_LOCK"
    SELECTIVE_RECOVERY = "SELECTIVE_RECOVERY"
    EMERGENCY_IMPULSE = "EMERGENCY_IMPULSE"


@dataclass(frozen=True)
class AEState:
    # Economic/recovery state
    debt: float
    debt_limit: float
    debt_drift: float
    debt_drift_limit: float
    mae: float
    mae_limit: float
    recovery_age_s: float
    recovery_age_limit_s: float

    # Market/execution state
    tail_probability: float
    spread_stress: float
    volatility_stress: float
    shock_score: float

    # Account safety state
    drawdown_pct: float
    margin_level_pct: float

    # Estimated probability of Economic-BE first passage inside chosen horizon
    natural_recovery_probability: float


@dataclass(frozen=True)
class ReflectedRecoveryConfig:
    # Hazard weights. Defaults sum to 1.0.
    w_debt: float = 0.22
    w_debt_drift: float = 0.18
    w_mae: float = 0.16
    w_recovery_age: float = 0.12
    w_tail: float = 0.18
    w_spread: float = 0.07
    w_volatility: float = 0.07

    # Natural-recovery credit: larger p_NR delays intervention.
    recovery_credit: float = 0.35

    # Reflected action boundaries for adjusted hazard score.
    theta_stop_add: float = 0.25
    theta_reduce: float = 0.45
    theta_hedge: float = 0.65
    theta_recovery: float = 0.82

    # Hard safety overrides.
    max_drawdown_pct: float = 5.0
    min_margin_level_pct: float = 500.0
    shock_threshold: float = 1.0

    # Maximum fraction reduced at a reflected boundary touch.
    max_reduce_fraction: float = 0.50


@dataclass(frozen=True)
class ControlDecision:
    action: AEAction
    hazard: float
    adjusted_hazard: float
    natural_recovery_probability: float
    intervention_fraction: float
    reason: str


def _clip01(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)


def _ratio(value: float, limit: float) -> float:
    if limit <= 0:
        raise ValueError("normalization limit must be positive")
    return _clip01(abs(float(value)) / float(limit))


def recovery_hazard(state: AEState, cfg: ReflectedRecoveryConfig) -> float:
    """DERIVED path-state hazard proxy H_t in [0, 1].

    H_t = sum_i w_i * normalized_state_i.

    This is intentionally explicit and auditable. It is the first low-dimensional
    implementation step before any learned high-dimensional value-gradient model.
    """
    weights = (
        cfg.w_debt,
        cfg.w_debt_drift,
        cfg.w_mae,
        cfg.w_recovery_age,
        cfg.w_tail,
        cfg.w_spread,
        cfg.w_volatility,
    )
    if any(w < 0 for w in weights):
        raise ValueError("hazard weights must be non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("hazard weights must have positive sum")

    components = (
        _ratio(state.debt, state.debt_limit),
        _ratio(max(state.debt_drift, 0.0), state.debt_drift_limit),
        _ratio(state.mae, state.mae_limit),
        _ratio(state.recovery_age_s, state.recovery_age_limit_s),
        _clip01(state.tail_probability),
        _clip01(state.spread_stress),
        _clip01(state.volatility_stress),
    )
    return float(sum(w * x for w, x in zip(weights, components)) / total)


def adjusted_recovery_hazard(state: AEState, cfg: ReflectedRecoveryConfig) -> tuple[float, float]:
    """Return (raw hazard H_t, recovery-adjusted score S_t).

    S_t = clip(H_t - lambda * p_NR, 0, 1)

    A high Economic-BE first-passage probability therefore expands the no-action
    region; low recovery probability makes the controller intervene earlier.
    """
    h = recovery_hazard(state, cfg)
    p_nr = _clip01(state.natural_recovery_probability)
    s = _clip01(h - cfg.recovery_credit * p_nr)
    return h, s


def _validate_boundaries(cfg: ReflectedRecoveryConfig) -> None:
    xs = (cfg.theta_stop_add, cfg.theta_reduce, cfg.theta_hedge, cfg.theta_recovery)
    if not all(0.0 <= x <= 1.0 for x in xs):
        raise ValueError("all thresholds must be inside [0,1]")
    if not (xs[0] < xs[1] < xs[2] < xs[3]):
        raise ValueError("thresholds must satisfy stop_add < reduce < hedge < recovery")


def reflected_reduce_fraction(score: float, cfg: ReflectedRecoveryConfig) -> float:
    """Minimum proportional intervention after REDUCE boundary contact.

    The fraction grows continuously from zero at theta_reduce to
    max_reduce_fraction at theta_hedge. This approximates a reflected/singular
    controller instead of making a full discrete liquidation immediately.
    """
    if cfg.max_reduce_fraction < 0 or cfg.max_reduce_fraction > 1:
        raise ValueError("max_reduce_fraction must be in [0,1]")
    if score <= cfg.theta_reduce:
        return 0.0
    span = cfg.theta_hedge - cfg.theta_reduce
    if span <= 0:
        return cfg.max_reduce_fraction
    u = _clip01((score - cfg.theta_reduce) / span)
    return float(cfg.max_reduce_fraction * u)


def decide_reflected_recovery(state: AEState, cfg: ReflectedRecoveryConfig | None = None) -> ControlDecision:
    """Convert state into an auditable BOT action.

    Hard safety overrides are checked first. Otherwise action is selected by the
    recovery-adjusted reflected boundaries.
    """
    cfg = cfg or ReflectedRecoveryConfig()
    _validate_boundaries(cfg)

    h, s = adjusted_recovery_hazard(state, cfg)
    p_nr = _clip01(state.natural_recovery_probability)

    # Impulse/safety layer overrides normal reflected control.
    if state.shock_score >= cfg.shock_threshold:
        return ControlDecision(
            AEAction.EMERGENCY_IMPULSE, h, s, p_nr, 1.0,
            "shock score breached impulse threshold",
        )
    if state.drawdown_pct >= cfg.max_drawdown_pct:
        return ControlDecision(
            AEAction.EMERGENCY_IMPULSE, h, s, p_nr, 1.0,
            "hard drawdown safety boundary breached",
        )
    if state.margin_level_pct <= cfg.min_margin_level_pct:
        return ControlDecision(
            AEAction.EMERGENCY_IMPULSE, h, s, p_nr, 1.0,
            "hard margin safety boundary breached",
        )

    if s < cfg.theta_stop_add:
        return ControlDecision(AEAction.WAIT, h, s, p_nr, 0.0, "inside no-action region")
    if s < cfg.theta_reduce:
        return ControlDecision(AEAction.STOP_ADD, h, s, p_nr, 0.0, "first reflected boundary reached")
    if s < cfg.theta_hedge:
        fraction = reflected_reduce_fraction(s, cfg)
        return ControlDecision(AEAction.REDUCE, h, s, p_nr, fraction, "reduce boundary reached; minimal proportional intervention")
    if s < cfg.theta_recovery:
        return ControlDecision(AEAction.HEDGE_LOCK, h, s, p_nr, 1.0, "hedge-lock boundary reached")
    return ControlDecision(AEAction.SELECTIVE_RECOVERY, h, s, p_nr, 1.0, "tail/recovery boundary reached")


def soft_action_value(score: float, threshold: float, temperature: float = 0.05) -> float:
    """Optional smooth boundary gate in [0,1] for later optimization experiments."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = (float(score) - float(threshold)) / temperature
    z = min(max(z, -60.0), 60.0)
    return float(1.0 / (1.0 + math.exp(-z)))
