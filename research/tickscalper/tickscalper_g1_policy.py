"""Risk-governed replacement for a loss-amplifying recovery grid.

The controller is deliberately independent of any entry signal. It incorporates
only *risk-shrinking* AE Math controls:
- Dynamic Drawdown Reserve (DDR)
- drawdown velocity/acceleration throttle
- First-Passage recovery-probability gate

Every mathematical term returns a multiplier in [0, 1]. No term may increase
a lot, layer count, or exposure beyond the base G1 decision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    OPEN = "OPEN"
    ADD = "ADD"
    HOLD = "HOLD"
    EXIT = "EXIT"
    HALT = "HALT"


@dataclass(frozen=True)
class G1Config:
    max_adds: int = 2
    max_total_risk_r: float = 2.0
    hard_stop_r: float = 1.0
    max_hold_seconds: int = 45 * 60
    max_spread_r: float = 0.15
    daily_drawdown_limit: float = 0.05
    total_drawdown_limit: float = 0.15
    max_consecutive_losses: int = 2
    # AE Math supervisor: controls can only reduce exposure.
    ddr_cap: float = 0.15
    ddr_gamma: float = 2.0
    dd_velocity_lambda: float = 2.5
    dd_acceleration_lambda: float = 1.0
    min_first_passage_recovery_probability: float = 0.55


@dataclass(frozen=True)
class BasketState:
    side: str
    adds: int
    total_risk_r: float
    unrealized_r: float
    elapsed_seconds: int
    spread_r: float
    daily_drawdown: float
    total_drawdown: float
    consecutive_losses: int
    entry_quality: float
    adverse_confirmation: bool
    # Inputs come from a separately validated raw-tick estimator. None means
    # unknown: G1 fails closed for an ADD but does not invent a probability.
    drawdown_velocity: float = 0.0
    drawdown_acceleration: float = 0.0
    first_passage_recovery_probability: float | None = None


class G1Policy:
    """Fail-closed controller; no unconditional geometric recovery is allowed."""

    def __init__(self, config: G1Config = G1Config()) -> None:
        self.config = config

    def _hard_halt(self, state: BasketState) -> bool:
        return (
            state.daily_drawdown >= self.config.daily_drawdown_limit
            or state.total_drawdown >= self.config.total_drawdown_limit
            or state.consecutive_losses >= self.config.max_consecutive_losses
        )

    def ae_math_multiplier(self, state: BasketState) -> float:
        """Return an exposure multiplier in [0, 1], never an amplifier."""
        if self._hard_halt(state) or state.spread_r > self.config.max_spread_r:
            return 0.0
        if state.first_passage_recovery_probability is None:
            return 0.0
        if state.first_passage_recovery_probability < self.config.min_first_passage_recovery_probability:
            return 0.0
        # Dynamic Drawdown Reserve: capacity falls continuously to zero at cap.
        d = max(0.0, state.total_drawdown)
        reserve = max(0.0, 1.0 - d / self.config.ddr_cap) ** self.config.ddr_gamma
        # A rapidly worsening DD never receives larger exposure.
        velocity_guard = math.exp(
            -self.config.dd_velocity_lambda * max(0.0, state.drawdown_velocity)
            -self.config.dd_acceleration_lambda * max(0.0, state.drawdown_acceleration)
        )
        return max(0.0, min(1.0, reserve, velocity_guard))

    def entry_action(self, state: BasketState) -> Action:
        if self._hard_halt(state):
            return Action.HALT
        if state.spread_r > self.config.max_spread_r:
            return Action.HOLD
        return Action.OPEN if state.entry_quality >= 0.0 else Action.HOLD

    def manage_action(self, state: BasketState) -> Action:
        if self._hard_halt(state):
            return Action.HALT
        if state.unrealized_r <= -self.config.hard_stop_r:
            return Action.EXIT
        if state.elapsed_seconds >= self.config.max_hold_seconds:
            return Action.EXIT
        # Adds are capped, signal-confirmed, and subject to AE Math shrinkage.
        if (
            state.adds < self.config.max_adds
            and state.total_risk_r < self.config.max_total_risk_r
            and state.adverse_confirmation
            and state.entry_quality > 0.0
            and self.ae_math_multiplier(state) > 0.0
        ):
            return Action.ADD
        return Action.HOLD
