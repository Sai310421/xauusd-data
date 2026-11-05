"""Risk-governed replacement for a loss-amplifying recovery grid.

This is deliberately independent of any entry signal. A Nautilus strategy calls
this policy before each entry/add/exit so risk behavior is reproducible and testable.
"""
from __future__ import annotations

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
    breakeven_r: float = 1.0
    trail_start_r: float = 2.0
    max_hold_seconds: int = 45 * 60
    max_spread_r: float = 0.15
    daily_drawdown_limit: float = 0.05
    total_drawdown_limit: float = 0.15
    max_consecutive_losses: int = 2


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


class G1Policy:
    """A fail-closed controller: no grid-like add is permitted by default."""

    def __init__(self, config: G1Config = G1Config()) -> None:
        self.config = config

    def entry_action(self, state: BasketState) -> Action:
        if state.daily_drawdown >= self.config.daily_drawdown_limit:
            return Action.HALT
        if state.total_drawdown >= self.config.total_drawdown_limit:
            return Action.HALT
        if state.consecutive_losses >= self.config.max_consecutive_losses:
            return Action.HALT
        if state.spread_r > self.config.max_spread_r:
            return Action.HOLD
        return Action.OPEN if state.entry_quality >= 0.0 else Action.HOLD

    def manage_action(self, state: BasketState) -> Action:
        if state.daily_drawdown >= self.config.daily_drawdown_limit:
            return Action.HALT
        if state.total_drawdown >= self.config.total_drawdown_limit:
            return Action.HALT
        if state.unrealized_r <= -self.config.hard_stop_r:
            return Action.EXIT
        if state.elapsed_seconds >= self.config.max_hold_seconds:
            return Action.EXIT
        # Adds are limited, risk-capped, and require a renewed signal; they never
        # use a fixed geometric multiplier or an unconditional adverse-price rule.
        if (
            state.adds < self.config.max_adds
            and state.total_risk_r < self.config.max_total_risk_r
            and state.adverse_confirmation
            and state.entry_quality > 0.0
            and state.spread_r <= self.config.max_spread_r
        ):
            return Action.ADD
        return Action.HOLD
