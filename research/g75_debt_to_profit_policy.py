from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable


class RecoveryMode(str, Enum):
    RECOVER = "RECOVER"
    BE_PROTECT = "BE_PROTECT"
    PROFIT_EXTEND = "PROFIT_EXTEND"
    HOLD = "HOLD"
    RISK_CONTROL = "RISK_CONTROL"


@dataclass(frozen=True)
class RecoveryProbabilities:
    p300: float
    p600: float
    p1800: float
    p3600: float
    p6h: float
    p12h: float
    p24h: float

    def validate(self) -> None:
        xs = [self.p300, self.p600, self.p1800, self.p3600, self.p6h, self.p12h, self.p24h]
        if any(x < 0.0 or x > 1.0 for x in xs):
            raise ValueError("recovery probabilities must be in [0, 1]")
        if xs != sorted(xs):
            raise ValueError("recovery probabilities must be non-decreasing by horizon")


@dataclass(frozen=True)
class DebtState:
    initial_debt: float
    remaining_debt: float
    realized_rescue_profit: float
    accumulated_costs: float
    floating_dd_pct: float
    margin_level_pct: float
    basket_age_s: float

    def validate(self) -> None:
        if self.initial_debt < 0 or self.remaining_debt < 0:
            raise ValueError("debt cannot be negative")
        if self.accumulated_costs < 0:
            raise ValueError("costs cannot be negative")
        if self.floating_dd_pct < 0:
            raise ValueError("floating DD cannot be negative")
        if self.margin_level_pct < 0:
            raise ValueError("margin level cannot be negative")
        if self.basket_age_s < 0:
            raise ValueError("basket age cannot be negative")


@dataclass(frozen=True)
class RescueBudget:
    slots: int
    total_risk_budget: float
    used_risk_budget: float = 0.0

    def validate(self) -> None:
        if self.slots not in (3, 5, 10):
            raise ValueError("slots must be one of 3, 5, 10")
        if self.total_risk_budget < 0 or self.used_risk_budget < 0:
            raise ValueError("risk budget cannot be negative")
        if self.used_risk_budget > self.total_risk_budget:
            raise ValueError("used risk cannot exceed total risk budget")

    @property
    def remaining_risk_budget(self) -> float:
        return self.total_risk_budget - self.used_risk_budget

    @property
    def per_slot_risk_if_equal(self) -> float:
        return self.remaining_risk_budget / self.slots


@dataclass(frozen=True)
class PolicyInputs:
    debt: DebtState
    probabilities: RecoveryProbabilities
    budget: RescueBudget
    predicted_mfe: float
    required_move_to_be: float
    rescue_expected_net_profit_per_trade: float
    rescue_profit_velocity_per_hour: float
    min_margin_level_pct: float = 500.0
    max_floating_dd_pct: float = 15.0


@dataclass(frozen=True)
class PolicyDecision:
    mode: RecoveryMode
    rescue_enabled: bool
    rescue_slots: int
    max_risk_per_next_slot: float
    required_profit_per_remaining_slot: float
    coverage_ratio: float
    reason: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mode"] = self.mode.value
        return d


def required_profit_per_slot(remaining_debt: float, slots: int, cost_buffer: float = 0.0) -> float:
    if slots <= 0:
        raise ValueError("slots must be positive")
    return max(0.0, remaining_debt + cost_buffer) / slots


def choose_mode(x: PolicyInputs) -> PolicyDecision:
    x.debt.validate()
    x.probabilities.validate()
    x.budget.validate()

    if x.debt.margin_level_pct < x.min_margin_level_pct or x.debt.floating_dd_pct > x.max_floating_dd_pct:
        return PolicyDecision(
            mode=RecoveryMode.RISK_CONTROL,
            rescue_enabled=False,
            rescue_slots=x.budget.slots,
            max_risk_per_next_slot=0.0,
            required_profit_per_remaining_slot=required_profit_per_slot(
                x.debt.remaining_debt, x.budget.slots, x.debt.accumulated_costs
            ),
            coverage_ratio=0.0,
            reason="margin_or_dd_guard",
        )

    if x.debt.remaining_debt <= 0.0:
        return PolicyDecision(
            mode=RecoveryMode.PROFIT_EXTEND,
            rescue_enabled=x.rescue_expected_net_profit_per_trade > 0.0,
            rescue_slots=x.budget.slots,
            max_risk_per_next_slot=x.budget.per_slot_risk_if_equal,
            required_profit_per_remaining_slot=0.0,
            coverage_ratio=float("inf") if x.required_move_to_be <= 0 else x.predicted_mfe / x.required_move_to_be,
            reason="economic_be_reached",
        )

    coverage = (
        float("inf")
        if x.required_move_to_be <= 0
        else max(0.0, x.predicted_mfe) / x.required_move_to_be
    )
    required = required_profit_per_slot(
        x.debt.remaining_debt,
        x.budget.slots,
        x.debt.accumulated_costs,
    )

    # If natural recovery remains highly probable within the next practical horizon,
    # avoid increasing rescue intensity. The threshold is deliberately explicit and
    # deterministic so it can be calibrated from Raw Bid/Ask evidence later.
    if x.debt.basket_age_s <= 3600 and x.probabilities.p3600 >= 0.90:
        return PolicyDecision(
            mode=RecoveryMode.HOLD,
            rescue_enabled=False,
            rescue_slots=x.budget.slots,
            max_risk_per_next_slot=0.0,
            required_profit_per_remaining_slot=required,
            coverage_ratio=coverage,
            reason="high_natural_recovery_probability_to_1h",
        )

    if x.debt.basket_age_s <= 21600 and x.probabilities.p6h >= 0.95 and x.rescue_profit_velocity_per_hour <= 0:
        return PolicyDecision(
            mode=RecoveryMode.BE_PROTECT,
            rescue_enabled=False,
            rescue_slots=x.budget.slots,
            max_risk_per_next_slot=0.0,
            required_profit_per_remaining_slot=required,
            coverage_ratio=coverage,
            reason="high_natural_recovery_probability_to_6h_no_positive_rescue_velocity",
        )

    rescue_ok = (
        x.rescue_expected_net_profit_per_trade > 0.0
        and x.rescue_profit_velocity_per_hour > 0.0
        and x.budget.remaining_risk_budget > 0.0
        and coverage >= 1.0
    )

    return PolicyDecision(
        mode=RecoveryMode.RECOVER if rescue_ok else RecoveryMode.BE_PROTECT,
        rescue_enabled=rescue_ok,
        rescue_slots=x.budget.slots,
        max_risk_per_next_slot=x.budget.per_slot_risk_if_equal if rescue_ok else 0.0,
        required_profit_per_remaining_slot=required,
        coverage_ratio=coverage,
        reason=(
            "positive_rescue_expectancy_and_coverage"
            if rescue_ok
            else "insufficient_rescue_expectancy_or_coverage"
        ),
    )


def compare_slot_budgets(total_risk_budget: float, slots: Iterable[int] = (3, 5, 10)) -> list[dict]:
    out = []
    for n in slots:
        b = RescueBudget(slots=n, total_risk_budget=total_risk_budget)
        b.validate()
        out.append(
            {
                "slots": n,
                "total_risk_budget": total_risk_budget,
                "per_slot_risk_if_equal": b.per_slot_risk_if_equal,
            }
        )
    return out
