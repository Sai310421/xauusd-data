from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    WAIT = "WAIT"
    ENTRY_LONG = "ENTRY_LONG"
    ENTRY_SHORT = "ENTRY_SHORT"
    ADD = "ADD"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    REVERSE = "REVERSE"
    HEDGE = "HEDGE"
    NATURAL_RECOVERY = "NATURAL_RECOVERY"
    BE_RECOVERY = "BE_RECOVERY"
    EXIT = "EXIT"


@dataclass
class ActionPolicy:
    lambda_dd: float = 1.0
    lambda_tail: float = 1.0
    lambda_cost: float = 1.0
    lambda_ruin: float = 1.0

    def q(self, expected_pnl: float, expected_dd: float, tail_risk: float, cost: float, ruin_risk: float) -> float:
        return (
            expected_pnl
            - self.lambda_dd * expected_dd
            - self.lambda_tail * tail_risk
            - self.lambda_cost * cost
            - self.lambda_ruin * ruin_risk
        )

    def choose(self, candidates: dict[Action, dict]) -> tuple[Action, dict[Action, float]]:
        scores = {a: self.q(**v) for a, v in candidates.items()}
        best = max(scores, key=scores.get)
        return best, scores
