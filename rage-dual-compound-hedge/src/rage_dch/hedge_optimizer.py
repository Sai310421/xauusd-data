"""Adapter-facing hedge optimizer with no broker or order-routing responsibility."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .h1_signal_adapter import SignalStateSynchronizer
from .protective_hedge import HedgeIntensity, ProtectiveHedgeParams, compute_hedge_intensity


@dataclass(frozen=True)
class HedgeDecision:
    """One auditable non-execution hedge recommendation."""

    signal_id: str
    model_version: str
    macro_crisis_score: float
    hedge_execution_risk: float
    intensity: HedgeIntensity


class HedgeOptimizer:
    """Combine a held H1 macro state with independently measured execution risk."""

    def __init__(
        self,
        synchronizer: SignalStateSynchronizer,
        params: ProtectiveHedgeParams,
    ) -> None:
        self._synchronizer = synchronizer
        self._params = params

    def decide(
        self,
        now: datetime,
        base_hedge_ratio: float,
        hedge_execution_risk: float,
    ) -> HedgeDecision:
        """Return a recommendation only; the caller owns all order submission."""
        signal = self._synchronizer.state_for_tick(now)
        macro_score = self._synchronizer.held_macro_score
        intensity = compute_hedge_intensity(
            base_hedge_ratio=base_hedge_ratio,
            macro_crisis_score=macro_score,
            hedge_execution_risk=hedge_execution_risk,
            params=self._params,
        )
        return HedgeDecision(
            signal_id=signal.signal_id,
            model_version=signal.model_version,
            macro_crisis_score=macro_score,
            hedge_execution_risk=hedge_execution_risk,
            intensity=intensity,
        )
