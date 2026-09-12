from __future__ import annotations

"""Final MultiEDGE architecture scaffold.

Goal: operate 8-12 independent EDGE bots and let a Supervisor select/expose
only the bots appropriate for the detected market regime.

This module intentionally separates:
1) EDGE validity/selection metrics
2) independent BOT signal generation
3) regime routing
4) portfolio/risk arbitration
5) exit model

No EDGE is promoted unless PF >= 1.20 on Raw BidAsk validation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Regime(str, Enum):
    TREND = "TREND"
    RANGE_FOLLOW = "RANGE_FOLLOW"
    REVERSAL = "REVERSAL"
    UNKNOWN = "UNKNOWN"


class EdgeStatus(str, Enum):
    REJECT = "REJECT"
    HOLD = "HOLD"
    CANDIDATE = "CANDIDATE"
    PROMOTE = "PROMOTE"
    STRONG = "STRONG"


@dataclass(frozen=True)
class EdgeMetrics:
    pf: float
    expectancy: float
    n: int
    frequency_per_day: float
    stability: float          # 0..1, time-split / regime-split consistency
    cost_survival: float      # 0..1, survives spread/slippage stress
    independence: float       # 0..1, low redundancy vs other promoted EDGE
    mfe_mae_ratio: float

    def status(self) -> EdgeStatus:
        if self.pf < 1.0 or self.expectancy <= 0:
            return EdgeStatus.REJECT
        if self.pf < 1.20:
            return EdgeStatus.HOLD
        if self.pf < 1.50:
            return EdgeStatus.CANDIDATE
        if self.pf < 2.00:
            return EdgeStatus.PROMOTE
        return EdgeStatus.STRONG

    def utility(self) -> float:
        """Final selection utility; PF gate remains separate and mandatory."""
        ev = max(self.expectancy, 0.0)
        freq_term = max(self.frequency_per_day, 0.0) ** 0.5
        return ev * freq_term * self.stability * self.cost_survival * self.independence


@dataclass
class EdgeBotSpec:
    edge_id: str
    family: Regime
    role: str
    enabled: bool = True
    metrics: Optional[EdgeMetrics] = None
    tags: List[str] = field(default_factory=list)

    @property
    def eligible(self) -> bool:
        return bool(
            self.enabled
            and self.metrics is not None
            and self.metrics.pf >= 1.20
            and self.metrics.expectancy > 0
        )


# 12-slot final library. Bots are independent; empty/weak slots are not forced into service.
EDGE_LIBRARY: Dict[str, EdgeBotSpec] = {
    # Trend family (4)
    "T1_EMA21_FLOW": EdgeBotSpec("T1_EMA21_FLOW", Regime.TREND, "Directional flow / pullback continuation", tags=["M1", "EMA21"]),
    "T2_MTF_ALIGN": EdgeBotSpec("T2_MTF_ALIGN", Regime.TREND, "Adaptive MTF alignment continuation", tags=["M1", "M5", "M15"]),
    "T3_BREAK_RETEST_POI": EdgeBotSpec("T3_BREAK_RETEST_POI", Regime.TREND, "Break -> structural retest -> continuation", tags=["BOS", "RETEST"]),
    "T4_ACCELERATION": EdgeBotSpec("T4_ACCELERATION", Regime.TREND, "Momentum / displacement acceleration", tags=["DISPLACEMENT"]),

    # Range-follow family (4)
    "R1_RANGE_DIRECTIONAL_BIAS": EdgeBotSpec("R1_RANGE_DIRECTIONAL_BIAS", Regime.RANGE_FOLLOW, "Directional continuation inside range", tags=["RANGE", "EMA21"]),
    "R2_COMPRESSION_EXPANSION": EdgeBotSpec("R2_COMPRESSION_EXPANSION", Regime.RANGE_FOLLOW, "Compression -> expansion continuation", tags=["COMPRESSION"]),
    "R3_MICRO_BREAK_RETEST": EdgeBotSpec("R3_MICRO_BREAK_RETEST", Regime.RANGE_FOLLOW, "Micro-break -> retest -> hold", tags=["MICRO_BREAK"]),
    "R4_RANGE_LIQUIDITY_RUN": EdgeBotSpec("R4_RANGE_LIQUIDITY_RUN", Regime.RANGE_FOLLOW, "Range-side liquidity run continuation", tags=["LIQUIDITY"]),

    # Reversal family (4)
    "V1_SWEEP_MSS": EdgeBotSpec("V1_SWEEP_MSS", Regime.REVERSAL, "Liquidity sweep + MSS/CHOCH", tags=["SWEEP", "MSS"]),
    "V2_EXTREME_REJECTION": EdgeBotSpec("V2_EXTREME_REJECTION", Regime.REVERSAL, "Volatility/price extreme + rejection", tags=["EXTREME"]),
    "V3_IFVG_REVERSAL": EdgeBotSpec("V3_IFVG_REVERSAL", Regime.REVERSAL, "Failed FVG -> IFVG reversal", tags=["IFVG"]),
    "V4_BPR_REVERSAL": EdgeBotSpec("V4_BPR_REVERSAL", Regime.REVERSAL, "BPR/POI reversal with structure confirmation", tags=["BPR"]),
}


class MultiEdgeSupervisor:
    """Regime router + quality selector. Execution stays inside each independent BOT."""

    def __init__(self, max_active_edges: int = 5):
        self.max_active_edges = max_active_edges

    def select(self, regime: Regime, library: Dict[str, EdgeBotSpec] = EDGE_LIBRARY) -> List[EdgeBotSpec]:
        candidates = [x for x in library.values() if x.family == regime and x.eligible]
        candidates.sort(key=lambda x: x.metrics.utility() if x.metrics else 0.0, reverse=True)
        return candidates[: self.max_active_edges]

    def portfolio_gate(self, selected: List[EdgeBotSpec]) -> bool:
        # Minimum requirement: at least one validated independent EDGE is active.
        return any(x.eligible for x in selected)


PROMOTION_RULES = {
    "hard_pf_min": 1.20,
    "hard_expectancy_min": 0.0,
    "preferred_pf": 1.50,
    "strong_pf": 2.00,
    "min_n_initial": 100,
    "min_stability": 0.60,
    "min_cost_survival": 0.60,
    "min_independence": 0.40,
    "max_active_edges_per_regime": 5,
    "target_total_promoted_edges": [8, 12],
}
