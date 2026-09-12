from dataclasses import dataclass, field
from typing import Dict
from enum import Enum

@dataclass
class MarketState:
    timestamp_ns: int
    bid: float
    ask: float
    spread: float
    timeframe: str = "M1"
    regime: str = "UNKNOWN"
    liquidity_score: float = 0.0
    volatility_z: float = 0.0

@dataclass
class ICTState:
    direction: int = 0
    score: float = 0.0
    sweep: bool = False
    mss: bool = False
    choch: bool = False
    bos: bool = False
    fvg: bool = False
    ifvg: bool = False
    order_block: bool = False
    ote: bool = False
    smt: bool = False
    po3: bool = False

    @property
    def confluence(self) -> int:
        return sum([self.sweep,self.mss,self.choch,self.bos,self.fvg,self.ifvg,self.order_block,self.ote,self.smt,self.po3])

@dataclass
class AISupervisorState:
    direction_prob: float = 0.5
    confidence: float = 0.0
    approve: bool = True
    reject_reason: str = ""

@dataclass
class HarmonicBoostState:
    direction: int = 0
    score: float = 0.0
    pattern: str = ""

@dataclass
class Decision:
    action: str = "HOLD"
    score: float = 0.0
    reason: str = ""
    components: Dict[str, float] = field(default_factory=dict)

@dataclass
class ExitState:
    entry_price: float = 0.0
    current_price: float = 0.0
    pnl: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0
    atr: float = 0.0
    bars_held: int = 0
    seconds_held: float = 0.0
    structure_invalidated: bool = False
    opposite_mss: bool = False
    opposite_choch: bool = False
    liquidity_target_hit: bool = False
    ai_reject_now: bool = False
    spread: float = 0.0

@dataclass
class ExitDecision:
    action: str = "HOLD"
    reason: str = ""
    lock_fraction: float = 0.0
    trail_distance: float = 0.0

class ZoneLifecycle(str, Enum):
    CREATED="CREATED"; ACTIVE="ACTIVE"; RETESTED="RETESTED"; REJECTED="REJECTED"; FILLED="FILLED"; BROKEN="BROKEN"; INVALIDATED="INVALIDATED"

@dataclass
class FVGZone:
    direction: int = 0
    lower: float = 0.0
    upper: float = 0.0
    created_ts_ns: int = 0
    lifecycle: ZoneLifecycle = ZoneLifecycle.CREATED
    broken_ts_ns: int = 0
    break_direction: int = 0
    retested_ts_ns: int = 0

    @property
    def width(self) -> float:
        return max(0.0,self.upper-self.lower)

@dataclass
class IFVGState:
    source_fvg: FVGZone | None = None
    active: bool = False
    direction: int = 0
    activated_ts_ns: int = 0
    retested: bool = False
    rejected: bool = False

    @property
    def valid_transition(self) -> bool:
        return self.source_fvg is not None and self.source_fvg.lifecycle == ZoneLifecycle.BROKEN and self.source_fvg.break_direction != 0 and self.active

@dataclass
class BPRZone:
    bullish_fvg: FVGZone | None = None
    bearish_fvg: FVGZone | None = None
    lower: float = 0.0
    upper: float = 0.0
    created_ts_ns: int = 0
    lifecycle: ZoneLifecycle = ZoneLifecycle.CREATED
    retest_direction: int = 0
    rejection_direction: int = 0

    @property
    def width(self) -> float:
        return max(0.0,self.upper-self.lower)

    @property
    def valid_overlap(self) -> bool:
        if self.bullish_fvg is None or self.bearish_fvg is None:
            return False
        lo=max(self.bullish_fvg.lower,self.bearish_fvg.lower)
        hi=min(self.bullish_fvg.upper,self.bearish_fvg.upper)
        return hi>lo and abs(lo-self.lower)<1e-12 and abs(hi-self.upper)<1e-12
