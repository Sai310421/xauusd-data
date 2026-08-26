from __future__ import annotations

"""NOTE-HFT Original Reality Parity Core v2.

This module is NOT an alpha reconstruction.
It preserves execution invariants observable in the public NOTE HFT source and
later author commentary:
- authentic a_cond_num / b_cond_num are external inputs
- entry is eligible only at broker spread == 0
- fixed 3 second create permit baseline
- position observation triggers close-first behavior
- reality degradation is modeled by latency/slippage/reject/stale-feed, not by
  charging a synthetic positive spread to an accepted zero-spread entry

No EMA/ATR/spike/proxy alpha is generated here.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

D = Decimal

BASELINE_CYCLES = 176_483
BASELINE_EXECUTIONS = 352_966
BASELINE_BUY = 88_223
BASELINE_SELL = 88_260
BASELINE_WR = D("72.71")
BASELINE_PF = D("1.74")
BASELINE_MAX_DD = D("3.97")


@dataclass(frozen=True)
class AuthenticSignal:
    a_cond_num: D
    b_cond_num: D
    valid: bool = True

    def side(self, threshold: D = D("0")) -> int:
        if not self.valid:
            return 0
        buy = self.b_cond_num > threshold and self.a_cond_num < 0
        sell = self.a_cond_num > threshold and self.b_cond_num < 0
        if buy and not sell:
            return 1
        if sell and not buy:
            return -1
        return 0


@dataclass(frozen=True)
class Quote:
    ts_ns: int
    bid: D
    ask: D

    @property
    def spread(self) -> D:
        return self.ask - self.bid


@dataclass
class RealityProfile:
    name: str
    latency_ms: D
    reject_rate: D = D("0")
    stale_ref_rate: D = D("0")
    # slippage is to be measured from quote movement over latency in a real replay.
    # This fallback field is only for sensitivity tests when empirical latency-path
    # slippage cannot yet be computed.
    fallback_slippage_price: D = D("0")


REALITY_PROFILES = {
    "NORMAL": RealityProfile("NORMAL", D("20"), D("0.001"), D("0.001"), D("0")),
    "STRESS": RealityProfile("STRESS", D("80"), D("0.010"), D("0.010"), D("0")),
    "TAIL": RealityProfile("TAIL", D("250"), D("0.040"), D("0.050"), D("0")),
}


class OriginalExecutionState:
    """Minimal state machine preserving the public execution skeleton."""

    def __init__(self, create_cooldown_sec: D = D("3"), threshold: D = D("0")):
        self.create_cooldown_ns = int(create_cooldown_sec * D("1000000000"))
        self.threshold = threshold
        self.next_create_ns = 0
        self.position = 0  # +1 long, -1 short, 0 flat
        self.cycles = 0
        self.executions = 0
        self.buy_cycles = 0
        self.sell_cycles = 0
        self.zero_spread_ticks = 0
        self.entry_eligible_ticks = 0
        self.blocked_nonzero_spread = 0
        self.blocked_cooldown = 0
        self.blocked_missing_signal = 0

    def on_quote(self, q: Quote, signal: Optional[AuthenticSignal]) -> str:
        # Original public structure closes an observed position before opening a
        # new one. Do not require or fabricate an opposite alpha to close.
        if self.position != 0:
            self.position = 0
            self.executions += 1
            self.cycles += 1
            return "CLOSE"

        if q.spread == 0:
            self.zero_spread_ticks += 1
        else:
            self.blocked_nonzero_spread += 1
            return "BLOCK_SPREAD"

        if q.ts_ns < self.next_create_ns:
            self.blocked_cooldown += 1
            return "BLOCK_COOLDOWN"

        if signal is None or not signal.valid:
            self.blocked_missing_signal += 1
            return "BLOCK_SIGNAL"

        side = signal.side(self.threshold)
        if side == 0:
            return "NO_SIGNAL"

        self.entry_eligible_ticks += 1
        self.position = side
        self.executions += 1
        self.next_create_ns = q.ts_ns + self.create_cooldown_ns
        if side > 0:
            self.buy_cycles += 1
            return "BUY"
        self.sell_cycles += 1
        return "SELL"

    def report(self) -> dict:
        retention = self.cycles / BASELINE_CYCLES if BASELINE_CYCLES else 0.0
        return {
            "cycles": self.cycles,
            "executions": self.executions,
            "buy_cycles": self.buy_cycles,
            "sell_cycles": self.sell_cycles,
            "zero_spread_ticks": self.zero_spread_ticks,
            "entry_eligible_ticks": self.entry_eligible_ticks,
            "blocked_nonzero_spread": self.blocked_nonzero_spread,
            "blocked_cooldown": self.blocked_cooldown,
            "blocked_missing_signal": self.blocked_missing_signal,
            "n_retention": retention,
            "parity_99pct": retention >= 0.99,
        }
