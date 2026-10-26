"""Main-engine seam.

The null engine trades nothing. That is intentional: this package must not
invent an XAUUSD edge. Existing AMOS scripts stay where they are and are
named, not executed, by the candidate registry.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Quote:
    ts: int
    bid: float
    ask: float


@dataclass(frozen=True)
class OrderIntent:
    instrument: str
    side: str
    qty: float
    reason: str
    engine: str


class NullMainEngine:
    name = "null-no-alpha"

    def desired(self, quote: Quote) -> list[OrderIntent]:
        if quote.ask < quote.bid or quote.bid <= 0:
            raise ValueError("invalid quote")
        return []


# Reuse targets. None of these are imported or backtested here.
MAIN_CANDIDATE_PATHS = (
    "research/ae_allweather_nautilus_raw_ab_v9_drmpc.py",
    "research/armada_nautilus_raw_cell.py",
    "research/bigplayer_rawtick_5tf_21d_bt.py",
)
