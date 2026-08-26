"""NOTE-HFT Reconstructed Alpha v3

Reconstructs only the missing UDP decision fragment while keeping the public
execution core frozen.

Evidence-preserving default:
- public code waits for 5 ask/bid samples before main_loop starts
- BUY requires b_cond_num > 0 and a_cond_num < 0
- SELL requires a_cond_num > 0 and b_cond_num < 0
- UDP payload is price-only (data1=ask, data2=bid)

The parsimonious reconstruction is a coherent directional delta over the
same rolling window:
    a_cond = old_ask - new_ask
    b_cond = new_bid - old_bid
Thus an upward quote move yields (a<0,b>0) and a downward move yields
(a>0,b<0), exactly matching the frozen public conditions.

This module NEVER changes the frozen execution conditions. Window is exposed
only for parity testing; default=5 follows the explicit warm-up in the source.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Deque, Optional, Tuple

D = Decimal

@dataclass(frozen=True)
class AlphaSnapshot:
    a_cond_num: Decimal
    b_cond_num: Decimal
    signal: int  # +1 BUY, -1 SELL, 0 NONE
    ready: bool

class ReconstructedDirectionalAlpha:
    def __init__(self, window: int = 5, threshold: Decimal = D("0")):
        if window < 2:
            raise ValueError("window must be >=2")
        self.window = int(window)
        self.threshold = D(str(threshold))
        self.ask: Deque[Decimal] = deque(maxlen=self.window)
        self.bid: Deque[Decimal] = deque(maxlen=self.window)

    def update(self, ask, bid) -> AlphaSnapshot:
        a = D(str(ask)); b = D(str(bid))
        if a < b:
            raise ValueError("ask must be >= bid")
        self.ask.append(a); self.bid.append(b)
        if len(self.ask) < self.window:
            return AlphaSnapshot(D("0"), D("0"), 0, False)

        old_ask = self.ask[0]
        old_bid = self.bid[0]
        new_ask = self.ask[-1]
        new_bid = self.bid[-1]

        # Sign orientation is dictated by the frozen public entry conditions.
        a_cond = old_ask - new_ask
        b_cond = new_bid - old_bid

        buy = b_cond > self.threshold and a_cond < 0
        sell = a_cond > self.threshold and b_cond < 0
        sig = 1 if buy and not sell else (-1 if sell and not buy else 0)
        return AlphaSnapshot(a_cond, b_cond, sig, True)


def reconstruct(ask, bid, window: int = 5) -> Tuple[Decimal, Decimal, int]:
    """Stateless convenience for tests over exactly one window."""
    if len(ask) != len(bid) or len(ask) < window:
        raise ValueError("ask/bid must contain at least window samples")
    eng = ReconstructedDirectionalAlpha(window=window)
    out: Optional[AlphaSnapshot] = None
    for a,b in zip(ask[-window:], bid[-window:]):
        out = eng.update(a,b)
    assert out is not None
    return out.a_cond_num, out.b_cond_num, out.signal
