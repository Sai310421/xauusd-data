from __future__ import annotations

"""Evidence-driven recovery of the four redacted NOTE-HFT expressions.

This is NOT fitted to N.  It is the minimum-complexity reconstruction implied by:
- explicit 5-sample warm-up in main()
- ask_list / bid_list state
- downstream opposite-sign BUY/SELL conditions
- author's statement that UDP is the decision price feed
- author's statement that frozen/degenerate values do not trigger

Visible source has no list-maintenance lines, so `push()` restores that separately
from the four redacted expressions.
"""

from collections import deque
from dataclasses import dataclass
from decimal import Decimal

D = Decimal

@dataclass(frozen=True)
class Snapshot:
    slot1: bool
    slot2: bool
    a_cond_num: Decimal
    b_cond_num: Decimal
    signal: int

class FourSlotRecovery:
    def __init__(self, window: int = 5):
        if window != 5:
            raise ValueError("Recovery v4 fixes window=5 from the explicit source warm-up")
        self.ask_list = deque(maxlen=5)
        self.bid_list = deque(maxlen=5)

    def push(self, udp_ask, udp_bid) -> Snapshot:
        ask = D(str(udp_ask))
        bid = D(str(udp_bid))
        if ask < bid:
            raise ValueError("ask < bid")

        # Structural omission repair (not one of the four displayed 00000 slots).
        self.ask_list.append(ask)
        self.bid_list.append(bid)

        # SLOT 1: ask-history validity gate.
        slot1 = len(self.ask_list) >= 5

        # SLOT 2: bid-history validity gate, symmetric to slot 1.
        slot2 = len(self.bid_list) >= 5

        if slot1 and slot2:
            # SLOT 3: ask-side signed movement. Up move => negative.
            a_cond_num = self.ask_list[0] - self.ask_list[-1]

            # SLOT 4: bid-side signed movement. Up move => positive.
            b_cond_num = self.bid_list[-1] - self.bid_list[0]
        else:
            a_cond_num = D("0")
            b_cond_num = D("0")

        buy = b_cond_num > 0 and a_cond_num < 0
        sell = a_cond_num > 0 and b_cond_num < 0
        signal = 1 if buy and not sell else (-1 if sell and not buy else 0)
        return Snapshot(slot1, slot2, a_cond_num, b_cond_num, signal)
