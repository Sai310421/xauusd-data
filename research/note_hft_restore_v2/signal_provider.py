from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from collections import deque
from typing import Deque, Optional, Protocol, Tuple


@dataclass(frozen=True)
class SignalState:
    a_cond_num: Decimal
    b_cond_num: Decimal
    source: str
    valid: bool


class SignalProvider(Protocol):
    def update(
        self,
        *,
        udp_ask: Decimal,
        udp_bid: Decimal,
        ask_history: Deque[Decimal],
        bid_history: Deque[Decimal],
        payload: Optional[dict] = None,
    ) -> SignalState: ...


class FailClosedSignalProvider:
    """Default provider for the Frozen baseline.

    The published source redacts the decision formula. This provider deliberately
    refuses to invent a substitute. No valid signal => no entry.
    """

    def update(self, *, udp_ask, udp_bid, ask_history, bid_history, payload=None):
        return SignalState(Decimal("0"), Decimal("0"), "FAIL_CLOSED_REDACTED", False)


class ExplicitPayloadSignalProvider:
    """Use only when an authentic upstream process supplies the recovered values.

    Expected fields in the UDP JSON payload:
      a_cond_num, b_cond_num

    This does not derive or alter the signal; it only transports recovered values.
    """

    def update(self, *, udp_ask, udp_bid, ask_history, bid_history, payload=None):
        if not payload:
            return SignalState(Decimal("0"), Decimal("0"), "PAYLOAD_MISSING", False)
        if "a_cond_num" not in payload or "b_cond_num" not in payload:
            return SignalState(Decimal("0"), Decimal("0"), "PAYLOAD_SIGNAL_MISSING", False)
        try:
            a = Decimal(str(payload["a_cond_num"]))
            b = Decimal(str(payload["b_cond_num"]))
        except Exception:
            return SignalState(Decimal("0"), Decimal("0"), "PAYLOAD_SIGNAL_INVALID", False)
        return SignalState(a, b, "AUTHENTIC_PAYLOAD", True)


class LocalSignalState:
    """Structural restoration around the redacted block.

    Keeps bounded price history because the published source creates ask_list and
    bid_list and waits for at least five observations, but the public excerpt lacks
    the update lines. The history itself does not create trading decisions.
    """

    def __init__(self, provider: Optional[SignalProvider] = None, max_history: int = 256):
        self.ask_list: Deque[Decimal] = deque(maxlen=max_history)
        self.bid_list: Deque[Decimal] = deque(maxlen=max_history)
        self.udp_ask: Optional[Decimal] = None
        self.udp_bid: Optional[Decimal] = None
        self.udp_sp: Optional[Decimal] = None
        self.udp_mid: Optional[Decimal] = None
        self.a_cond_num = Decimal("0")
        self.b_cond_num = Decimal("0")
        self.signal_valid = False
        self.signal_source = "UNINITIALIZED"
        self.provider: SignalProvider = provider or FailClosedSignalProvider()

    def ingest_c_data(self, data: dict) -> SignalState:
        self.udp_ask = Decimal(str(data["data1"]))
        self.udp_bid = Decimal(str(data["data2"]))
        self.udp_sp = self.udp_ask - self.udp_bid
        self.udp_mid = (self.udp_ask + self.udp_bid) / Decimal("2")

        # Structural restoration only: public source waits on these lists, therefore
        # they must be populated somewhere in the deleted block.
        self.ask_list.append(self.udp_ask)
        self.bid_list.append(self.udp_bid)

        state = self.provider.update(
            udp_ask=self.udp_ask,
            udp_bid=self.udp_bid,
            ask_history=self.ask_list,
            bid_history=self.bid_list,
            payload=data,
        )
        self.a_cond_num = state.a_cond_num
        self.b_cond_num = state.b_cond_num
        self.signal_valid = state.valid
        self.signal_source = state.source
        return state

    def entry_conditions(self, c_n_of: Decimal = Decimal("0")) -> Tuple[bool, bool]:
        if not self.signal_valid:
            return False, False
        s_ct_cond = self.b_cond_num > c_n_of and self.a_cond_num < 0
        l_ct_cond = self.a_cond_num > c_n_of and self.b_cond_num < 0
        return s_ct_cond, l_ct_cond
