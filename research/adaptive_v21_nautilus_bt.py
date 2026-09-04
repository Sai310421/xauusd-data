from __future__ import annotations

"""Fib/ICT reconstruction gate — SOURCE OF TRUTH RESET.

This module intentionally does NOT run profitability BT yet.
It restores the original design baseline and makes exact process-order replay a
hard prerequisite. No generic Fib-touch entry, no guessed detector, no OHLC.

Formal execution remains Raw Bid/Ask QuoteTick only:
  BUY entry Ask / BUY exit Bid
  SELL entry Bid / SELL exit Ask
"""

from dataclasses import dataclass
from enum import Enum

TF_SECONDS = {"M1": 60, "M5": 300, "M15": 900}


class Stage(str, Enum):
    WAIT = "WAIT"
    SETUP = "SETUP"
    FIB_PRZ = "FIB_PRZ"
    TRIGGER = "TRIGGER"
    EXECUTION_GATE = "EXECUTION_GATE"
    ENTRY = "ENTRY"


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    process: tuple[str, ...]
    primary: str
    ratios: tuple[float, ...]
    executable: bool = False


# Original Design Baseline. Process order is part of the algorithm and MUST NOT
# be reordered, skipped, retrospectively satisfied, or replaced by level touch.
STRATEGY_SPECS = {
    "A0_OTE": StrategySpec(
        "A0_OTE",
        ("LIQUIDITY", "SWEEP", "MSS_CHOCH", "DISPLACEMENT", "ANCHOR_LOCK", "OTE_PRZ", "ENTRY_TRIGGER", "EXECUTION_GATE", "ENTRY"),
        "OTE_PRIMARY_0.705_0.708",
        (0.50, 0.618, 0.705, 0.708, 0.786),
    ),
    "B0_XABCD_D_PRZ": StrategySpec(
        "B0_XABCD_D_PRZ",
        ("X", "A", "B", "C", "D_PRZ", "QML_HTF", "LIQUIDITY_LTF", "SWEEP_LTF", "MSS_LTF", "ENTRY_TRIGGER", "EXECUTION_GATE", "ENTRY"),
        "PATTERN_SPECIFIC_D_PRZ",
        (),
    ),
    "POP": StrategySpec("POP", ("POP_SETUP", "POP_FIB_ZONE", "POP_TRIGGER", "EXECUTION_GATE", "ENTRY"), "POP_NATIVE", (0.5, 0.559, 0.669, 0.786)),
    "GOLD_SILVER": StrategySpec("GOLD_SILVER", ("GOLD_SILVER_SETUP", "GOLD_SILVER_ZONE", "GS_TRIGGER", "EXECUTION_GATE", "ENTRY"), "GS_NATIVE", (0.232, 0.25, 0.688, 0.705, 0.708, 0.718, 0.786, 0.822)),
    "CRT": StrategySpec("CRT", ("CRT_SETUP", "CRT_DELIVERY", "CRT_FIB_LEVEL", "CRT_TRIGGER", "EXECUTION_GATE", "ENTRY"), "CRT_NATIVE", (-0.40, -0.29, -0.255, -0.21, 0.0, 1.0, 1.47, 1.55, 2.56, 2.60, 2.64)),
    "ORDER_FLOW": StrategySpec("ORDER_FLOW", ("ORDER_FLOW_SETUP", "OF_FIB_LEVEL", "OF_TRIGGER", "EXECUTION_GATE", "ENTRY"), "OF_NATIVE", (0.0, 0.25, 0.5, 0.75, 1.0)),
    "PREMIUM_DISCOUNT": StrategySpec("PREMIUM_DISCOUNT", ("DEALING_RANGE", "EQ_CONTEXT", "PD_LOCATION", "PD_TRIGGER", "EXECUTION_GATE", "ENTRY"), "EQ_0.5_CONTEXTUAL", (0.0, 0.5, 1.0)),
    "MONKEY": StrategySpec("MONKEY", ("MONKEY_SETUP", "RETRACE_ZONE", "MONKEY_TRIGGER", "EXECUTION_GATE", "ENTRY"), "MONKEY_0.63_0.78", (0.63, 0.78)),
    "FIB_SNR": StrategySpec("FIB_SNR", ("SNR_SETUP", "SNR_LEVEL", "SNR_TRIGGER", "EXECUTION_GATE", "ENTRY"), "SNR_NATIVE", (-0.29, -0.255, -0.21, 1.55, 2.47, 2.64)),
    "TARGET_PRICES": StrategySpec("TARGET_PRICES", ("TARGET_SETUP", "TARGET_LEVEL", "TARGET_TRIGGER", "EXECUTION_GATE", "ENTRY"), "TARGET_NATIVE", (-1.0, -2.0, -2.5, -3.0, -4.0)),
    "STD_DEV_FIB": StrategySpec("STD_DEV_FIB", ("STD_SETUP", "STD_LEVEL", "STD_TRIGGER", "EXECUTION_GATE", "ENTRY"), "STD_NATIVE", ()),
    "CIRCLE_CLASSIC": StrategySpec("CIRCLE_CLASSIC", ("ANCHOR_A", "ANCHOR_B", "CIRCLE_GEOMETRY", "PRICE_TIME_CONFLUENCE", "TRIGGER", "EXECUTION_GATE", "ENTRY"), "CIRCLE_CLASSIC", (0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618)),
    "CIRCLE_GOLDEN": StrategySpec("CIRCLE_GOLDEN", ("ANCHOR_A", "ANCHOR_B", "GOLDEN_CIRCLE", "PRICE_TIME_CONFLUENCE", "TRIGGER", "EXECUTION_GATE", "ENTRY"), "CIRCLE_GOLDEN", (0.618, 1.618)),
    "CIRCLE_TIME": StrategySpec("CIRCLE_TIME", ("ANCHOR_A", "ANCHOR_B", "TIME_PROJECTION", "TIME_WINDOW", "PRICE_CONFIRMATION", "TRIGGER", "EXECUTION_GATE", "ENTRY"), "TIME_NATIVE", (0.236, 0.382, 0.5, 0.618, 1.0, 1.618, 2.618, 4.236)),
    "CIRCLE_HARMONIC": StrategySpec("CIRCLE_HARMONIC", ("XABCD_VALID", "D_PRZ", "HARMONIC_CIRCLE", "PRICE_TIME_CONFLUENCE", "TRIGGER", "EXECUTION_GATE", "ENTRY"), "HARMONIC_D_PRZ", (0.382, 0.5, 0.618, 0.707, 0.786, 1.0, 1.272, 1.618, 2.618, 2.886)),
    "CIRCLE_LIQUIDITY": StrategySpec("CIRCLE_LIQUIDITY", ("LIQUIDITY_MAP", "SWEEP", "CIRCLE_LIQUIDITY", "MSS", "TRIGGER", "EXECUTION_GATE", "ENTRY"), "LIQUIDITY_CIRCLE", (0.5, 0.618, 0.786, 1.0, 1.618)),
    "CIRCLE_MSNR": StrategySpec("CIRCLE_MSNR", ("MSNR_SETUP", "MSNR_CIRCLE", "MSNR_TRIGGER", "EXECUTION_GATE", "ENTRY"), "EXPERIMENTAL_MSNR", (1.2, 4.5, 4.83)),
}


class OrderedProcessGate:
    """Causal state gate. A later event cannot back-fill an earlier event."""
    def __init__(self, spec: StrategySpec):
        self.spec = spec
        self.index = 0
        self.event_times: list[float] = []

    @property
    def expected(self):
        return self.spec.process[self.index] if self.index < len(self.spec.process) else None

    def accept(self, event: str, ts: float) -> bool:
        if self.expected != event:
            return False
        if self.event_times and ts < self.event_times[-1]:
            return False
        self.event_times.append(ts)
        self.index += 1
        return True

    @property
    def complete(self) -> bool:
        return self.index == len(self.spec.process)

    @property
    def entry_allowed(self) -> bool:
        return self.complete and self.spec.executable


def fib_retrace(a: float, b: float, r: float, bull: bool) -> float:
    return b - r * (b - a) if bull else b + r * (a - b)


def normalized_range_position(price: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("invalid range")
    return (price - low) / (high - low)


def raw_execution_price(side: str, action: str, bid: float, ask: float) -> float:
    if side == "LONG":
        return ask if action == "ENTRY" else bid
    if side == "SHORT":
        return bid if action == "ENTRY" else ask
    raise ValueError(side)


def validate_source_of_truth() -> None:
    # No strategy may end anywhere except ENTRY and every one must pass an
    # explicit EXECUTION_GATE immediately before ENTRY.
    for spec in STRATEGY_SPECS.values():
        assert spec.process[-1] == "ENTRY", spec.strategy_id
        assert spec.process[-2] == "EXECUTION_GATE", spec.strategy_id
        assert not spec.executable, "Reconstruction phase must not emit live/BT entries"


if __name__ == "__main__":
    validate_source_of_truth()
    print({
        "status": "SOURCE_OF_TRUTH_RESTORED",
        "formal_bt_enabled": False,
        "reason": "Exact per-strategy detectors must be reconstructed and fixture-tested before profitability BT",
        "input": "RAW_BID_ASK_QUOTETICK_ONLY",
        "ohlc": False,
        "strategies_registered": len(STRATEGY_SPECS),
    })
