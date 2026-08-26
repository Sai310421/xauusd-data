from decimal import Decimal as D
from note_hft_original_reality_v2 import AuthenticSignal, OriginalExecutionState, Quote


def q(sec: int, bid: str, ask: str) -> Quote:
    return Quote(sec * 1_000_000_000, D(bid), D(ask))


def test_zero_spread_is_hard_entry_gate():
    s = OriginalExecutionState()
    sig = AuthenticSignal(D("-1"), D("1"))
    assert s.on_quote(q(0, "100", "100.01"), sig) == "BLOCK_SPREAD"
    assert s.executions == 0
    assert s.on_quote(q(1, "100", "100"), sig) == "BUY"
    assert s.executions == 1


def test_close_first_does_not_require_opposite_alpha():
    s = OriginalExecutionState()
    buy = AuthenticSignal(D("-1"), D("1"))
    assert s.on_quote(q(0, "100", "100"), buy) == "BUY"
    # Same BUY alpha is still present, but public execution skeleton closes an
    # observed long before considering another create.
    assert s.on_quote(q(1, "100.1", "100.1"), buy) == "CLOSE"
    assert s.cycles == 1
    assert s.executions == 2


def test_three_second_create_permit():
    s = OriginalExecutionState()
    buy = AuthenticSignal(D("-1"), D("1"))
    assert s.on_quote(q(0, "100", "100"), buy) == "BUY"
    assert s.on_quote(q(1, "100", "100"), buy) == "CLOSE"
    assert s.on_quote(q(2, "100", "100"), buy) == "BLOCK_COOLDOWN"
    assert s.on_quote(q(3, "100", "100"), buy) == "BUY"


def test_missing_authentic_signal_fails_closed():
    s = OriginalExecutionState()
    assert s.on_quote(q(0, "100", "100"), None) == "BLOCK_SIGNAL"
    assert s.executions == 0


if __name__ == "__main__":
    for f in [
        test_zero_spread_is_hard_entry_gate,
        test_close_first_does_not_require_opposite_alpha,
        test_three_second_create_permit,
        test_missing_authentic_signal_fails_closed,
    ]:
        f()
        print("PASS", f.__name__)
