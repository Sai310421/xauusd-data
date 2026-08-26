from decimal import Decimal
from signal_provider import LocalSignalState, FailClosedSignalProvider, ExplicitPayloadSignalProvider
from parity_gate import evaluate


def test_fail_closed():
    s = LocalSignalState(FailClosedSignalProvider())
    for i in range(5):
        st = s.ingest_c_data({"data1": "100.0", "data2": "100.0"})
    assert len(s.ask_list) == 5
    assert len(s.bid_list) == 5
    assert s.entry_conditions() == (False, False)
    assert not st.valid


def test_authentic_payload_long_short():
    s = LocalSignalState(ExplicitPayloadSignalProvider())
    s.ingest_c_data({"data1": "100", "data2": "100", "a_cond_num": "-1", "b_cond_num": "1"})
    assert s.entry_conditions(Decimal("0")) == (True, False)
    s.ingest_c_data({"data1": "100", "data2": "100", "a_cond_num": "1", "b_cond_num": "-1"})
    assert s.entry_conditions(Decimal("0")) == (False, True)


def test_parity_gate():
    r = evaluate(n=176483, buy=88223, sell=88260, wr_pct=72.71, pf=1.74, maxdd_pct=3.97)
    assert r["structural"]["pass"]
    r2 = evaluate(n=95, buy=48, sell=47)
    assert not r2["structural"]["pass"]


if __name__ == "__main__":
    test_fail_closed()
    test_authentic_payload_long_short()
    test_parity_gate()
    print("PASS 3/3")
