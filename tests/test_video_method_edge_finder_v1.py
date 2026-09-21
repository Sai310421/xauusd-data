from research.video_method_edge_finder_v1 import _bootstrap_metrics,_trades,_rf,_risk_mc

def test_bootstrap_positive_lane_has_positive_ev_and_low_noloss():
    trades=[.001]*80+[-.0005]*20
    m=_bootstrap_metrics(trades,paths=200,ntrades=100,seed=11)
    assert m["EV"]>0 and m["MedProfit"]>0
    assert m["NoLossPct"]<3
    assert m["WinRatePct"]==80

def test_numeric_trade_rule_direction_and_hold():
    rets=[.01,.02,.01,-.01,.005,.004,.003]
    long=_trades(rets,.009,"Long",hold=2)
    short=_trades(rets,.009,"Short",hold=2)
    assert long and short
    assert all(isinstance(x,float) for x in long+short)

def test_recovery_factor_positive_path():
    assert _rf([1,-.2,1])>1

def test_risk_mc_is_bounded():
    r=_risk_mc([.001,-.0005,.0015,-.0002],.01,paths=100,ntrades=50,seed=1)
    assert 0<=r["median_maxdd_pct"]<=100
    assert 0<=r["ruin50_pct"]<=100
