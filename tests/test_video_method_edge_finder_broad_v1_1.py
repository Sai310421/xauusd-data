from research.video_method_edge_finder_broad_v1_1 import _feature,_threshold,_trades_family,_stage_ok

def test_four_numeric_families():
    rets=[.01,-.005,.007,-.01,.004,.006,-.002,.008]*8
    prices=[]; p=100.
    for r in rets: p*=1+r; prices.append(p)
    for fam in ("IMPULSE_CONT","IMPULSE_REV","STREAK3_CONT","MEAN_REVERT20"):
        f=_feature(prices,rets,fam)
        th=_threshold(f,.5)
        assert th>=0
        x=_trades_family(rets,f,th,"Long",fam)
        assert isinstance(x,list)

def test_fail_closed_stage_chain_primitives():
    good={"MedRF":2.0,"NoLossPct":2.0,"EV":.1,"Trades":100}
    weak={"MedRF":1.2,"NoLossPct":5.0,"EV":.1,"Trades":50}
    assert _stage_ok(good,True)
    assert not _stage_ok(weak,True)
    assert _stage_ok(weak,False)
