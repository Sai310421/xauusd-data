import math
from research.chris_quant_raw_bidask_wfo_v1_2 import (
    _basket, _cvar_losses, _maxdd, _risk_packet, _optimize, _aligned, _wfo
)

def _assets(n=500):
    return [
        [.0007+((i%5)-2)*.0003 for i in range(n)],
        [.0002+((i%7)-3)*.0004 for i in range(n)],
        [.0004+((i%3)-1)*.0005 for i in range(n)],
    ]

def test_risk_packet_is_deterministic_and_fail_closed_context():
    r=_basket(_assets(120),[.5,.3,.2])
    a=_risk_packet(r,paths=120,horizon=20,seed=7)
    b=_risk_packet(r,paths=120,horizon=20,seed=7)
    assert a==b
    assert 0<=a["first_passage_tail"]<=1
    assert 0<=a["first_passage_recovery"]<=1
    assert a["wasserstein_stressed_cvar"]>=a["cvar_loss"]
    assert 0<=a["ae_tightening_cap"]<=1
    assert "NOT_CALIBRATED" in a["crystal_ball_context"]

def test_optimizer_is_deterministic_and_normalized():
    rows=_assets(120)
    a=_optimize(rows,[1/3]*3,seed=11,iterations=1,candidates=4)
    b=_optimize(rows,[1/3]*3,seed=11,iterations=1,candidates=4)
    assert a==b
    assert abs(sum(abs(x) for x in a[0])-1)<1e-10
    assert math.isfinite(a[1])

def test_walk_forward_is_chronological():
    folds=_wfo(_assets(500),train=240,test=80,step=80,seed=13,max_folds=3)
    assert len(folds)==3
    assert all(f["train_end"]==f["test_start"] for f in folds)
    assert all(f["test_start"]<f["test_end"] for f in folds)
    assert all(0<=f["ae_tightening_cap"]<=1 for f in folds)

def test_alignment_uses_intersection_only():
    syms,keys,rows=_aligned({"A":{1:.1,2:.2,3:.3},"B":{2:.4,3:.5,4:.6}})
    assert syms==["A","B"]
    assert keys==[2,3]
    assert rows==[[.2,.3],[.4,.5]]

def test_drawdown_and_cvar_are_nonnegative():
    assert _maxdd([.1,-.2,.05])>0
    assert _cvar_losses([0,.1,.2,.3],.75)>=0
