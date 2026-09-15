from research.mql5_edge_factory.metrics import metrics, matched_delta
from research.mql5_edge_factory.reconstruction import score
from research.mql5_edge_factory.classify import classify


def test_metrics():
    m=metrics([2,-1,3,-1])
    assert m.n==4
    assert m.wr_pct==50.0
    assert m.pf==2.5
    assert m.expectancy==0.75
    assert m.max_dd==1.0


def test_delta():
    d=matched_delta([2,-1,3,-1],[1,-1,1,-1])
    assert d['delta_expectancy']>0


def test_match():
    r=[{'side':'buy','entry_ts':10,'exit_ts':20,'add_count':1}]
    m=[{'side':'buy','entry_ts':12,'exit_ts':25,'add_count':1}]
    s=score(r,m)
    assert s.composite==1.0


def test_classifier_never_auto_promotes():
    v=classify(300,1.0,2.0,5.0,0.5,True)
    assert v['grade']=='S'
    assert v['library_ready'] is False
