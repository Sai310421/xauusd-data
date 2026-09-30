from research.mql5_edge_factory.overlay_gate import assess, OVERLAY_ABLATION


def test_overlay_candidate_gate_passes_reference_screen():
    r=assess(monthly_n=100,pf=1.01,wr_pct=65.0,base_expectancy=0.01)
    assert r.eligible


def test_overlay_candidate_gate_rejects_weak_base():
    r=assess(monthly_n=99,pf=0.99,wr_pct=64.9,base_expectancy=-0.01)
    assert not r.eligible
    assert len(r.reasons)==4


def test_required_ablation():
    assert OVERLAY_ABLATION == ('BASE','BASE_SPLIT3','BASE_G75','BASE_SPLIT3_G75')
