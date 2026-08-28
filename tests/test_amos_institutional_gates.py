from research.amos_live_parity import evaluate as parity
from research.amos_portfolio_gate import evaluate as portfolio
from research.amos_reality_grade import evaluate as grade
from research.amos_robustness_gate import evaluate as robust


def test_robustness_missing_is_invalid():
    r = robust({})
    assert r["status"] == "INVALID"


def test_robustness_full_pass():
    p = {k: {"status": "PASS"} for k in ("wfo","monte_carlo","parameter_stability","regime_segmentation","master_checker")}
    assert robust(p)["status"] == "PASS"


def test_live_parity_missing_is_invalid():
    assert parity({}, 0.25)["status"] == "INVALID"


def test_live_parity_breach_demotes():
    f = ("return","pf","drawdown","slippage","fill_rate","exposure","unresolved_inventory","basket_age")
    p = {"expected": {k: 1 for k in f}, "observed": {k: 1 for k in f}}
    p["observed"]["pf"] = 2
    assert parity(p, 0.25)["status"] == "DEMOTE"


def test_portfolio_missing_is_invalid():
    assert portfolio({})["status"] == "INVALID"


def test_grade_never_emits_authoritative_grade_without_prerequisites():
    assert grade({})["grade"] == "INVALID"
