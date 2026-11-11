"""Decision-only contracts for the Phase 9–10 extension."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.governance import production_accepted
from src.multiasset import portfolio_decision, smallcap_metrics


def test_smallcap_and_portfolio_action() -> None:
    stress, breadth, warning = smallcap_metrics(-2.2, 0.1, 0.2)
    decision = portfolio_decision(0.74, 0.68, stress, breadth, False, 0.7, 0.3, 0.0)
    assert warning and decision.hedge_action == "Reduce Equity Increase Gold"


def test_governance_rejects_any_missing_gate() -> None:
    accepted, failures = production_accepted({"auc_roc": 0.7})
    assert not accepted and "future_leakage" in failures
