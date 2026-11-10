"""Safety tests for H1-to-tick state and hedge risk separation."""
from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rage_dch.h1_signal_adapter import SignalStateSynchronizer, SignalValidationError
from rage_dch.hedge_optimizer import HedgeOptimizer
from rage_dch.protective_hedge import (
    HedgeInputError,
    ProtectiveHedgeParams,
    compute_hedge_intensity,
)


NOW = datetime(2026, 1, 1, tzinfo=UTC)


def payload(risk: float = 77.0) -> dict[str, object]:
    """Return one READY contract fixture."""
    return {
        "schema_version": "h1-macro-signal/v1",
        "signal_id": "signal-001",
        "as_of_utc": NOW.isoformat(),
        "valid_until_utc": (NOW + timedelta(minutes=90)).isoformat(),
        "signal_status": "READY",
        "model_version": "shadow-v001",
        "portfolio_risk": risk,
        "portfolio_regime": "HIGH",
        "psi_status": "NORMAL",
        "source_hash": "a" * 64,
        "probability_ordering_pass": True,
        "feature_schema_pass": True,
        "manifest_integrity_pass": True,
    }


def parameters() -> ProtectiveHedgeParams:
    return ProtectiveHedgeParams(h_max=0.8, lambda_macro=1.0)


def test_portfolio_risk_normalizes_once() -> None:
    sync = SignalStateSynchronizer()
    accepted = sync.accept(payload(77.0), NOW)
    assert accepted.macro_crisis_score == pytest.approx(0.77)
    assert sync.held_macro_score == pytest.approx(0.77)


def test_crisis_escalation_is_immediate_and_deescalation_is_smoothed() -> None:
    sync = SignalStateSynchronizer(downward_smoothing_alpha=0.25)
    sync.accept(payload(20.0), NOW)
    sync.accept(payload(80.0), NOW)
    assert sync.held_macro_score == pytest.approx(0.8)
    sync.accept(payload(40.0), NOW)
    assert sync.held_macro_score == pytest.approx(0.7)


def test_stopped_or_expired_signal_is_rejected() -> None:
    sync = SignalStateSynchronizer()
    stopped = payload()
    stopped["signal_status"] = "STOPPED"
    with pytest.raises(SignalValidationError, match="SIGNAL_STOPPED"):
        sync.accept(stopped, NOW)
    expired = payload()
    expired["valid_until_utc"] = (NOW - timedelta(minutes=1)).isoformat()
    with pytest.raises(SignalValidationError, match="HEDGE_INPUT_STALE"):
        sync.accept(expired, NOW)


def test_macro_never_reduces_hedge_when_execution_risk_is_fixed() -> None:
    low = compute_hedge_intensity(0.2, 0.2, 0.3, parameters())
    high = compute_hedge_intensity(0.2, 0.9, 0.3, parameters())
    assert high.final_hedge_ratio >= low.final_hedge_ratio


def test_execution_risk_never_increases_hedge_when_macro_is_fixed() -> None:
    low = compute_hedge_intensity(0.2, 0.8, 0.0, parameters())
    high = compute_hedge_intensity(0.2, 0.8, 1.0, parameters())
    assert high.final_hedge_ratio <= low.final_hedge_ratio


def test_execution_risk_rejects_portfolio_scale() -> None:
    with pytest.raises(HedgeInputError, match="hedge_execution_risk"):
        compute_hedge_intensity(0.2, 0.77, 77.0, parameters())


def test_optimizer_is_non_execution_and_fails_closed_when_stale() -> None:
    sync = SignalStateSynchronizer()
    sync.accept(payload(), NOW)
    decision = HedgeOptimizer(sync, parameters()).decide(NOW, 0.2, 0.3)
    assert decision.intensity.final_hedge_ratio <= 0.8
    with pytest.raises(SignalValidationError, match="HEDGE_INPUT_STALE"):
        sync.state_for_tick(NOW + timedelta(minutes=91))
