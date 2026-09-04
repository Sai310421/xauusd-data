from research.ae_reflected_recovery_controller import (
    AEAction,
    AEState,
    ReflectedRecoveryConfig,
    decide_reflected_recovery,
)


def make_state(**overrides):
    base = dict(
        debt=10.0,
        debt_limit=100.0,
        debt_drift=0.0,
        debt_drift_limit=10.0,
        mae=10.0,
        mae_limit=100.0,
        recovery_age_s=10.0,
        recovery_age_limit_s=300.0,
        tail_probability=0.05,
        spread_stress=0.05,
        volatility_stress=0.05,
        shock_score=0.0,
        drawdown_pct=1.0,
        margin_level_pct=1000.0,
        natural_recovery_probability=0.8,
    )
    base.update(overrides)
    return AEState(**base)


def test_low_hazard_waits():
    d = decide_reflected_recovery(make_state())
    assert d.action == AEAction.WAIT
    assert d.intervention_fraction == 0.0


def test_same_loss_high_recovery_probability_intervenes_less():
    common = dict(
        debt=65.0,
        debt_drift=5.0,
        mae=60.0,
        recovery_age_s=180.0,
        tail_probability=0.45,
        spread_stress=0.30,
        volatility_stress=0.30,
    )
    high_recovery = decide_reflected_recovery(make_state(**common, natural_recovery_probability=0.9))
    low_recovery = decide_reflected_recovery(make_state(**common, natural_recovery_probability=0.05))
    assert high_recovery.adjusted_hazard < low_recovery.adjusted_hazard


def test_reduce_is_minimal_not_full_liquidation():
    cfg = ReflectedRecoveryConfig(
        recovery_credit=0.0,
        theta_stop_add=0.10,
        theta_reduce=0.20,
        theta_hedge=0.80,
        theta_recovery=0.95,
        max_reduce_fraction=0.50,
    )
    d = decide_reflected_recovery(
        make_state(
            debt=60.0,
            debt_drift=6.0,
            mae=60.0,
            recovery_age_s=180.0,
            tail_probability=0.40,
            spread_stress=0.30,
            volatility_stress=0.30,
            natural_recovery_probability=0.0,
        ),
        cfg,
    )
    assert d.action == AEAction.REDUCE
    assert 0.0 < d.intervention_fraction < 0.50


def test_shock_overrides_reflected_policy():
    d = decide_reflected_recovery(make_state(shock_score=1.2))
    assert d.action == AEAction.EMERGENCY_IMPULSE
    assert d.intervention_fraction == 1.0


def test_drawdown_override():
    d = decide_reflected_recovery(make_state(drawdown_pct=5.1))
    assert d.action == AEAction.EMERGENCY_IMPULSE


def test_margin_override():
    d = decide_reflected_recovery(make_state(margin_level_pct=450.0))
    assert d.action == AEAction.EMERGENCY_IMPULSE
