from research.g75_debt_to_profit_policy import (
    DebtState,
    PolicyInputs,
    RecoveryMode,
    RecoveryProbabilities,
    RescueBudget,
    choose_mode,
    compare_slot_budgets,
)


def probs(**kw):
    base = dict(p300=0.8, p600=0.85, p1800=0.89, p3600=0.91, p6h=0.97, p12h=0.98, p24h=0.99)
    base.update(kw)
    return RecoveryProbabilities(**base)


def debt(**kw):
    base = dict(initial_debt=100.0, remaining_debt=100.0, realized_rescue_profit=0.0,
                accumulated_costs=0.0, floating_dd_pct=5.0, margin_level_pct=800.0,
                basket_age_s=1200.0)
    base.update(kw)
    return DebtState(**base)


def test_equal_total_risk_budget_across_3_5_10_slots():
    rows = compare_slot_budgets(30.0)
    assert [r["slots"] for r in rows] == [3, 5, 10]
    assert [r["per_slot_risk_if_equal"] for r in rows] == [10.0, 6.0, 3.0]
    assert all(r["total_risk_budget"] == 30.0 for r in rows)


def test_high_1h_natural_recovery_prefers_hold_over_forced_rescue():
    x = PolicyInputs(
        debt=debt(basket_age_s=1800.0), probabilities=probs(p3600=0.92),
        budget=RescueBudget(slots=5, total_risk_budget=20.0),
        predicted_mfe=2.0, required_move_to_be=1.0,
        rescue_expected_net_profit_per_trade=5.0, rescue_profit_velocity_per_hour=10.0,
    )
    d = choose_mode(x)
    assert d.mode is RecoveryMode.HOLD
    assert d.rescue_enabled is False


def test_margin_guard_disables_rescue():
    x = PolicyInputs(
        debt=debt(margin_level_pct=400.0), probabilities=probs(),
        budget=RescueBudget(slots=3, total_risk_budget=20.0),
        predicted_mfe=2.0, required_move_to_be=1.0,
        rescue_expected_net_profit_per_trade=5.0, rescue_profit_velocity_per_hour=10.0,
    )
    d = choose_mode(x)
    assert d.mode is RecoveryMode.RISK_CONTROL
    assert d.rescue_enabled is False


def test_positive_expectancy_and_coverage_enable_recovery_after_1h():
    x = PolicyInputs(
        debt=debt(basket_age_s=7200.0), probabilities=probs(p3600=0.89, p6h=0.94),
        budget=RescueBudget(slots=10, total_risk_budget=30.0),
        predicted_mfe=1.2, required_move_to_be=1.0,
        rescue_expected_net_profit_per_trade=1.0, rescue_profit_velocity_per_hour=2.0,
    )
    d = choose_mode(x)
    assert d.mode is RecoveryMode.RECOVER
    assert d.rescue_enabled is True
    assert d.max_risk_per_next_slot == 3.0


def test_be_reached_transitions_to_profit_extend():
    x = PolicyInputs(
        debt=debt(remaining_debt=0.0), probabilities=probs(),
        budget=RescueBudget(slots=5, total_risk_budget=20.0),
        predicted_mfe=1.0, required_move_to_be=0.0,
        rescue_expected_net_profit_per_trade=0.5, rescue_profit_velocity_per_hour=1.0,
    )
    d = choose_mode(x)
    assert d.mode is RecoveryMode.PROFIT_EXTEND
    assert d.rescue_enabled is True
