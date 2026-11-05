from research.tickscalper.tickscalper_g1_policy import Action, BasketState, G1Policy


def state(**changes):
    values = dict(
        side="BUY", adds=0, total_risk_r=1.0, unrealized_r=0.0,
        elapsed_seconds=10, spread_r=0.01, daily_drawdown=0.0,
        total_drawdown=0.0, consecutive_losses=0, entry_quality=1.0,
        adverse_confirmation=False,
    )
    values.update(changes)
    return BasketState(**values)


def test_halts_before_open_after_daily_drawdown():
    assert G1Policy().entry_action(state(daily_drawdown=0.05)) == Action.HALT


def test_hard_stop_exits_without_recovery_grid():
    assert G1Policy().manage_action(state(unrealized_r=-1.0)) == Action.EXIT


def test_add_requires_new_confirmation_and_risk_capacity():
    policy = G1Policy()
    assert policy.manage_action(state(unrealized_r=-0.2)) == Action.HOLD
    assert policy.manage_action(state(unrealized_r=-0.2, adverse_confirmation=True)) == Action.ADD
    assert policy.manage_action(state(unrealized_r=-0.2, adverse_confirmation=True, adds=2)) == Action.HOLD
