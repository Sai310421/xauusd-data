from research.math_core import (
    FirstPassageState, Kalman1DDrift, OnlineRegimeState, BivariateExpHawkes,
    JumpTailState, RoughnessState, Action, ActionPolicy, RiskConstrainedKelly,
    UnifiedMathState,
)


def main():
    fp = FirstPassageState.from_state(2000.0, 1.0, 1.0, mu=0.02, sigma=0.3)
    assert 0.0 <= fp.p_up <= 1.0

    kf = Kalman1DDrift()
    last_k = None
    reg = OnlineRegimeState(window=64)
    jt = JumpTailState(window=128)
    rv = RoughnessState(window=128)
    hawkes = BivariateExpHawkes()
    last_reg = last_jt = last_rv = last_h = None
    p = 2000.0
    for i in range(160):
        p += (0.01 if i % 7 else -0.005)
        last_k = kf.update(p)
        last_reg = reg.update(p)
        last_jt = jt.update(p)
        last_rv = rv.update(p)
        last_h = hawkes.update(i * 0.25, 1 if i % 3 else -1)

    policy = ActionPolicy()
    best, scores = policy.choose({
        Action.WAIT: dict(expected_pnl=0.0, expected_dd=0.0, tail_risk=0.0, cost=0.0, ruin_risk=0.0),
        Action.ENTRY_LONG: dict(expected_pnl=max(fp.edge, 0.0), expected_dd=0.02, tail_risk=last_jt["cvar"], cost=0.01, ruin_risk=0.001),
    })
    assert best in scores

    kelly = RiskConstrainedKelly()
    size = kelly.size_fraction(max(fp.p_up, 0.5), 1.5, current_dd=0.01)
    assert 0.0 <= size <= kelly.max_fraction

    state = UnifiedMathState(
        fp_edge=fp.edge, fp_p_up=fp.p_up,
        kalman_velocity=last_k["velocity"], kalman_innovation_z=last_k["innovation_z"],
        regime_trend_prob=last_reg["trend_prob"], regime_meanrev_prob=last_reg["meanrev_prob"],
        regime_breakout_prob=last_reg["breakout_prob"], regime_chaos_prob=last_reg["chaos_prob"],
        change_prob=last_reg["change_prob"], hawkes_imbalance=last_h["imbalance"],
        hawkes_branching=last_h["branching_ratio"], jump_prob=last_jt["jump_prob"],
        jump_ratio=last_jt["jump_ratio"], cvar=last_jt["cvar"],
        hurst_proxy=last_rv["hurst_proxy"], roughness=last_rv["roughness"],
    )
    print({"status": "MATH_CORE_SMOKE_OK", "best_action": best.value, "kelly_fraction": size, "state": state.as_dict()})


if __name__ == "__main__":
    main()
