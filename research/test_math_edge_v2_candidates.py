from research.ae_multifiltration_eprocess import EProcess, MultiFiltrationGovernor, bounded_signal
from research.ae_dr_mpc_recovery import RecoveryState, Disturbance, disturbance_affine_control


def test_eprocess_nonnegative_and_evidence_rises_on_positive_sequence():
    ep = EProcess(bet_fraction=0.25)
    start = ep.value
    for _ in range(8):
        ep.update(0.8)
    assert ep.value > start
    assert ep.value > 0
    assert ep.running_max >= ep.value


def test_eprocess_falls_on_negative_sequence():
    ep = EProcess(bet_fraction=0.25)
    for _ in range(8):
        ep.update(-0.8)
    assert 0 < ep.value < 1.0


def test_multifiltration_governor_requires_evidence():
    g = MultiFiltrationGovernor(min_evidence=1.05)
    assert not g.allow_entry()
    for tf in ("M1", "M5", "M15", "H1"):
        for _ in range(6):
            g.update(tf, 0.85)
    assert g.allow_entry()
    assert g.fused_value() >= 1.05


def test_bounded_signal_limits():
    assert bounded_signal(10.0) == 1.0
    assert bounded_signal(-10.0) == -1.0


def low_state():
    return RecoveryState(0.10,0.05,0.10,0.10,0.05,0.05,0.05,0.05)


def high_state():
    return RecoveryState(0.95,0.90,0.85,0.80,0.75,0.70,0.80,0.90)


def test_drmpc_low_stress_preserves_adds():
    c = disturbance_affine_control(low_state(), Disturbance(0.0,0.0,0.0,0.0))
    assert c.add_scale > 0.5
    assert c.reduce_fraction < 0.25
    assert c.hedge_scale < 0.25


def test_drmpc_high_stress_derisks():
    c = disturbance_affine_control(high_state(), Disturbance(1.0,1.0,1.0,1.0))
    assert c.add_scale < 0.25
    assert c.reduce_fraction > 0.25
    assert c.hedge_scale > 0.25
    assert c.robust_score > 0.5


def test_shock_monotonicity():
    s = RecoveryState(0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5)
    calm = disturbance_affine_control(s, Disturbance(0,0,0,0))
    shock = disturbance_affine_control(s, Disturbance(1,1,1,1))
    assert shock.add_scale <= calm.add_scale
    assert shock.reduce_fraction >= calm.reduce_fraction
    assert shock.hedge_scale >= calm.hedge_scale
