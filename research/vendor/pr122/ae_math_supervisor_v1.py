"""AE Math Supervisor v1.

External risk/recovery/sizing supervisor for AMOS/G75.
The signal generator / Frozen Core is intentionally not modified.

ORIGINAL-source mathematics represented here:
- Risk-constrained Kelly negative-moment constraint.
- First-passage / scale-function logic (v1 uses the drifted-Brownian closed-form special case).
- Receding-horizon MPC.
- 1-Wasserstein DRO dual penalty for linear exposure under an unconstrained support model.

AE-specific composition (DERIVED): convert those outputs into a conservative sizing multiplier.
This module is research code; promotion requires Raw Tick OOS validation and realistic costs.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Callable, Iterable, Sequence


EPS = 1e-12


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        raise ValueError("empty sample")
    return sum(xs) / len(xs)


@dataclass(frozen=True)
class KellyConfig:
    enabled: bool = True
    lambda_risk: float = 2.0
    max_fraction: float = 1.0
    grid_points: int = 400
    min_wealth_relative: float = 1e-6


@dataclass(frozen=True)
class RecoveryConfig:
    enabled: bool = True
    ruin_level: float = 0.0
    recovery_level: float = 1.0
    min_probability: float = 0.50
    full_probability: float = 0.80


@dataclass(frozen=True)
class WassersteinConfig:
    enabled: bool = True
    rho: float = 0.05
    min_robust_edge: float = 0.0
    full_robust_edge: float = 0.002


@dataclass(frozen=True)
class MPCConfig:
    enabled: bool = True
    horizon: int = 4
    discount: float = 0.98
    dd_penalty: float = 2.0
    debt_penalty: float = 1.0
    inventory_penalty: float = 0.25
    cost_penalty: float = 1.0


@dataclass(frozen=True)
class SupervisorConfig:
    kelly: KellyConfig = KellyConfig()
    recovery: RecoveryConfig = RecoveryConfig()
    wasserstein: WassersteinConfig = WassersteinConfig()
    mpc: MPCConfig = MPCConfig()
    hard_min_margin_level: float = 500.0
    soft_margin_level: float = 650.0
    max_multiplier: float = 1.0


@dataclass(frozen=True)
class AEState:
    equity: float
    peak_equity: float
    debt: float
    inventory: float
    margin_level: float
    recovery_coordinate: float

    @property
    def drawdown_fraction(self) -> float:
        if self.peak_equity <= 0:
            return 1.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)


@dataclass(frozen=True)
class MPCAction:
    name: str
    size_multiplier: float
    expected_profit: float
    dd_delta: float
    debt_delta: float
    inventory_delta: float
    cost: float


@dataclass(frozen=True)
class SupervisorDecision:
    multiplier: float
    kelly_multiplier: float
    recovery_multiplier: float
    wasserstein_multiplier: float
    mpc_multiplier: float
    margin_multiplier: float
    mpc_action: str
    recovery_probability: float
    robust_edge: float
    reason: str


def risk_constrained_kelly_fraction(
    returns: Sequence[float],
    cfg: KellyConfig = KellyConfig(),
) -> float:
    """Largest sampled fraction satisfying E[(1+fR)^(-lambda)] <= 1.

    This directly implements the negative-moment risk constraint on one-period
    wealth relatives. It does NOT claim the finite-state theorem extends unchanged
    to dependent path-wise AE returns; validation must be OOS/path based.
    """
    if not cfg.enabled:
        return cfg.max_fraction
    if cfg.lambda_risk <= 0 or cfg.grid_points < 2:
        raise ValueError("invalid KellyConfig")
    rs = list(returns)
    if not rs:
        return 0.0

    best_f = 0.0
    best_growth = float("-inf")
    for i in range(cfg.grid_points + 1):
        f = cfg.max_fraction * i / cfg.grid_points
        relatives = [1.0 + f * r for r in rs]
        if min(relatives) <= cfg.min_wealth_relative:
            continue
        neg_moment = _mean([w ** (-cfg.lambda_risk) for w in relatives])
        if neg_moment > 1.0 + 1e-10:
            continue
        growth = _mean([log(w) for w in relatives])
        if growth >= best_growth:
            best_growth = growth
            best_f = f
    return max(0.0, min(cfg.max_fraction, best_f))


def brownian_first_passage_recovery_probability(
    x: float,
    lower: float,
    upper: float,
    drift: float,
    volatility: float,
) -> float:
    """P_x(tau_upper < tau_lower) for dX=mu dt + sigma dW.

    This is the closed-form diffusion special case of the scale-function /
    first-passage framework. Full jump-Levy scale-function numerics are a v2 item.
    """
    if upper <= lower:
        raise ValueError("upper must exceed lower")
    if x <= lower:
        return 0.0
    if x >= upper:
        return 1.0
    if volatility <= 0:
        if drift > 0:
            return 1.0
        if drift < 0:
            return 0.0
        return (x - lower) / (upper - lower)

    y = x - lower
    b = upper - lower
    if abs(drift) < 1e-12:
        return y / b

    z_y = -2.0 * drift * y / (volatility * volatility)
    z_b = -2.0 * drift * b / (volatility * volatility)
    # Stable enough for the bounded AE state ranges used in v1.
    num = 1.0 - exp(max(-700.0, min(700.0, z_y)))
    den = 1.0 - exp(max(-700.0, min(700.0, z_b)))
    if abs(den) < EPS:
        return y / b
    return max(0.0, min(1.0, num / den))


def probability_to_multiplier(p: float, cfg: RecoveryConfig) -> float:
    if not cfg.enabled:
        return 1.0
    if p <= cfg.min_probability:
        return 0.0
    if p >= cfg.full_probability:
        return 1.0
    return (p - cfg.min_probability) / (cfg.full_probability - cfg.min_probability)


def wasserstein_linear_robust_edge(
    scenario_returns: Sequence[Sequence[float]],
    exposure: Sequence[float],
    rho: float,
) -> float:
    """Worst-case mean for linear payoff <exposure, X> under W1 ambiguity.

    For an unconstrained support and Euclidean transport cost, duality gives
    empirical mean payoff - rho * ||exposure||_2. This is intentionally a narrow,
    auditable v1 primitive rather than a generic claim for all DRO formulations.
    """
    if rho < 0:
        raise ValueError("rho must be non-negative")
    if not scenario_returns:
        return float("-inf")
    d = len(exposure)
    if d == 0 or any(len(row) != d for row in scenario_returns):
        raise ValueError("dimension mismatch")
    payoffs = [sum(w * x for w, x in zip(exposure, row)) for row in scenario_returns]
    norm = sum(w * w for w in exposure) ** 0.5
    return _mean(payoffs) - rho * norm


def robust_edge_to_multiplier(edge: float, cfg: WassersteinConfig) -> float:
    if not cfg.enabled:
        return 1.0
    if edge <= cfg.min_robust_edge:
        return 0.0
    if edge >= cfg.full_robust_edge:
        return 1.0
    return (edge - cfg.min_robust_edge) / (cfg.full_robust_edge - cfg.min_robust_edge)


def mpc_select_action(
    state: AEState,
    actions: Sequence[MPCAction],
    cfg: MPCConfig = MPCConfig(),
    transition: Callable[[AEState, MPCAction], AEState] | None = None,
) -> MPCAction:
    """Small discrete receding-horizon MPC by exhaustive search.

    The optimizer evaluates action sequences, but only the first action is returned.
    This is suitable for a bounded AE action set (e.g. STOP/HALF/NORMAL/RECOVER).
    """
    if not actions:
        raise ValueError("no MPC actions")
    if not cfg.enabled:
        return min(actions, key=lambda a: abs(a.size_multiplier - 1.0))

    def default_transition(s: AEState, a: MPCAction) -> AEState:
        next_equity = max(EPS, s.equity + a.expected_profit)
        peak = max(s.peak_equity, next_equity)
        return AEState(
            equity=next_equity,
            peak_equity=peak,
            debt=max(0.0, s.debt + a.debt_delta),
            inventory=max(0.0, s.inventory + a.inventory_delta),
            margin_level=s.margin_level,
            recovery_coordinate=s.recovery_coordinate,
        )

    trans = transition or default_transition

    def stage_cost(s: AEState, a: MPCAction) -> float:
        projected_dd = max(0.0, s.drawdown_fraction + a.dd_delta)
        projected_debt = max(0.0, s.debt + a.debt_delta)
        projected_inv = max(0.0, s.inventory + a.inventory_delta)
        return (
            -a.expected_profit
            + cfg.dd_penalty * projected_dd * projected_dd
            + cfg.debt_penalty * projected_debt * projected_debt
            + cfg.inventory_penalty * projected_inv * projected_inv
            + cfg.cost_penalty * max(0.0, a.cost)
        )

    best_first = actions[0]
    best_total = float("inf")

    def search(s: AEState, depth: int, total: float, first: MPCAction | None) -> None:
        nonlocal best_first, best_total
        if depth >= cfg.horizon:
            if total < best_total:
                best_total = total
                best_first = first or actions[0]
            return
        disc = cfg.discount ** depth
        for a in actions:
            nxt = trans(s, a)
            c = total + disc * stage_cost(s, a)
            if c >= best_total:
                continue
            search(nxt, depth + 1, c, first or a)

    search(state, 0, 0.0, None)
    return best_first


def margin_multiplier(margin_level: float, hard_min: float, soft: float) -> float:
    if margin_level <= hard_min:
        return 0.0
    if margin_level >= soft:
        return 1.0
    return (margin_level - hard_min) / (soft - hard_min)


class AEMathSupervisorV1:
    """Conservative compositor. Frozen Core emits direction; supervisor sizes only."""

    def __init__(self, cfg: SupervisorConfig = SupervisorConfig()) -> None:
        self.cfg = cfg

    def decide(
        self,
        state: AEState,
        one_period_returns: Sequence[float],
        recovery_drift: float,
        recovery_volatility: float,
        dro_scenarios: Sequence[Sequence[float]],
        dro_exposure: Sequence[float],
        mpc_actions: Sequence[MPCAction],
    ) -> SupervisorDecision:
        k = risk_constrained_kelly_fraction(one_period_returns, self.cfg.kelly)
        p = brownian_first_passage_recovery_probability(
            x=state.recovery_coordinate,
            lower=self.cfg.recovery.ruin_level,
            upper=self.cfg.recovery.recovery_level,
            drift=recovery_drift,
            volatility=recovery_volatility,
        )
        r = probability_to_multiplier(p, self.cfg.recovery)
        robust_edge = wasserstein_linear_robust_edge(
            dro_scenarios, dro_exposure, self.cfg.wasserstein.rho
        )
        w = robust_edge_to_multiplier(robust_edge, self.cfg.wasserstein)
        action = mpc_select_action(state, mpc_actions, self.cfg.mpc)
        m = max(0.0, min(1.0, action.size_multiplier))
        g = margin_multiplier(
            state.margin_level,
            self.cfg.hard_min_margin_level,
            self.cfg.soft_margin_level,
        )
        multiplier = max(0.0, min(self.cfg.max_multiplier, k, r, w, m, g))

        gates = {"kelly": k, "recovery": r, "wasserstein": w, "mpc": m, "margin": g}
        binding = min(gates, key=gates.get)
        reason = f"binding_gate={binding}; frozen_core_signal_unchanged"
        return SupervisorDecision(
            multiplier=multiplier,
            kelly_multiplier=k,
            recovery_multiplier=r,
            wasserstein_multiplier=w,
            mpc_multiplier=m,
            margin_multiplier=g,
            mpc_action=action.name,
            recovery_probability=p,
            robust_edge=robust_edge,
            reason=reason,
        )
