# Math EDGE Dictionary v1

This document separates ORIGINAL source mathematics from AE DERIVED / COMPOSED mappings and defines promotion gates.

## Status ladder
DISCOVERED -> FORMALIZED -> IMPLEMENTED -> PROXY_PASS -> NAUTILUS_PASS -> RAW_BIDASK_PASS -> BROKER_REALITY_PASS -> VERIFIED

No candidate may skip evidence levels.

## 1. Multi-Variable Conformal Joint Reachable Set
Status: FORMALIZED
Priority: A

### ORIGINAL class
Use vector-valued nonconformity scores and jointly calibrated prediction sets rather than one scalar score / one scalar threshold.

Generic set form:
C_t(alpha) = { y in R^d : S(y, x_t) in A_alpha }
where A_alpha is calibrated so that coverage is at least 1-alpha under the source assumptions.

### AE DERIVED mapping
Target vector:
y_t = [MFE_T, MAE_T, tau_EBE, TailDepth]^T

Use the calibrated joint set to route action:
- aggressive size only if lower MFE bound is positive enough,
- reject if MAE / TailDepth upper bounds violate DD budget,
- Natural Recovery only if tau_EBE upper quantile is inside the allowed horizon.

Candidate action score:
AE_SafeEdge = L(MFE) - lambda_MAE U(MAE) - lambda_tau U(tau_EBE) - lambda_tail U(TailDepth)

This AE_SafeEdge equation is COMPOSED, not an ORIGINAL source equation.

Failure modes: regime drift, insufficient calibration sample, dimensional conservatism, weak exchangeability.

## 2. Certified Wasserstein Robust Utility
Status: FORMALIZED
Priority: A

### ORIGINAL class
max_x inf_{P: W1(P, P_hat) <= epsilon} E_P[U(x^T R)]

The research contribution of interest is finite tractable approximation / certification of robust utility and near-optimality under Wasserstein ambiguity.

### AE DERIVED mapping
R can represent strategy-state outcomes rather than assets:
R = [ProfitRate, -DebtDrift, -TailLoss, -RecoveryTime].

AE chooses policy / sizing vector x against a distributional neighborhood, not only the empirical distribution.

Failure modes: epsilon miscalibration, support mismatch, excessive conservatism, state vector instability.

## 3. Conformal Kelly Uncertainty Governor
Status: FORMALIZED
Priority: B+

### ORIGINAL class
Kelly approximation f* approximately mu / sigma^2 combined with conformal interval width as an uncertainty scale.

### AE DERIVED mapping
f_AE = clip(f_Kelly * g(width) * h(miscoverage), f_min, f_max)

Recommended first implementation:
g(width) = 1 / (1 + beta * width_norm)
h(miscoverage) = exp(-gamma * excess_miscoverage)

This exact governor is AE DERIVED.

Use for lot, add intensity, maximum layers, and recovery aggressiveness. Do not use interval coverage as proof of expected profit.

Failure modes: unstable width, lag after regime breaks, sizing oscillation, calibration/return disconnect.

## 4. Hurst / Non-Brownian DD Scaling
Status: FORMALIZED
Priority: B / Risk Engine

### ORIGINAL class
For self-similar Gaussian long-memory processes, dispersion scaling differs from Brownian sqrt(T); relative correction scales with T^(H-1/2).

### AE DERIVED mapping
TailBoundary(T) = TailBoundary_BM(T) * T^(H_t - 1/2)

Use only as a calibration modifier after H is estimated with uncertainty. Never promote a single short-window H estimate directly to execution control.

Failure modes: H estimator variance, non-fBM market microstructure, jumps, regime switching.

## 5. High-Dimensional Reflected Recovery Controller
Status: IMPLEMENTED
Priority: A / Control OS Candidate
Implementation: `research/ae_reflected_recovery_controller.py`
Tests: `research/test_ae_reflected_recovery_controller.py`

### ORIGINAL class
Multidimensional singular / reflected stochastic control keeps a controlled state inside an admissible region by applying finite-variation control only when the state reaches the intervention boundary.

Generic controlled diffusion:

dX_t = b(X_t)dt + Sigma(X_t)dW_t + G(X_t)dU_t

A typical value function is:

V(x) = inf_U E_x[ integral exp(-rho t)c(X_t)dt + integral exp(-rho t)k(X_t)^T d|U_t| ]

The HJB variational inequality contains gradient constraints. For control direction g_j:

-k_j^+ <= g_j(x)^T grad V(x) <= k_j^-

inside the no-action region. At the boundary, minimal intervention is applied to reflect the state toward the admissible region.

### AE DERIVED mapping
First implementation uses an explicit low-dimensional state instead of an opaque learned 30-100D policy:

X_t = [Debt, DebtDrift, MAE, RecoveryAge, TailProbability, SpreadStress, VolatilityStress, ShockScore, DD, MarginLevel, p_NaturalRecovery].

Recovery hazard:

H_t = sum_i w_i * normalized_state_i

Natural-recovery adjusted hazard:

S_t = clip(H_t - lambda_NR * p_NR,t, 0, 1)

where:

p_NR,t = P(tau_EBE <= T | X_t).

Action mapping:

- S_t < theta_1 -> WAIT
- theta_1 <= S_t < theta_2 -> STOP_ADD
- theta_2 <= S_t < theta_3 -> REDUCE by the minimum reflected fraction
- theta_3 <= S_t < theta_4 -> HEDGE_LOCK
- S_t >= theta_4 -> SELECTIVE_RECOVERY

Shock, hard DD, and margin safety boundaries override the reflected policy and route to EMERGENCY_IMPULSE.

The initial proportional REDUCE amount is continuous between the REDUCE and HEDGE boundaries rather than forcing full liquidation. This is an AE DERIVED approximation to minimal reflected intervention.

### State -> Action -> Intended effect
- Low hazard + high Economic-BE probability -> WAIT -> preserve natural recovery and avoid churn.
- Rising hazard -> STOP_ADD -> stop increasing debt while retaining recovery optionality.
- Medium-high hazard -> REDUCE minimally -> move the state back toward the admissible region while limiting transaction cost.
- High hazard -> HEDGE_LOCK -> freeze further directional debt growth.
- Tail/recovery boundary -> SELECTIVE_RECOVERY -> isolate the bad state and route recovery deliberately.
- Shock / DD / margin breach -> EMERGENCY_IMPULSE -> bypass slow reflection for discontinuous risk.

### Failure modes
- Bad normalization limits or weights can make the hazard score meaningless.
- p_NR estimation error can delay needed intervention or trigger intervention too early.
- Fixed boundaries may lag regime changes.
- Jump/gap events can cross several reflected boundaries instantaneously.
- A low-dimensional explicit proxy may omit relevant state interactions.
- The implementation is not evidence of edge until same-catalog Raw Bid/Ask A/B validation passes.

### Required next validation
Run BASE vs REFLECTED under identical Raw Bid/Ask QuoteTick data, execution assumptions, strategy SHA and exposure reporting. First promotion target is PROXY_PASS only after unit tests and deterministic synthetic path tests; canonical promotion requires Nautilus Raw Bid/Ask evidence.

## Promotion gates
All candidate A/B tests must use the same raw Bid/Ask catalog hash, same execution assumptions, same baseline strategy hash, and same initial display basis.

Common view:
- Initial display equity: $1,000
- Horizon: 21 business days normalized where valid
- Monthly21
- Daily compound rate
- WR
- N / 21D and N / Day
- PF
- RF
- MaxDD
- Net profit

Minimum directional promotion rule for a risk/sizing EDGE:
1. OOS or equivalent held-out segment required.
2. No material N collapse unless explicitly intended and compensated by higher expectancy.
3. MaxDD must not worsen materially.
4. At least one of PF, RF, or return must improve without hidden leverage increase.
5. Mean exposure / gross risk must be reported so the comparison cannot win only by taking more risk.
6. RAW_BIDASK / Nautilus evidence must include run_id, SHA, catalog hash, strategy hash, and artifact.

## Explicit rejection memory
Standalone short-horizon OU Natural-Recovery routing is currently REJECTED as a primary selector based on prior OOS structure-gate results. OU may re-enter only inside Regime-Switching / Jump / First-Passage compositions.