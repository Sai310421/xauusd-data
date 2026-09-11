# Math EDGE Dictionary v2 — 2026-09-11 additions

This file extends `docs/MATH_EDGE_DICTIONARY_v1.md`. It preserves the rule that ORIGINAL source mathematics must be separated from AE DERIVED / COMPOSED mappings.

Status ladder: DISCOVERED -> FORMALIZED -> IMPLEMENTED -> PROXY_PASS -> NAUTILUS_PASS -> RAW_BIDASK_PASS -> BROKER_REALITY_PASS -> VERIFIED.

## 6. Multi-Filtration e-Process Evidence Governor
Status: IMPLEMENTED
Priority: A+
Implementation target: `research/ae_multifiltration_eprocess.py`
Validation target: `research/amos_reversal_v1_v4_rawtick_bt.py` V4 / Adaptive MTF.

### ORIGINAL class
An e-process is a nonnegative adapted process E_t with anytime-valid optional-stopping control under the null, generically satisfying E_P[E_tau] <= 1 for admissible stopping times tau. When evidence streams live on different filtrations, coarse-filtration evidence cannot be naively merged with fine-filtration evidence without adjustment. A valid lifting/adjustment can be built from an increasing adjuster A satisfying the integral condition

int_1^infinity A(e) / e^2 de <= 1.

An adjusted running maximum can then be combined with a fine-filtration e-process while preserving anytime validity.

### AE DERIVED mapping
Treat each TF / model evidence stream as a distinct information filtration:

F_M1,t, F_M5,t, F_M15,t, F_H1,t.

For each completed signal, convert a bounded score into a one-step betting e-value using a predictable bet fraction lambda in [0,1):

E_{t+1} = E_t * (1 + lambda * Z_t),  Z_t in [-1,1].

This multiplicative update is an AE implementation choice; it is not claimed as the unique ORIGINAL construction.

Coarser streams are adjusted before fusion. AE confidence is then used only as a gate / sizing modifier, never as proof that expected trading profit is positive.

Candidate gate:

EntryAllowed_t = 1{ E_fused,t >= E_min and DrawdownGate_t = 1 }.

### First A/B target
BASE = AMOS Reversal V4 Adaptive MTF.
TEST = same V4 signals and execution, with e-process evidence governor applied before entry.

Primary metrics: PF, RF, WR, N, EV_R/trade, MaxDD_R, false-entry proxy, OOS stability. Same Raw Tick source and execution assumptions required.

### Failure modes
- Misspecified one-step evidence increments can destroy practical power even if nonnegative.
- Dependence between TF streams can make naive products invalid; only approved adjusted fusion may be used.
- e-values measure evidence against a null, not trading expectancy.
- Very conservative adjusters can collapse N.
- Regime drift can make the historical null/alternative definition stale.

## 7. Drawdown-Distance Dynamic De-Risking (DDR)
Status: FORMALIZED
Priority: A

### ORIGINAL class
Continuous drawdown-aware leverage controls make position fraction a smooth function of remaining distance to a drawdown boundary rather than a hard on/off threshold. A useful generic family is

f_lambda(d) = kappa * (1 - exp(-lambda d / b)),

with d >= 0 the distance to the drawdown boundary and b a drawdown budget / scale.

### AE DERIVED mapping
For a strategy already proven to have positive native expectancy:

f_AE,t = f_base,t * (1 - exp(-lambda * d_t / b)).

This candidate is forbidden as a rescue layer for negative-expectancy baselines. Promotion requires BASE EV > 0, PF > 1 and Native Fill > 0 before DDR is tested.

### First target
Do not use current Raw/Nautilus G75 Frozen Core as BASE because it is negative / non-parity. Apply DDR only after a positive Native Order candidate exists.

Failure modes: leverage collapse near the boundary, stale DD scale, parameter error, jump-through-boundary risk, reduced recovery speed.

## 8. Disturbance-Affine Distributionally Robust MPC (DA-DR-MPC)
Status: IMPLEMENTED
Priority: A
Implementation target: `research/ae_dr_mpc_recovery.py`
Validation target: `research/ae_reflected_recovery_controller.py`.

### ORIGINAL class
For a controlled system

x_{k+1} = A x_k + B u_k + D w_k,

a disturbance-affine policy uses causal lower-triangular feedback on realized disturbances, generically

u_k = c_k + sum_{j<k} K_{k,j} w_j.

Distributionally robust MPC optimizes over an ambiguity set around the empirical disturbance law, often Wasserstein-based, with worst-case objective / chance or CVaR-style constraints.

### AE DERIVED mapping
Use a compact recovery state, not the full 100D AE state initially:

x = [DebtRatio, DebtDriftRatio, MAERatio, RecoveryAgeRatio, TailProb, SpreadStress, VolStress, DD ratio].

Control vector:

u = [add_scale, reduce_fraction, hedge_scale, recovery_scale].

Disturbance proxy:

w = [price_shock, spread_shock, volatility_shock, recovery_forecast_error].

First implementation is an auditable affine controller / constraint projection proxy, not a solved full Wasserstein min-max MPC. It must therefore be labeled AE DERIVED until the robust optimization problem is solved explicitly.

### First A/B target
BASE = existing Reflected Recovery Controller.
TEST = Reflected Recovery + DA disturbance-affine anticipation layer.

Primary metrics: hard-DD breach count, debt peak, debt residence time, recovery completion rate, intervention turnover, hedge frequency, MaxDD, tail loss, exposure.

### Failure modes
- Wrong linearization can produce harmful control directions.
- Disturbance estimates may be nonstationary.
- High-dimensional K matrices become expensive and unstable.
- Excessive robustness can over-hedge and destroy recovery expectancy.
- This layer cannot create positive directional EDGE by itself.

## 9. Risk-Sensitive / H-infinity Shock Governor
Status: FORMALIZED
Priority: A- / Tail Control

### ORIGINAL class
Risk-sensitive stochastic control replaces a pure mean criterion with exponential sensitivity, generically

J_theta = (1/theta) log E[ exp(theta * G) ],

or its cost-sign equivalent. In linear-quadratic settings, increasing risk sensitivity connects conceptually to worst-case / H-infinity control.

### AE DERIVED mapping
Use as a shock governor on an already-valid strategy / recovery controller:

Cost_AE = ExpectedLoss + eta_tail * TailAmplification + eta_shock * ShockSensitivity.

Candidate actions are lot reduction, add suppression, hedge-lock and emergency impulse; not directional entry generation.

Failure modes: theta / penalty over-tuning, over-conservatism, model mismatch, jump risk beyond local quadratic assumptions.

## Current validation order
1. AMOS Reversal V4 + Multi-Filtration e-Process.
2. Reflected Recovery + DA-DR-MPC proxy.
3. Risk-Sensitive shock overlay on whichever of 1/2 reaches positive native evidence.
4. DDR only on a positive Native Order baseline; never on current negative G75 Raw implementation.

## Promotion rule for these additions
No promotion from IMPLEMENTED to PROXY_PASS without deterministic tests. No trading claim without same-source A/B. No final promotion without Native Order / Raw BidAsk evidence where execution is relevant.
