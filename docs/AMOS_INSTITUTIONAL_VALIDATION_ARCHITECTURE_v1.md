# AMOS Institutional Validation Architecture v1

## Purpose
Turn the existing Raw Bid/Ask Nautilus evidence grid into a governed research-to-production validation system. This document defines gates only; it does not claim fund-grade status or strategy performance.

## Promotion lifecycle
CANDIDATE -> WR1 -> WR2 -> WR3 -> WR4 -> WR5 -> SHADOW -> LIMITED_CAPITAL -> PRODUCTION

Any missing mandatory evidence = INVALID. INVALID is never converted to PASS or REJECT.

## Compute lanes
- CORE: low-frequency Trend/ICT candidates.
- BOOST_RANGE: BOOST and independent RANGE_CB candidates.
- MODULE_RECOVERY: complementary modules and strictly limited Recovery/ZR.
- ROBUSTNESS: WFO, Monte Carlo, parameter stability, regime segmentation and stress.

All lanes preserve evidence under `results/ae-bt/<experiment_id>/` and use the common Raw Bid/Ask reusable runner.

## WR5 mandatory evidence
A WR5 decision requires measured or explicitly not-applicable values for: spread, round-trip commission, slippage, execution delay, swap, cashback assumptions, mark-to-market equity, floating drawdown, current open-position state, aggregate exposure, margin level, basket age, unresolved inventory, and event/price-pitch budget.

If any applicable field is absent, `wr5_status=INVALID`.

## Live parity / calibration
For each promoted strategy compare expected vs observed values for return, PF, drawdown, slippage, fill rate, exposure, unresolved inventory and basket age. Store expected and observed values separately; never overwrite BT evidence with live observations.

Recommended lifecycle:
1. SHADOW: no capital, execution/fill observation only.
2. LIMITED_CAPITAL: constrained allocation after WR1-WR5 and robustness pass.
3. PRODUCTION: only after live-parity tolerances are satisfied.
4. DEMOTE/KILL: automatic eligibility when observed safety or parity limits are breached.

## Portfolio governance
Portfolio promotion additionally requires engine contribution, pairwise/cross-engine dependence, aggregate exposure, event budget, margin stress and concentration checks. Combining individually profitable engines is not sufficient.

## Evidence contract
Every experiment must retain: experiment_id, run_id, git_sha, lane_class, runner, dataset/catalog manifest, Raw Bid/Ask inventory, configuration, cost assumptions, test period/OOS period, sizing, metrics, open-state snapshot, WR statuses, rejection/invalid reasons and rollback reference.

## Classification
Every accepted candidate must be exactly one of CORE, BOOST, RANGE_CB, MODULE, PORTFOLIO. Unverified candidates remain CANDIDATE/EXPERIMENTAL and must not be presented as promoted.

## Final AMOS acceptance constraints
- monthly trading return >= 40%
- monthly cashback return >= 10%
- combined monthly return >= 50%
- max DD <= 15%, target <= 10%
- PF >= 1.8
- estimated ruin probability < 1%
- rescue rate <= 5%
- ZR dependency 2-5%
- BE-or-better exits >= 90%

These thresholds do not override WR1-WR5, OOS, robustness or evidence requirements.

## Reality Grade
A grade may only be emitted after all mandatory evidence exists. Suggested labels: A+, A, B, C, INVALID. Grade logic must be deterministic and versioned. Until implemented and calibrated, no grade is authoritative.

## Separation of responsibility
`Sai310421/my-ea-factory-ICT`: strategy/AE source and promotion PRs.
`Sai310421/xauusd-data`: Raw Bid/Ask Nautilus execution and immutable/reproducible BT evidence.

## Next implementation gates
1. Machine-readable evidence schema and validator.
2. Deterministic WR5 INVALID/PASS/REJECT checker.
3. Robustness evidence aggregator.
4. Shadow/live parity schema and deviation checker.
5. Portfolio risk/contribution aggregator.
6. Versioned Reality Grade only after calibration data exists.
