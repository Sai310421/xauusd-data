# CHECKPOINT 2026-09-21 — Video Method Broad Numeric EDGE Finder v1.1

## Goal
Continue the uploaded video's "numbers first, patterns later" process on real Raw Bid/Ask data after v1.0 produced no strict OOS-confirmed EDGE.

## Sweep
- Raw Bid/Ask QuoteTick only; no OHLC fallback.
- 6 symbols.
- M5 numeric view.
- 4 transparent AMOS numeric families:
  - IMPULSE_CONT
  - IMPULSE_REV
  - STREAK3_CONT
  - MEAN_REVERT20
- 4 train-only threshold quantiles per family: 50/65/80/90%.
- 2 directions.
- Total: 192 lanes.
- Monte Carlo: 2,000 paths x 400 trades.
- Selection: first 60% discovery -> next 20% validation -> last 20% untouched final OOS.
- No lane may be selected from validation/OOS results alone.

## Evidence
Run: 35602882815
SHA: 2e46feed512682d9305da8c7ca7e9a029e82e8f2
Artifact ID: 10640810062
Artifact: video-edge-broad-v11-35602882815

Results:
- discovery candidates: 16
- validation passed: 3
- final OOS confirmed: **0**

The three validation survivors were:
1. EURUSD / IMPULSE_CONT / Long / P1
   - IS: MedRF 3.660, NoLoss 1.40%, EV +0.05301, N=413
   - VAL: MedRF 2.368, NoLoss 5.30%, EV +0.03125, N=137
   - FINAL OOS: MedRF -0.665, NoLoss 87.10%, EV -0.01283, N=116
2. EURUSD / STREAK3_CONT / Long / P2
   - IS: MedRF 3.005, NoLoss 3.00%, EV +0.05780, N=249
   - VAL: MedRF 2.627, NoLoss 3.85%, EV +0.03820, N=92
   - FINAL OOS: MedRF -0.988, NoLoss 100%, EV -0.10057, N=32
3. EURUSD / IMPULSE_CONT / Long / P3
   - IS: MedRF 3.772, NoLoss 1.30%, EV +0.07656, N=121
   - VAL: MedRF 2.278, NoLoss 5.90%, EV +0.03853, N=45
   - FINAL OOS: insufficient trades for lane metric.

## Finding
No lane survives strict three-stage chronological validation. The dominant apparent bias is EURUSD Long continuation, but it breaks in the final period. It is therefore a regime-dependent observation, not a promoted EDGE.

This is precisely the failure mode the video-style sweep can hide if the full sample is filtered without untouched OOS. AMOS keeps the source workflow but adds fail-closed chronology.

## Next search
Move to the canonical 90-day Raw Tick horizon and use rolling walk-forward stability rather than one 30-day regime split. Add session/time-of-day as a purely numerical conditioning axis before adding any visual chart pattern.

No production weighting. No execution authority.
