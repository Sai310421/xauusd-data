# MultiEDGE Final Selection Standard

## Target
Build **8–12 independent EDGE BOTs**. Each BOT owns its own signal/entry state. A Supervisor detects regime and selects only validated BOTs for that regime.

## Families
- Trend: target 3–4 promoted BOTs
- Range-follow: target 2–4 promoted BOTs
- Reversal: target 2–4 promoted BOTs

No family quota is forced if the EDGE does not pass the gate.

## Mandatory promotion gate
An EDGE cannot enter the final BOT pool unless:

- PF >= **1.20**
- Expectancy > 0
- Raw BidAsk verification, no external OHLC resample
- Cost/slippage stress survives
- Time-split/regime-split stability is acceptable
- Sufficient N; initial target N >= 100, then larger out-of-sample verification

Classification:
- PF < 1.00: reject
- 1.00 <= PF < 1.20: hold / reusable component only
- 1.20 <= PF < 1.50: candidate
- 1.50 <= PF < 2.00: promote candidate
- PF >= 2.00: strong candidate

## Final utility
PF is a hard gate, not the ranking score.

After PF >= 1.20:

`Utility = max(EV, 0) * sqrt(Frequency) * Stability * CostSurvival * Independence`

This avoids selecting a high-PF but rare/fragile/redundant EDGE.

## Independence
Do not combine all signals with AND. Each EDGE remains a BOT. The Supervisor routes regime and ranks eligible BOTs. Correlated BOTs receive an independence penalty or are prevented from taking duplicated exposure.

## Initial 12 slots
### Trend
1. T1_EMA21_FLOW
2. T2_MTF_ALIGN
3. T3_BREAK_RETEST_POI
4. T4_ACCELERATION

### Range-follow
5. R1_RANGE_DIRECTIONAL_BIAS
6. R2_COMPRESSION_EXPANSION
7. R3_MICRO_BREAK_RETEST
8. R4_RANGE_LIQUIDITY_RUN

### Reversal
9. V1_SWEEP_MSS
10. V2_EXTREME_REJECTION
11. V3_IFVG_REVERSAL
12. V4_BPR_REVERSAL

These are slots, not pre-approved EDGE. Failed slots are replaced.

## Exit
Entry EDGE and Exit EDGE are validated separately. Current common strong exit candidate: Dynamic ATR Trail. A BOT is not promoted just because an exit rescues PF; entry quality, MFE/MAE and robustness remain separately measured.

## Current state
- OB∩FVG standalone: HOLD/component, because only one tested depth barely exceeded PF 1 and did not reach PF 1.20.
- OB∩IFVG and OB∩BPR: under Raw Tick validation.

## Build order
1. Validate independent EDGE candidates under identical Raw BidAsk conditions.
2. Promote only PF >= 1.20 and positive-EV EDGE.
3. Run stability/cost/out-of-sample gates.
4. Measure pairwise signal/PnL correlation and compute Independence.
5. Fill 8–12 final BOT slots.
6. Add Regime Supervisor and portfolio exposure gate.
7. Run combined multi-BOT Raw Tick portfolio BT.
