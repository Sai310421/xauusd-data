# DexG AMD Liquidity Cycle v6 — Final Retest Entry

## Evidence from v5
V5 Broad+Disp immediate entry produced first-passage WR 40.3958% at 1s / $0.10 (1041 favorable first, 1536 adverse first, 1197 unresolved; N=3774). Among resolved passages, adverse-first is therefore 59.6042%.

This is a diagnostic first-passage asymmetry, not broker PnL. It confirms that the v5 immediate trigger is too early for the anchored direction.

## Direction hypothesis retained
FinalDirection = InitialSweepDirection remains a hypothesis to test, not a proven rule.

## v6 causal sequence
Accumulation
→ InitialSweep (direction anchor)
→ OppositeLiquidityHarvest (strict or broad/internal)
→ FirstReturn / v5 ARM
→ AdverseExcursion
→ FinalRetest
→ SecondReclaim
→ SecondMicroCISD
→ ConfirmedRawTickDisplacement
→ ENTRY in InitialSweepDirection

The v5 signal is downgraded from ENTRY to ARM.

## Mandatory causal constraints
- Completed context only before ARM.
- After ARM, use current/past QuoteTicks only.
- Never use a completed future D bar to validate an earlier entry.
- Never search backward for entry after a future condition becomes known.
- Raw Bid/Ask only; no OHLC-resample fallback.

## A/B lanes
1. V6_ARM_BASELINE — preserve v5 immediate entry as control.
2. V6_AE05_SECOND — require adverse excursion >= $0.05, then second reclaim + second micro-CISD + displacement.
3. V6_AE10_SECOND — require adverse excursion >= $0.10, then second reclaim + second micro-CISD + displacement.
4. V6_FINAL_RETEST — require adverse excursion, recovery, final retest that holds, then confirmed displacement.
5. V6_INVERSE_ARM — opposite direction at ARM, diagnostic only; must use side-correct Bid/Ask before any PnL interpretation.

## Final-retest state machine
ARM
→ WAIT_ADVERSE
→ ADVERSE_SEEN
→ RECOVERED
→ RETESTING
→ RETEST_HELD
→ SECOND_CISD
→ DISPLACEMENT_CONFIRMED
→ ENTRY

Invalidate setup if price exceeds the opposite-harvest extreme by a configurable structural buffer or if no second trigger arrives within the maximum ARM lifetime.

## Required metrics
For each lane:
- arms, entries, ARM→ENTRY conversion
- long/short count
- adverse excursion distribution before entry
- ARM→entry latency distribution
- first passage at 1/3/5/10/15 sec × $0.10/$0.20/$0.30
- MFE/MAE
- strict external sweep vs broad/internal harvest attribution
- manipulation count
- IFVG state and HTF context
- spread at ARM and entry
- side-correct executable entry price

## Gate
Do not promote v6 to trading logic unless a causal lane materially exceeds 50% on the key second-scalp first-passage tests with adequate N and remains positive after spread/slippage/commission assumptions.