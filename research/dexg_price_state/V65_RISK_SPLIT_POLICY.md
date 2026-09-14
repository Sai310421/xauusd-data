# DexG v6.5 Adaptive Risk + 3-Split Policy

## Purpose
Convert a validated structural edge into higher capital efficiency without fabricating profitability.

## Core rules

### 1. Low-DD -> increase risk, not entry frequency
Risk sizing is allowed to scale only after broker-cost PnL is available.

Risk multiplier ladder (initial test grid):
- MaxDD <= 2.0%: 1.50x baseline risk
- 2.0% < MaxDD <= 3.5%: 1.25x
- 3.5% < MaxDD <= 5.0%: 1.00x
- MaxDD > 5.0%: reduce risk / reject for Prop lane

Hard rule: scaling must preserve signal logic. Do not loosen entry filters merely to raise N.

### 2. Low N -> 3-split entry
When trade count is too low for the target capital-turnover objective, preserve one signal but split its exposure into 3 independent tranches.

Default split:
- L1: 50%
- L2: 30%
- L3: 20%

The split does not create three independent signals and must not triple statistical N in reporting.

### 3. Split timing for second-scalp probe lane
Test these causal variants:
- SPLIT_A: 50% at ARM, 30% on first favorable micro-displacement, 20% on retest without structural invalidation.
- SPLIT_B: 50% at ARM, 30% after +0.10 favorable excursion, 20% after +0.20 favorable excursion.
- SPLIT_C: 50% at ARM, 30% on tick-acceleration confirmation, 20% on PA confirmation (failed break / wick rejection / close-location confirmation).

No averaging down in the base model. Adds are profit-direction only.

### 4. Position-size constraint
Total exposure of L1+L2+L3 equals the risk budget of one original signal. Risk scaling and split-entry are separate controls:

TotalRisk = BaselineRisk * RiskMultiplier
L1 = 0.50 * TotalRisk
L2 = 0.30 * TotalRisk
L3 = 0.20 * TotalRisk

### 5. Required v6.5 metrics
Report both unscaled and scaled results:
- Real signal count N
- Filled tranche count
- Avg tranches per signal
- WR / PF / Expectancy
- Net PnL after spread, commission, slippage, cashback
- MaxDD % / $
- RF
- Monthly return (21 business-day normalized only after valid broker-PnL lifecycle)
- Total cashback and cashback contribution %
- Exposure time
- Max concurrent exposure
- Risk multiplier used

### 6. Selection objective
Do not maximize monthly return alone.

Primary objective:
maximize NetReturn subject to MaxDD ceiling and positive net expectancy.

Preferred comparison:
Score = NetReturn / max(MaxDD, epsilon)

with minimum constraints:
- Net expectancy > 0
- PF > 1.0 before scaling
- No increase in ruin probability from split/add logic

## Interpretation
- If DD is low but expectancy is positive, increase risk.
- If N is low, use 3-split execution to improve capital deployment and exit flexibility, but do not count tranches as independent signals.
- If expectancy is negative after execution costs, neither leverage nor 3-split is permitted as a fix; entry/exit edge must be repaired first.
