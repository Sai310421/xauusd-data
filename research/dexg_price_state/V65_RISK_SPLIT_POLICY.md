# DexG v6.5 Adaptive Risk + 3-Split Policy

## Purpose
Convert a validated structural edge into higher capital efficiency without fabricating profitability.

## Core rules

### 1. Low DD -> increase risk
Risk sizing is allowed to scale only after broker-cost PnL is available and net expectancy is positive.

Initial risk grid:
- MaxDD <= 2.0%: test 1.00x / 1.25x / 1.50x / 2.00x
- 2.0% < MaxDD <= 3.5%: test 1.00x / 1.25x / 1.50x
- 3.5% < MaxDD <= 5.0%: test 0.75x / 1.00x / 1.25x
- MaxDD > 5.0%: reduce risk or reject for Prop lane

Do not loosen signal conditions just to raise frequency. Risk is scaled only while the same edge survives broker costs.

### 2. Low N -> use 3-split execution
When real signal count is too low for the desired capital-turnover objective, keep the signal definition unchanged but execute the same valid setup through three predeclared tranches.

Default exposure split:
- L1: 50%
- L2: 30%
- L3: 20%

Important reporting rule:
- N_signal = number of independent setup signals.
- N_fill = number of actually filled tranches/orders.
- N_signal must never be multiplied by three.
- N_fill and total lots are reported separately because they affect turnover, commission and cashback.

### 3. Corrected 3-split entry design
The previous rule "adds are profit-direction only" was too restrictive for the intended low-N solution. v6.5 must compare both staggered-entry and confirmation-add execution while keeping total risk fixed.

Test these causal variants:

#### SPLIT_ZONE_50_30_20
Preplanned three-price entry inside the still-valid PA/liquidity setup zone.
- L1 50%: first ARM/entry trigger.
- L2 30%: first causal retest/pullback into the valid entry zone.
- L3 20%: deeper retest toward the PA/structure boundary, only while hard invalidation has not fired.

This is not martingale sizing: lot fractions are fixed before entry and never increase because price moved against the position.

#### SPLIT_CONFIRM_50_30_20
- L1 50%: ARM/entry trigger.
- L2 30%: favorable micro-displacement / tick acceleration.
- L3 20%: PA confirmation such as failed break, wick rejection, reclaim/close-location confirmation or valid retest.

#### SPLIT_HYBRID_50_30_20
- L1 50%: ARM/entry trigger.
- L2 30%: whichever arrives first, valid retest or favorable displacement.
- L3 20%: second independent confirmation, with structural invalidation still false.

### 4. PA requirements for tranches
A tranche may fill only while the original causal setup remains valid. At minimum evaluate:
- wick rejection vs acceptance
- close location inside bar/range
- body/range ratio
- compression -> expansion
- failed break / liquidity probe
- reclaim / loss of local level
- tick velocity and acceleration

L2/L3 are cancelled immediately after hard invalidation.

### 5. Position-size constraint
Risk scaling and 3-split execution are independent controls.

TotalRisk = BaselineRisk * RiskMultiplier
L1 = 0.50 * TotalRisk
L2 = 0.30 * TotalRisk
L3 = 0.20 * TotalRisk

Total planned risk of L1+L2+L3 may not exceed TotalRisk. No martingale multiplier is permitted inside the split.

### 6. Low-N interpretation
Three splits do not create three independent statistical opportunities, but they can improve:
- average entry price
- fill probability
- partial-exit flexibility
- turnover / traded lots when more than one tranche fills
- cashback contribution

Therefore both signal-level and order-level KPIs must be shown.

### 7. Required v6.5 metrics
Report both unscaled and adaptive-risk results:
- N_signal
- N_fill
- fill conversion by L1/L2/L3
- average filled tranches per signal
- total lots
- WR
- PF
- net expectancy per signal and per filled order
- Net PnL after spread, commission, slippage and cashback
- MaxDD % and $
- RF
- ruin / stop-out rate if applicable
- 21-business-day normalized return only after a valid broker-PnL lifecycle exists
- total cashback and cashback contribution %
- exposure time
- max concurrent exposure
- selected risk multiplier

### 8. Selection objective
Do not maximize monthly return alone.

Primary objective:
maximize NetReturn subject to a MaxDD ceiling and positive net expectancy.

Preferred score:
Score = NetReturn / max(MaxDD, epsilon)

Minimum gates before risk-up:
- Net expectancy > 0 after execution costs
- PF > 1.0
- risk-up must not violate the DD ceiling
- split logic must not materially increase ruin probability

## Interpretation
- If DD is low and net expectancy remains positive, raise risk.
- If N_signal is low, use the 50/30/20 3-split execution to exploit each valid setup more efficiently.
- Staggered retest entries are allowed only as preplanned fixed-risk tranches, not as loss-driven martingale averaging.
- If expectancy is negative after execution costs, neither leverage nor 3-split is allowed to hide the problem; entry/exit edge must be repaired first.
