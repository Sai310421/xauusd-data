# C0 Multi-TF Independent Entry Spec v1.1

## Scope
C0 = QM + Harmonic + DrawOnLiquidity + LTF sweep/MSS style entry engine.

Independent entry timeframes:
- M1
- M5
- M15
- H1
- H2
- H4
- D1

Each TF runs as an independent state machine:

C0_M1 || C0_M5 || C0_M15 || C0_H1 || C0_H2 || C0_H4 || C0_D1

No higher-TF confirmation is mandatory. Cross-TF agreement is a boost only.

## Per-TF state
Each timeframe owns independent:
- swing state
- X/A/B/C/D candidate
- harmonic score
- QM zone
- liquidity sweep state
- MSS/CHOCH state
- entry basket
- stop/targets
- G75 state
- risk ledger

## Core sequence
For bullish setup:
1. detect causal swing chain
2. form XABCD candidate
3. validate harmonic ratio bands
4. detect QM demand overlap
5. detect local liquidity sweep/reclaim
6. confirm structure break (MSS/CHOCH)
7. create 3-part entry basket
8. manage TP1/TP2/runner
9. optionally hand off to same-direction G75 profit chase

Bearish is symmetric.

## Initial ratio bands from image reconstruction
- B: 0.697-0.706
- C: 0.654-0.667
- D: 0.825-0.835
- expansion reference: 1.255-1.302

These are reconstruction parameters, not proven source code values. Backtest must test tolerance robustness.

## Entry split
Default: 50/30/20.

## G75 bridge
Direction is inherited from C0 initial basket only.
No reverse-direction add and no loss-side averaging.
Initial research values:
- trigger 0.12
- add 0.025
- reversal 0.20
- max adds 10

## Risk gates
Prop research gate:
- Max floating DD <= 5.0%
- Max daily DD <= 3.5%
- preserve per-TF N as much as possible

## M1 special handling
M1 is not subordinated to M5/M15. It can enter independently.
Because M1 fires more often, keep a separate M1 basket/risk ledger and report its KPI separately.

## Cross-TF boost
When two or more TFs independently signal same direction within an overlap window, raise confidence but do not suppress standalone signals.
Example:
- M1 BUY + M5 BUY => boost
- M15 BUY + H1 BUY => boost
- H4/D1 agreement => strongest context boost

## Backtest output required
For each TF and combined portfolio:
- N
- WR
- PF
- Net return
- 21-business-day normalized return
- Max closed DD
- Max floating DD
- Max daily DD
- average RR
- TP1 hit rate
- TP2 hit rate
- runner hit rate
- G75 add count
- average/max active layers
- ruin / hard-rule breach count

## Data rule
Do not generate higher TF test bars by OHLC resampling for final BT. Use native/Catalog data for M1/M5/M15/H1/H2/H4/D1. Temporary engineering smoke tests must be clearly labeled as proxy only.
