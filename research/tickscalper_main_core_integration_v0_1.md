# TickScalper Main Core Integration v0.1

Date: 2026-09-27
Status: DERIVED RESEARCH SPEC — source originals remain READ ONLY

## Goal
Increase N materially beyond M5 strategies while preserving PF/DD through source-backed microstructure gates.

## Source-backed modules

### A. Micro Direction Core — nexobanks-prep/XAUUSD-
Use as the high-N direction/timing engine:
- 30-tick ring buffer
- 10-tick net momentum
- momentum threshold 0.03 price units
- >=3 consecutive ticks in the same direction
- weighted momentum confirmation
- tick-MA confirmation (10-tick fast proxy vs 20-tick slow)
- spread monitor over recent ticks
- 50 ms management timer
- 500 ms minimum trade spacing
- 900 ms maximum hold
IMPORTANT: upstream claims such as ~50 trades/min and 70-80% WR are targets, not verified KPIs.

### B. Anti-Spike / Confirmation Gate — n30dyn4m1c/gold-pro-scalper
Add as an independent gate, not as a replacement for Direction:
- signal persistence: >=3 consecutive confirming ticks OR >=2 seconds sustained
- last-10-tick velocity buffer
- if 10-tick span <=10 sec AND absolute 10-tick move > 0.25 * M1 ATR: block entry 3 sec
- max-spread filter
- optional ADX / H1 / ATR regime controls
- TickRobust cost-gate variant is a negative/control benchmark

### C. Execution Gate — NadirAliOfficial/snipe-fx-ea
A/B only after A+B:
- virtual pending trigger
- pending expiry 5 sec
- optional M1 anchor level
- optional pullback/limit entry
- execution slippage rejection
- cooldown after loss
Do NOT enable all options by default; isolate their effect.

## Baseline derived architecture

RawTick
  -> TickBuffer(30)
  -> Momentum10 + ConsecutiveTicks>=3 + WeightedMomentum + TickMA
  -> SpreadGate
  -> N30 Persistence/VelocitySafety
  -> Entry
  -> 50ms position monitor
  -> Exit A/B
  -> Cost/Risk ledger

No Martingale. MaxPositions=1 for Core validation.

## Clock ablation
- true tick
- 50 / 100 / 250 / 500 ms
- 1 / 2 / 3 / 5 / 10 sec
- 3 / 5 / 10 / 20 / 30 ticks
- hybrid N-ticks OR T-seconds

## Exit ablation
1. 900 ms max hold + any-positive-profit close after >=100 ms
2. fixed TP/SL from upstream HFT source
3. N30 Z/mean reversion exit
4. time-only control
5. cost-aware minimum-net-profit exit

## Required KPI
N/day, entries/minute, WR, PF, EV, Net, Return%, MaxDD%, max consecutive losses,
average/median hold ms, ticks/trade, spread cost, slippage cost,
MAE/MFE, rejected-by-spread count, rejected-by-velocity count.

## Promotion gates
BAR/OHLC results cannot promote this module.
Promotion requires causal bid/ask Raw Tick replay, real spread, latency/slippage stress,
OOS/walk-forward, and event-order parity tests.

## Provenance rule
All upstream files remain immutable. This specification is AMOS-derived and must not
be represented as the original authors' combined strategy.
