# AMOS Fib v0.30 STDV LocalOpt — Validation Candidate

Source package supplied by user on 2026-09-11.

## Claimed research reference (source package)
- Instrument: XAUUSD
- Entry: Fib / PD Array immediate price entry
- 3-leg structure
- Fib TP priority
- ATR runner
- Profit-funded pyramid
- STDV winner extension
- Research N: 390
- WR: 66.92%
- PF: 8.01
- RF: 22.09
- Max DD: 4.00%
- Daily: 3.34%
- 21 trading-day return: 99.47%
- OOS block 4: WR 57.58%, PF 7.98, RF 10.92, DD 4.00%, 21-day return 120.79%

These are source-package Python/M1 OHLC research metrics and are NOT accepted as Raw BidAsk / Nautilus / MT5 Real Tick proof.

## Core parameters to preserve
- Swing lookback: 48
- Retracement levels: 0.382 / 0.500 / 0.618
- TP multipliers: 1.000 / 1.272 / 1.618
- PD Array tolerance: 0.35 ATR
- ATR: 14, long ATR: 50, EMA9
- Normal 3-leg weights: 0.20 / 0.15 / 0.65
- Expansion weights: 0.15 / 0.15 / 0.70
- Short-run weights: 0.15 / 0.15 / 0.70
- Profit pyramid multiplier: 1.50
- Pyramid max locked fraction: 0.30
- STDV target: 2.65
- STDV runner fraction: 0.24
- STDV stop projection: 0.55
- Spread guard: 120 points
- Minimum margin level: 500%
- Daily DD gate: 4%
- Total DD gate: 9%

## Validation plan
1. Static logic audit against source MQL5.
2. Rebuild exact causal entry/exit state machine for Raw BidAsk validation.
3. XAUUSD M1/M5/M15 Raw BidAsk OOS tests. No OHLC-resample fallback.
4. Ablation tests:
   - core Fib only
   - +PD Array gate
   - +ATR runner
   - +profit-funded pyramid
   - +STDV extension
5. Local sensitivity around STDV target 2.65, runner 0.24, stop 0.55.
6. Compare against current G75 v16 Stage3 policy on PF / expectancy / DD / trade count / tail loss.
7. MT5 Every Tick Based on Real Ticks remains a separate certification step.

## Audit warning found in source
The current MQL5 PDArrayGate is explicitly a placeholder based mainly on EMA9 distance / ATR tolerance, not a full OB/FVG/BPR/IFVG PD Array object engine. Raw validation must therefore report both `placeholder_gate` and future `full_pdarray_gate` separately; they must not be conflated.

## Status
QUEUED_FOR_RAW_BIDASK_PARITY_PORT
