# A+B+C Shared Equity Boost BT v1

Purpose: run a causal shared-equity backtest of A, B, C on one $1,000 account and boost only when independent EDGE signals overlap in the same direction.

## Engines
- A: XAUUSD M5 SquareRoot/SNR + same-direction G75. Use current reconstructed A proxy parameters from the best prop-compliant family as baseline.
- B: XAUUSD M15 source RSI-level engine, RSI14 research parameter, touch/reentry variants separated; EMA9 exit; same-direction G75. B must be revalidated in this run (do not reuse previously quoted unverified aggregate numbers).
- C: C0 QM_Harmonic_DrawOnLiquidity, EURUSD M1 + XAUUSD M1, current balanced causal detector + ATR-normalized same-direction G75.

## Boost rule
- Each EDGE keeps independent entry firing. Boost is never an entry gate.
- Boost only when a new signal overlaps an already-active independent EDGE on the SAME SYMBOL and SAME DIRECTION.
- Opposite-direction overlap => no boost.
- 1 active confirming EDGE: 1.25x risk.
- 2 active confirming EDGEs: 1.50x risk.
- Cap total projected shared-equity risk so projected Max Floating DD does not exceed 5%.
- No cross-symbol boost (e.g. EURUSD C does not boost XAU A/B).

## Shared account
- Initial equity: $1,000.
- One chronological event ledger across all engines.
- Simultaneous positions allowed.
- Risk sizing uses current shared equity at entry.
- Compute shared floating DD and daily DD from concurrent open positions, not by summing standalone DDs.

## Data / causality
- Native data only; no OHLC resampling.
- Period target: 2026-02-25 through 2026-05-26 where common native data is available.
- No lookahead: confirmed pivots only after right bars; signal at close, fill next bar open unless strategy limit-fill logic explicitly requires bar-range touch.
- Proxy limitations must be labeled: OHLC path ambiguity, no spread/slippage/swap unless modeled.

## Required outputs
- Overall: N, WR, PF, Net, Final Equity, Month21, MaxDD closed, Max Floating DD, Max Daily DD, maximum simultaneous positions, maximum simultaneous risk, Boost count.
- Per engine: A/B/C N, WR, PF, Net, G75 adds, G75 PnL.
- Boost attribution: 1.0x / 1.25x / 1.50x trade count and PnL; overlap matrix A-B, A-C, B-C, A-B-C.
- Baseline comparison: same unified run with Boost OFF vs Boost ON.

Integrity rule: do not call this production/MT5-real-tick performance. B values must come from this fresh run, not prior unverified summaries.
