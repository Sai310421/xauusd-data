# RAW Bid/Ask Only Backtest Policy

Effective immediately, OHLC-resample-based backtests are prohibited for canonical Nautilus BT evidence in this repository.

## Canonical rule
A result may be promoted to `NAUTILUS_BT`, `RAW_BIDASK_PASS`, `BROKER_REALITY_BT`, or `VERIFIED` only when the strategy is replayed from raw Bid/Ask quote data (Nautilus `QuoteTick` or equivalent raw bid/ask event stream) and the execution assumptions are recorded in the manifest.

The following are not acceptable as canonical BT evidence:
- M1 OHLC -> M5/M15 resample replay
- synthetic intrabar ordering
- mid-price-only bars used as execution prices
- OHLC proxy spread/slippage used in place of raw Bid/Ask replay

Such historical outputs remain archived research only and must be labeled `PROXY_BT`.

## Standard universe
Symbols:
- XAUUSD
- EURUSD
- GBPUSD
- USDJPY
- AUDUSD
- USDCHF

Trading timeframes:
- M1
- M5
- M15

M5 and M15 bars may be constructed by Nautilus from the raw Bid/Ask tick stream for signal logic, but fills, spread, ordering and execution must remain driven by the underlying raw quote stream. Pre-resampling M1 OHLC into higher timeframes for BT is prohibited.

## Standard invocation
When a user says `Nautilus BTして`, the canonical path is:

`Raw Bid/Ask Catalog -> Catalog load -> 6 symbols x M1/M5/M15 -> portfolio replay -> KPI artifact`

If the required raw catalog is absent or incomplete, the runner must fail closed and report the missing symbols/range. It must never silently fall back to OHLC.

## Portfolio outputs
Every run must report at minimum:
- N / WR / PF / RF / MaxDD
- Net PnL / Monthly21 / Daily
- symbol x timeframe KPIs
- spread / slippage / commission / latency assumptions where applicable
- dataset hash / strategy hash / config hash
- run_id / commit SHA / Nautilus version

## Historical note
Earlier MinimumSpike MTF experiments used M1 OHLC resampling and are therefore retained as `PROXY_BT` only. They are not canonical evidence under this policy.
