# AE High-N 2Edge + Math Controller Raw BT

Status: requested for Nautilus Raw Bid/Ask execution.

## Required ablation
1. A = HFT Microstructure
2. B = Harmonic/MTF
3. AB = combined without Math Controller
4. AB100D = combined with AE Math Controller

## Hard conditions
- XAUUSD first
- M1 / M5 / M15
- Nautilus ParquetDataCatalog QuoteTick
- Raw Bid/Ask only
- OHLC resample execution prohibited
- Initial balance USD 1,000
- Leverage 1:2000
- Native order/fill lifecycle

## Required output
- N
- Net PnL
- PF
- WR
- Max DD
- expectancy/trade
- signals submitted
- blocked/pass rate
- controller action distribution
- A vs B vs AB vs AB100D deltas

Do not report synthetic or estimated KPI as measured results.
