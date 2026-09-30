# GoldeBrave v4 — MT5 ↔ Nautilus Raw BidAsk parity protocol

## Objective
Reconstruct `GoldeBrave_v4.mq5` v4.20 without changing its trading thesis, then measure divergence from the original MT5 Strategy Tester result.

## Source-preserved defaults
- Symbol: XAUUSD
- Fixed lot: 0.10
- PivotTF: H1
- FastTF: M15
- BarrierTF: M15
- ZigZag: Depth=12, Deviation=5, Backstep=3
- Lookback: 600
- Layer A cap: 5/side (+2 in ADX trend regime)
- Layer B cap: 3/side
- TradeHours: 11,15,16,17,18 (EA shifted server-hour logic preserved)
- ATR short/long: 14/480
- ADX: 14, trend >=25, range <=18
- SL: max(4.0 USD, 1.2*ATR), capped at 12.0 USD
- TP: max(9.0 USD, 2.4*ATR), capped at 30.0 USD, then regime multiplier
- Break-even: trigger 1.2 USD, lock 0.2 USD, volatility-scaled
- Trail: trigger 2.0 USD; prior M1 extreme +/-3.0 USD, H1-near-extreme gate
- Layer C: minimum 3 entries/day, daily high/low breakout boost
- Spread break: 2.5 USD-equivalent in the EA unit model; 120s cooldown and <50% recovery

## Layer audit tags
The original source sends the same comment `A` for both ZigZag layers. The Nautilus reconstruction keeps behavior unchanged but assigns internal audit labels:
- `A_H1`: PivotTF H1 ZigZag
- `B_M15`: FastTF M15 ZigZag
- `C_DAILY`: daily high/low minimum-entry boost

## Data/execution rule
OHLC-resampled execution is prohibited. Raw Bid/Ask QuoteTicks are the execution source. Nautilus INTERNAL M1/M15/H1 BID bars are generated causally from those ticks for indicator/state logic. Pending stop fills, SL, TP and BE are evaluated against the executable quote side (ask for buy entry / short exit; bid for sell entry / long exit).

## Parity gates
The first report must compare:
1. trade count
2. win rate
3. profit factor
4. net profit / return
5. maximum drawdown
6. long/short counts if MT5 export is supplied
7. equity-curve shape/correlation if MT5 time-series export is supplied
8. entry timestamps if MT5 deals export is supplied
9. Layer A/B/C contribution in Nautilus

### Interpretation
- Trade-count mismatch first → time/session, ZigZag or pending-order semantics are wrong; do not optimize profits.
- Count matches but PF/WR mismatch → fill side, spread, SL/TP, BE/trailing or broker cost model.
- PF matches but DD mismatch → equity accounting, concurrent-position semantics or lot/contract assumptions.

## Known missing broker inputs
A screenshot cannot provide exact tester inputs such as broker `SYMBOL_TRADE_STOPS_LEVEL`, commission, tester spread/slippage model, exact date range and broker symbol contract specification. These are reported as parity risks rather than guessed.

## Arena integration
A completed `summary.json` follows `amos.arena.v1` core fields and is eligible for AMOS Dual-Arena only after the Raw BidAsk parity run succeeds. A smoke/syntax result is never promoted to performance evidence.
