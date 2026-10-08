# AMOS Triple Engine + Virtual Entry v2.00

正式ベース:
- 初期資金: $1,000
- Exness Pro / XAUUSD
- TariTari cashback: 初期モデル $6 / completed round-turn lot（設定可能）
- Shared equity risk budget: A 35% / B 30% / C 35%
- Hard DD: 15%
- 新規停止: DD 14%
- Margin Level floor: 600%
- Formal validation: Nautilus / RAW BID-ASK TICK ONLY
- OHLC は正式KPIに使用しない

## A: CB Harvest
A1 Tick Follow / A2 Micro Reversion / A3 Spread Compression / A4 Turnover Cycle / A5 G75 Pursuit。
目的は Gross RT Lots、CB/DD、(Trading PnL + CB)/DD。

## B: Trend Break
B1 Donchian / B2 ATR Expansion / B3 MSS-BOS / B4 Session Break / B5 G75 Pursuit。
目的は PF、RR、Trend Capture、Return/DD。

## C: High-WR Capital Builder
C1 Z-score MR / C2 Bollinger Re-entry / C3 RSI-Stoch Confluence / C4 VWAP Deviation / C5 Regime Micro Reversion。
目的は WR、PF、Loss Cluster、Return/DD。

## Virtual Entry
15ロジックを常時仮想稼働。Virtual PnL/Virtual CBはReal Equityに加算しない。
最低N、PF、EV、Virtual DDを通過したロジックだけ Shadow -> Limited Live に昇格。

## Raw Tick
`timestamp,bid,ask` が必須。OHLC-only入力は runner が拒否する。
