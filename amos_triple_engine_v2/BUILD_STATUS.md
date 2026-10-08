# Build status

- 3 engines: **A / B / C**
- Candidate entries: **5 each = 15 total**
- Virtual Entry: **implemented**
- Start capital / allocator: **$1,000 shared equity, A 35% / B 30% / C 35%**
- Exness Pro / TariTari profile: **configured**
- Formal data policy: **RAW BID/ASK TICK ONLY**
- OHLC: **rejected by runner**
- Formal KPI: **not run yet**

## Why KPI is not yet printed
`Sai310421/xauusd-data` currently documents its 90-day XAUUSD data as **mid-quote OHLCV** with no broker spread/slippage. Therefore it is deliberately not used as Raw Tick evidence.

`Sai310421/xauusd-duka-feed` is a cache/sync pipeline. Its committed books are MTF aggregates; README says raw M1 BI5 stays in Actions cache. The next run must expose actual bid/ask tick bytes or a raw catalog to the backtest runner.

No synthetic or OHLC number is substituted for Raw Tick KPI.
