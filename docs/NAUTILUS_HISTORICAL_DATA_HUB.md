# Nautilus Historical Data Hub

This repository has one canonical source for a Nautilus backtest:

`Dukascopy BI5 -> Raw Bid/Ask QuoteTick -> ParquetDataCatalog -> Nautilus BT`

The `Raw BidAsk Catalog Factory` workflow is the entry point. It accepts a UTC
start date, a calendar-day count, and a space-separated symbol list. Its
default is a 90-day XAUUSD catalog, which replaces one-day-only smoke data for
research runs.

## Use from GitHub Actions

1. Open **Actions -> Raw BidAsk Catalog Factory -> Run workflow**.
2. Keep `symbols=XAUUSD`; choose the required `start` and `days`.
3. Read the uploaded `raw-bidask-factory-<run_id>` artifact before treating the
   result as usable. It contains the selected interval, code hashes, catalog
   inventory, and `catalog_manifest.json`.
4. A successful run caches the exact catalog under the cache key printed in
   `run_identity.txt`. Backtest workflows which need the same interval must
   restore that exact key before loading `catalog/raw_bidask`.

## Non-negotiable separation

| Dataset | Role | May produce `NAUTILUS_BT` evidence |
|---|---|---|
| `parquet/XAUUSD_M*_90d_20260526.parquet` | mid-quote OHLC research comparison | No |
| `xauusd-duka-feed/books/xauusd_mtf.json` | aggregated cache/book | No |
| `catalog/raw_bidask` created by the factory | Dukascopy raw Bid/Ask `QuoteTick` | Yes, after its guard and native fill evidence pass |

No workflow may substitute the OHLC data when the raw catalog is absent. The
factory is fail-closed: no `COMPLETE` manifest or zero QuoteTicks for the
requested symbol makes the requested historical range unavailable.

## Cache identity

The cache key is deterministic:

`raw-bidask-duka-<start>-<days>-<symbols>-v3`

It is an immutable snapshot. Changing dates, the number of days, or symbols
creates a separate data lineage rather than silently mixing data periods.
