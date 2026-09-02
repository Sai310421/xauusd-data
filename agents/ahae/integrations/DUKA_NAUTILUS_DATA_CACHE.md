# DUKA_NAUTILUS_DATA_CACHE

Canonical cache repo: `Sai310421/xauusd-duka-feed`

- `books/duka-sync.json` — coverage ledger
- `books/xauusd_mtf.json` — M5/M15/H1/H4/D1 book
- `scripts/duka_sync.py` — cache-first M1 gap fill
- Actions: `duka-sync` (hourly + dispatch)

Current ledger window in `books/duka-sync.json`:
- start `2025-09-01`
- end `2026-08-28`
- policy: cache-first, bulk-first, daily M1 only for gaps

QC cloud does **not** read this cache. QC uses QuantConnect CFD QuoteBars.
Nautilus canonical BT must keep using this Duka cache / raw BidAsk catalog.
