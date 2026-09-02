# Shared BidAsk Catalog Bridge

Saved: 2026-09-03

One Dukascopy BI5 source. Two physical trees. Three verification labels. No silent fallback.

## Layout

```
Dukascopy BI5
    -> research/shared_bidask/materialize.py
        -> catalog/raw_bidask     Nautilus ParquetDataCatalog
        -> lean_data/             Lean zip/CSV
        -> results/shared-bidask/shared_manifest.json
```

| Artifact | Engine | Label |
|---|---|---|
| `catalog/raw_bidask` | Nautilus | `NAUTILUS_BT` / `RAW_BIDASK_PASS` |
| `lean_data/` | Lean CLI local | `LEAN_LOCAL_BT` |
| QC official CFD | QuantConnect Cloud | `QC_CLOUD_BT` |

QC Cloud cannot read either tree. Do not promote Cloud KPIs to the other two labels.

## Commands

Lean-only smoke (no Nautilus install):

```bash
python research/shared_bidask/materialize.py --start 2026-07-27 --days 2 --symbols XAUUSD
python research/shared_bidask/guard.py --require-lean
```

Both trees:

```bash
python research/shared_bidask/materialize.py --start 2026-07-27 --days 2 --symbols XAUUSD --write-nautilus
```

Existing 6-symbol 30d Nautilus factory is unchanged:

`Actions -> Raw BidAsk Catalog Factory`

New dual writer:

`Actions -> Shared BidAsk Materialize`

## Lean paths written

- XAUUSD tick: `lean_data/cfd/dukascopy/tick/xauusd/YYYYMMDD_quote.zip`
- XAUUSD hour QuoteBar: `lean_data/cfd/dukascopy/hour/xauusd.zip`
- FX tick: `lean_data/forex/dukascopy/tick/<pair>/YYYYMMDD_quote.zip`

CSV quote tick: `ms,bid,ask,bid_size,ask_size,DUKA,,0`

Point Lean `data-folder` at `lean_data`. Market name is `dukascopy`, not `oanda`.

## Custom data on Lean local

`research/qc/dukascopy_quote_tick.py` reads the zip/CSV tree. Cloud Object Store upload of those zips is optional and stays `QC_CUSTOM_BT`, never `RAW_BIDASK_PASS`.
