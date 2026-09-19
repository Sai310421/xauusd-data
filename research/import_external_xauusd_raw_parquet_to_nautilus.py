from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Currency, Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.persistence.wranglers import QuoteTickDataWrangler

SIM = Venue("SIM")


def make_instrument() -> CurrencyPair:
    return CurrencyPair(
        instrument_id=InstrumentId(Symbol("XAUUSD"), SIM),
        raw_symbol=Symbol("XAUUSD"),
        base_currency=Currency.from_str("XAU"),
        quote_currency=Currency.from_str("USD"),
        price_precision=3,
        size_precision=2,
        price_increment=Price.from_str("0.001"),
        size_increment=Quantity.from_str("0.01"),
        ts_event=0,
        ts_init=0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise SystemExit("no parquet inputs")

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    catalog_path = Path(args.catalog)
    catalog_path.mkdir(parents=True, exist_ok=True)
    catalog = ParquetDataCatalog(str(catalog_path.resolve()))

    instrument = make_instrument()
    catalog.write_data([instrument])
    wrangler = QuoteTickDataWrangler(instrument=instrument)

    total_in = 0
    total_kept = 0
    patched_sizes = 0
    first_ts = None
    last_ts = None
    per_file = []

    for p in files:
        df = pd.read_parquet(p, columns=["ts", "bid", "ask", "bid_vol", "ask_vol"])
        total_in += len(df)
        ts = pd.to_datetime(df["ts"], utc=True)
        mask = (ts >= start) & (ts <= end)
        q = df.loc[mask].copy()
        if q.empty:
            per_file.append({"file": p.name, "rows_in": len(df), "rows_kept": 0})
            continue

        q["datetime"] = pd.to_datetime(q["ts"], utc=True)
        q = q.rename(
            columns={
                "bid": "bid_price",
                "ask": "ask_price",
                "bid_vol": "bid_size",
                "ask_vol": "ask_size",
            }
        )
        q = q[["datetime", "bid_price", "ask_price", "bid_size", "ask_size"]]
        q = q.dropna()
        q = q[q["ask_price"] >= q["bid_price"]]
        q = q.sort_values("datetime").drop_duplicates("datetime", keep="last")

        bad_bid = q["bid_size"] <= 0
        bad_ask = q["ask_size"] <= 0
        patched_sizes += int(bad_bid.sum() + bad_ask.sum())
        q.loc[bad_bid, "bid_size"] = 1.0
        q.loc[bad_ask, "ask_size"] = 1.0

        q = q.set_index("datetime")
        ticks = wrangler.process(q)
        if ticks:
            catalog.write_data(ticks)
            total_kept += len(ticks)
            ft = q.index[0]
            lt = q.index[-1]
            first_ts = ft if first_ts is None or ft < first_ts else first_ts
            last_ts = lt if last_ts is None or lt > last_ts else last_ts

        per_file.append({"file": p.name, "rows_in": len(df), "rows_kept": len(ticks)})

    if total_kept <= 0:
        raise SystemExit("no ticks kept in requested interval")

    manifest = {
        "status": "COMPLETE",
        "verification": "AMOS_COMPLETE_CANDIDATE_RAW_XAUUSD_IMPORT",
        "source_repository": "simom1/XAUUSD-history",
        "source_kind": "tick Bid/Ask parquet",
        "source_origin": "Dukascopy Historical Data (per upstream README)",
        "instrument_id": instrument.id.value,
        "requested_start": args.start,
        "requested_end": args.end,
        "first_tick": str(first_ts),
        "last_tick": str(last_ts),
        "rows_scanned": total_in,
        "ticks_written": total_kept,
        "size_values_patched_nonpositive": patched_sizes,
        "price_values_modified": False,
        "ohlc_resample_used": False,
        "execution_data": "QuoteTick Bid/Ask",
        "files": per_file,
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
