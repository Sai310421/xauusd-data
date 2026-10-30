"""Build a Nautilus XAUUSD QuoteTick catalog from the public Dukascopy archive."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from build_raw_bidask_catalog_duka import SIM, SYMBOLS, fetch_hour, make_instrument, sha256_tree
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.persistence.wranglers import QuoteTickDataWrangler

ARCHIVE_ROOT = "https://raw.githubusercontent.com/luxidevil/bybit-data/main/datasets/dukascopy-xauusd"
INDEX_URL = f"{ARCHIVE_ROOT}/daily-file-index.csv"
CHUNK_INDEX_URL = f"{ARCHIVE_ROOT}/chunk-index.json"
HEADERS = {"User-Agent": "amos-nautilus-public-history/1.0", "Accept": "*/*"}

def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()

def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()

def requested_days(start: dt.datetime, days: int) -> list[dt.date]:
    return [(start + dt.timedelta(days=i)).date() for i in range(days)]

def read_daily_index() -> dict[str, dict[str, str]]:
    rows = csv.DictReader(io.StringIO(download(INDEX_URL).decode("utf-8-sig")))
    return {row["date_utc"]: row for row in rows}

def read_chunk_index() -> dict[str, dict[str, object]]:
    payload = json.loads(download(CHUNK_INDEX_URL).decode("utf-8"))
    return {str(row["file"]): row for row in payload["assets"]}

def public_rows(blob: bytes) -> list[tuple[dt.datetime, float, float, float, float]]:
    with gzip.GzipFile(fileobj=io.BytesIO(blob), mode="rb") as zipped:
        text = io.TextIOWrapper(zipped, encoding="utf-8", newline="")
        reader = csv.DictReader(text)
        required = {"timestamp_ms", "bid_price", "ask_price", "bid_volume_native", "ask_volume_native"}
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError("public XAUUSD archive has an unexpected CSV schema")
        rows = []
        for row in reader:
            bid, ask = float(row["bid_price"]), float(row["ask_price"])
            if bid <= 0 or ask <= 0 or ask < bid:
                raise ValueError("invalid public Bid/Ask quote")
            timestamp = dt.datetime.fromtimestamp(int(row["timestamp_ms"]) / 1000, tz=dt.timezone.utc)
            rows.append((timestamp, bid, ask, float(row["bid_volume_native"]), float(row["ask_volume_native"])))
        return rows

def direct_rows(day: dt.datetime) -> tuple[list[tuple[dt.datetime, float, float, float, float]], dict[str, int]]:
    rows: list[tuple[dt.datetime, float, float, float, float]] = []
    statuses: dict[str, int] = {}
    for hour in range(24):
        hour_rows, status = fetch_hour("XAUUSD", SYMBOLS["XAUUSD"]["scale"], day.replace(hour=hour))
        statuses[str(status)] = statuses.get(str(status), 0) + 1
        rows.extend(hour_rows)
    return rows, statuses

def write_rows(catalog: ParquetDataCatalog, wrangler: QuoteTickDataWrangler, rows: list[tuple[dt.datetime, float, float, float, float]]) -> int:
    if not rows:
        return 0
    frame = pd.DataFrame(rows, columns=["datetime", "bid_price", "ask_price", "bid_size", "ask_size"])
    frame = frame.sort_values("datetime", kind="stable").set_index("datetime")
    ticks = wrangler.process(frame)
    catalog.write_data(ticks)
    return len(ticks)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-07-27")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--catalog", default="catalog/raw_bidask")
    args = parser.parse_args()
    if args.days <= 0:
        raise SystemExit("--days must be positive")

    start = dt.datetime.fromisoformat(args.start).replace(tzinfo=dt.timezone.utc)
    selected_days = requested_days(start, args.days)
    daily_index, chunk_index = read_daily_index(), read_chunk_index()
    by_chunk: dict[str, list[dict[str, str]]] = {}
    fallback_days: list[dt.date] = []
    for day in selected_days:
        row = daily_index.get(day.isoformat())
        if row is None:
            fallback_days.append(day)
        else:
            by_chunk.setdefault(row["chunk"], []).append(row)

    catalog_path = Path(args.catalog)
    catalog_path.mkdir(parents=True, exist_ok=True)
    catalog = ParquetDataCatalog(str(catalog_path.resolve()))
    instrument = make_instrument("XAUUSD", SYMBOLS["XAUUSD"])
    catalog.write_data([instrument])
    wrangler = QuoteTickDataWrangler(instrument=instrument)
    written: dict[str, int] = {}
    public_evidence: list[dict[str, object]] = []

    for relative_chunk, rows in by_chunk.items():
        filename = Path(relative_chunk).name
        expected = chunk_index.get(filename)
        if expected is None:
            raise SystemExit(f"missing public chunk checksum: {filename}")
        blob = download(f"{ARCHIVE_ROOT}/{relative_chunk}")
        if sha256(blob) != expected["sha256"]:
            raise SystemExit(f"public chunk checksum mismatch: {filename}")
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            for row in rows:
                daily_blob = archive.read(row["csv_inside_zip"])
                if sha256(daily_blob) != row["sha256"]:
                    raise SystemExit(f"public daily checksum mismatch: {row['date_utc']}")
                parsed = public_rows(daily_blob)
                if len(parsed) != int(row["records"]):
                    raise SystemExit(f"public daily record count mismatch: {row['date_utc']}")
                written[row["date_utc"]] = write_rows(catalog, wrangler, parsed)
        public_evidence.append({"chunk": relative_chunk, "sha256": expected["sha256"], "days": [r["date_utc"] for r in rows]})

    fallback_evidence: dict[str, dict[str, object]] = {}
    for day in fallback_days:
        rows, statuses = direct_rows(dt.datetime.combine(day, dt.time.min, tzinfo=dt.timezone.utc))
        if any(code not in {"200", "404"} for code in statuses):
            raise SystemExit(f"Dukascopy tail retrieval failed for {day}: {statuses}")
        written[day.isoformat()] = write_rows(catalog, wrangler, rows)
        fallback_evidence[day.isoformat()] = {"ticks": written[day.isoformat()], "http_status_counts": statuses}

    if set(written) != {day.isoformat() for day in selected_days}:
        raise SystemExit("incomplete requested date coverage")
    manifest = {
        "status": "COMPLETE",
        "data_kind": "RAW_BIDASK",
        "source": "public GitHub Dukascopy archive with direct Dukascopy tail fallback",
        "source_archive": "luxidevil/bybit-data/datasets/dukascopy-xauusd",
        "instrument_id": instrument.id.value,
        "symbols": ["XAUUSD"],
        "start": args.start,
        "days": args.days,
        "end_exclusive": (start + dt.timedelta(days=args.days)).date().isoformat(),
        "ohlc_resample_used": False,
        "public_chunks": public_evidence,
        "direct_duka_tail": fallback_evidence,
        "daily_quote_ticks": written,
        "total_ticks": sum(written.values()),
        "catalog_sha256": sha256_tree(catalog_path),
    }
    (catalog_path / "catalog_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
