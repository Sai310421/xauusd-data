from __future__ import annotations

import datetime as dt
import io
import zipfile
from collections import defaultdict
from pathlib import Path


def lean_tick_dir(root: Path, security_type: str, market: str, symbol: str) -> Path:
    return root / security_type / market / "tick" / symbol.lower()


def lean_hour_path(root: Path, security_type: str, market: str, symbol: str) -> Path:
    return root / security_type / market / "hour" / f"{symbol.lower()}.zip"


def write_quote_tick_day(
    root: Path,
    security_type: str,
    market: str,
    symbol: str,
    day: dt.datetime,
    rows: list[tuple],
) -> dict:
    out_dir = lean_tick_dir(root, security_type, market, symbol)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_s = day.strftime("%Y%m%d")
    zip_path = out_dir / f"{date_s}_quote.zip"
    midnight = day.replace(hour=0, minute=0, second=0, microsecond=0)
    buf = io.StringIO()
    for ts, bid, ask, bid_sz, ask_sz in rows:
        ms = int((ts - midnight).total_seconds() * 1000)
        if ms < 0:
            continue
        buf.write(f"{ms},{bid},{ask},{bid_sz},{ask_sz},DUKA,,0\n")
    payload = buf.getvalue().encode("utf-8")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{date_s}_{symbol.lower()}_quote_tick.csv", payload)
    return {
        "path": str(zip_path),
        "ticks": len(rows),
        "bytes": zip_path.stat().st_size,
    }


def hour_quote_bars(rows: list[tuple]) -> list[tuple]:
    buckets: dict[dt.datetime, list] = defaultdict(list)
    for ts, bid, ask, bid_sz, ask_sz in rows:
        hour = ts.replace(minute=0, second=0, microsecond=0)
        buckets[hour].append((bid, ask, bid_sz, ask_sz))
    bars = []
    for hour in sorted(buckets):
        chunk = buckets[hour]
        bids = [x[0] for x in chunk]
        asks = [x[1] for x in chunk]
        bars.append(
            (
                hour,
                bids[0],
                max(bids),
                min(bids),
                bids[-1],
                chunk[-1][2],
                asks[0],
                max(asks),
                min(asks),
                asks[-1],
                chunk[-1][3],
            )
        )
    return bars


def write_hour_quote_bars(
    root: Path,
    security_type: str,
    market: str,
    symbol: str,
    bars_by_day: dict[str, list[tuple]],
) -> dict:
    out_path = lean_hour_path(root, security_type, market, symbol)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for date_s, bars in sorted(bars_by_day.items()):
            buf = io.StringIO()
            for bar in bars:
                hour = bar[0]
                stamp = hour.strftime("%Y%m%d %H:%M")
                rest = ",".join(str(x) for x in bar[1:])
                buf.write(f"{stamp},{rest}\n")
            zf.writestr(f"{date_s}_{symbol.lower()}_quote.csv", buf.getvalue())
    return {"path": str(out_path), "days": len(bars_by_day), "bytes": out_path.stat().st_size}


def write_symbol_properties(root: Path) -> Path:
    path = root / "symbol-properties" / "symbol-properties-database.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "market,symbol,type,quote_currency,contract_multiplier,minimum_price_variation,lot_size,market_ticker,minimum_order_size,price_magnifier,security_type",
        "dukascopy,XAUUSD,cfd,USD,1,0.001,1,XAUUSD,1,1,cfd",
        "dukascopy,EURUSD,forex,USD,1,0.00001,1,EURUSD,1000,1,forex",
        "dukascopy,GBPUSD,forex,USD,1,0.00001,1,GBPUSD,1000,1,forex",
        "dukascopy,USDJPY,forex,JPY,1,0.001,1,USDJPY,1000,1,forex",
        "dukascopy,AUDUSD,forex,USD,1,0.00001,1,AUDUSD,1000,1,forex",
        "dukascopy,USDCHF,forex,CHF,1,0.00001,1,USDCHF,1000,1,forex",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path
