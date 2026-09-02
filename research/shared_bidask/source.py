from __future__ import annotations

import datetime as dt
import lzma
import struct
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

REC = struct.Struct(">3i2f")
HEADERS = {
    "User-Agent": "shared-bidask-duka/1.0",
    "Accept": "*/*",
    "Connection": "close",
}

SYMBOLS = {
    "XAUUSD": {
        "pair": "XAU/USD",
        "scale": 1000.0,
        "price_precision": 3,
        "lean_security_type": "cfd",
        "lean_market": "dukascopy",
    },
    "EURUSD": {
        "pair": "EUR/USD",
        "scale": 100000.0,
        "price_precision": 5,
        "lean_security_type": "forex",
        "lean_market": "dukascopy",
    },
    "GBPUSD": {
        "pair": "GBP/USD",
        "scale": 100000.0,
        "price_precision": 5,
        "lean_security_type": "forex",
        "lean_market": "dukascopy",
    },
    "USDJPY": {
        "pair": "USD/JPY",
        "scale": 1000.0,
        "price_precision": 3,
        "lean_security_type": "forex",
        "lean_market": "dukascopy",
    },
    "AUDUSD": {
        "pair": "AUD/USD",
        "scale": 100000.0,
        "price_precision": 5,
        "lean_security_type": "forex",
        "lean_market": "dukascopy",
    },
    "USDCHF": {
        "pair": "USD/CHF",
        "scale": 100000.0,
        "price_precision": 5,
        "lean_security_type": "forex",
        "lean_market": "dukascopy",
    },
}


def iter_days(start: dt.datetime, days: int):
    for i in range(days):
        yield start + dt.timedelta(days=i)


def fetch_hour(symbol: str, scale: float, t: dt.datetime):
    urls = [
        f"https://datafeed.dukascopy.com/datafeed/{symbol}/{t.year}/{t.month-1:02d}/{t.day:02d}/{t.hour:02d}h_ticks.bi5",
        f"https://www.dukascopy.com/datafeed/{symbol}/{t.year}/{t.month-1:02d}/{t.day:02d}/{t.hour:02d}h_ticks.bi5",
    ]
    last = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = r.read()
            if not raw:
                continue
            dec = lzma.decompress(raw)
            rows = []
            for i in range(0, len(dec) - REC.size + 1, REC.size):
                ms, ask_i, bid_i, ask_v, bid_v = REC.unpack_from(dec, i)
                ts = t + dt.timedelta(milliseconds=ms)
                ask = ask_i / scale
                bid = bid_i / scale
                if ask <= 0 or bid <= 0 or ask < bid:
                    continue
                rows.append((ts, bid, ask, float(bid_v), float(ask_v)))
            return rows, 200
        except urllib.error.HTTPError as e:
            last = e.code
        except Exception:
            last = -1
    return [], last


def fetch_day(symbol: str, scale: float, day: dt.datetime, workers: int = 32):
    hours = [day.replace(hour=h, minute=0, second=0, microsecond=0) for h in range(24)]
    rows = []
    status_counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_hour, symbol, scale, h): h for h in hours}
        for fut in as_completed(futs):
            part, status = fut.result()
            rows.extend(part)
            status_counts[str(status)] = status_counts.get(str(status), 0) + 1
    rows.sort(key=lambda x: x[0])
    dedup = {}
    for row in rows:
        dedup[row[0]] = row
    cleaned = [dedup[k] for k in sorted(dedup)]
    return cleaned, status_counts
