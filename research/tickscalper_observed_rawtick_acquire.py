#!/usr/bin/env python3
"""Download an exact UTC XAUUSD Dukascopy Raw Bid/Ask slice with evidence.

This is data acquisition only. It intentionally has no strategy entry logic.
The workflow begins with one observed UTC day; ranges are explicit inputs.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as dt
import json
import lzma
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

REC = struct.Struct(">3i2f")
HOSTS = ("https://datafeed.dukascopy.com/datafeed", "https://www.dukascopy.com/datafeed")
HEADERS = {"User-Agent": "AMOS-TickScalper-observed-replay/1.0", "Accept": "*/*"}


def hour_url(day: dt.date, hour: int) -> str:
    return f"XAUUSD/{day.year}/{day.month - 1:02d}/{day.day:02d}/{hour:02d}h_ticks.bi5"


def fetch_hour(job: tuple[dt.date, int]) -> tuple[list[tuple], dict]:
    day, hour = job
    origin = dt.datetime(day.year, day.month, day.day, hour, tzinfo=dt.timezone.utc)
    detail = {"day": str(day), "hour": hour, "status": None, "ticks": 0}
    last_error = None
    # Dukascopy intermittently returns 503. Retry the whole host pair before
    # declaring an hour unusable; never silently accept a partial day.
    for attempt in range(4):
        for host in HOSTS:
            try:
                request = urllib.request.Request(f"{host}/{hour_url(day, hour)}", headers=HEADERS)
                with urllib.request.urlopen(request, timeout=45) as response:
                    payload = response.read()
                data = lzma.decompress(payload)
                if len(data) % REC.size:
                    raise ValueError(f"bad BI5 record boundary: {len(data)}")
                rows = []
                for offset in range(0, len(data), REC.size):
                    ms, ask_i, bid_i, ask_v, bid_v = REC.unpack_from(data, offset)
                    bid, ask = bid_i / 1000.0, ask_i / 1000.0
                    if not (0 <= ms < 3_600_000 and 0 < bid <= ask):
                        continue
                    rows.append((origin + dt.timedelta(milliseconds=ms), bid, ask, float(bid_v), float(ask_v)))
                detail.update(status=200, ticks=len(rows), host=host, attempts=attempt + 1)
                return rows, detail
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}"
                if exc.code == 404:
                    detail.update(status=404, ticks=0, host=host, attempts=attempt + 1)
                    return [], detail
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        if attempt < 3:
            time.sleep(2 ** attempt)
    detail.update(status="error", error=last_error, attempts=4)
    return [], detail

def dates(start: dt.date, end: dt.date):
    day = start
    while day <= end:
        if day.weekday() != 5:
            yield day
        day += dt.timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--end", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.end < args.start:
        raise SystemExit("--end must not precede --start")

    jobs = [(day, hour) for day in dates(args.start, args.end) for hour in range(24)]
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        result = list(pool.map(fetch_hour, jobs))

    evidence = [detail for _, detail in result]
    failed = [x for x in evidence if x["status"] == "error"]
    rows = [row for part, _ in result for row in part]
    if failed or not rows:
        raise SystemExit(json.dumps({"status": "BLOCKED", "failed_hours": failed, "ticks": len(rows)}))

    frame = pd.DataFrame(rows, columns=["timestamp", "bid", "ask", "bid_size", "ask_size"])
    frame = frame.sort_values(["timestamp", "bid", "ask"]).drop_duplicates(["timestamp", "bid", "ask"])
    frame["spread"] = frame["ask"] - frame["bid"]
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out / "xauusd_raw_bidask.parquet", index=False)
    summary = {
        "status": "PASS",
        "source": "direct Dukascopy BI5 Bid/Ask",
        "timezone": "UTC",
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "hours_requested": len(jobs),
        "hours_ok": sum(x["status"] == 200 for x in evidence),
        "ticks": int(len(frame)),
        "min_bid": float(frame.bid.min()),
        "max_ask": float(frame.ask.max()),
        "median_spread": float(frame.spread.median()),
        "hourly": evidence,
        "strategy_logic_used": False,
    }
    (args.out / "raw_bidask_evidence.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
