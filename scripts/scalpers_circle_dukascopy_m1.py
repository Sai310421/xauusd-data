#!/usr/bin/env python3
"""Build M1 OHLC + tick_volume from Dukascopy RAW BI5 ticks.

This is intentionally RAW-tick-origin data, not an OHLC resample proxy.
Each Dukascopy hourly .bi5 file is LZMA-decoded and every tick is parsed,
validated, then aggregated to UTC M1 bars.

Target use: Scalpers Circle Republic x BigPlayerDetector BT.
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import io
import lzma
import struct
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

BASE = "https://datafeed.dukascopy.com/datafeed"
REC = struct.Struct(">3i2f")  # millis, ask, bid, ask_vol, bid_vol
DEFAULT_SYMBOLS = ["EURUSD","GBPUSD","USDCHF","AUDUSD","USDCAD","NZDUSD"]

@dataclass
class Job:
    symbol: str
    day: date
    hour: int


def url_for(j: Job) -> str:
    # Dukascopy month is zero-based.
    return f"{BASE}/{j.symbol}/{j.day.year}/{j.day.month-1:02d}/{j.day.day:02d}/{j.hour:02d}h_ticks.bi5"


def divider(symbol: str) -> float:
    # All current target pairs are 5-decimal FX. Keep generic JPY support.
    return 1e3 if "JPY" in symbol else 1e5


def fetch(job: Job, retries: int = 4) -> list[tuple]:
    url = url_for(job)
    body = None
    for k in range(retries):
        try:
            req = Request(url, headers={"User-Agent":"AMOS-Nautilus-DataFabric/1.0"})
            with urlopen(req, timeout=30) as r:
                body = r.read()
            break
        except HTTPError as e:
            if e.code == 404:
                return []
            if k == retries-1: raise
        except (URLError, TimeoutError):
            if k == retries-1: raise
        time.sleep(1.5*(k+1))
    if not body:
        return []
    try:
        raw = lzma.decompress(body)
    except lzma.LZMAError:
        # Some upstream edge cases may deliver empty/corrupt files: fail closed.
        raise RuntimeError(f"LZMA decode failed: {url}")
    if len(raw) % REC.size != 0:
        raise RuntimeError(f"Bad BI5 length {len(raw)} for {url}")

    div = divider(job.symbol)
    base = datetime(job.day.year, job.day.month, job.day.day, job.hour, tzinfo=timezone.utc)
    out=[]
    last_ms=-1
    for off in range(0, len(raw), REC.size):
        ms, ask_i, bid_i, ask_v, bid_v = REC.unpack_from(raw, off)
        if ms < 0 or ms >= 3_600_000 or ms < last_ms:
            raise RuntimeError(f"Non-monotonic/invalid timestamp in {url}")
        last_ms=ms
        ask=ask_i/div; bid=bid_i/div
        if bid <= 0 or ask <= 0 or bid > ask:
            raise RuntimeError(f"Invalid bid/ask in {url}: {bid}/{ask}")
        ts=base + timedelta(milliseconds=ms)
        mid=(bid+ask)/2.0
        out.append((ts, bid, ask, mid, float(ask_v), float(bid_v)))
    return out


def jobs(symbol: str, start: date, end: date):
    d=start
    while d <= end:
        # Dukascopy has weekend gaps; skip Saturday, keep Sunday because FX opens late UTC.
        if d.weekday() != 5:
            for h in range(24):
                yield Job(symbol,d,h)
        d += timedelta(days=1)


def aggregate_symbol(symbol: str, start: date, end: date, outdir: Path, workers: int):
    js=list(jobs(symbol,start,end))
    rows=[]
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(fetch,j):j for j in js}
        for n,f in enumerate(cf.as_completed(futs),1):
            j=futs[f]
            try:
                rows.extend(f.result())
            except Exception as e:
                raise RuntimeError(f"{symbol} {j.day} {j.hour:02d}: {e}") from e
            if n % 250 == 0:
                print(symbol, "hours", n, "/", len(js), "ticks", len(rows), flush=True)
    if not rows:
        raise RuntimeError(f"No raw ticks for {symbol}")

    df=pd.DataFrame(rows,columns=["timestamp","bid","ask","mid","ask_volume","bid_volume"])
    df=df.sort_values("timestamp").drop_duplicates(subset=["timestamp","bid","ask"])
    df=df.set_index("timestamp")

    # Raw tick -> M1. tick_volume is literal number of parsed ticks in that minute.
    g=df["mid"].resample("1min")
    m1=g.ohlc()
    m1.columns=["open","high","low","close"]
    m1["tick_volume"]=g.count()
    m1["bid_open"]=df["bid"].resample("1min").first()
    m1["ask_open"]=df["ask"].resample("1min").first()
    m1["spread_open"]=(m1["ask_open"]-m1["bid_open"])
    m1=m1[m1["tick_volume"]>0].reset_index()

    if not m1["timestamp"].is_monotonic_increasing:
        raise RuntimeError(f"M1 timestamp order failed for {symbol}")
    if (m1[["open","high","low","close"]] <= 0).any().any():
        raise RuntimeError(f"Non-positive M1 price for {symbol}")

    outdir.mkdir(parents=True,exist_ok=True)
    csvp=outdir/f"{symbol}.csv"
    pqp=outdir/f"{symbol}.parquet"
    m1.to_csv(csvp,index=False)
    m1.to_parquet(pqp,index=False)
    print(symbol, "raw_ticks", len(df), "m1", len(m1), "->", csvp)
    return {"symbol":symbol,"raw_ticks":len(df),"m1_rows":len(m1),"csv":str(csvp),"parquet":str(pqp)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-06-14", help="UTC date, includes warmup")
    ap.add_argument("--end", default="2026-09-08")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--out", default="artifacts/scalpers_circle_m1")
    ap.add_argument("--workers", type=int, default=10)
    a=ap.parse_args()
    start=date.fromisoformat(a.start); end=date.fromisoformat(a.end)
    if end < start: raise SystemExit("end < start")
    out=Path(a.out)
    summary=[]
    for s in a.symbols:
        summary.append(aggregate_symbol(s.upper(),start,end,out,a.workers))
    pd.DataFrame(summary).to_csv(out/"manifest.csv",index=False)

if __name__ == "__main__":
    main()
