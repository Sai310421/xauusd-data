#!/usr/bin/env python3
"""Resilient Dukascopy RAW BI5 -> M1 builder for Scalpers Circle reverse engineering.

Properties:
- RAW tick origin only (no OHLC proxy input)
- per-hour disk cache / resume
- exponential backoff + jitter for transient 429/5xx/network errors
- missing-hour queue instead of aborting an 87-day symbol on one transient failure
- final retry pass and manifest of unresolved hours
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, hashlib, lzma, random, struct, time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import pandas as pd

BASE="https://datafeed.dukascopy.com/datafeed"
REC=struct.Struct(">3i2f")
DEFAULT_SYMBOLS=["EURUSD","GBPUSD","USDCHF","AUDUSD","USDCAD","NZDUSD"]
TRANSIENT={408,425,429,500,502,503,504}

@dataclass(frozen=True)
class Job:
    symbol:str; day:date; hour:int

def url_for(j):
    return f"{BASE}/{j.symbol}/{j.day.year}/{j.day.month-1:02d}/{j.day.day:02d}/{j.hour:02d}h_ticks.bi5"

def divider(symbol): return 1e3 if "JPY" in symbol else 1e5

def cache_path(cache:Path,j:Job):
    return cache/j.symbol/f"{j.day:%Y%m%d}_{j.hour:02d}.bi5"

def download_hour(j:Job,cache:Path,retries=8,timeout=45):
    cp=cache_path(cache,j)
    if cp.exists() and cp.stat().st_size>0: return cp.read_bytes(),"cache"
    cp.parent.mkdir(parents=True,exist_ok=True)
    url=url_for(j)
    for k in range(retries):
        try:
            req=Request(url,headers={"User-Agent":"AMOS-Nautilus-DataFabric/1.1","Accept":"*/*"})
            with urlopen(req,timeout=timeout) as r: body=r.read()
            if body:
                tmp=cp.with_suffix(".tmp"); tmp.write_bytes(body); tmp.replace(cp)
                return body,"network"
            return b"","empty"
        except HTTPError as e:
            if e.code==404: return b"","404"
            if e.code not in TRANSIENT or k==retries-1: raise
        except (URLError,TimeoutError,ConnectionError) as e:
            if k==retries-1: raise
        delay=min(90.0,2.0*(2**k))+random.uniform(0,1.5)
        time.sleep(delay)
    raise RuntimeError("retry exhausted")

def parse(j:Job,body:bytes):
    if not body:return []
    try: raw=lzma.decompress(body)
    except lzma.LZMAError as e: raise RuntimeError(f"LZMA decode failed {url_for(j)}") from e
    if len(raw)%REC.size: raise RuntimeError(f"Bad BI5 length {len(raw)} {url_for(j)}")
    div=divider(j.symbol); base=datetime(j.day.year,j.day.month,j.day.day,j.hour,tzinfo=timezone.utc)
    out=[]; last=-1
    for off in range(0,len(raw),REC.size):
        ms,ask_i,bid_i,av,bv=REC.unpack_from(raw,off)
        if ms<0 or ms>=3600000 or ms<last: raise RuntimeError(f"Invalid timestamp {url_for(j)}")
        last=ms; ask=ask_i/div; bid=bid_i/div
        if bid<=0 or ask<=0 or bid>ask: raise RuntimeError(f"Invalid quote {url_for(j)}")
        out.append((base+timedelta(milliseconds=ms),bid,ask,(bid+ask)/2,float(av),float(bv)))
    return out

def fetch(j,cache,retries):
    body,source=download_hour(j,cache,retries=retries)
    return parse(j,body),source

def jobs(symbol,start,end):
    d=start
    while d<=end:
        if d.weekday()!=5:
            for h in range(24):yield Job(symbol,d,h)
        d+=timedelta(days=1)

def aggregate_symbol(symbol,start,end,outdir,cache,workers,retries,final_retries):
    js=list(jobs(symbol,start,end)); rows=[]; failed=[]; sources={"cache":0,"network":0,"404":0,"empty":0}
    def run_batch(batch,attempt_retries):
        local_failed=[]
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs={ex.submit(fetch,j,cache,attempt_retries):j for j in batch}
            for n,f in enumerate(cf.as_completed(futs),1):
                j=futs[f]
                try:
                    rr,src=f.result(); rows.extend(rr); sources[src]=sources.get(src,0)+1
                except Exception as e:
                    local_failed.append((j,str(e)))
                if n%250==0: print(symbol,"hours",n,"/",len(batch),"ticks",len(rows),"failed",len(local_failed),flush=True)
        return local_failed
    failed=run_batch(js,retries)
    if failed:
        print(symbol,"final retry queue",len(failed),flush=True); time.sleep(15)
        failed=run_batch([j for j,_ in failed],final_retries)
    outdir.mkdir(parents=True,exist_ok=True)
    miss=outdir/f"{symbol}_missing_hours.csv"
    pd.DataFrame([{"symbol":j.symbol,"day":str(j.day),"hour":j.hour,"error":e} for j,e in failed]).to_csv(miss,index=False)
    if failed: raise RuntimeError(f"{symbol}: unresolved hours={len(failed)}; see {miss}")
    if not rows: raise RuntimeError(f"No raw ticks for {symbol}")
    df=pd.DataFrame(rows,columns=["timestamp","bid","ask","mid","ask_volume","bid_volume"]).sort_values("timestamp").drop_duplicates(["timestamp","bid","ask"]).set_index("timestamp")
    g=df["mid"].resample("1min"); m1=g.ohlc(); m1.columns=["open","high","low","close"]
    m1["tick_volume"]=g.count(); m1["bid_open"]=df.bid.resample("1min").first(); m1["ask_open"]=df.ask.resample("1min").first(); m1["spread_open"]=m1.ask_open-m1.bid_open
    m1=m1[m1.tick_volume>0].reset_index()
    if not m1.timestamp.is_monotonic_increasing: raise RuntimeError(f"timestamp order failed {symbol}")
    if (m1[["open","high","low","close"]]<=0).any().any(): raise RuntimeError(f"bad price {symbol}")
    csvp=outdir/f"{symbol}.csv"; pqp=outdir/f"{symbol}.parquet"; m1.to_csv(csvp,index=False); m1.to_parquet(pqp,index=False)
    return {"symbol":symbol,"raw_ticks":len(df),"m1_rows":len(m1),"cached_hours":sources.get("cache",0),"network_hours":sources.get("network",0),"csv":str(csvp),"parquet":str(pqp)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--start",default="2026-06-14"); ap.add_argument("--end",default="2026-09-08"); ap.add_argument("--symbols",nargs="+",default=DEFAULT_SYMBOLS); ap.add_argument("--out",default="artifacts/scalpers_circle_m1"); ap.add_argument("--cache",default=".cache/dukascopy_bi5"); ap.add_argument("--workers",type=int,default=4); ap.add_argument("--retries",type=int,default=6); ap.add_argument("--final-retries",type=int,default=10)
    a=ap.parse_args(); start=date.fromisoformat(a.start); end=date.fromisoformat(a.end)
    if end<start: raise SystemExit("end < start")
    out=Path(a.out); cache=Path(a.cache); summary=[]
    for s in a.symbols: summary.append(aggregate_symbol(s.upper(),start,end,out,cache,a.workers,a.retries,a.final_retries))
    pd.DataFrame(summary).to_csv(out/"manifest.csv",index=False)
if __name__=="__main__": main()
