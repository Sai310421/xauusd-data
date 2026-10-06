from __future__ import annotations
import argparse, concurrent.futures as cf, datetime as dt, hashlib, json, lzma, os, struct, time, urllib.error, urllib.request
from pathlib import Path
import pandas as pd
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Currency, Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.persistence.wranglers import QuoteTickDataWrangler

REC=struct.Struct(">3I2f")
HOST="https://datafeed.dukascopy.com/datafeed"
SCALE=1000.0
SIM=Venue("SIM")

def instrument():
 return CurrencyPair(instrument_id=InstrumentId(Symbol("XAUUSD"),SIM),raw_symbol=Symbol("XAUUSD"),
  base_currency=Currency.from_str("XAU"),quote_currency=Currency.from_str("USD"),
  price_precision=3,size_precision=2,price_increment=Price.from_str("0.001"),size_increment=Quantity.from_str("0.01"),ts_event=0,ts_init=0)

def fetch_hour(task):
 day,h,out=task
 p=out/f"{day.isoformat()}_{h:02d}.bi5"
 if p.exists() and p.stat().st_size>0:return {"day":str(day),"hour":h,"status":"hit","bytes":p.stat().st_size}
 url=f"{HOST}/XAUUSD/{day.year}/{day.month-1:02d}/{day.day:02d}/{h:02d}h_ticks.bi5"
 for attempt in range(3):
  try:
   req=urllib.request.Request(url,headers={"User-Agent":"amos-direct-duka-nautilus/1"})
   with urllib.request.urlopen(req,timeout=35) as r:data=r.read()
   if data:
    p.write_bytes(data);return {"day":str(day),"hour":h,"status":"ok","bytes":len(data)}
  except urllib.error.HTTPError as e:
   if e.code in (404,403):return {"day":str(day),"hour":h,"status":f"http_{e.code}","bytes":0}
  except Exception as e:
   if attempt==2:return {"day":str(day),"hour":h,"status":"error","error":repr(e),"bytes":0}
  time.sleep(1+attempt)
 return {"day":str(day),"hour":h,"status":"empty","bytes":0}

def decode_day(day,raw_dir):
 rows=[]
 for h in range(24):
  p=raw_dir/f"{day.isoformat()}_{h:02d}.bi5"
  if not p.exists() or p.stat().st_size==0:continue
  try:raw=lzma.decompress(p.read_bytes())
  except Exception:continue
  origin=dt.datetime(day.year,day.month,day.day,h,tzinfo=dt.timezone.utc)
  for i in range(0,len(raw)-REC.size+1,REC.size):
   ms,ask,bid,askv,bidv=REC.unpack_from(raw,i)
   if not ask or not bid:continue
   bp=bid/SCALE;ap=ask/SCALE
   if bp<=0 or ap<bp:continue
   rows.append((origin+dt.timedelta(milliseconds=ms),bp,ap,float(bidv) if bidv>0 else 1.0,float(askv) if askv>0 else 1.0))
 if not rows:return None
 q=pd.DataFrame(rows,columns=["datetime","bid_price","ask_price","bid_size","ask_size"])
 q=q.sort_values("datetime").drop_duplicates("datetime",keep="last").set_index("datetime")
 return q

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--start",required=True);ap.add_argument("--end",required=True);ap.add_argument("--catalog",required=True)
 ap.add_argument("--work",required=True);ap.add_argument("--workers",type=int,default=24);ap.add_argument("--manifest",required=True)
 a=ap.parse_args()
 start=pd.Timestamp(a.start,tz="UTC");end=pd.Timestamp(a.end,tz="UTC")
 raw_dir=Path(a.work)/"bi5";raw_dir.mkdir(parents=True,exist_ok=True)
 days=[];d=start.date()
 while d<end.date():
  # Dukascopy weekends mostly empty; skip Saturday/Sunday to reduce requests.
  if d.weekday()<5:days.append(d)
  d+=dt.timedelta(days=1)
 tasks=[(d,h,raw_dir) for d in days for h in range(24)]
 print(f"FETCH tasks={len(tasks)} workers={a.workers}",flush=True)
 with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
  fetch=list(ex.map(fetch_hour,tasks))
 print("FETCH_DONE",json.dumps({"ok":sum(x["status"] in ("ok","hit") for x in fetch),"miss":sum(x["status"] not in ("ok","hit") for x in fetch)}),flush=True)

 cat_path=Path(a.catalog);cat_path.mkdir(parents=True,exist_ok=True);cat=ParquetDataCatalog(str(cat_path.resolve()))
 inst=instrument();cat.write_data([inst]);wr=QuoteTickDataWrangler(instrument=inst)
 total=0;daily=[]
 for d in days:
  q=decode_day(d,raw_dir)
  if q is None or q.empty:
   daily.append({"day":str(d),"ticks":0});continue
  med=float(q.bid_price.median());mn=float(q.bid_price.min());mx=float(q.bid_price.max());spr=float((q.ask_price-q.bid_price).median())
  # Fail closed on obvious symbol/scale corruption.
  if not (1000.0<med<10000.0):raise RuntimeError(f"PRICE_SCALE_FAIL {d} median={med}")
  ticks=wr.process(q);cat.write_data(ticks);n=len(ticks);total+=n
  daily.append({"day":str(d),"ticks":n,"bid_min":mn,"bid_median":med,"bid_max":mx,"median_spread":spr})
  print("DAY",daily[-1],flush=True)
 if total<=0:raise RuntimeError("no direct Dukascopy ticks written")
 manifest={"verification":"DIRECT_DUKASCOPY_BI5_XAUUSD","source":HOST,"scale":SCALE,"start":str(start),"end_exclusive":str(end),
  "ticks_written":total,"fetch_status_counts":{k:sum(x["status"]==k for x in fetch) for k in sorted(set(x["status"] for x in fetch))},"daily":daily}
 Path(a.manifest).parent.mkdir(parents=True,exist_ok=True);Path(a.manifest).write_text(json.dumps(manifest,indent=2),encoding="utf-8")
 print(json.dumps({"ticks_written":total,"days_with_ticks":sum(x["ticks"]>0 for x in daily),"start":str(start),"end":str(end)},indent=2),flush=True)
if __name__=="__main__":main()
