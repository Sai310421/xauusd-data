#!/usr/bin/env python3
import argparse,csv,datetime as dt,hashlib,lzma,statistics,struct,subprocess,tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REC=struct.Struct(">IIIff")
HOST="https://datafeed.dukascopy.com/datafeed"
def fetch(url):
 with tempfile.NamedTemporaryFile() as tmp:
  r=subprocess.run(["curl","-sS","-L","--http1.1","-m","10","-A","Mozilla/5.0","-o",tmp.name,"-w","%{http_code}",url],capture_output=True,text=True)
  code=(r.stdout or "0").strip()
  tmp.seek(0); data=tmp.read()
  if code=="200" and data:return data
  print(f"FETCH {code} bytes={len(data)} {url}")
  return b""
def main():
 p=argparse.ArgumentParser();p.add_argument("--date",required=True);p.add_argument("--out",required=True);a=p.parse_args()
 day=dt.date.fromisoformat(a.date); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
 rows=[]; hashes=[]
 def one(h):
  url=f"{HOST}/XAUUSD/{day.year}/{day.month-1:02d}/{day.day:02d}/{h:02d}h_ticks.bi5"
  return h,fetch(url)
 with ThreadPoolExecutor(max_workers=8) as ex:
  blobs=list(ex.map(one,range(24)))
 for h,blob in sorted(blobs):
  if not blob: continue
  hashes.append(hashlib.sha256(blob).hexdigest())
  try: raw=lzma.decompress(blob)
  except Exception as e: raise SystemExit(f"FAIL-CLOSED decompress hour={h}: {e}")
  if len(raw)%REC.size: raise SystemExit(f"FAIL-CLOSED record alignment hour={h}")
  for i in range(0,len(raw),REC.size):
   ms,ask_i,bid_i,av,bv=REC.unpack_from(raw,i)
   ask,bid=ask_i/1000.0,bid_i/1000.0
   ts=dt.datetime(day.year,day.month,day.day,tzinfo=dt.timezone.utc).timestamp()+h*3600+ms/1000
   rows.append((ts,bid,ask,av,bv))
 if len(rows)<1000: raise SystemExit(f"FAIL-CLOSED insufficient ticks: {len(rows)}")
 prices=[(x[1]+x[2])/2 for x in rows]; spreads=[x[2]-x[1] for x in rows]
 med=statistics.median(prices)
 if not (500<med<10000): raise SystemExit(f"FAIL-CLOSED implausible XAU median {med}")
 if any(x[2]<x[1] for x in rows): raise SystemExit("FAIL-CLOSED ask<bid")
 if any(rows[i][0]<rows[i-1][0] for i in range(1,len(rows))): raise SystemExit("FAIL-CLOSED nonmonotonic timestamp")
 distinct_ms=len({int(x[0]*1000) for x in rows})
 if distinct_ms<1000: raise SystemExit("FAIL-CLOSED insufficient millisecond diversity")
 with out.open("w",newline="") as f:
  z=csv.writer(f);z.writerow(["timestamp","bid","ask","bid_volume","ask_volume"]);z.writerows((t,b,a,bv,av) for t,b,a,av,bv in rows)
 meta=out.with_suffix(".meta.txt")
 meta.write_text(f"date={day}\nticks={len(rows)}\nmedian_price={med}\nmedian_spread={statistics.median(spreads)}\ndistinct_ms={distinct_ms}\nscale=1000\nhour_hashes={','.join(hashes)}\n")
 print(meta.read_text())
if __name__=="__main__":main()
