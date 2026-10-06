#!/usr/bin/env python3
import datetime as dt, json, lzma, struct, urllib.request
REC=struct.Struct(">3I2f")
HOST="https://datafeed.dukascopy.com/datafeed"
SCALE=1000.0
SAMPLES=[
 ("2026-03-09T15:00:00Z",2026,3,9,15),
 ("2026-04-01T00:00:00Z",2026,4,1,0),
 ("2026-05-07T02:00:00Z",2026,5,7,2),
]
def fetch(y,m,d,h):
 url=f"{HOST}/XAUUSD/{y}/{m-1:02d}/{d:02d}/{h:02d}h_ticks.bi5"
 req=urllib.request.Request(url,headers={"User-Agent":"amos-duka-audit/1"})
 with urllib.request.urlopen(req,timeout=30) as r: raw=lzma.decompress(r.read())
 bids=[];asks=[]
 for i in range(0,len(raw)-REC.size+1,REC.size):
  ms,ask,bid,askv,bidv=REC.unpack_from(raw,i)
  if ask and bid:
   bids.append(bid/SCALE);asks.append(ask/SCALE)
 if not bids:return {"url":url,"ticks":0}
 s=sorted(bids);a=sorted(asks)
 return {"url":url,"ticks":len(bids),"bid_min":min(bids),"bid_median":s[len(s)//2],"bid_max":max(bids),
         "ask_median":a[len(a)//2],"median_spread":a[len(a)//2]-s[len(s)//2]}
out={}
for label,y,m,d,h in SAMPLES:
 try:out[label]=fetch(y,m,d,h)
 except Exception as e:out[label]={"error":repr(e)}
print(json.dumps(out,indent=2))
