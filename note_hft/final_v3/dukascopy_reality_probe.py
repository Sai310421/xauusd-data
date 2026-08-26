#!/usr/bin/env python3
import argparse, datetime as dt, json, lzma, struct, urllib.request, urllib.error
from pathlib import Path
from reconstructed_alpha import ReconstructedDirectionalAlpha

REC=struct.Struct('>3i2f')
BASE_URL='https://datafeed.dukascopy.com/datafeed/XAUUSD/{y}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5'
PRICE_SCALE=1000.0

def iter_hours(start,end):
    t=start.replace(minute=0,second=0,microsecond=0)
    while t<end:
        yield t; t+=dt.timedelta(hours=1)

def fetch_hour(t):
    url=BASE_URL.format(y=t.year,m=t.month-1,d=t.day,h=t.hour)
    try:
        with urllib.request.urlopen(url, timeout=20) as r: raw=r.read()
    except Exception:
        return []
    if not raw: return []
    try: dec=lzma.decompress(raw)
    except Exception: return []
    out=[]
    for i in range(0,len(dec)-REC.size+1,REC.size):
        ms,ask_i,bid_i,av,bv=REC.unpack_from(dec,i)
        ts=t+dt.timedelta(milliseconds=ms)
        out.append((ts.timestamp(),ask_i/PRICE_SCALE,bid_i/PRICE_SCALE))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--start',required=True)
    ap.add_argument('--days',type=int,default=14)
    ap.add_argument('--window',type=int,default=5)
    ap.add_argument('-o','--out',default='dukascopy_reality_probe.json')
    a=ap.parse_args()
    start=dt.datetime.fromisoformat(a.start).replace(tzinfo=dt.timezone.utc)
    end=start+dt.timedelta(days=a.days)
    alpha=ReconstructedDirectionalAlpha(window=a.window)
    ticks=zero=raw_buy=raw_sell=permit_buy=permit_sell=0
    last_signal_ts=-1e30
    for h in iter_hours(start,end):
        for ts,ask,bid in fetch_hour(h):
            ticks+=1
            if ask==bid: zero+=1
            r=alpha.update(ask,bid)
            sig=r.signal
            if sig==1: raw_buy+=1
            elif sig==-1: raw_sell+=1
            if sig and ts-last_signal_ts>=3.0:
                if sig==1: permit_buy+=1
                else: permit_sell+=1
                last_signal_ts=ts
    permit_n=permit_buy+permit_sell
    result={
      'status':'OK','source':'Dukascopy XAUUSD tick BidAsk','start':start.isoformat(),'end':end.isoformat(),
      'days':a.days,'window':a.window,'ticks':ticks,'zero_spread_ticks':zero,
      'zero_spread_ratio':(zero/ticks if ticks else None),
      'raw_signal_buy':raw_buy,'raw_signal_sell':raw_sell,'raw_signal_n':raw_buy+raw_sell,
      'permit3s_buy':permit_buy,'permit3s_sell':permit_sell,'permit3s_n':permit_n,
      'baseline':{'N':176483,'BUY':88223,'SELL':88260,'WR':72.71,'PF':1.74,'MaxDD':3.97},
      'notes':['Dukascopy is reference-market Bid/Ask, not Exness/HFM execution feed.','zero_spread_ticks tests whether Dukascopy itself can satisfy the Frozen zero-spread entry gate.','WR/PF/DD require an execution-price replay and are not fabricated here.']
    }
    Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
