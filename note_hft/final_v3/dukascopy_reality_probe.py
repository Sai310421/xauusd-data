#!/usr/bin/env python3
import argparse, datetime as dt, json, lzma, struct, urllib.request, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from reconstructed_alpha import ReconstructedDirectionalAlpha

REC=struct.Struct('>3i2f')
BASE_URLS=[
 'https://datafeed.dukascopy.com/datafeed/XAUUSD/{y}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5',
 'https://www.dukascopy.com/datafeed/XAUUSD/{y}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5']
PRICE_SCALE=1000.0
HEADERS={'User-Agent':'duka/0.2.1','Accept':'*/*','Connection':'close'}

def iter_hours(start,end):
    t=start.replace(minute=0,second=0,microsecond=0)
    while t<end:
        yield t; t+=dt.timedelta(hours=1)

def fetch_hour(t):
    last_err=None
    for tpl in BASE_URLS:
        url=tpl.format(y=t.year,m=t.month-1,d=t.day,h=t.hour)
        try:
            req=urllib.request.Request(url,headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as r: raw=r.read()
            if not raw: continue
            dec=lzma.decompress(raw)
            out=[]
            for i in range(0,len(dec)-REC.size+1,REC.size):
                ms,ask_i,bid_i,av,bv=REC.unpack_from(dec,i)
                ts=t+dt.timedelta(milliseconds=ms)
                out.append((ts.timestamp(),ask_i/PRICE_SCALE,bid_i/PRICE_SCALE))
            return out,200
        except urllib.error.HTTPError as e:
            last_err=e.code
        except Exception:
            last_err=-1
    return [],last_err

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--start',required=True)
    ap.add_argument('--days',type=int,default=14)
    ap.add_argument('--window',type=int,default=5)
    ap.add_argument('--workers',type=int,default=24)
    ap.add_argument('-o','--out',default='dukascopy_reality_probe.json')
    a=ap.parse_args()
    start=dt.datetime.fromisoformat(a.start).replace(tzinfo=dt.timezone.utc)
    end=start+dt.timedelta(days=a.days)
    hours=list(iter_hours(start,end)); all_ticks=[]; status_counts={}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs={ex.submit(fetch_hour,h):h for h in hours}
        for fut in as_completed(futs):
            rows,status=fut.result(); all_ticks.extend(rows)
            status_counts[str(status)]=status_counts.get(str(status),0)+1
    all_ticks.sort(key=lambda x:x[0])
    alpha=ReconstructedDirectionalAlpha(window=a.window)
    ticks=zero=raw_buy=raw_sell=permit_buy=permit_sell=0; last_signal_ts=-1e30
    for ts,ask,bid in all_ticks:
        ticks+=1; zero += int(ask==bid)
        r=alpha.update(ask,bid); sig=r.signal
        raw_buy += int(sig==1); raw_sell += int(sig==-1)
        if sig and ts-last_signal_ts>=3.0:
            permit_buy += int(sig==1); permit_sell += int(sig==-1); last_signal_ts=ts
    result={'status':'OK' if ticks else 'NO_DATA','source':'Dukascopy XAUUSD tick BidAsk','start':start.isoformat(),'end':end.isoformat(),'days':a.days,'window':a.window,'workers':a.workers,'hours_requested':len(hours),'http_status_counts':status_counts,'ticks':ticks,'zero_spread_ticks':zero,'zero_spread_ratio':(zero/ticks if ticks else None),'raw_signal_buy':raw_buy,'raw_signal_sell':raw_sell,'raw_signal_n':raw_buy+raw_sell,'permit3s_buy':permit_buy,'permit3s_sell':permit_sell,'permit3s_n':permit_buy+permit_sell,'baseline':{'N':176483,'BUY':88223,'SELL':88260,'WR':72.71,'PF':1.74,'MaxDD':3.97},'notes':['Dukascopy is reference-market Bid/Ask, not Exness/HFM execution feed.','WR/PF/DD require execution-price replay and are not fabricated.']}
    Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
