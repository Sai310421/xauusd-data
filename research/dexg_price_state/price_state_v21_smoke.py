from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog

EPS=1e-12

@dataclass
class Bar:
    ts:int; o:float; h:float; l:float; c:float


def bars_from_ticks(ticks, sec=60):
    out=[]; cur=None; bucket=None
    for q in ticks:
        p=float(q.bid_price)
        b=(int(q.ts_event)//1_000_000_000)//sec
        if bucket is None or b!=bucket:
            if cur: out.append(cur)
            bucket=b; cur=Bar(int(q.ts_event),p,p,p,p)
        else:
            cur.h=max(cur.h,p); cur.l=min(cur.l,p); cur.c=p
    if cur: out.append(cur)
    return out


def candle_metrics(b):
    r=max(b.h-b.l,EPS)
    return {
        'body':abs(b.c-b.o)/r,
        'upper':(b.h-max(b.o,b.c))/r,
        'lower':(min(b.o,b.c)-b.l)/r,
        'close_loc':(b.c-b.l)/r,
    }


def fvg3(bars):
    if len(bars)<3:return None
    a,_,c=bars[-3:]
    if a.h<c.l:return ('UP',(a.h+c.l)/2)
    if a.l>c.h:return ('DOWN',(a.l+c.h)/2)
    return None


def cisd(bars, lookback=8):
    if len(bars)<3:return None
    cur=bars[-1]; hist=bars[-lookback:-1]
    lb=next((b for b in reversed(hist) if b.c<b.o),None)
    lu=next((b for b in reversed(hist) if b.c>b.o),None)
    if lb and cur.c>lb.o:return 'UP'
    if lu and cur.c<lu.o:return 'DOWN'
    return None


def wick_probe(prev,cur,th=.45):
    m=candle_metrics(cur)
    if m['upper']>=th:
        return 'UPPER_REJECTION' if cur.c<prev.h and m['close_loc']<.55 else 'UPPER_PROBE'
    if m['lower']>=th:
        return 'LOWER_REJECTION' if cur.c>prev.l and m['close_loc']>.45 else 'LOWER_PROBE'
    return 'NONE'


def run(catalog, variant='BASE', max_ticks=0):
    cat=ParquetDataCatalog(catalog)
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    if not ticks: raise SystemExit('no XAUUSD QuoteTicks')
    if max_ticks and len(ticks)>max_ticks: ticks=ticks[:max_ticks]
    bars=bars_from_ticks(ticks,60)
    if len(bars)<20: raise SystemExit('insufficient bars')

    n=0; wins=0; gross_win=0.0; gross_loss=0.0; max_dd=0.0; eq=0.0; peak=0.0
    mfe_sum=0.0; mae_sum=0.0; no_progress=0; destinations=0; dest_hits=0
    horizon=5
    for i in range(8,len(bars)-horizon-1):
        hist=bars[:i+1]; b=bars[i]; prev=bars[i-1]
        side=1 if b.c>b.o else -1
        wp=wick_probe(prev,b)
        ci=cisd(hist)
        fv=fvg3(hist)
        pa=candle_metrics(b)

        enter=True
        if variant in ('PA_WICK','CISD','FVG_ICT','FULL'):
            if side>0: enter = (wp in ('LOWER_REJECTION','LOWER_PROBE')) or pa['body']>.55
            else: enter = (wp in ('UPPER_REJECTION','UPPER_PROBE')) or pa['body']>.55
        if enter and variant in ('CISD','FVG_ICT','FULL'):
            enter = (ci=='UP' and side>0) or (ci=='DOWN' and side<0)
        if enter and variant in ('FVG_ICT','FULL'):
            if fv:
                enter = (fv[0]=='UP' and side>0) or (fv[0]=='DOWN' and side<0)
            else:
                enter=False
        if not enter: continue

        entry=b.c
        fut=bars[i+1:i+1+horizon]
        exits=[x.c for x in fut]
        highs=[x.h for x in fut]; lows=[x.l for x in fut]
        pnl=(exits[-1]-entry)*side
        mfe=(max(highs)-entry) if side>0 else (entry-min(lows))
        mae=(entry-min(lows)) if side>0 else (max(highs)-entry)
        mfe_sum+=max(0,mfe); mae_sum+=max(0,mae)

        if fv:
            destinations+=1
            tp=fv[1]
            hit=any((x.h>=tp if side>0 else x.l<=tp) for x in fut)
            dest_hits+=int(hit)

        first_progress=max(abs(x.c-entry) for x in fut[:2])
        if first_progress < max((b.h-b.l)*0.20,1e-9): no_progress+=1

        n+=1
        if pnl>0: wins+=1; gross_win+=pnl
        else: gross_loss+=abs(pnl)
        eq+=pnl; peak=max(peak,eq); max_dd=max(max_dd,peak-eq)

    pf=gross_win/gross_loss if gross_loss>0 else (999.0 if gross_win>0 else 0.0)
    return {
      'variant':variant,'raw_bidask':True,'execution_note':'signals from completed M1 bars built from raw QuoteTicks; evaluation uses forward raw-derived path, no external OHLC dataset',
      'ticks':len(ticks),'bars':len(bars),'n':n,'wins':wins,'wr':wins/n if n else 0.0,'pf':pf,
      'expectancy_price':eq/n if n else 0.0,'max_dd_price':max_dd,'avg_mfe':mfe_sum/n if n else 0.0,'avg_mae':mae_sum/n if n else 0.0,
      'no_progress_rate':no_progress/n if n else 0.0,'destination_hit_rate':dest_hits/destinations if destinations else None,'destination_cases':destinations,
      'status':'RESEARCH_SMOKE_ONLY_NOT_BROKER_REALITY'
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--variant',choices=['BASE','PA_WICK','CISD','FVG_ICT','FULL'],required=True); ap.add_argument('--max-ticks',type=int,default=0); a=ap.parse_args()
    res=run(a.catalog,a.variant,a.max_ticks)
    d=Path('results/dexg-price-state')/a.experiment_id; d.mkdir(parents=True,exist_ok=True)
    p=d/f'{a.variant}.json'; p.write_text(json.dumps(res,indent=2),encoding='utf-8'); print(json.dumps(res,indent=2))

if __name__=='__main__': main()
