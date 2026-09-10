#!/usr/bin/env python3
from __future__ import annotations
import argparse, datetime as dt, json, sys
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent
SRC=HERE/'shared_bidask'
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from source import SYMBOLS, fetch_day, iter_days

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--symbol',default='XAUUSD')
    ap.add_argument('--start',default='2026-05-31')
    ap.add_argument('--days',type=int,default=90)
    ap.add_argument('--workers',type=int,default=24)
    ap.add_argument('--out',default='raw/XAUUSD_dukascopy_ticks.parquet')
    a=ap.parse_args(); s=a.symbol.upper()
    if s not in SYMBOLS: raise SystemExit(f'unsupported symbol {s}')
    start=dt.datetime.fromisoformat(a.start).replace(tzinfo=dt.timezone.utc)
    rows=[]; counts={}; written=0
    for day in iter_days(start,a.days):
        part,st=fetch_day(s,SYMBOLS[s]['scale'],day,workers=a.workers)
        for k,v in st.items(): counts[k]=counts.get(k,0)+v
        if part: rows.extend(part); written+=1
    if not rows: raise SystemExit('NO_RAW_TICKS_FETCHED')
    df=pd.DataFrame(rows,columns=['datetime','bid','ask','bid_size','ask_size'])
    df=df.drop_duplicates('datetime',keep='last').sort_values('datetime')
    df['volume']=df['bid_size']+df['ask_size']
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); df.to_parquet(out,index=False)
    m={'status':'COMPLETE','source':'Dukascopy BI5 QuoteTick Bid/Ask','ohlc_resample_used':False,'symbol':s,'start':a.start,'days':a.days,'rows':len(df),'written_days':written,'http_status_counts':counts,'path':str(out)}
    out.with_suffix('.manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    print(json.dumps(m,indent=2))
if __name__=='__main__': main()
