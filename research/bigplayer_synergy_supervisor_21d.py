from __future__ import annotations

# External supervisor experiment. BigPlayer formula constants/logic are frozen.
# Raw Dukascopy QuoteTick -> direct bars; no OHLC-to-OHLC resampling.
# All TFs use the SAME fixed 5-minute execution horizon from raw ticks.

import argparse, concurrent.futures as cf, datetime as dt, json, lzma, math, struct, urllib.request, urllib.error
from pathlib import Path
import numpy as np
import pandas as pd

LOOKBACK=120; VOL_SIGMA=2.0; RANGE_MULT=1.5; WICK_RATIO=1.2; POINT=0.01
REC=struct.Struct('>3i2f'); HOSTS=('https://datafeed.dukascopy.com/datafeed','https://www.dukascopy.com/datafeed')
HEADERS={'User-Agent':'bigplayer-synergy/1.0','Accept':'*/*'}
TF_SECONDS={'M1':60,'M5':300,'M15':900,'M30':1800,'H1':3600}; OUT=Path('results/bigplayer_synergy_supervisor_21d')

def business_days(start,n):
    out=[]; d=start
    while len(out)<n:
        if d.weekday()<5: out.append(d)
        d+=dt.timedelta(days=1)
    return out

def fetch_hour(day,hour):
    origin=dt.datetime(day.year,day.month,day.day,hour,tzinfo=dt.timezone.utc); rel=f'XAUUSD/{day.year}/{day.month-1:02d}/{day.day:02d}/{hour:02d}h_ticks.bi5'
    last=None
    for host in HOSTS:
        try:
            req=urllib.request.Request(f'{host}/{rel}',headers=HEADERS)
            with urllib.request.urlopen(req,timeout=25) as r: raw=r.read()
            dec=lzma.decompress(raw); rows=[]
            for i in range(0,len(dec)-REC.size+1,REC.size):
                ms,ai,bi,av,bv=REC.unpack_from(dec,i); ask=ai/1000.; bid=bi/1000.
                if ask>0 and bid>0 and ask>=bid: rows.append((origin+dt.timedelta(milliseconds=ms),bid,ask,float(bv),float(av)))
            return rows,200
        except urllib.error.HTTPError as e: last=e.code
        except Exception: last=-1
    return [],last

def load_ticks(days,workers):
    jobs=[(d,h) for d in days for h in range(24)]; rows=[]; status={}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs=[ex.submit(fetch_hour,d,h) for d,h in jobs]
        for f in cf.as_completed(futs):
            r,s=f.result(); rows.extend(r); status[str(s)]=status.get(str(s),0)+1
    if not rows: raise SystemExit('no raw ticks')
    rows.sort(key=lambda x:x[0]); x=pd.DataFrame(rows,columns=['datetime','bid','ask','bid_size','ask_size']); x.datetime=pd.to_datetime(x.datetime,utc=True); x=x.drop_duplicates('datetime',keep='last').reset_index(drop=True); x['mid']=(x.bid+x.ask)/2.; x['spread']=x.ask-x.bid; x['tick_volume']=1.
    (OUT/'download_status.json').write_text(json.dumps({'hours':len(jobs),'status':status,'ticks':len(x)},indent=2))
    return x

def bars(ticks,sec):
    x=ticks.assign(bucket=ticks.datetime.dt.floor(f'{sec}s')); g=x.groupby('bucket',sort=True)
    return g.agg(open=('mid','first'),high=('mid','max'),low=('mid','min'),close=('mid','last'),volume=('tick_volume','sum'),spread=('spread','mean')).reset_index().rename(columns={'bucket':'datetime'})

def edges(b):
    o=b.copy(); rng=o.high-o.low; vol=o.volume.astype(float); vm=vol.shift(1).rolling(LOOKBACK,min_periods=LOOKBACK).mean(); vs=vol.shift(1).rolling(LOOKBACK,min_periods=LOOKBACK).std(ddof=0); rm=rng.shift(1).rolling(LOOKBACK,min_periods=LOOKBACK).mean(); z=(vol-vm)/vs.replace(0,np.nan); body=(o.close-o.open).abs(); bs=body.clip(lower=POINT); br=body/rng.replace(0,np.nan); up=o.high-o[['open','close']].max(axis=1); lo=o[['open','close']].min(axis=1)-o.low; hv=(z>=VOL_SIGMA)&(rng>0)
    imb=pd.Series(0,index=o.index,dtype='int8'); gate=hv&((rng/rm)>=RANGE_MULT)&(br>=.60); imb.loc[gate&(o.close>o.open)]=1; imb.loc[gate&(o.close<o.open)]=-1
    ab=pd.Series(0,index=o.index,dtype='int8'); ll=hv&(lo>=bs*WICK_RATIO)&(lo>up); lu=hv&(up>=bs*WICK_RATIO)&(up>lo); ab.loc[ll]=1; ab.loc[lu]=-1; o['IMBALANCE']=imb; o['ABSORPTION']=ab
    # External source-derived supervisor features; no BigPlayer formula changes.
    prev=o.close.shift(1); tr=pd.concat([(o.high-o.low),(o.high-prev).abs(),(o.low-prev).abs()],axis=1).max(axis=1); o['atr14']=tr.rolling(14,min_periods=14).mean(); o['ema20']=o.close.ewm(span=20,adjust=False).mean(); o['ema50']=o.close.ewm(span=50,adjust=False).mean(); o['ema200']=o.close.ewm(span=200,adjust=False).mean(); o['trend_dir']=np.where((o.ema20>o.ema50)&(o.ema50>o.ema200),1,np.where((o.ema20<o.ema50)&(o.ema50<o.ema200),-1,0)); o['atr_med120']=o.atr14.shift(1).rolling(120,min_periods=120).median(); o['spread_med120']=o.spread.shift(1).rolling(120,min_periods=120).median()
    return o

def exit_at_5m(ticks,entry_time,direction,qty,cost):
    a=ticks.datetime.searchsorted(entry_time,side='left'); target=entry_time+pd.Timedelta(minutes=5); z=ticks.datetime.searchsorted(target,side='left');
    if a>=len(ticks) or z>=len(ticks): return None
    entry=float(ticks.mid.iat[a]); ex=float(ticks.mid.iat[z]); return entry,ex,direction*(ex-entry)*qty-cost,ticks.datetime.iat[a],ticks.datetime.iat[z]

def collect(ticks,b,tf,edge,filters,initial,lot,contract,cost):
    sig=b[edge].to_numpy(); qty=lot*contract; out=[]
    for i in np.flatnonzero(sig!=0):
        d=int(sig[i]); et=b.datetime.iat[i]+pd.Timedelta(seconds=TF_SECONDS[tf]); ok=True
        if 'TREND' in filters: ok &= int(b.trend_dir.iat[i])==d
        if 'REGIME' in filters: ok &= pd.notna(b.atr_med120.iat[i]) and b.atr14.iat[i]>=b.atr_med120.iat[i]
        if 'SPREAD' in filters: ok &= pd.notna(b.spread_med120.iat[i]) and b.spread.iat[i]<=b.spread_med120.iat[i]
        if 'SESSION' in filters: ok &= et.hour in range(6,21)
        if not ok: continue
        r=exit_at_5m(ticks,et,d,qty,cost)
        if r: en,ex,pnl,eit,xit=r; out.append((tf,edge,d,eit,xit,en,ex,pnl))
    return pd.DataFrame(out,columns=['tf','edge','direction','entry_time','exit_time','entry','exit','pnl'])

def metrics(t,initial):
    if t.empty:return dict(N=0,N_per_day=0.,WR=0.,PF=0.,RF=0.,net_profit=0.,return_pct=0.,daily_return_pct=0.,max_dd_pct=0.,max_dd_usd=0.,final_balance=initial)
    t=t.sort_values('entry_time'); gp=t.loc[t.pnl>0,'pnl'].sum(); gl=-t.loc[t.pnl<0,'pnl'].sum(); net=t.pnl.sum(); pf=gp/gl if gl>0 else math.inf; eq=initial+t.pnl.cumsum(); peak=np.maximum.accumulate(np.r_[initial,eq.values])[1:]; dd=peak-eq.values; mdd=float(dd.max()); rf=net/mdd if mdd>0 else math.inf
    return dict(N=len(t),N_per_day=len(t)/21.,WR=(t.pnl>0).mean()*100.,PF=pf,RF=rf,net_profit=net,return_pct=net/initial*100.,daily_return_pct=net/initial*100./21.,max_dd_pct=float(np.max(np.where(peak>0,dd/peak*100.,0))),max_dd_usd=mdd,final_balance=initial+net)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--start',default='2026-07-27'); ap.add_argument('--workers',type=int,default=48); ap.add_argument('--initial',type=float,default=1000.); ap.add_argument('--lot',type=float,default=.01); ap.add_argument('--contract',type=float,default=100.); ap.add_argument('--cost',type=float,default=0.); a=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True); days=business_days(dt.date.fromisoformat(a.start),21); ticks=load_ticks(days,a.workers)
    built={tf:edges(bars(ticks,s)) for tf,s in TF_SECONDS.items()}; selected=[('M1','IMBALANCE'),('M5','ABSORPTION')]
    configs={'BASE':(), 'TREND':('TREND',), 'REGIME':('REGIME',), 'SPREAD':('SPREAD',), 'SESSION':('SESSION',), 'TREND_REGIME':('TREND','REGIME'), 'TREND_SPREAD':('TREND','SPREAD'), 'REGIME_SPREAD':('REGIME','SPREAD'), 'TREND_REGIME_SPREAD':('TREND','REGIME','SPREAD'), 'TREND_REGIME_SPREAD_SESSION':('TREND','REGIME','SPREAD','SESSION')}
    rows=[]; frames={}
    for name,fs in configs.items():
        parts=[collect(ticks,built[tf],tf,edge,fs,a.initial,a.lot,a.contract,a.cost) for tf,edge in selected]; t=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(); frames[name]=t; rows.append({'config':name,'filters':'+'.join(fs) or 'NONE',**metrics(t,a.initial)})
    res=pd.DataFrame(rows); base=res.iloc[0]
    for c in ['N','WR','PF','RF','return_pct','max_dd_pct']:
        res[f'delta_{c}_vs_base']=res[c]-base[c]
    res.to_csv(OUT/'summary_21d.csv',index=False)
    for n,t in frames.items(): t.to_csv(OUT/f'trades_{n}.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps({'formula_policy':'FROZEN_BIGPLAYER_2EDGE_NO_INTERNAL_CHANGES','selected_edges':selected,'bar_policy':'raw QuoteTick direct aggregation; no OHLC resample','execution':'common fixed 5-minute raw-tick horizon across all TFs','supervisor':'external gates only','configs':{k:list(v) for k,v in configs.items()},'cost_usd':a.cost},indent=2))
    print(res.to_string(index=False))
if __name__=='__main__': main()
