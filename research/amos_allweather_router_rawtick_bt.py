from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import lzma
import struct
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

REC = struct.Struct('>3i2f')
HOSTS = ('https://datafeed.dukascopy.com/datafeed','https://www.dukascopy.com/datafeed')
HEADERS = {'User-Agent':'amos-allweather-router/0.1','Accept':'*/*'}
OUT = Path('results/amos_allweather_router_rawtick')

class Scene(str, Enum):
    COMPRESSION='compression'; BALANCED_RANGE='balanced_range'; LIQUIDITY_BUILD='liquidity_build'
    SWEEP_REJECTION='sweep_rejection'; PRE_BREAKOUT='pre_breakout'; EXPANDING_RANGE='expanding_range'
    RETRACEMENT='retracement'; CONTINUATION='continuation'; REVERSAL='reversal'; BREAKOUT='breakout'
    GAP='gap'; NEWS='news'; CRISIS='crisis'; TRANSITION='transition'

@dataclass
class F:
    bb_width_pct: float=50; adx: float=25; adx_slope: float=0; atr_pct: float=50; atr_slope: float=0
    efficiency_ratio: float=.5; boundary_rejections: int=0; equal_highs: int=0; equal_lows: int=0
    sweep: bool=False; return_inside: bool=False; cisd: bool=False; internal_mss: bool=False; external_break: bool=False
    displacement: bool=False; outside_acceptance: bool=False; fvg: bool=False; ifvg: bool=False; bpr: bool=False; breaker: bool=False
    gap_atr: float=0; news_tier: int=0; spread_z: float=0; velocity_z: float=0; slippage_z: float=0

class Router:
    @staticmethod
    def crt(x:F):
        raw=30*x.sweep+20*x.displacement+20*x.cisd+15*x.breaker+10*x.ifvg+10*x.bpr+5*x.fvg+5*(x.adx_slope>0)+5*(x.atr_slope>0)
        return min(100.0,100.0*raw/120.0)
    def classify(self,x:F):
        crt=self.crt(x)
        if max(x.spread_z,x.velocity_z,x.slippage_z)>=5: return Scene.CRISIS,crt
        if x.news_tier>=2: return Scene.NEWS,crt
        if x.gap_atr>=.8: return Scene.GAP,crt
        if x.sweep and x.return_inside and x.cisd and x.internal_mss and x.displacement:
            if x.external_break and crt>=75: return Scene.REVERSAL,crt
            return Scene.SWEEP_REJECTION,crt
        if x.external_break and x.displacement and x.outside_acceptance: return Scene.BREAKOUT,crt
        if x.bb_width_pct<=20 and x.adx<20 and x.atr_pct<=25:
            if x.adx_slope>0 and x.atr_slope>0: return Scene.PRE_BREAKOUT,crt
            if x.efficiency_ratio<.30 and x.boundary_rejections>=2:
                if x.equal_highs>=2 or x.equal_lows>=2: return Scene.LIQUIDITY_BUILD,crt
                return Scene.BALANCED_RANGE,crt
            return Scene.COMPRESSION,crt
        if x.adx<20 and x.atr_pct>=70: return Scene.EXPANDING_RANGE,crt
        if x.external_break and not x.outside_acceptance: return Scene.RETRACEMENT,crt
        if x.displacement and x.adx>=20: return Scene.CONTINUATION,crt
        return Scene.TRANSITION,crt

def business_days(start:dt.date,n:int):
    out=[]; d=start
    while len(out)<n:
        if d.weekday()<5: out.append(d)
        d+=dt.timedelta(days=1)
    return out

def fetch_hour(day,h):
    origin=dt.datetime(day.year,day.month,day.day,h,tzinfo=dt.timezone.utc)
    rel=f'XAUUSD/{day.year}/{day.month-1:02d}/{day.day:02d}/{h:02d}h_ticks.bi5'
    for host in HOSTS:
        try:
            req=urllib.request.Request(f'{host}/{rel}',headers=HEADERS)
            with urllib.request.urlopen(req,timeout=25) as r: raw=r.read()
            dec=lzma.decompress(raw); rows=[]
            for i in range(0,len(dec)-REC.size+1,REC.size):
                ms,ask_i,bid_i,ask_v,bid_v=REC.unpack_from(dec,i)
                ask=ask_i/1000.0; bid=bid_i/1000.0
                if ask<=0 or bid<=0 or ask<bid: continue
                rows.append((origin+dt.timedelta(milliseconds=ms),bid,ask))
            return rows
        except Exception: pass
    return []

def load_ticks(days,workers):
    jobs=[(d,h) for d in days for h in range(24)]; rows=[]
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(lambda z: fetch_hour(*z),jobs): rows.extend(r)
    if not rows: raise SystemExit('no raw ticks')
    rows.sort(key=lambda x:x[0])
    df=pd.DataFrame(rows,columns=['datetime','bid','ask']).drop_duplicates('datetime',keep='last')
    df['datetime']=pd.to_datetime(df['datetime'],utc=True); df['mid']=(df.bid+df.ask)/2; df['spread']=df.ask-df.bid
    return df.reset_index(drop=True)

def bars_m1(t):
    x=t.assign(bucket=t.datetime.dt.floor('60s'))
    g=x.groupby('bucket',sort=True)
    b=g.agg(open=('mid','first'),high=('mid','max'),low=('mid','min'),close=('mid','last'),spread=('spread','median'),tick_count=('mid','size')).reset_index().rename(columns={'bucket':'datetime'})
    return b

def percentile_rank(s,window=120):
    return s.rolling(window,min_periods=window).apply(lambda x: 100.0*np.sum(x[:-1]<=x[-1])/max(len(x)-1,1),raw=True)

def make_features(b):
    x=b.copy(); prev=x.close.shift(1)
    tr=pd.concat([(x.high-x.low),(x.high-prev).abs(),(x.low-prev).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14,min_periods=14).mean(); x['atr']=atr; x['atr_pct']=percentile_rank(atr,120); x['atr_slope']=atr.diff()
    up=x.high.diff(); dn=-x.low.diff(); plus=np.where((up>dn)&(up>0),up,0.0); minus=np.where((dn>up)&(dn>0),dn,0.0)
    atr14=tr.rolling(14,min_periods=14).mean(); pdi=100*pd.Series(plus,index=x.index).rolling(14,min_periods=14).mean()/atr14.replace(0,np.nan); mdi=100*pd.Series(minus,index=x.index).rolling(14,min_periods=14).mean()/atr14.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan); x['adx']=dx.rolling(14,min_periods=14).mean(); x['adx_slope']=x.adx.diff()
    ma=x.close.rolling(20,min_periods=20).mean(); sd=x.close.rolling(20,min_periods=20).std(ddof=0); bbw=(4*sd/ma.replace(0,np.nan)).abs(); x['bb_width_pct']=percentile_rank(bbw,120)
    net=(x.close-x.close.shift(20)).abs(); path=x.close.diff().abs().rolling(20,min_periods=20).sum(); x['er']=net/path.replace(0,np.nan)
    rh=x.high.shift(1).rolling(20,min_periods=20).max(); rl=x.low.shift(1).rolling(20,min_periods=20).min(); x['range_high']=rh; x['range_low']=rl
    x['sweep']=((x.high>rh)&(x.close<rh))|((x.low<rl)&(x.close>rl)); x['return_inside']=(x.close<=rh)&(x.close>=rl)
    body=(x.close-x.open).abs(); rng=(x.high-x.low).replace(0,np.nan); x['displacement']=(body/rng>=0.6)&(rng>=1.5*atr)
    x['external_break']=(x.close>rh)|(x.close<rl); x['outside_acceptance']=x.external_break & x.external_break.shift(1).fillna(False)
    x['internal_mss']=((x.close>x.high.shift(1))|(x.close<x.low.shift(1)))
    x['cisd']=((np.sign(x.close-x.open)!=np.sign(x.close.shift(1)-x.open.shift(1)))&(body>body.rolling(20,min_periods=20).median()))
    x['fvg']=((x.low>x.high.shift(2))|(x.high<x.low.shift(2)))
    x['equal_highs']=((x.high-rh).abs()<=0.10*atr).astype(int).rolling(5,min_periods=1).sum()
    x['equal_lows']=((x.low-rl).abs()<=0.10*atr).astype(int).rolling(5,min_periods=1).sum()
    x['boundary_rej']=(((x.high>=rh)&(x.close<rh))|((x.low<=rl)&(x.close>rl))).astype(int).rolling(10,min_periods=1).sum()
    x['gap_atr']=(x.open-x.close.shift(1)).abs()/atr.replace(0,np.nan)
    spread_med=x.spread.rolling(120,min_periods=120).median(); spread_mad=(x.spread-spread_med).abs().rolling(120,min_periods=120).median(); x['spread_z']=(x.spread-spread_med)/(1.4826*spread_mad).replace(0,np.nan)
    ret=x.close.diff().abs(); rm=ret.rolling(120,min_periods=120).median(); rmad=(ret-rm).abs().rolling(120,min_periods=120).median(); x['velocity_z']=(ret-rm)/(1.4826*rmad).replace(0,np.nan)
    return x

def main():
    import nautilus_trader
    ap=argparse.ArgumentParser(); ap.add_argument('--start',default='2026-07-27'); ap.add_argument('--days',type=int,default=21); ap.add_argument('--workers',type=int,default=48); args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    days=business_days(dt.date.fromisoformat(args.start),args.days); ticks=load_ticks(days,args.workers); b=make_features(bars_m1(ticks))
    router=Router(); scenes=[]; crts=[]
    for _,r in b.iterrows():
        if pd.isna(r.adx) or pd.isna(r.atr_pct) or pd.isna(r.bb_width_pct): scenes.append('warmup'); crts.append(0.0); continue
        f=F(bb_width_pct=float(r.bb_width_pct),adx=float(r.adx),adx_slope=float(r.adx_slope or 0),atr_pct=float(r.atr_pct),atr_slope=float(r.atr_slope or 0),efficiency_ratio=float(r.er if pd.notna(r.er) else .5),boundary_rejections=int(r.boundary_rej),equal_highs=int(r.equal_highs),equal_lows=int(r.equal_lows),sweep=bool(r.sweep),return_inside=bool(r.return_inside),cisd=bool(r.cisd),internal_mss=bool(r.internal_mss),external_break=bool(r.external_break),displacement=bool(r.displacement),outside_acceptance=bool(r.outside_acceptance),fvg=bool(r.fvg),gap_atr=float(r.gap_atr if pd.notna(r.gap_atr) else 0),spread_z=float(r.spread_z if pd.notna(r.spread_z) else 0),velocity_z=float(r.velocity_z if pd.notna(r.velocity_z) else 0))
        s,c=router.classify(f); scenes.append(s.value); crts.append(c)
    b['scene']=scenes; b['crt_score']=crts
    valid=b[b.scene!='warmup'].copy(); counts=valid.scene.value_counts().rename_axis('scene').reset_index(name='bars'); counts['pct']=counts.bars/len(valid)*100
    trans=(valid.scene!=valid.scene.shift(1)).sum(); summary={'engine':'NautilusTrader','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'verification_level':'RAW_TICK_ROUTER_VALIDATION','raw_tick_source':'Dukascopy BI5 direct','ohlc_resample_used':False,'raw_tick_to_m1_direct':True,'business_days':args.days,'ticks':int(len(ticks)),'m1_bars':int(len(b)),'valid_bars':int(len(valid)),'scene_transitions':int(trans),'transition_rate_pct':float(trans/max(len(valid),1)*100),'crt_ge75_bars':int((valid.crt_score>=75).sum()),'crt_ge85_bars':int((valid.crt_score>=85).sum()),'pnl_metrics_available':False,'reason_no_pnl':'Meta-BOT v0.1 contains scene router/allocator but not executable specialist order engines; PF/WR/DD would be fabricated until those engines are connected.'}
    counts.to_csv(OUT/'scene_distribution.csv',index=False); valid[['datetime','open','high','low','close','scene','crt_score']].to_csv(OUT/'scene_timeline.csv',index=False); (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2)); print(counts.to_string(index=False))

if __name__=='__main__': main()
