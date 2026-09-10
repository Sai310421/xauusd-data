#!/usr/bin/env python3
"""AMOS Reversal V1-V4 cross-version backtest runner.

Design goals:
- one deterministic runner for V1..V4 and M1/M5/M15/H1
- raw-tick execution contract (fails closed if no tick source is found)
- no look-ahead; features use completed bars only
- structural SL, baseline TP=2R for generation comparability
- explicit diagnostics for MTF leadership/propagation/context

This first repository runner is intentionally self-contained. It converts raw ticks to bars
only for *signal state construction*; fills are evaluated against tick bid/ask, not OHLC.
"""
from __future__ import annotations

import argparse, json, math, os, random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

TF_MIN = {"M1":1,"M5":5,"M15":15,"H1":60}

@dataclass
class Trade:
    version: str; timeframe: str; mode: str; side: int
    entry_time: str; exit_time: str; entry: float; stop: float; target: float
    exit_price: float; r: float; result: str
    leader_tf: str=""; handoff: int=0; conflict: float=0.0
    structure_score: float=0.0; context_score: float=0.0; probability: float=0.0


def cli():
    p=argparse.ArgumentParser()
    p.add_argument('--version',required=True,choices=['V1','V2','V3','V4'])
    p.add_argument('--timeframe',required=True,choices=list(TF_MIN))
    p.add_argument('--mode',required=True,choices=['BASE','AGGRESSIVE','STANDARD','CONSERVATIVE'])
    p.add_argument('--symbol',default='XAUUSD')
    p.add_argument('--dataset',default='repository')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--rr',type=float,default=2.0)
    p.add_argument('--out',required=True)
    return p.parse_args()


def discover_tick_source(symbol:str)->Path:
    env=os.environ.get('AMOS_RAW_TICK_PATH')
    candidates=[]
    if env: candidates.append(Path(env))
    roots=[Path('ticks'),Path('raw'),Path('data'),Path('parquet'),Path('csv')]
    pats=[f'*{symbol}*tick*.parquet',f'*{symbol}*Tick*.parquet',f'*{symbol}*ticks*.parquet',
          f'*{symbol}*tick*.csv',f'*{symbol}*Tick*.csv',f'*{symbol}*ticks*.csv']
    for root in roots:
        if root.exists():
            for pat in pats: candidates.extend(root.rglob(pat))
    for p in candidates:
        if p.exists() and p.is_file(): return p
    raise FileNotFoundError(
        f'Raw tick source for {symbol} not found. Set AMOS_RAW_TICK_PATH. '
        'Runner refuses OHLC substitution by design.'
    )


def load_ticks(path:Path)->pd.DataFrame:
    d=pd.read_parquet(path) if path.suffix.lower()=='.parquet' else pd.read_csv(path)
    low={c.lower():c for c in d.columns}
    tcol=next((low[k] for k in ['timestamp','time','datetime','date_time','ts'] if k in low),None)
    if tcol is None: raise ValueError('tick timestamp column not found')
    bid=next((low[k] for k in ['bid','bid_price','price_bid'] if k in low),None)
    ask=next((low[k] for k in ['ask','ask_price','price_ask'] if k in low),None)
    if bid is None or ask is None: raise ValueError('raw tick source must contain bid and ask')
    vol=next((low[k] for k in ['volume','tick_volume','size'] if k in low),None)
    out=pd.DataFrame({'time':pd.to_datetime(d[tcol],utc=True,errors='coerce'),
                      'bid':pd.to_numeric(d[bid],errors='coerce'),
                      'ask':pd.to_numeric(d[ask],errors='coerce')})
    out['vol']=pd.to_numeric(d[vol],errors='coerce').fillna(1.0) if vol else 1.0
    out=out.dropna().sort_values('time').drop_duplicates('time',keep='last').reset_index(drop=True)
    if out.empty: raise ValueError('empty tick source')
    return out


def bars_from_ticks(t:pd.DataFrame, tf:str)->pd.DataFrame:
    x=t.set_index('time')
    mid=(x.bid+x.ask)/2
    rule=f'{TF_MIN[tf]}min'
    o=mid.resample(rule,label='right',closed='right').ohlc()
    o['volume']=x.vol.resample(rule,label='right',closed='right').sum()
    o['spread']=(x.ask-x.bid).resample(rule,label='right',closed='right').mean()
    return o.dropna().reset_index()


def atr(b,n=14):
    pc=b.close.shift(1); tr=pd.concat([(b.high-b.low),(b.high-pc).abs(),(b.low-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0).rolling(n).mean(); dn=(-d.clip(upper=0)).rolling(n).mean()
    rs=up/(dn+1e-12); return 100-100/(1+rs)

def adx_proxy(b,n=14):
    # Stable regime proxy: directional range / total true range, scaled 0..100.
    net=(b.close-b.close.shift(n)).abs(); tr=atr(b,n)*n
    return (100*net/(tr+1e-12)).clip(0,100)

def vwap(b):
    tp=(b.high+b.low+b.close)/3; return (tp*b.volume).cumsum()/(b.volume.cumsum()+1e-12)


def feature_frame(b:pd.DataFrame)->pd.DataFrame:
    z=b.copy(); z['atr']=atr(z); z['rsi']=rsi(z.close); z['adx']=adx_proxy(z); z['vwap']=vwap(z)
    z['vol_med']=z.volume.rolling(20).median(); z['range']=z.high-z.low; z['body']=(z.close-z.open).abs()
    z['disp']=z.body/(z.atr+1e-12)
    # completed-bar swing levels; shift ensures no current-bar leakage
    z['swing_hi']=z.high.shift(1).rolling(5).max(); z['swing_lo']=z.low.shift(1).rolling(5).min()
    z['prev_hi2']=z.high.shift(2); z['prev_lo2']=z.low.shift(2)
    # FVG using fully completed bars: bullish low(t) > high(t-2), bearish high(t) < low(t-2)
    z['bull_fvg']=(z.low>z.high.shift(2)); z['bear_fvg']=(z.high<z.low.shift(2))
    return z


def sigmoid(x): return 1/(1+math.exp(-max(-40,min(40,x))))


def signal_at(z:pd.DataFrame,i:int,version:str,mode:str):
    if i<30: return None
    r=z.iloc[i]; p=z.iloc[i-1]
    if not np.isfinite(r.atr) or r.atr<=0: return None
    # Liquidity sweep + reclaim
    bull_sweep = r.low < r.swing_lo and r.close > r.swing_lo
    bear_sweep = r.high > r.swing_hi and r.close < r.swing_hi
    side=1 if bull_sweep else (-1 if bear_sweep else 0)
    if side==0: return None
    # CISD proxy: close crosses prior opposite-delivery open/body anchor
    cisd=(r.close>p.open and r.close>p.close) if side==1 else (r.close<p.open and r.close<p.close)
    # MSS: close breaks nearest completed short-term structure
    mss=(r.close>z.high.shift(1).rolling(3).max().iloc[i]) if side==1 else (r.close<z.low.shift(1).rolling(3).min().iloc[i])
    displacement=r.disp>=0.8
    fvg=bool(r.bull_fvg if side==1 else r.bear_fvg)
    # IFVG/BPR approximations are explicit, deterministic state features.
    prior_opp = bool(z.bear_fvg.iloc[max(0,i-8):i].any()) if side==1 else bool(z.bull_fvg.iloc[max(0,i-8):i].any())
    ifvg=prior_opp and displacement
    bpr=fvg and prior_opp
    structure=(1.0*bull_sweep if side==1 else 1.0*bear_sweep)+1.0*cisd+1.2*mss+1.0*displacement+0.7*fvg+0.5*ifvg+0.5*bpr

    if version=='V1':
        ok=cisd and mss and displacement and (fvg or ifvg or bpr)
        return (side,structure,0.0,0.50,'') if ok else None

    threshold={'AGGRESSIVE':2.7,'STANDARD':3.5,'CONSERVATIVE':4.2}.get(mode,3.5)
    if version=='V2':
        ok=structure>=threshold and cisd
        return (side,structure,0.0,sigmoid(structure-3.5),'') if ok else None

    # V3 local adaptive-lead proxy: freshness/velocity and structural maturity.
    velocity=min(1.5, abs(r.close-p.close)/(r.atr+1e-12))
    freshness=1.0
    adaptive=0.55*structure+0.30*velocity+0.15*freshness
    if version=='V3':
        prob=sigmoid(-2.1+0.72*structure+0.35*velocity)
        gate={'AGGRESSIVE':0.50,'STANDARD':0.54,'CONSERVATIVE':0.58}[mode]
        return (side,structure,adaptive,prob,'SELF') if prob>=gate else None

    # V4 context fusion: momentum/regime/location/participation are soft evidence only.
    mom=((r.rsi-50)/25.0)*side
    regime=max(-1,min(1,(r.adx-20)/20.0))
    loc=((r.close-r.vwap)/(r.atr+1e-12))*side if np.isfinite(r.vwap) else 0.0
    participation=min(2.0,r.volume/(r.vol_med+1e-12))-1.0 if np.isfinite(r.vol_med) else 0.0
    context=0.30*mom+0.25*regime+0.25*loc+0.20*participation
    prob=sigmoid(-2.35+0.67*structure+0.30*adaptive+0.45*context)
    gate={'AGGRESSIVE':0.52,'STANDARD':0.56,'CONSERVATIVE':0.60}[mode]
    return (side,structure,context,prob,'ADAPTIVE') if prob>=gate else None


def simulate(ticks:pd.DataFrame,b:pd.DataFrame,version:str,tf:str,mode:str,rr:float):
    z=feature_frame(b); trades=[]
    times=ticks.time.values
    last_exit=pd.Timestamp.min.tz_localize('UTC')
    for i in range(30,len(z)-1):
        sig=signal_at(z,i,version,mode)
        if sig is None: continue
        side,structure,ctx,prob,leader=sig
        sig_time=z.time.iloc[i]
        if sig_time<=last_exit: continue
        # next tick after completed signal bar prevents same-bar hindsight fill
        j=int(np.searchsorted(times,np.datetime64(sig_time.to_datetime64()),side='right'))
        if j>=len(ticks): break
        q=ticks.iloc[j]
        entry=float(q.ask if side==1 else q.bid)
        atrv=float(z.atr.iloc[i]); swing=float(z.swing_lo.iloc[i] if side==1 else z.swing_hi.iloc[i])
        stop=min(swing,entry-0.35*atrv) if side==1 else max(swing,entry+0.35*atrv)
        risk=abs(entry-stop)
        if risk<=0 or not np.isfinite(risk): continue
        target=entry+side*rr*risk
        exit_price=None; exit_t=None; result='OPEN'; rval=0.0
        # tick-exact barrier evaluation, max 5 TF bars to keep reversal test local
        horizon=sig_time+pd.Timedelta(minutes=TF_MIN[tf]*5)
        for k in range(j+1,len(ticks)):
            x=ticks.iloc[k]
            if x.time>horizon: break
            px=float(x.bid if side==1 else x.ask)  # executable close side
            if side==1:
                if px<=stop: exit_price=px; result='LOSS'; rval=-1.0; exit_t=x.time; break
                if px>=target: exit_price=px; result='WIN'; rval=rr; exit_t=x.time; break
            else:
                if px>=stop: exit_price=px; result='LOSS'; rval=-1.0; exit_t=x.time; break
                if px<=target: exit_price=px; result='WIN'; rval=rr; exit_t=x.time; break
        if exit_price is None:
            k=min(len(ticks)-1,max(j+1,int(np.searchsorted(times,np.datetime64(horizon.to_datetime64()),side='right')-1)))
            x=ticks.iloc[k]; exit_price=float(x.bid if side==1 else x.ask); exit_t=x.time
            rval=side*(exit_price-entry)/risk; result='TIME'
        trades.append(Trade(version,tf,mode,side,str(sig_time),str(exit_t),entry,stop,target,exit_price,float(rval),result,
                            leader_tf=leader,structure_score=float(structure),context_score=float(ctx),probability=float(prob)))
        last_exit=exit_t
    return trades


def metrics(trades:list[Trade]):
    rs=np.array([x.r for x in trades],dtype=float)
    n=len(rs); wins=int((rs>0).sum()); losses=int((rs<=0).sum()); wr=100*wins/n if n else 0.0
    gp=rs[rs>0].sum() if n else 0.; gl=-rs[rs<0].sum() if n else 0.; pf=gp/gl if gl>0 else (999.0 if gp>0 else 0.0)
    curve=np.cumsum(rs) if n else np.array([0.]); peak=np.maximum.accumulate(curve); dd=peak-curve; maxdd=float(dd.max()) if len(dd) else 0.
    net=float(rs.sum()) if n else 0.; rf=net/maxdd if maxdd>0 else (999.0 if net>0 else 0.)
    return {'N':n,'wins':wins,'losses':losses,'WR_pct':wr,'avg_R':float(rs.mean()) if n else 0.,'PF':float(pf),
            'EV_R_per_trade':float(rs.mean()) if n else 0.,'net_R':net,'max_DD_R':maxdd,'RF':float(rf)}


def main():
    a=cli(); random.seed(a.seed); np.random.seed(a.seed)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    src=discover_tick_source(a.symbol); ticks=load_ticks(src); bars=bars_from_ticks(ticks,a.timeframe)
    trades=simulate(ticks,bars,a.version,a.timeframe,a.mode,a.rr); m=metrics(trades)
    pd.DataFrame([asdict(x) for x in trades]).to_csv(out/'trades.csv',index=False)
    pd.DataFrame({'time':bars.time}).to_csv(out/'telemetry.csv',index=False)
    m.update({'version':a.version,'timeframe':a.timeframe,'mode':a.mode,'symbol':a.symbol,'rr_target':a.rr,
              'period_start':str(ticks.time.iloc[0]),'period_end':str(ticks.time.iloc[-1]),'raw_tick_source':str(src),
              'verification_level':'RAW_TICK_SIGNAL_AND_EXECUTION'})
    (out/'metrics.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps({'args':vars(a),'raw_tick_source':str(src),'rows':len(ticks),'bars':len(bars)},indent=2),encoding='utf-8')
    print(json.dumps(m,indent=2))

if __name__=='__main__': main()
