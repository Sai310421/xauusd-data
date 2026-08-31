from __future__ import annotations
# Finish screen: BigPlayer formulas frozen. External liquidity + execution management only.
import argparse, datetime as dt, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import research.bigplayer_synergy_supervisor_21d as core

OUT=Path('results/bigplayer_finish_liquidity_exit_21d'); INITIAL=1000.; QTY=1.0

def swing_sweeps(b,lookback=50):
    x=b.copy(); ph=x.high.shift(1).rolling(lookback,min_periods=lookback).max(); pl=x.low.shift(1).rolling(lookback,min_periods=lookback).min()
    x['bull_sweep']=(x.low<pl)&(x.close>pl)
    x['bear_sweep']=(x.high>ph)&(x.close<ph)
    return x

def trade_path(ticks,entry_time,direction,atr,mode='FIXED5',boost=1.0):
    a=ticks.datetime.searchsorted(entry_time,side='left'); z=ticks.datetime.searchsorted(entry_time+pd.Timedelta(minutes=5),side='left')
    if a>=len(ticks) or z>=len(ticks) or z<=a:return None
    p=ticks.iloc[a:z+1]; entry=float(p.ask.iloc[0] if direction>0 else p.bid.iloc[0]); last=float(p.bid.iloc[-1] if direction>0 else p.ask.iloc[-1])
    if mode=='FIXED5': ex=last
    else:
        # External source-derived exit geometry, evaluated on raw bid/ask path.
        sl_mult,tp_mult,trail_act=(0.3,0.6,0.2) if mode=='TESTBOT' else (1.0,2.0,1.2)
        sl=entry-direction*sl_mult*atr; tp=entry+direction*tp_mult*atr; trail_on=False; best=entry; ex=last
        for _,r in p.iloc[1:].iterrows():
            px=float(r.bid if direction>0 else r.ask); best=max(best,px) if direction>0 else min(best,px)
            fav=direction*(best-entry)
            if mode=='TESTBOT' and fav>=trail_act*atr: trail_on=True
            if mode=='FAST' and fav>=1.2*atr: trail_on=True
            trail=(best-direction*trail_act*atr) if trail_on else None
            stop_hit=(px<=sl if direction>0 else px>=sl)
            tp_hit=(px>=tp if direction>0 else px<=tp)
            trail_hit=trail_on and (px<=trail if direction>0 else px>=trail)
            if stop_hit or tp_hit or trail_hit: ex=px; break
    pnl=direction*(ex-entry)*QTY*boost
    return entry,ex,pnl

def metrics(t):
    if t.empty:return dict(N=0,N_per_day=0.,WR=0.,PF=0.,RF=0.,net_profit=0.,return_pct=0.,daily_return_pct=0.,max_dd_pct=0.,max_dd_usd=0.,final_balance=INITIAL)
    t=t.sort_values('entry_time'); p=t.pnl; gp=p[p>0].sum(); gl=-p[p<0].sum(); net=p.sum(); eq=INITIAL+p.cumsum(); peak=np.maximum.accumulate(np.r_[INITIAL,eq.values])[1:]; dd=peak-eq.values; mdd=float(dd.max());
    return dict(N=len(t),N_per_day=len(t)/21.,WR=(p>0).mean()*100.,PF=gp/gl if gl>0 else math.inf,RF=net/mdd if mdd>0 else math.inf,net_profit=net,return_pct=net/INITIAL*100.,daily_return_pct=net/INITIAL*100./21.,max_dd_pct=float(np.max(np.where(peak>0,dd/peak*100.,0))),max_dd_usd=mdd,final_balance=INITIAL+net)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--start',default='2026-07-27'); ap.add_argument('--workers',type=int,default=48); a=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    days=core.business_days(dt.date.fromisoformat(a.start),21); ticks=core.load_ticks(days,a.workers)
    built={tf:swing_sweeps(core.edges(core.bars(ticks,s))) for tf,s in core.TF_SECONDS.items()}
    selected=[('M1','IMBALANCE'),('M5','ABSORPTION')]
    raw=[]
    for tf,edge in selected:
        b=built[tf]; sig=b[edge].to_numpy()
        for i in np.flatnonzero(sig!=0):
            d=int(sig[i]); et=b.datetime.iat[i]+pd.Timedelta(seconds=core.TF_SECONDS[tf])
            # preserve winning proxy TREND+REGIME gate exactly as prior screen
            ok=(int(b.trend_dir.iat[i])==d) and pd.notna(b.atr_med120.iat[i]) and (b.atr14.iat[i]>=b.atr_med120.iat[i])
            if not ok: continue
            raw.append({'tf':tf,'edge':edge,'direction':d,'entry_time':et,'atr':float(b.atr14.iat[i]),'liq':bool(b.bull_sweep.iat[i] if d>0 else b.bear_sweep.iat[i])})
    s=pd.DataFrame(raw).sort_values('entry_time'); s['minute']=pd.to_datetime(s.entry_time,utc=True).dt.floor('min'); c=s.groupby(['minute','direction']).size().rename('agree').reset_index(); s=s.merge(c,on=['minute','direction'],how='left'); s['boost']=np.where(s.agree>=2,2.0,1.0)
    configs=[('BIDASK_FIXED5',False,'FIXED5'),('LIQ_BOOST_FIXED5',True,'FIXED5'),('BIDASK_TESTBOT_EXIT',False,'TESTBOT'),('BIDASK_FAST_EXIT',False,'FAST'),('LIQ_TESTBOT',True,'TESTBOT'),('LIQ_FAST',True,'FAST')]
    rows=[]
    for name,liq_only,mode in configs:
        out=[]
        for _,r in s.iterrows():
            if liq_only and not r.liq: continue
            q=trade_path(ticks,r.entry_time,int(r.direction),float(r.atr),mode,float(r.boost))
            if q: en,ex,pnl=q; out.append({**r.to_dict(),'entry':en,'exit':ex,'pnl':pnl})
        t=pd.DataFrame(out); rows.append({'config':name,'liquidity_required':liq_only,'exit_mode':mode,**metrics(t)}); t.to_csv(OUT/f'trades_{name}.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'summary_21d.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps({'formula_policy':'FROZEN_BIGPLAYER_NO_INTERNAL_CHANGES','entry_supervisor':'same proxy TREND+REGIME winner from prior screen','mtf_boost':'same-minute same-direction x2.0','liquidity':'50-bar prior swing sweep: pierce then close back','execution':'raw bid/ask, 5-minute max horizon','exit_branches':{'TESTBOT':'ATR14 SL0.3 / TP0.6 / trail activation-distance0.2','FAST':'ATR14 SL1.0 / TP2.0 / trail after1.2 ATR'}},indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
