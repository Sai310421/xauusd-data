from __future__ import annotations
# BigPlayer formulas frozen. External execution-only Economic BE / First-Passage screen.
import argparse, datetime as dt, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research.bigplayer_synergy_supervisor_21d as core

OUT=Path('results/bigplayer_economic_be_firstpassage_21d'); INITIAL=1000.; QTY=1.0

def metrics(t):
    if t.empty:return dict(N=0,N_per_day=0.,WR=0.,PF=0.,RF=0.,net_profit=0.,return_pct=0.,daily_return_pct=0.,max_dd_pct=0.,max_dd_usd=0.,final_balance=INITIAL)
    t=t.sort_values('entry_time'); p=t.pnl; gp=p[p>0].sum(); gl=-p[p<0].sum(); net=p.sum(); eq=INITIAL+p.cumsum(); peak=np.maximum.accumulate(np.r_[INITIAL,eq.values])[1:]; dd=peak-eq.values; mdd=float(dd.max())
    return dict(N=len(t),N_per_day=len(t)/21.,WR=(p>0).mean()*100.,PF=gp/gl if gl>0 else math.inf,RF=net/mdd if mdd>0 else math.inf,net_profit=net,return_pct=net/INITIAL*100.,daily_return_pct=net/INITIAL*100./21.,max_dd_pct=float(np.max(np.where(peak>0,dd/peak*100.,0))),max_dd_usd=mdd,final_balance=INITIAL+net)

def first_passage(ticks,entry_time,direction,horizon_min,target_spread_mult,boost):
    a=ticks.datetime.searchsorted(entry_time,side='left'); z=ticks.datetime.searchsorted(entry_time+pd.Timedelta(minutes=horizon_min),side='left')
    if a>=len(ticks) or z>=len(ticks) or z<=a:return None
    p=ticks.iloc[a:z+1]
    entry=float(p.ask.iloc[0] if direction>0 else p.bid.iloc[0]); spr=float(p.ask.iloc[0]-p.bid.iloc[0]); target=target_spread_mult*spr
    exit_px=float(p.bid.iloc[-1] if direction>0 else p.ask.iloc[-1]); hit=False; hit_time=p.datetime.iloc[-1]
    for _,r in p.iloc[1:].iterrows():
        px=float(r.bid if direction>0 else r.ask); netmove=direction*(px-entry)
        if netmove>=target:
            exit_px=px; hit=True; hit_time=r.datetime; break
    pnl=direction*(exit_px-entry)*QTY*boost
    return entry,exit_px,pnl,hit,hit_time,spr,target

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--start',default='2026-07-27'); ap.add_argument('--workers',type=int,default=48); a=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    days=core.business_days(dt.date.fromisoformat(a.start),21); ticks=core.load_ticks(days,a.workers)
    built={tf:core.edges(core.bars(ticks,s)) for tf,s in core.TF_SECONDS.items()}; selected=[('M1','IMBALANCE'),('M5','ABSORPTION')]; raw=[]
    for tf,edge in selected:
        b=built[tf]; sig=b[edge].to_numpy()
        for i in np.flatnonzero(sig!=0):
            d=int(sig[i]); et=b.datetime.iat[i]+pd.Timedelta(seconds=core.TF_SECONDS[tf]); ok=(int(b.trend_dir.iat[i])==d) and pd.notna(b.atr_med120.iat[i]) and (b.atr14.iat[i]>=b.atr_med120.iat[i])
            if ok: raw.append({'tf':tf,'edge':edge,'direction':d,'entry_time':et})
    s=pd.DataFrame(raw).sort_values('entry_time'); s['minute']=pd.to_datetime(s.entry_time,utc=True).dt.floor('min'); c=s.groupby(['minute','direction']).size().rename('agree').reset_index(); s=s.merge(c,on=['minute','direction'],how='left'); s['boost']=np.where(s.agree>=2,2.0,1.0)
    rows=[]
    for horizon in [5,15,30,60]:
        for k in [0.5,1.0,2.0,3.0]:
            out=[]
            for _,r in s.iterrows():
                q=first_passage(ticks,r.entry_time,int(r.direction),horizon,k,float(r.boost))
                if q:
                    en,ex,pnl,hit,ht,spr,target=q; out.append({**r.to_dict(),'entry':en,'exit':ex,'pnl':pnl,'target_hit':hit,'exit_time':ht,'entry_spread':spr,'required_move':target})
            t=pd.DataFrame(out); m=metrics(t); hit_rate=float(t.target_hit.mean()*100.) if not t.empty else 0.
            rows.append({'config':f'FP_{horizon}m_{k:g}xSpread','horizon_min':horizon,'target_spread_mult':k,'target_hit_rate_pct':hit_rate,**m}); t.to_csv(OUT/f'trades_FP_{horizon}m_{k:g}x.csv',index=False)
    res=pd.DataFrame(rows).sort_values(['return_pct','PF'],ascending=False); res.to_csv(OUT/'summary_21d.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps({'formula_policy':'FROZEN_BIGPLAYER_NO_INTERNAL_CHANGES','entry_supervisor':'same TREND+REGIME proxy winner','mtf_boost':'same-minute same-direction x2.0','execution':'raw bid/ask','economic_be_definition':'BUY exit bid >= entry ask + k*entry_spread; SELL exit ask <= entry bid - k*entry_spread','first_passage':'take first target hit; otherwise mark-to-market at horizon','horizons_min':[5,15,30,60],'target_spread_mult':[0.5,1.0,2.0,3.0]},indent=2))
    print(res.to_string(index=False))
if __name__=='__main__':main()
