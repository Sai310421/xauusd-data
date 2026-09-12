#!/usr/bin/env python3
from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amos_reversal_v1_v4_rawtick_bt_v2 as m


def build_signals_retrace(z, v, mode):
    diag={'sweep':0,'cisd':0,'mss':0,'disp':0,'poi':0,'mtf_confirm':0,'accepted':0}
    out=[]
    states={1:None,-1:None}
    for i in range(30,len(z)-3):
        r=z.iloc[i]; p=z.iloc[i-1]
        if not np.isfinite(r.atr) or r.atr<=0: continue
        for side in (1,-1):
            s=states[side]
            if s and i-s['sweep_i']>24: states[side]=None
        bs=bool(r.low<r.swing_lo and r.close>r.swing_lo)
        ss=bool(r.high>r.swing_hi and r.close<r.swing_hi)
        if bs:
            diag['sweep']+=1; states[1]={'sweep_i':i,'extreme':float(r.low),'mss_ref':float(z.high.iloc[max(0,i-3):i].max()),'cisd_i':None}
        if ss:
            diag['sweep']+=1; states[-1]={'sweep_i':i,'extreme':float(r.high),'mss_ref':float(z.low.iloc[max(0,i-3):i].min()),'cisd_i':None}
        for side in (1,-1):
            s=states[side]
            if not s: continue
            if s['cisd_i'] is None and i>s['sweep_i'] and i-s['sweep_i']<=8:
                cisd=(r.close>p.open and r.close>p.close) if side==1 else (r.close<p.open and r.close<p.close)
                if cisd:
                    s['cisd_i']=i; diag['cisd']+=1
            if s['cisd_i'] is None or i<=s['cisd_i']: continue
            mss=(r.close>s['mss_ref']) if side==1 else (r.close<s['mss_ref'])
            if not mss or i-s['cisd_i']>12: continue
            diag['mss']+=1
            # Require displacement after MSS.
            if r.disp < 0.65:
                states[side]=None; continue
            diag['disp']+=1

            # Build POI zone from FVG / IFVG / BPR around displacement leg.
            bull_fvg=bool(r.bull_fvg); bear_fvg=bool(r.bear_fvg)
            prior_bear=bool(z.bear_fvg.iloc[max(0,i-8):i].any())
            prior_bull=bool(z.bull_fvg.iloc[max(0,i-8):i].any())
            fvg = bull_fvg if side==1 else bear_fvg
            ifvg = (prior_bear and side==1) or (prior_bull and side==-1)
            bpr = fvg and ((prior_bear and side==1) or (prior_bull and side==-1))
            if not (fvg or ifvg or bpr):
                states[side]=None; continue
            diag['poi']+=1

            # Zone anchored to 2-bar FVG where possible; otherwise use 0.35 ATR pocket behind close.
            if side==1 and bull_fvg:
                zl=float(z.high.iloc[i-2]); zh=float(r.low)
            elif side==-1 and bear_fvg:
                zl=float(r.high); zh=float(z.low.iloc[i-2])
            else:
                width=max(0.15*float(r.atr),1e-9)
                if side==1:
                    zh=float(r.close-0.15*r.atr); zl=float(zh-width)
                else:
                    zl=float(r.close+0.15*r.atr); zh=float(zl+width)
            if zl>zh: zl,zh=zh,zl

            # Wait up to 6 bars for actual retrace into POI; do not enter on MSS/displacement bar.
            retrace_i=None
            for j in range(i+1,min(i+7,len(z)-1)):
                q=z.iloc[j]
                touched = (q.low<=zh and q.high>=zl)
                if touched:
                    retrace_i=j; break
            if retrace_i is None:
                states[side]=None; continue

            q=z.iloc[retrace_i]
            st=3.2 + 1.0 + 0.7*float(fvg) + 0.5*float(ifvg) + 0.5*float(bpr)
            ctx=0.0; leader=''
            # V2 strength gate after true retrace.
            if v=='V1':
                ok=True
            elif v=='V2':
                ok=st >= {'AGGRESSIVE':4.0,'STANDARD':4.4,'CONSERVATIVE':4.8}.get(mode,4.4)
            else:
                # Lightweight MTF proxy: retrace candle must reject POI in reversal direction.
                reject=(q.close>q.open and q.close>=zl+(zh-zl)*0.5) if side==1 else (q.close<q.open and q.close<=zl+(zh-zl)*0.5)
                ok=reject
                if ok: diag['mtf_confirm']+=1
                leader='MTF'
                if v=='V4' and ok:
                    mom=max(-1.5,min(1.5,((q.rsi-50)/25)*side)) if np.isfinite(q.rsi) else 0.0
                    reg=max(-1,min(1,(q.regime-20)/20)) if np.isfinite(q.regime) else 0.0
                    loc=max(-2,min(2,((q.close-q.vwap)/(q.atr+1e-12))*side)) if np.isfinite(q.vwap) else 0.0
                    part=min(2,q.volume/(q.vol_med+1e-12))-1 if np.isfinite(q.vol_med) and q.vol_med>0 else 0.0
                    ctx=.30*mom+.25*reg+.25*loc+.20*part
                    gate={'AGGRESSIVE':-0.15,'STANDARD':0.00,'CONSERVATIVE':0.15}[mode]
                    ok = ctx >= gate
                    leader='ADAPTIVE'
            if ok:
                diag['accepted']+=1
                pr=1/(1+math.exp(-(.8+0.45*st+0.35*ctx)))
                out.append((retrace_i,side,st,ctx,pr,leader,s['extreme']))
            states[side]=None
    return out,diag


def simulate_retrace(t,b,v,tf,mode,rr):
    z=m.feat(b); signals,diag=build_signals_retrace(z,v,mode); trades=[]
    tv=np.fromiter((pd.Timestamp(x).value for x in t['time']),dtype=np.int64,count=len(t))
    last_exit_ns=np.int64(-1)
    diag.update({'blocked_overlap':0,'no_entry_tick':0,'invalid_risk':0,'no_horizon_ticks':0,'executed':0})
    for i,side,st,ctx,pr,leader,sweep_extreme in signals:
        sig_t=pd.Timestamp(z.time.iloc[i]); sig_t=sig_t.tz_localize('UTC') if sig_t.tzinfo is None else sig_t.tz_convert('UTC')
        sig_ns=np.int64(sig_t.value)
        if sig_ns<=last_exit_ns: diag['blocked_overlap']+=1; continue
        j=int(np.searchsorted(tv,sig_ns,side='right'))
        if j>=len(t): diag['no_entry_tick']+=1; continue
        q=t.iloc[j]; entry=float(q.ask if side==1 else q.bid)
        av=float(z.atr.iloc[i]); stop=min(sweep_extreme,entry-.20*av) if side==1 else max(sweep_extreme,entry+.20*av)
        risk=abs(entry-stop)
        if risk<=0 or not np.isfinite(risk): diag['invalid_risk']+=1; continue
        target=entry+side*rr*risk
        # Give retrace entries more time: 16 bars rather than 8.
        horizon_ns=np.int64((sig_t+pd.Timedelta(minutes=m.TF_MIN[tf]*16)).value)
        end=min(int(np.searchsorted(tv,horizon_ns,side='right')),len(t))
        if end<=j+1: diag['no_horizon_ticks']+=1; continue
        exit_px=None; exit_t=None; rval=None; result='TIME'
        for k in range(j+1,end):
            x=t.iloc[k]; px=float(x.bid if side==1 else x.ask)
            if (side==1 and px<=stop) or (side==-1 and px>=stop):
                exit_px=px; exit_t=pd.Timestamp(x.time); rval=side*(px-entry)/risk; result='LOSS'; break
            if (side==1 and px>=target) or (side==-1 and px<=target):
                exit_px=px; exit_t=pd.Timestamp(x.time); rval=side*(px-entry)/risk; result='WIN'; break
        if exit_px is None:
            x=t.iloc[end-1]; exit_px=float(x.bid if side==1 else x.ask); exit_t=pd.Timestamp(x.time); rval=side*(exit_px-entry)/risk
        exit_t=exit_t.tz_localize('UTC') if exit_t.tzinfo is None else exit_t.tz_convert('UTC')
        trades.append(m.Trade(v,tf,mode,side,str(sig_t),str(exit_t),entry,stop,target,exit_px,float(rval),result,leader,float(st),float(ctx),float(pr)))
        last_exit_ns=np.int64(exit_t.value); diag['executed']+=1
    return trades,diag

m.build_signals=build_signals_retrace
m.simulate=simulate_retrace
if __name__=='__main__':
    m.main()
