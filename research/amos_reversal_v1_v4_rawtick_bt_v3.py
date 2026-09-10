#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import pandas as pd
import research.amos_reversal_v1_v4_rawtick_bt_v2 as m


def simulate_fixed(t,b,v,tf,mode,rr):
    z=m.feat(b)
    signals,diag=m.build_signals(z,v,mode)
    trades=[]
    # Force every tick timestamp to Python Timestamp.value (UTC epoch ns).
    tv=np.fromiter((pd.Timestamp(x).value for x in t['time']),dtype=np.int64,count=len(t))
    last_exit_ns=np.int64(-1)
    diag.update({'blocked_overlap':0,'no_entry_tick':0,'invalid_risk':0,'no_horizon_ticks':0,'executed':0})
    for i,side,st,ctx,pr,leader,sweep_extreme in signals:
        sig_t=pd.Timestamp(z.time.iloc[i])
        if sig_t.tzinfo is None: sig_t=sig_t.tz_localize('UTC')
        else: sig_t=sig_t.tz_convert('UTC')
        sig_ns=np.int64(sig_t.value)
        if sig_ns<=last_exit_ns:
            diag['blocked_overlap']+=1; continue
        j=int(np.searchsorted(tv,sig_ns,side='right'))
        if j>=len(t):
            diag['no_entry_tick']+=1; continue
        q=t.iloc[j]
        entry=float(q.ask if side==1 else q.bid)
        av=float(z.atr.iloc[i])
        stop=min(sweep_extreme,entry-.35*av) if side==1 else max(sweep_extreme,entry+.35*av)
        risk=abs(entry-stop)
        if risk<=0 or not np.isfinite(risk):
            diag['invalid_risk']+=1; continue
        target=entry+side*rr*risk
        horizon_ns=np.int64((sig_t+pd.Timedelta(minutes=m.TF_MIN[tf]*8)).value)
        end=int(np.searchsorted(tv,horizon_ns,side='right'))
        end=min(end,len(t))
        if end<=j+1:
            diag['no_horizon_ticks']+=1; continue
        exit_px=None; exit_t=None; rval=None; result='TIME'
        for k in range(j+1,end):
            x=t.iloc[k]
            px=float(x.bid if side==1 else x.ask)
            if (side==1 and px<=stop) or (side==-1 and px>=stop):
                exit_px=px; exit_t=pd.Timestamp(x.time); rval=side*(px-entry)/risk; result='LOSS'; break
            if (side==1 and px>=target) or (side==-1 and px<=target):
                exit_px=px; exit_t=pd.Timestamp(x.time); rval=side*(px-entry)/risk; result='WIN'; break
        if exit_px is None:
            x=t.iloc[end-1]
            exit_px=float(x.bid if side==1 else x.ask)
            exit_t=pd.Timestamp(x.time)
            rval=side*(exit_px-entry)/risk
        if exit_t.tzinfo is None: exit_t=exit_t.tz_localize('UTC')
        else: exit_t=exit_t.tz_convert('UTC')
        trades.append(m.Trade(v,tf,mode,side,str(sig_t),str(exit_t),entry,stop,target,exit_px,float(rval),result,leader,float(st),float(ctx),float(pr)))
        last_exit_ns=np.int64(exit_t.value)
        diag['executed']+=1
    return trades,diag

m.simulate=simulate_fixed

if __name__=='__main__':
    m.main()
