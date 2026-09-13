#!/usr/bin/env python3
from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amos_reversal_v1_v4_rawtick_bt_v2 as m

TFS = ['M1','M5','M15','H1']
TF_IDX = {tf:i for i,tf in enumerate(TFS)}
HORIZON_BARS = {'BASE':20,'AGGRESSIVE':12,'STANDARD':20,'CONSERVATIVE':32}


def _utc(x):
    t=pd.Timestamp(x)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def _fvg_zone(z,i,side):
    if i < 2: return None
    r=z.iloc[i]
    if side==1 and bool(r.bull_fvg):
        lo=float(z.high.iloc[i-2]); hi=float(r.low)
        return (min(lo,hi),max(lo,hi),'FVG')
    if side==-1 and bool(r.bear_fvg):
        lo=float(r.high); hi=float(z.low.iloc[i-2])
        return (min(lo,hi),max(lo,hi),'FVG')
    return None


def _prior_fvg_zones(z,i,side,lookback=12):
    out=[]
    for k in range(max(2,i-lookback),i):
        q=z.iloc[k]
        if side==1 and bool(q.bear_fvg):
            lo=float(q.high); hi=float(z.low.iloc[k-2]); out.append((k,min(lo,hi),max(lo,hi),'BEAR_FVG'))
        elif side==-1 and bool(q.bull_fvg):
            lo=float(z.high.iloc[k-2]); hi=float(q.low); out.append((k,min(lo,hi),max(lo,hi),'BULL_FVG'))
    return out


def _poi_zone(z,i,side):
    """Return actual FVG / inverted-FVG / BPR overlap bounds; no synthetic ATR pocket."""
    current=_fvg_zone(z,i,side)
    prior=_prior_fvg_zones(z,i,side)
    r=z.iloc[i]
    inverted=[]
    for k,lo,hi,kind in prior:
        # IFVG exists only after price closes through the opposite FVG in reversal direction.
        inv=(float(r.close)>hi) if side==1 else (float(r.close)<lo)
        if inv: inverted.append((k,lo,hi,'IFVG'))
    if current and inverted:
        _,clo,chi,_=current
        # BPR = real overlap between current directional FVG and an eligible inverted opposite imbalance.
        overlaps=[]
        for _,ilo,ihi,_ in inverted:
            lo=max(clo,ilo); hi=min(chi,ihi)
            if lo<=hi: overlaps.append((lo,hi))
        if overlaps:
            lo,hi=overlaps[-1]
            return (lo,hi,'BPR')
    if current: return current
    if inverted:
        _,lo,hi,_=inverted[-1]
        return (lo,hi,'IFVG')
    return None


def build_tf_events(z, tf):
    """Independent per-TF FSM. Events use only completed bars and preserve causal ordering."""
    events=[]; states={1:None,-1:None}
    for i in range(30,len(z)-1):
        r=z.iloc[i]; p=z.iloc[i-1]
        if not np.isfinite(r.atr) or r.atr<=0: continue
        for side in (1,-1):
            s=states[side]
            if s and i-s['sweep_i']>28: states[side]=None
        bs=bool(r.low<r.swing_lo and r.close>r.swing_lo)
        ss=bool(r.high>r.swing_hi and r.close<r.swing_hi)
        if bs:
            states[1]={'sweep_i':i,'extreme':float(r.low),'mss_ref':float(z.high.iloc[max(0,i-3):i].max()),'cisd_i':None,'mss_i':None}
        if ss:
            states[-1]={'sweep_i':i,'extreme':float(r.high),'mss_ref':float(z.low.iloc[max(0,i-3):i].min()),'cisd_i':None,'mss_i':None}
        for side in (1,-1):
            s=states[side]
            if not s: continue
            if s['cisd_i'] is None:
                if i>s['sweep_i'] and i-s['sweep_i']<=8:
                    cisd=(r.close>p.open and r.close>p.close) if side==1 else (r.close<p.open and r.close<p.close)
                    if cisd: s['cisd_i']=i
                continue
            if s['mss_i'] is None:
                if i-s['cisd_i']>12:
                    states[side]=None; continue
                if i<=s['cisd_i']: continue
                mss=(r.close>s['mss_ref']) if side==1 else (r.close<s['mss_ref'])
                if mss: s['mss_i']=i
                else: continue
            # Displacement may occur after MSS, not necessarily on the same bar.
            if i<s['mss_i'] or i-s['mss_i']>4:
                if i-s['mss_i']>4: states[side]=None
                continue
            if float(r.disp)<0.65: continue
            zone=_poi_zone(z,i,side)
            if zone is None: continue
            lo,hi,poi_type=zone
            vel=min(1.5,abs(float(r.close)-float(p.close))/(float(r.atr)+1e-12))
            st=4.2 + 0.8*(poi_type=='FVG') + 1.0*(poi_type=='IFVG') + 1.2*(poi_type=='BPR') + 0.35*vel
            events.append({'tf':tf,'bar_i':i,'time':_utc(r.time),'side':side,'score':float(st),'extreme':float(s['extreme']),'zone_lo':float(lo),'zone_hi':float(hi),'poi_type':poi_type})
            states[side]=None
    return events


def latest_event(events, ts, tf):
    cutoff=_utc(ts)
    chosen=None
    fresh=pd.Timedelta(minutes=m.TF_MIN[tf]*8)
    for e in events:
        if e['time']<=cutoff and cutoff-e['time']<=fresh:
            chosen=e
        elif e['time']>cutoff:
            break
    return chosen


def mtf_fusion(all_events, touch_t, exec_tf, side):
    """True completed-bar cross-TF evidence at touch time; no future bars."""
    support=0.0; conflict=0.0; details=[]; leader_tf=exec_tf; leader_score=-1e9
    ei=TF_IDX[exec_tf]
    for tf in TFS:
        e=latest_event(all_events[tf],touch_t,tf)
        if e is None: continue
        ti=TF_IDX[tf]
        if ti<ei: role='LOWER_PROPAGATE'; w=0.70 + 0.10*(ei-ti-1)
        elif ti==ei: role='SELF'; w=1.00
        else: role='HIGHER_CONFIRM'; w=1.05 + 0.15*(ti-ei-1)
        freshness_min=max(0.0,(_utc(touch_t)-e['time']).total_seconds()/60.0)
        fresh_factor=max(0.25,1.0-freshness_min/(m.TF_MIN[tf]*8.0))
        strength=w*fresh_factor*(0.75+0.08*min(6.0,e['score']))
        if e['side']==side:
            support+=strength
            if strength>leader_score: leader_score=strength; leader_tf=tf
            details.append((tf,role,'SUPPORT',strength))
        else:
            # Higher-TF conflict is intentionally more expensive than lower-TF disagreement.
            penalty=strength*(1.45 if ti>ei else 1.10)
            conflict+=penalty
            details.append((tf,role,'CONFLICT',penalty))
    net=support-conflict
    higher_conflict=any(d[1]=='HIGHER_CONFIRM' and d[2]=='CONFLICT' for d in details)
    return net,support,conflict,higher_conflict,leader_tf,details


def first_touch_tick(t, start_ns, end_ns, side, lo, hi):
    tv=pd.to_datetime(t['time'],utc=True).astype('int64').to_numpy(dtype=np.int64)
    a=int(np.searchsorted(tv,np.int64(start_ns),side='right'))
    b=int(np.searchsorted(tv,np.int64(end_ns),side='right'))
    b=min(b,len(t))
    for k in range(a,b):
        q=t.iloc[k]
        px=float(q.ask if side==1 else q.bid)
        if lo<=px<=hi:
            return k
    return None


def context_score(z, bar_i, side):
    q=z.iloc[bar_i]; ctx=0.0
    if np.isfinite(q.rsi): ctx += .30*max(-1.5,min(1.5,((q.rsi-50)/25)*side))
    if np.isfinite(q.regime): ctx += .25*max(-1,min(1,(q.regime-20)/20))
    if np.isfinite(q.vwap): ctx += .25*max(-2,min(2,((q.close-q.vwap)/(q.atr+1e-12))*side))
    if np.isfinite(q.vol_med) and q.vol_med>0: ctx += .20*(min(2,q.volume/(q.vol_med+1e-12))-1)
    return float(ctx)


def simulate_true_mtf(t, target_b, v, tf, mode, rr):
    bars_by={x:m.bars(t,x) for x in TFS}
    feats={x:m.feat(bars_by[x]) for x in TFS}
    all_events={x:build_tf_events(feats[x],x) for x in TFS}
    target_events=all_events[tf]
    z=feats[tf]
    tv=pd.to_datetime(t['time'],utc=True).astype('int64').to_numpy(dtype=np.int64)
    trades=[]; last_exit_ns=np.int64(-1)
    diag={'events':len(target_events),'poi_touch':0,'mtf_support':0,'mtf_conflict':0,'context_pass':0,'accepted':0,'blocked_overlap':0,'executed':0}
    mtf_gate={'AGGRESSIVE':0.70,'STANDARD':1.10,'CONSERVATIVE':1.55}.get(mode,1.10)
    ctx_gate={'AGGRESSIVE':-0.15,'STANDARD':0.00,'CONSERVATIVE':0.15}.get(mode,0.0)
    for e in target_events:
        disp_close=e['time']; end_touch=disp_close+pd.Timedelta(minutes=m.TF_MIN[tf]*6)
        k=first_touch_tick(t,disp_close.value,end_touch.value,e['side'],e['zone_lo'],e['zone_hi'])
        if k is None: continue
        diag['poi_touch']+=1
        q=t.iloc[k]; touch_t=_utc(q.time); touch_ns=np.int64(touch_t.value)
        if touch_ns<=last_exit_ns:
            diag['blocked_overlap']+=1; continue
        net,sup,conf,hconf,leader,details=mtf_fusion(all_events,touch_t,tf,e['side'])
        if sup>0: diag['mtf_support']+=1
        if conf>0: diag['mtf_conflict']+=1
        ok=True; ctx=0.0
        if v=='V1':
            ok=True
        elif v=='V2':
            min_st={'AGGRESSIVE':4.4,'STANDARD':4.8,'CONSERVATIVE':5.2}.get(mode,4.8)
            ok=e['score']>=min_st
        elif v=='V3':
            ok=(net>=mtf_gate and not hconf)
        else:
            ok=(net>=mtf_gate and not hconf)
            # Context uses latest completed target-TF bar at or before touch, never the unfinished touch bar.
            bi=int(np.searchsorted(pd.to_datetime(z.time,utc=True).astype('int64').to_numpy(),touch_ns,side='right')-1)
            bi=max(0,min(bi,len(z)-1)); ctx=context_score(z,bi,e['side'])
            if ctx>=ctx_gate: diag['context_pass']+=1
            ok=ok and ctx>=ctx_gate
        if not ok: continue
        diag['accepted']+=1
        side=e['side']; entry=float(q.ask if side==1 else q.bid); av=float(z.atr.iloc[e['bar_i']])
        stop=min(e['extreme'],entry-.20*av) if side==1 else max(e['extreme'],entry+.20*av)
        risk=abs(entry-stop)
        if not np.isfinite(risk) or risk<=0: continue
        target=entry+side*rr*risk
        hb=HORIZON_BARS.get(mode,20); horizon=touch_t+pd.Timedelta(minutes=m.TF_MIN[tf]*hb)
        end=int(np.searchsorted(tv,np.int64(horizon.value),side='right')); end=min(end,len(t))
        if end<=k+1: continue
        exit_px=None; exit_t=None; rval=None; result='TIME'
        for j in range(k+1,end):
            x=t.iloc[j]; px=float(x.bid if side==1 else x.ask)
            if (side==1 and px<=stop) or (side==-1 and px>=stop):
                exit_px=px; exit_t=_utc(x.time); rval=side*(px-entry)/risk; result='LOSS'; break
            if (side==1 and px>=target) or (side==-1 and px<=target):
                exit_px=px; exit_t=_utc(x.time); rval=side*(px-entry)/risk; result='WIN'; break
        if exit_px is None:
            x=t.iloc[end-1]; exit_px=float(x.bid if side==1 else x.ask); exit_t=_utc(x.time); rval=side*(exit_px-entry)/risk
        pr=1/(1+math.exp(-max(-40,min(40,0.55*e['score']+0.50*net+0.35*ctx-3.0))))
        trades.append(m.Trade(v,tf,mode,side,str(touch_t),str(exit_t),entry,stop,target,exit_px,float(rval),result,leader,float(e['score']+net),float(ctx),float(pr)))
        last_exit_ns=np.int64(exit_t.value); diag['executed']+=1
    return trades,diag

m.simulate=simulate_true_mtf

if __name__=='__main__':
    m.main()
