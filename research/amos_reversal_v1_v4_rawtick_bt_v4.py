#!/usr/bin/env python3
from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amos_reversal_v1_v4_rawtick_bt_v2 as m
import amos_reversal_v1_v4_rawtick_bt_v3 as execfix


def sigm(x):
    return 1/(1+math.exp(-max(-40,min(40,x))))


def build_signals_repaired(z,v,mode):
    diag={'sweep':0,'cisd':0,'mss':0,'disp':0,'poi':0,'mtf_confirm':0,'v3_candidate':0,'accepted':0}
    out=[]; states={1:None,-1:None}
    gates={'AGGRESSIVE':(0.50,0.00),'STANDARD':(0.56,0.15),'CONSERVATIVE':(0.62,0.30)}
    for i in range(60,len(z)-1):
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
            if i-s['cisd_i']>12:
                states[side]=None; continue
            mss=(r.close>s['mss_ref']) if side==1 else (r.close<s['mss_ref'])
            if not mss: continue
            diag['mss']+=1
            disp=bool(r.disp>=0.65)
            if disp: diag['disp']+=1
            fvg=bool(r.bull_fvg if side==1 else r.bear_fvg)
            opp=bool(z.bear_fvg.iloc[max(0,i-8):i].any()) if side==1 else bool(z.bull_fvg.iloc[max(0,i-8):i].any())
            ifvg=opp and disp; bpr=fvg and opp; poi=(fvg or ifvg or bpr)
            if poi: diag['poi']+=1
            # Multi-horizon structure confirmation: approximates M1->M5->M15->H1 hierarchy
            h5=(r.close-z.close.iloc[i-5])*side
            h15=(r.close-z.close.iloc[i-15])*side
            h60=(r.close-z.close.iloc[i-60])*side
            votes=int(h5>0)+int(h15>0)+int(h60>0)
            mtf_ok=votes>=2
            if mtf_ok: diag['mtf_confirm']+=1
            vel=min(1.5,abs(r.close-p.close)/(r.atr+1e-12))
            st=3.2+1.0*disp+0.8*fvg+0.6*ifvg+0.6*bpr+0.35*votes
            base_pr=sigm(-3.0+0.72*st+0.25*vel)
            ctx=0.0; leader=''
            if np.isfinite(r.rsi): ctx += 0.25*max(-1.5,min(1.5,((r.rsi-50)/25)*side))
            if np.isfinite(r.regime): ctx += 0.20*max(-1,min(1,(r.regime-20)/20))
            if np.isfinite(r.vwap): ctx += 0.25*max(-2,min(2,((r.close-r.vwap)/(r.atr+1e-12))*side))
            if np.isfinite(r.vol_med) and r.vol_med>0: ctx += 0.15*(min(2,r.volume/(r.vol_med+1e-12))-1)
            ctx += 0.15*((votes-1.5)/1.5)
            ok=False; pr=base_pr
            if v=='V1':
                ok=disp and poi
            elif v=='V2':
                # sequential structure + displacement + POI; mode changes strictness
                min_st={'AGGRESSIVE':4.1,'STANDARD':4.6,'CONSERVATIVE':5.1}.get(mode,4.6)
                ok=disp and poi and st>=min_st
            elif v=='V3':
                pr_gate,_=gates[mode]
                ok=disp and poi and mtf_ok and base_pr>=pr_gate
                leader='MTF'
                if ok: diag['v3_candidate']+=1
            else:
                pr_gate,ctx_gate=gates[mode]
                pr=sigm(math.log(max(1e-6,base_pr)/max(1e-6,1-base_pr))+0.55*ctx)
                ok=disp and poi and mtf_ok and base_pr>=pr_gate and ctx>=ctx_gate
                leader='ADAPTIVE_MTF'
                if disp and poi and mtf_ok and base_pr>=pr_gate: diag['v3_candidate']+=1
            if ok:
                diag['accepted']+=1; out.append((i,side,st,ctx,pr,leader,s['extreme']))
            states[side]=None
    return out,diag

m.build_signals=build_signals_repaired
m.simulate=execfix.simulate_fixed

if __name__=='__main__':
    m.main()
