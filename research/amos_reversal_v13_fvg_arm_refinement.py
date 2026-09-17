#!/usr/bin/env python3
from __future__ import annotations
import json, os, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import amos_reversal_v1_v4_rawtick_bt_v2 as m
import amos_reversal_v8_ablation_m1_standard as v8
import amos_reversal_v11_ote_ablation as v11
import amos_reversal_v12_ote_arm_ablation as v12

TF='M1'; MODE='STANDARD'; RR=4.0
WINDOWS=[15,30,45,60,90,120]
# No OTE. FVG only. Thresholds are predeclared, not fitted after seeing outcomes.
ARM_STRENGTH={'ANY':None,'S50':0.50,'S70':0.70,'S85':0.85}
FVG_QUALITY={'ANY':None,'Q50':0.50,'Q70':0.70,'Q85':0.85}

def ns(x): return np.fromiter((pd.Timestamp(v).value for v in x),dtype=np.int64,count=len(x))

def arm_events_features(t):
    # Same causal 15s DexG structure as V12, adding only ex-ante observable strength features.
    tv=ns(t.time); bid=t.bid.to_numpy(float,copy=False); ask=t.ask.to_numpy(float,copy=False); mid=.5*(bid+ask)
    sec=(tv//1_000_000_000).astype(np.int64); buckets=sec//15
    bars=[]; events=[]; midbuf=[]; cur_bucket=None; cur=None
    watch=False; anchor=0; reclaimed=False; ret=0.; hex_=0.; strict_=False; dcount=0; reclaim_ns=None
    def br(b): return max(b['h']-b['l'],1e-12)
    def body(b): return abs(b['c']-b['o'])/br(b)
    def setup():
        nonlocal watch,anchor,reclaimed,ret,hex_,strict_,dcount,reclaim_ns
        watch=False;anchor=0;reclaimed=False;ret=0.;hex_=0.;strict_=False;dcount=1;reclaim_ns=None
        if len(bars)<3:return
        ai=len(bars)-3; A,S,H=bars[ai],bars[ai+1],bars[ai+2]
        if ai<v12.ACCUM_LB:return
        rs=[br(x) for x in bars[max(0,ai-v12.ACCUM_LB):ai]];med=float(np.median(rs)) if rs else br(A)
        if not (br(A)<=v12.ACCUM_RANGE_FACTOR*med or body(A)<=v12.ACCUM_BODY_RATIO):return
        up=(S['h']>A['h'] and S['c']<A['h']);dn=(S['l']<A['l'] and S['c']>A['l']);a=1 if up and not dn else(-1 if dn and not up else 0)
        if not a:return
        ar=br(A)
        if a>0:
            strict=H['l']<A['l'];travel=H['l']<min(S['o'],S['c'])-max(v12.TRAVEL_MIN,v12.TRAVEL_RF*ar);internal=H['l']<min(A['o'],A['c']);wick=(min(H['o'],H['c'])-H['l'])>=v12.HARVEST_WICK*br(H);ok=strict or(travel and(internal or wick));r=max(A['h'],S['o']);hx=H['l']
        else:
            strict=H['h']>A['h'];travel=H['h']>max(S['o'],S['c'])+max(v12.TRAVEL_MIN,v12.TRAVEL_RF*ar);internal=H['h']>max(A['o'],A['c']);wick=(H['h']-max(H['o'],H['c']))>=v12.HARVEST_WICK*br(H);ok=strict or(travel and(internal or wick));r=min(A['l'],S['o']);hx=H['h']
        if ok:watch=True;anchor=a;ret=r;hex_=hx;strict_=bool(strict)
    for i in range(len(tv)):
        b=int(buckets[i]);p=float(bid[i]);mp=float(mid[i])
        if cur_bucket is None:cur_bucket=b;cur={'o':p,'h':p,'l':p,'c':p};setup()
        elif b!=cur_bucket:
            bars.append(cur)
            if len(bars)>256:bars=bars[-256:]
            cur_bucket=b;cur={'o':p,'h':p,'l':p,'c':p};setup()
        else:cur['h']=max(cur['h'],p);cur['l']=min(cur['l'],p);cur['c']=p;dcount+=1
        midbuf.append(mp)
        if len(midbuf)>64:midbuf=midbuf[-64:]
        if not watch or dcount<5:continue
        if not reclaimed:
            reclaimed=mp>ret if anchor>0 else mp<ret
            if reclaimed:reclaim_ns=int(tv[i])
            else:continue
        lb=max(v12.MICRO_LB,8)
        if len(midbuf)<lb+1 or len(midbuf)<9:continue
        e=len(midbuf)-1;d1=midbuf[e]-midbuf[e-1];d2=midbuf[e-1]-midbuf[e-2];d3=midbuf[e-2]-midbuf[e-3]
        vel=((d1+d2+d3)/3.)*anchor;acc=(d1-d2)*anchor;travel3=abs(midbuf[e]-midbuf[e-3])
        if vel>0 and acc>=0 and travel3>=v12.DISP_MIN:
            reclaim_ms=0. if reclaim_ns is None else (int(tv[i])-reclaim_ns)/1e6
            events.append({'time_ns':int(tv[i]),'anchor':int(anchor),'inverse_side':int(-anchor),'vel':float(vel),'acc':float(acc),'travel3':float(travel3),'strict':int(strict_),'reclaim_ms':float(reclaim_ms),'harvest_distance':float(abs(ret-hex_))})
            watch=False
    return pd.DataFrame(events)

def percentile_score(s):
    if len(s)==0:return np.array([],float)
    return pd.Series(s).rank(method='average',pct=True).to_numpy(float)

def fvg_features(z,c):
    lo,hi,typ=c['zone'];i=int(c['disp_i']);atr=max(float(z.atr.iloc[i]),1e-12)
    width=(float(hi)-float(lo))/atr
    # candidate score and normalized FVG width are known before touch.
    return width,float(c['score'])

def latest_inverse(ev,et_ns,side,window_s):
    if len(ev)==0:return None
    arr=ev.time_ns.to_numpy(np.int64);j=int(np.searchsorted(arr,et_ns,side='right')-1)
    if j<0:return None
    row=ev.iloc[j]
    if et_ns-int(row.time_ns)>window_s*1_000_000_000:return None
    if int(row.inverse_side)!=side:return None
    return row

def metrics(tr):
    x=m.metrics(tr);rs=np.array([q.r for q in tr],float);pos=rs[rs>0];neg=rs[rs<0]
    x.update(avg_win_R=float(pos.mean()) if len(pos) else 0.,avg_loss_R=float(neg.mean()) if len(neg) else 0.,median_R=float(np.median(rs)) if len(rs) else 0.)
    return x

def main():
    src=m.resolve_source('XAUUSD','dukascopy_raw');t=m.load_ticks(src);z=m.feat(m.bars(t,TF));cands=[c for c in v8.detect_candidates(z) if c.get('zone') is not None and str(c['zone'][2])=='FVG']
    tv=ns(t.time);bid=t.bid.to_numpy(float,copy=False);ask=t.ask.to_numpy(float,copy=False)
    ev=arm_events_features(t)
    if len(ev):
        raw=(np.maximum(ev.vel.to_numpy(float),0.)/max(np.median(np.maximum(ev.vel,0.)),1e-12)+np.maximum(ev.acc.to_numpy(float),0.)/max(np.median(np.maximum(ev.acc,0.)),1e-12)+ev.travel3.to_numpy(float)/v12.DISP_MIN+0.5*ev.strict.to_numpy(float)+np.minimum(ev.harvest_distance.to_numpy(float)/max(np.median(ev.harvest_distance),1e-12),3.0))
        ev['arm_strength']=percentile_score(raw)
    else:ev['arm_strength']=[]
    sig=[]
    for sid,c in enumerate(cands):
        k=v11.first_touch(tv,bid,ask,c)
        if k is None or k>=len(t):continue
        side=int(c['side']);entry=float(ask[k] if side==1 else bid[k]);w,cs=fvg_features(z,c)
        sig.append({'signal_id':sid,'c':c,'k':int(k),'et_ns':int(tv[k]),'side':side,'entry':entry,'fvg_width_atr':w,'candidate_score':cs})
    if sig:
        qraw=np.array([np.log1p(max(s['fvg_width_atr'],0.))+0.25*s['candidate_score'] for s in sig]);qscore=percentile_score(qraw)
        for s,q in zip(sig,qscore):s['fvg_quality']=float(q)
    out=Path(os.environ.get('AMOS_V13_OUT','results/amos-v13-fvg-arm-refinement'));out.mkdir(parents=True,exist_ok=True);rows=[]
    ev.to_csv(out/'arm_events_features.csv',index=False)
    pd.DataFrame([{k:v for k,v in s.items() if k!='c'} for s in sig]).to_csv(out/'signal_universe.csv',index=False)
    for win in WINDOWS:
      for an,ath in ARM_STRENGTH.items():
       for qn,qth in FVG_QUALITY.items():
        tr=[];diag={'signals':len(sig),'arm_match':0,'arm_strength_pass':0,'fvg_quality_pass':0,'executed':0}
        # Fixed common signal universe; no variant-dependent overlap suppression.
        for s in sig:
            ar=latest_inverse(ev,s['et_ns'],s['side'],win)
            if ar is None:continue
            diag['arm_match']+=1
            if ath is not None and float(ar.arm_strength)<ath:continue
            diag['arm_strength_pass']+=1
            if qth is not None and s['fvg_quality']<qth:continue
            diag['fvg_quality_pass']+=1
            c=s['c'];k=s['k'];side=s['side'];entry=s['entry'];et=pd.Timestamp(t.time.iloc[k]);av=float(z.atr.iloc[c['disp_i']]);sweep=float(c['extreme']);risk=max(abs(entry-sweep),.50*av);stop=entry-side*risk;target=entry+side*RR*risk
            end=min(int(np.searchsorted(tv,(et+pd.Timedelta(minutes=120)).value,side='right')),len(t));result='TIME';xp=xt=rval=None
            for j in range(k+1,end):
                px=float(bid[j] if side==1 else ask[j])
                if (side==1 and px<=stop) or(side==-1 and px>=stop):result='LOSS';xp=px;xt=pd.Timestamp(t.time.iloc[j]);rval=side*(px-entry)/risk;break
                if (side==1 and px>=target) or(side==-1 and px<=target):result='WIN';xp=px;xt=pd.Timestamp(t.time.iloc[j]);rval=side*(px-entry)/risk;break
            if xp is None:
                j=end-1;xp=float(bid[j] if side==1 else ask[j]);xt=pd.Timestamp(t.time.iloc[j]);rval=side*(xp-entry)/risk
            note=f"SID={s['signal_id']};WIN={win};ARMQ={float(ar.arm_strength):.4f};FVGQ={s['fvg_quality']:.4f};WATR={s['fvg_width_atr']:.4f}"
            tr.append(m.Trade(f'V13_W{win}_{an}_{qn}',TF,MODE,side,str(et),str(xt),entry,stop,target,xp,float(rval),result,note,float(c['score']),0.,0.));diag['executed']+=1
        met=metrics(tr);met.update({'window_s':win,'arm_threshold':an,'fvg_threshold':qn,'RR_target':RR});met.update({f'diag_{k}':v for k,v in diag.items()});rows.append(met)
        pd.DataFrame([asdict(x) for x in tr]).to_csv(out/f'W{win}_{an}_{qn}_trades.csv',index=False)
    df=pd.DataFrame(rows);df.to_csv(out/'summary.csv',index=False);(out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    manifest={'purpose':'V13 FVG x ARM inverse refinement','ote':'NOT USED','symbol':'XAUUSD','tf':TF,'rr':RR,'windows_s':WINDOWS,'arm_strength_percentile_thresholds':ARM_STRENGTH,'fvg_quality_percentile_thresholds':FVG_QUALITY,'arm_strength_inputs':['velocity','acceleration','3-tick travel','strict harvest','harvest distance'],'fvg_quality_inputs':['FVG width/ATR','candidate score'],'execution':'raw Dukascopy bid/ask; same structural stop min 0.50ATR; RR4; 120m','comparison':'fixed common FVG first-touch signal universe; no variant-dependent overlap suppression','caveat':'Thresholds are screening percentiles computed on the full sample feature distribution and are not OOS-calibrated. Any apparent winner requires chronological OOS/walk-forward confirmation.'};(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(df.sort_values(['PF','EV_R_per_trade'],ascending=False).head(30).to_json(orient='records',indent=2))
if __name__=='__main__':main()
