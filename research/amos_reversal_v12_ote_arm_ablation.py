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

TF='M1'; MODE='STANDARD'; RR=4.0
OTE_MODES=v11.OTE_MODES
ARM_MODES=['NO_ARM','ARM_INVERSE_CONFIRM','ARM_ANCHOR_CONFIRM']
POI_GROUPS=['ALL','FVG','IFVG','BPR']
ARM_LOOKBACK_NS=60_000_000_000

# DexG_ARM_Inverse_Probe_v1_01 defaults
ACCUM_LB=10; ACCUM_RANGE_FACTOR=.90; ACCUM_BODY_RATIO=.40
TRAVEL_MIN=.05; TRAVEL_RF=.15; HARVEST_WICK=.25; MICRO_LB=8; DISP_MIN=.03

def ns(x): return np.fromiter((pd.Timestamp(v).value for v in x),dtype=np.int64,count=len(x))
def rng(b): return max(float(b[2]-b[3]),1e-12) # o,h,l,c tuple? corrected below

def arm_events(t):
    tv=ns(t.time); bid=t.bid.to_numpy(float,copy=False); ask=t.ask.to_numpy(float,copy=False); mid=.5*(bid+ask)
    sec=(tv//1_000_000_000).astype(np.int64); buckets=sec//15
    bars=[]; events=[]; midbuf=[]
    cur_bucket=None; cur=None; watch=False; anchor=0; reclaimed=False; ret=0.; dcount=0
    def br(b): return max(b['h']-b['l'],1e-12)
    def body(b): return abs(b['c']-b['o'])/br(b)
    def setup(new_bucket):
        nonlocal watch,anchor,reclaimed,ret,dcount
        watch=False; anchor=0; reclaimed=False; ret=0.; dcount=1
        if len(bars)<3: return
        ai=len(bars)-3; A,S,H=bars[ai],bars[ai+1],bars[ai+2]
        if ai<ACCUM_LB: return
        rs=[br(x) for x in bars[max(0,ai-ACCUM_LB):ai]]
        med=float(np.median(rs)) if rs else br(A)
        if not (br(A)<=ACCUM_RANGE_FACTOR*med or body(A)<=ACCUM_BODY_RATIO): return
        up=(S['h']>A['h'] and S['c']<A['h']); dn=(S['l']<A['l'] and S['c']>A['l'])
        a=1 if up and not dn else (-1 if dn and not up else 0)
        if not a: return
        ar=br(A)
        if a>0:
            strict=H['l']<A['l']; travel=H['l']<min(S['o'],S['c'])-max(TRAVEL_MIN,TRAVEL_RF*ar)
            internal=H['l']<min(A['o'],A['c']); wick=(min(H['o'],H['c'])-H['l'])>=HARVEST_WICK*br(H)
            ok=strict or (travel and (internal or wick)); r=max(A['h'],S['o'])
        else:
            strict=H['h']>A['h']; travel=H['h']>max(S['o'],S['c'])+max(TRAVEL_MIN,TRAVEL_RF*ar)
            internal=H['h']>max(A['o'],A['c']); wick=(H['h']-max(H['o'],H['c']))>=HARVEST_WICK*br(H)
            ok=strict or (travel and (internal or wick)); r=min(A['l'],S['o'])
        if ok: watch=True; anchor=a; ret=r
    for i in range(len(tv)):
        b=int(buckets[i]); p=float(bid[i]); mp=float(mid[i])
        if cur_bucket is None:
            cur_bucket=b; cur={'bucket':b,'o':p,'h':p,'l':p,'c':p}; setup(b)
        elif b!=cur_bucket:
            bars.append(cur)
            if len(bars)>256: bars=bars[-256:]
            cur_bucket=b; cur={'bucket':b,'o':p,'h':p,'l':p,'c':p}; setup(b)
        else:
            cur['h']=max(cur['h'],p); cur['l']=min(cur['l'],p); cur['c']=p; dcount+=1
        midbuf.append(mp)
        if len(midbuf)>64: midbuf=midbuf[-64:]
        if not watch or dcount<5: continue
        if not reclaimed:
            reclaimed = mp>ret if anchor>0 else mp<ret
            if not reclaimed: continue
        lb=max(MICRO_LB,8)
        if len(midbuf)<lb+1 or len(midbuf)<9: continue
        e=len(midbuf)-1; d1=midbuf[e]-midbuf[e-1]; d2=midbuf[e-1]-midbuf[e-2]; d3=midbuf[e-2]-midbuf[e-3]
        vel=((d1+d2+d3)/3.)*anchor; acc=(d1-d2)*anchor
        disp=vel>0 and acc>=0 and abs(midbuf[e]-midbuf[e-3])>=DISP_MIN
        if disp:
            events.append((int(tv[i]),int(anchor),int(-anchor)))
            watch=False
    return np.array(events,dtype=np.int64) if events else np.empty((0,3),dtype=np.int64)

def latest_arm(ev, et_ns):
    if len(ev)==0:return None
    j=int(np.searchsorted(ev[:,0],et_ns,side='right')-1)
    if j<0:return None
    if et_ns-int(ev[j,0])>ARM_LOOKBACK_NS:return None
    return ev[j]

def arm_pass(mode, evrow, side):
    if mode=='NO_ARM': return True
    if evrow is None:return False
    if mode=='ARM_INVERSE_CONFIRM': return int(evrow[2])==side
    if mode=='ARM_ANCHOR_CONFIRM': return int(evrow[1])==side
    return False

def simulate(t,z,cands,ev,ote_mode,band,arm_mode,poi_group):
    tv=ns(t.time); bid=t.bid.to_numpy(float,copy=False); ask=t.ask.to_numpy(float,copy=False); out=[]; last=-1
    diag={'candidates':0,'touch':0,'ote_pass':0,'arm_pass':0,'executed':0,'overlap':0}
    for c in cands:
        if c['disp_i'] is None or c['zone'] is None: continue
        poi=str(c['zone'][2]); diag['candidates']+=1
        if poi_group!='ALL' and poi!=poi_group: continue
        k=v11.first_touch(tv,bid,ask,c)
        if k is None or k>=len(t): continue
        diag['touch']+=1; et=pd.Timestamp(t.time.iloc[k]); side=int(c['side']); entry=float(ask[k] if side==1 else bid[k])
        ote=v11.ote_retracement(z,c,entry)
        if not v11.pass_ote(ote,band): continue
        diag['ote_pass']+=1
        ar=latest_arm(ev,et.value)
        if not arm_pass(arm_mode,ar,side): continue
        diag['arm_pass']+=1
        if et.value<=last: diag['overlap']+=1; continue
        av=float(z.atr.iloc[c['disp_i']]); sweep=float(c['extreme']); risk=max(abs(entry-sweep),.50*av); stop=entry-side*risk; target=entry+side*RR*risk
        end=min(int(np.searchsorted(tv,(et+pd.Timedelta(minutes=120)).value,side='right')),len(t))
        if end<=k+1: continue
        result='TIME';xp=xt=rval=None
        for j in range(k+1,end):
            px=float(bid[j] if side==1 else ask[j])
            if (side==1 and px<=stop) or (side==-1 and px>=stop): result='LOSS';xp=px;xt=pd.Timestamp(t.time.iloc[j]);rval=side*(px-entry)/risk;break
            if (side==1 and px>=target) or (side==-1 and px<=target): result='WIN';xp=px;xt=pd.Timestamp(t.time.iloc[j]);rval=side*(px-entry)/risk;break
        if xp is None:
            j=end-1;xp=float(bid[j] if side==1 else ask[j]);xt=pd.Timestamp(t.time.iloc[j]);rval=side*(xp-entry)/risk
        note=f'POI_{poi};OTE={ote:.6f};ARM={arm_mode}'
        out.append(m.Trade(f'V12_{ote_mode}_{arm_mode}_{poi_group}',TF,MODE,side,str(et),str(xt),entry,stop,target,xp,float(rval),result,note,float(c['score']),float(ote) if np.isfinite(ote) else 0.,0.));last=xt.value;diag['executed']+=1
    return out,diag

def metrics(tr):
    x=m.metrics(tr);rs=np.array([q.r for q in tr],float);pos=rs[rs>0];neg=rs[rs<0]
    x.update(avg_win_R=float(pos.mean()) if len(pos) else 0.,avg_loss_R=float(neg.mean()) if len(neg) else 0.,median_R=float(np.median(rs)) if len(rs) else 0.)
    return x

def main():
    src=m.resolve_source('XAUUSD','dukascopy_raw');t=m.load_ticks(src);z=m.feat(m.bars(t,TF));cands=v8.detect_candidates(z);ev=arm_events(t)
    out=Path(os.environ.get('AMOS_V12_OUT','results/amos-v12-ote-arm-ablation'));out.mkdir(parents=True,exist_ok=True);rows=[]
    pd.DataFrame(ev,columns=['time_ns','anchor','inverse_side']).to_csv(out/'arm_events.csv',index=False)
    for om,band in OTE_MODES.items():
        for am in ARM_MODES:
            for pg in POI_GROUPS:
                tr,diag=simulate(t,z,cands,ev,om,band,am,pg);met=metrics(tr);met.update({'OTE_mode':om,'ARM_mode':am,'POI_group':pg,'RR_target':RR,'ARM_events_total':int(len(ev))});met.update({f'diag_{k}':v for k,v in diag.items()});rows.append(met)
                pd.DataFrame([asdict(q) for q in tr]).to_csv(out/f'{om}_{am}_{pg}_trades.csv',index=False)
    df=pd.DataFrame(rows);df.to_csv(out/'summary.csv',index=False);(out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    manifest={'purpose':'V12 OTE x DexG ARM polarity ablation','symbol':'XAUUSD','tf':TF,'rr':RR,'arm_source':'causal Python port of DexG_ARM_Inverse_Probe_v1_01 defaults','arm_match':'latest ARM fire at or before S3 POI entry, max age 60s; no future ARM','arm_modes':ARM_MODES,'ote_modes':OTE_MODES,'poi_groups':POI_GROUPS,'execution':'raw Dukascopy bid/ask','stop':'sweep extreme with min 0.50 ATR','horizon_minutes':120,'caveat':'Python parity port is logic-matched but not proven tick-for-tick identical to MT5; each cell has its own no-overlap clock.'};(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(df.to_json(orient='records',indent=2))
if __name__=='__main__':main()
