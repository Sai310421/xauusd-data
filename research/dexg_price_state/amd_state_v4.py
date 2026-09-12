from __future__ import annotations
import argparse, json
from pathlib import Path
from statistics import median
from dataclasses import dataclass
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog

EPS=1e-12

@dataclass
class Bar:
    start_ns:int; end_ns:int; o:float; h:float; l:float; c:float; first_idx:int; last_idx:int

def load_ticks(catalog,max_ticks=0):
    cat=ParquetDataCatalog(catalog)
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    if max_ticks and len(ticks)>max_ticks: ticks=ticks[:max_ticks]
    return ticks

def bars_n(ticks,seconds):
    out=[]; bucket=None; cur=None
    span_ns=int(seconds*1e9)
    for idx,q in enumerate(ticks):
        ns=int(q.ts_event); b=(ns//span_ns); p=float(q.bid_price)
        if bucket is None or b!=bucket:
            if cur: out.append(cur)
            bucket=b; s=b*span_ns
            cur=Bar(s,s+span_ns,p,p,p,p,idx,idx)
        else:
            cur.h=max(cur.h,p); cur.l=min(cur.l,p); cur.c=p; cur.last_idx=idx
    if cur: out.append(cur)
    return out

def rng(b): return max(b.h-b.l,EPS)
def body_ratio(b): return abs(b.c-b.o)/rng(b)
def close_loc(b): return (b.c-b.l)/rng(b)

def completed_before(bs,ts_ns):
    # last completed bar before timestamp
    lo,hi=0,len(bs)-1; ans=None
    while lo<=hi:
        m=(lo+hi)//2
        if bs[m].end_ns<=ts_ns:
            ans=bs[m]; lo=m+1
        else:
            hi=m-1
    return ans

def accumulation_score(bs,i):
    if i<14:return 0.0
    A=bs[i-2]
    recent=bs[i-12:i-2]
    medr=median([rng(x) for x in recent]) if recent else rng(A)
    s=0.0
    if rng(A)<=0.80*medr:s+=0.55
    if body_ratio(A)<=0.35:s+=0.45
    return min(1.0,s)

def manipulation_side(prev,cur):
    bull = cur.l<prev.l and cur.c>prev.l
    bear = cur.h>prev.h and cur.c<prev.h
    if bull and not bear:return 1
    if bear and not bull:return -1
    return 0

def manipulation_features(bs,i):
    # Completed bars only. Do not require a second manipulation.
    # M1 = most recent manipulation vs A, M2 = previous opposite manipulation if present.
    if i<4:return 0,0,0
    A=bs[i-2]; M1=bs[i-1]
    side1=manipulation_side(A,M1)
    count=1 if side1 else 0
    opposite_before=0
    if i>=3:
        P=bs[i-3]
        side0=manipulation_side(P,A)
        if side0 and side1 and side0==-side1:
            count=2; opposite_before=1
    return side1,count,opposite_before

def find_recent_fvg(bs,i,lookback=8):
    # Return latest completed strict 3-bar FVG before current D.
    start=max(2,i-lookback)
    for k in range(i-1,start-1,-1):
        a,c=bs[k-2],bs[k]
        if a.h<c.l:
            return {'dir':1,'low':a.h,'high':c.l,'mid':(a.h+c.l)/2}
        if a.l>c.h:
            return {'dir':-1,'low':c.h,'high':a.l,'mid':(c.h+a.l)/2}
    return None

def ifvg_feature(fvg,side,p):
    # Optional confidence modifier, never a mandatory gate.
    # Long: prior bearish FVG invalidated upward. Short: prior bullish FVG invalidated downward.
    if not fvg:return 0.0
    if side>0 and fvg['dir']<0:
        if p>fvg['high']: return 1.0
        if p>fvg['mid']: return 0.5
    if side<0 and fvg['dir']>0:
        if p<fvg['low']: return 1.0
        if p<fvg['mid']: return 0.5
    return 0.0

def htf_shadow_score(ts_ns,p,side,b1,b5,b60):
    # HTF shadow/context: LTF pattern is treated as part of the currently forming HTF candle.
    # Use only completed HTF bars before current tick.
    score=0.0; used=0
    for series,w in ((b1,0.30),(b5,0.30),(b60,0.40)):
        b=completed_before(series,ts_ns)
        if not b: continue
        used+=1
        # continuation-friendly close/body orientation + proximity to prior bar range/POI
        orient=(b.c-b.o)*side
        if orient>0: score+=w*0.55
        cl=close_loc(b)
        if (side>0 and cl>=0.60) or (side<0 and cl<=0.40): score+=w*0.25
        # if current price is revisiting the completed HTF body, treat as POI support/resistance context
        lo=min(b.o,b.c); hi=max(b.o,b.c)
        if lo<=p<=hi: score+=w*0.20
    return min(1.0,score/(1.0 if used else 1.0))

def micro_cisd_velocity(ticks,j,side,lookback=8):
    if j<lookback+3:return (0.0,0.0,0.0)
    ps=[float(ticks[k].bid_price) for k in range(j-lookback,j+1)]
    prior=ps[-5:-1]
    ref=max(prior) if side>0 else min(prior)
    cisd=1.0 if ((side>0 and ps[-1]>ref) or (side<0 and ps[-1]<ref)) else 0.0
    d1=ps[-1]-ps[-2]; d2=ps[-2]-ps[-3]; d3=ps[-3]-ps[-4]
    v=((d1+d2+d3)/3.0)*side
    a=(d1-d2)*side
    vscore=1.0 if v>0 else 0.0
    ascore=1.0 if a>=0 else 0.0
    return cisd,vscore,ascore

def trigger_score(ticks,D,M,setup_side,m_count,fvg,b1,b5,b60,threshold=0.62):
    # D is event-driven: 15s is only an observation window.
    # No completed-D data, no backward trigger search.
    level=M.o
    reclaimed=False; retested=False
    for j in range(D.first_idx+3,D.last_idx+1):
        p=float(ticks[j].bid_price); ts=int(ticks[j].ts_event)
        if not reclaimed:
            reclaimed=(setup_side>0 and p>level) or (setup_side<0 and p<level)
            if not reclaimed: continue
        elif not retested:
            if abs(p-level)<=0.08:
                retested=True
        cisd,vel,acc=micro_cisd_velocity(ticks,j,setup_side)
        ifvg=ifvg_feature(fvg,setup_side,p)
        htf=htf_shadow_score(ts,p,setup_side,b1,b5,b60)
        dual=min(1.0,m_count/2.0)  # 1M=0.5, 2M=1.0; modifier only
        ret=1.0 if retested else 0.0
        reclaim=1.0
        score=(0.16*reclaim + 0.10*ret + 0.22*cisd + 0.17*vel + 0.10*acc +
               0.10*ifvg + 0.08*dual + 0.07*htf)
        if score>=threshold:
            return j,setup_side,{'score':score,'cisd':cisd,'velocity':vel,'acceleration':acc,
                                'ifvg':ifvg,'dual_m':dual,'htf_shadow':htf,'retest':ret}
    return None,None,None

def first_passage(ticks,start_idx,side,entry,seconds,dist):
    end=int(ticks[start_idx].ts_event)+int(seconds*1e9); mfe=mae=0.0
    for j in range(start_idx+1,len(ticks)):
        if int(ticks[j].ts_event)>end:break
        p=float(ticks[j].bid_price); fav=(p-entry)*side; adv=-(p-entry)*side
        mfe=max(mfe,fav); mae=max(mae,adv)
        if fav>=dist:return 1,mfe,mae
        if adv>=dist:return -1,mfe,mae
    return 0,mfe,mae

def run_variant(ticks,b15,b1,b5,b60,variant):
    horizons=[1,3,5,10,15]; dists=[0.10,0.20,0.30]
    stats={(h,d):{'w':0,'l':0,'none':0,'mfe':0.0,'mae':0.0} for h in horizons for d in dists}
    sig=longs=shorts=0; feat={'m1':0,'m2':0,'ifvg':0,'htf':0}
    thresholds={'V4_SCORE_LOOSE':0.54,'V4_SCORE_CORE':0.62,'V4_SCORE_STRICT':0.70}
    th=thresholds[variant]
    for i in range(14,len(b15)-2):
        accs=accumulation_score(b15,i)
        if accs<=0:continue
        side,m_count,_=manipulation_features(b15,i)
        if not side:continue
        D=b15[i]; M=b15[i-1]; fvg=find_recent_fvg(b15,i)
        idx,trade_side,meta=trigger_score(ticks,D,M,side,m_count,fvg,b1,b5,b60,th)
        if idx is None or idx>=len(ticks)-2:continue
        sig+=1; longs+=trade_side>0; shorts+=trade_side<0
        feat['m2']+=m_count>=2; feat['m1']+=m_count==1; feat['ifvg']+=meta['ifvg']>0; feat['htf']+=meta['htf_shadow']>0
        entry=float(ticks[idx].bid_price)
        for h in horizons:
            for d in dists:
                r,mfe,mae=first_passage(ticks,idx,trade_side,entry,h,d); s=stats[(h,d)]
                if r>0:s['w']+=1
                elif r<0:s['l']+=1
                else:s['none']+=1
                s['mfe']+=mfe; s['mae']+=mae
    rows=[]
    for h in horizons:
        for d in dists:
            s=stats[(h,d)]; dec=s['w']+s['l']; n=dec+s['none']
            rows.append({'horizon_s':h,'distance':d,'wins_first':s['w'],'losses_first':s['l'],'none':s['none'],
                         'first_passage_wr':s['w']/dec if dec else None,'coverage':dec/n if n else 0.0,
                         'avg_mfe':s['mfe']/n if n else 0.0,'avg_mae':s['mae']/n if n else 0.0})
    return {'variant':variant,'threshold':th,'signals':sig,'longs':longs,'shorts':shorts,'feature_counts':feat,'first_passage':rows}

def run(catalog,max_ticks=3000000):
    ticks=load_ticks(catalog,max_ticks)
    b15=bars_n(ticks,15); b1=bars_n(ticks,60); b5=bars_n(ticks,300); b60=bars_n(ticks,3600)
    variants=['V4_SCORE_LOOSE','V4_SCORE_CORE','V4_SCORE_STRICT']
    return {
      'raw_bidask':True,'ticks':len(ticks),'bars15':len(b15),'bars1m':len(b1),'bars5m':len(b5),'bars1h':len(b60),
      'causality':'All context bars are derived from raw QuoteTick and must be completed before the current tick. D is event-driven inside the current 15s observation window. No completed-D filter or backward entry search.',
      'design':{
        'double_manipulation':'feature/weight only; never mandatory',
        'ifvg':'confidence modifier only; never mandatory',
        'htf_shadow':'completed M1/M5/H1 context from raw ticks; LTF pattern interpreted as HTF candle formation context',
        'distribution_start':'score crossing from reclaim/retest + micro-CISD + velocity/acceleration + optional IFVG/dual-M/HTF context'
      },
      'weights':{'reclaim':0.16,'retest':0.10,'micro_cisd':0.22,'velocity':0.17,'acceleration':0.10,'ifvg':0.10,'dual_m':0.08,'htf_shadow':0.07},
      'variants':[run_variant(ticks,b15,b1,b5,b60,v) for v in variants],
      'status':'CAUSAL_V4_DISCOVERY_DIAGNOSTIC_NOT_BROKER_PNL'
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--max-ticks',type=int,default=3000000); a=ap.parse_args()
    res=run(a.catalog,a.max_ticks); d=Path('results/dexg-amd15')/a.experiment_id; d.mkdir(parents=True,exist_ok=True)
    p=d/'AMD_STATE_V4.json'; p.write_text(json.dumps(res,indent=2),encoding='utf-8'); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
