from __future__ import annotations

import json, math, os
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, median

RESULT_DIR=Path(os.getenv('RESULT_DIR','results/adaptive-v21')); RESULT_DIR.mkdir(parents=True,exist_ok=True)
OTE={'front':0.669,'core':0.708,'deep':0.786}

@dataclass
class Trade:
    side:str; entry:float; exit:float; pnl:float; mae:float; mfe:float; slices:int; exit_reason:str

def load_quotes():
    for p in [Path('data/xauusd_quotes.jsonl'),Path('data/xauusd_ticks.jsonl'),Path('books/xauusd_quotes.jsonl'),Path('books/xauusd_ticks.jsonl')]:
        if p.exists():
            with p.open(encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]
    raise FileNotFoundError('Raw Bid/Ask quote JSONL not found')
def bid(q): return float(q.get('bid',q.get('b',0)))
def ask(q): return float(q.get('ask',q.get('a',0)))
def robust_step(xs):
    d=[abs(xs[i]-xs[i-1]) for i in range(1,len(xs)) if xs[i]!=xs[i-1]]
    return max(median(d) if d else .001,.001)
def fib_retrace(impulse_start,impulse_end,r,bull):
    return impulse_end-r*(impulse_end-impulse_start) if bull else impulse_end+r*(impulse_start-impulse_end)

def detect_crt_delivery(hist):
    '''Tick-native CRT sequence: range -> liquidity sweep -> reclaim -> opposite delivery/MSS.
    Fib is anchored on the CONFIRMED delivery impulse, not on the swept range itself.
    Returns bull, impulse_start, impulse_end, sweep_extreme, eps.
    '''
    if len(hist)<480:return None
    B=[bid(q) for q in hist]; A=[ask(q) for q in hist]
    base=hist[:300]; bb=B[:300]; aa=A[:300]
    lo=min(bb); hi=max(aa); eps=max(robust_step(bb+aa)*3,.01)
    # bullish: sell-side liquidity swept, reclaimed, then delivery breaks opposite internal high
    for s in range(300,len(hist)-80):
        if B[s] < lo-eps:
            reclaim=next((j for j in range(s+1,min(s+80,len(hist))) if B[j]>lo),None)
            if reclaim is None: continue
            internal_hi=max(B[max(0,s-100):s])
            mss=next((j for j in range(reclaim+1,len(hist)) if B[j]>internal_hi+eps),None)
            if mss is None: continue
            sweep_low=min(B[s:mss+1]); impulse_start=sweep_low; impulse_end=max(B[reclaim:mss+1])
            if impulse_end-impulse_start>=10*eps:return True,impulse_start,impulse_end,sweep_low,eps
    # bearish: buy-side liquidity swept, reclaimed, then delivery breaks opposite internal low
    for s in range(300,len(hist)-80):
        if A[s] > hi+eps:
            reclaim=next((j for j in range(s+1,min(s+80,len(hist))) if A[j]<hi),None)
            if reclaim is None: continue
            internal_lo=min(A[max(0,s-100):s])
            mss=next((j for j in range(reclaim+1,len(hist)) if A[j]<internal_lo-eps),None)
            if mss is None: continue
            sweep_high=max(A[s:mss+1]); impulse_start=sweep_high; impulse_end=min(A[reclaim:mss+1])
            if impulse_start-impulse_end>=10*eps:return False,impulse_start,impulse_end,sweep_high,eps
    return None

def run():
    Q=load_quotes()
    if len(Q)<10000:raise RuntimeError('Insufficient raw quotes')
    trades=[]; setups=0; lookback=480; horizon=720; stride=120
    for i in range(lookback,len(Q)-horizon,stride):
        z=detect_crt_delivery(Q[i-lookback:i])
        if z is None:continue
        bull,a,b,sweep,eps=z; setups+=1
        levels=[fib_retrace(a,b,OTE[k],bull) for k in ('front','core','deep')]
        path=Q[i:i+horizon]; fills=[]; fillidx=[]
        # after confirmed delivery, wait for retracement into OTE. Never enter before delivery confirmation.
        for level in levels:
            j=next((j for j,q in enumerate(path) if (ask(q)<=level if bull else bid(q)>=level)),None)
            if j is not None:
                fills.append(ask(path[j]) if bull else bid(path[j])); fillidx.append(j)
        if not fills:continue
        entry=sum(fills)/len(fills)
        invalid=sweep-eps if bull else sweep+eps
        risk=entry-invalid if bull else invalid-entry
        if risk<=0:continue
        target=entry+1.5*risk if bull else entry-1.5*risk
        start=max(fillidx); marks=[]; exitpx=None; reason='TIME'
        for q in path[start:]:
            x=bid(q) if bull else ask(q); marks.append(x)
            if bull and x<=invalid:exitpx=x;reason='INVALIDATION';break
            if (not bull) and x>=invalid:exitpx=x;reason='INVALIDATION';break
            if bull and x>=target:exitpx=x;reason='TP_1.5R';break
            if (not bull) and x<=target:exitpx=x;reason='TP_1.5R';break
        if exitpx is None: exitpx=bid(path[-1]) if bull else ask(path[-1])
        d=1 if bull else -1; pnl=d*(exitpx-entry); ex=[d*(x-entry) for x in marks] or [pnl]
        trades.append(Trade('LONG' if bull else 'SHORT',entry,exitpx,pnl,min(ex),max(ex),len(fills),reason))
    wins=[t.pnl for t in trades if t.pnl>0]; losses=[-t.pnl for t in trades if t.pnl<0]
    gw=sum(wins);gl=sum(losses);pf=gw/gl if gl else (math.inf if gw else 0);wr=len(wins)/len(trades) if trades else 0;ev=sum(t.pnl for t in trades)/len(trades) if trades else 0
    eq=peak=dd=0
    for t in trades:eq+=t.pnl;peak=max(peak,eq);dd=max(dd,peak-eq)
    S={'verification_level':'CRT_DELIVERY_REAL_RAW_BID_ASK_OTE','quotes':len(Q),'setups':setups,'trades':len(trades),'wr':wr,'pf':pf,'ev_price_units':ev,'net_price_units':sum(t.pnl for t in trades),'max_dd_price_units':dd,'avg_mae':mean([t.mae for t in trades]) if trades else 0,'avg_mfe':mean([t.mfe for t in trades]) if trades else 0,'slice_fill_counts':{str(n):sum(t.slices==n for t in trades) for n in (1,2,3)},'exit_counts':{r:sum(t.exit_reason==r for t in trades) for r in ('TP_1.5R','INVALIDATION','TIME')},'ohlc_resample_used':False,'mid_price_used':False,'notes':['CRT range -> sweep -> reclaim -> opposite delivery/MSS -> lock delivery impulse -> OTE retracement -> entry.','OTE primary 0.708 with E1 0.669 / E3 0.786.','Long Ask entry/Bid exit; Short Bid entry/Ask exit.']}
    (RESULT_DIR/'summary.json').write_text(json.dumps(S,indent=2));(RESULT_DIR/'trades.json').write_text(json.dumps([asdict(t) for t in trades],indent=2));print(json.dumps(S,indent=2))
if __name__=='__main__':run()
