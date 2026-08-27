#!/usr/bin/env python3
import argparse, csv, json, math
from bisect import bisect_left
from collections import deque

BASELINE={"N":176483,"BUY":88223,"SELL":88260,"WR_pct":72.71,"PF":1.74,"MaxDD_pct":3.97}

def load_csv(path):
    rows=[]
    with open(path,newline='',encoding='utf-8-sig') as f:
        r=csv.DictReader(f)
        lower={c.lower():c for c in r.fieldnames}
        def col(*names):
            for n in names:
                if n in lower:return lower[n]
            raise KeyError(f'missing one of {names}; got {r.fieldnames}')
        tc=col('timestamp','ts','time','time_msc','datetime')
        bc=col('bid','b')
        ac=col('ask','a')
        for x in r:
            try:
                t=float(x[tc]); b=float(x[bc]); a=float(x[ac])
                if t<1e12: t*=1000.0
                if a>=b>0: rows.append((t,b,a))
            except: pass
    rows.sort(key=lambda x:x[0])
    return rows

def asof_idx(times,t):
    i=bisect_left(times,t)
    if i>=len(times): return len(times)-1
    return i

def run(ref,exe,window=5,cooldown_ms=3000,hold_ms=100,entry_latency_ms=15,close_latency_ms=15,spread_gate=0.0,initial_equity=100000.0,contract_value=100.0,lot=0.01):
    et=[x[0] for x in exe]
    asks=deque(maxlen=window); bids=deque(maxlen=window)
    next_create=-1e30
    pnl=[]; sides=[]; holds=[]
    for t,rb,ra in ref:
        asks.append(ra); bids.append(rb)
        if len(asks)<window or t<next_create: continue
        a_cond=asks[0]-asks[-1]
        b_cond=bids[-1]-bids[0]
        side=1 if (b_cond>0 and a_cond<0) else (-1 if (a_cond>0 and b_cond<0) else 0)
        if side==0: continue
        ei=asof_idx(et,t+entry_latency_ms)
        te,eb,ea=exe[ei]
        point=max(1e-12,abs(ea+eb)/2*1e-9)
        spread_pts=(ea-eb)/point
        if spread_gate>=0 and spread_pts>spread_gate+1e-12: continue
        xi=asof_idx(et,max(te,t+entry_latency_ms+hold_ms+close_latency_ms))
        tx,xb,xa=exe[xi]
        if side>0: edge=xb-ea
        else: edge=eb-xa
        usd=edge*contract_value*lot
        pnl.append(usd); sides.append(side); holds.append(tx-te)
        next_create=t+cooldown_ms
    n=len(pnl); wins=sum(1 for x in pnl if x>0)
    gp=sum(x for x in pnl if x>0); gl=-sum(x for x in pnl if x<0)
    eq=initial_equity; peak=eq; maxdd=0.0
    for x in pnl:
        eq+=x; peak=max(peak,eq)
        if peak>0: maxdd=max(maxdd,(peak-eq)/peak*100.0)
    buy=sum(1 for s in sides if s>0); sell=n-buy
    wr=100*wins/n if n else 0.0
    pf=gp/gl if gl>0 else (float('inf') if gp>0 else 0.0)
    return {"N":n,"BUY":buy,"SELL":sell,"WR_pct":wr,"PF":pf,"MaxDD_pct":maxdd,"NetPnL":sum(pnl),"EndingEquity":eq,"AvgHold_ms":sum(holds)/len(holds) if holds else 0.0,"baseline":BASELINE,"delta":{"N":n-BASELINE['N'],"WR_pct":wr-BASELINE['WR_pct'],"PF":pf-BASELINE['PF'],"MaxDD_pct":maxdd-BASELINE['MaxDD_pct']}}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',required=True)
    p.add_argument('--execution',required=True)
    p.add_argument('-o','--output',default='dual_feed_parity.json')
    p.add_argument('--spread-gate',type=float,default=0.0)
    p.add_argument('--hold-ms',type=int,default=100)
    p.add_argument('--entry-latency-ms',type=int,default=15)
    p.add_argument('--close-latency-ms',type=int,default=15)
    a=p.parse_args()
    ref=load_csv(a.reference); exe=load_csv(a.execution)
    if not ref or not exe: raise SystemExit('empty reference or execution feed')
    res=run(ref,exe,hold_ms=a.hold_ms,entry_latency_ms=a.entry_latency_ms,close_latency_ms=a.close_latency_ms,spread_gate=a.spread_gate)
    with open(a.output,'w',encoding='utf-8') as f: json.dump(res,f,indent=2,ensure_ascii=False)
    print(json.dumps(res,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
