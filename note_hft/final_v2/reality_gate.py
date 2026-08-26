#!/usr/bin/env python3
import csv,json,sys
from pathlib import Path

LATENCY_MS=[0,10,30,50,100]
CT_LOCK_SEC=3.0

def fnum(x):
    try:return float(x)
    except:return None

def main():
    if len(sys.argv)<2:
        print('usage: reality_gate.py bidask.csv [out.json]'); return 2
    p=Path(sys.argv[1]); out=Path(sys.argv[2]) if len(sys.argv)>2 else Path('reality_report.json')
    if not p.exists():
        rep={'status':'BLOCKED_DATA_ACCESS','reason':'Bid/Ask file missing','required_columns':['timestamp','bid','ask'],'ct_lock_sec':CT_LOCK_SEC}
        out.write_text(json.dumps(rep,indent=2)); print(out.read_text()); return 3
    rows=list(csv.DictReader(p.open(encoding='utf-8-sig')))
    aliases={k.lower():k for k in (rows[0].keys() if rows else [])}
    def col(*names):
        for n in names:
            if n.lower() in aliases:return aliases[n.lower()]
        return None
    tc=col('timestamp','time','datetime','ts','ts_ns'); bc=col('bid','b'); ac=col('ask','a')
    if not rows or not all((tc,bc,ac)):
        rep={'status':'BLOCKED_DATA_FORMAT','reason':'timestamp/bid/ask columns not detected','columns':list(rows[0].keys()) if rows else []}
        out.write_text(json.dumps(rep,indent=2)); print(out.read_text()); return 3
    samples=[]
    for r in rows:
        b=fnum(r.get(bc)); a=fnum(r.get(ac));
        if b is None or a is None: continue
        samples.append((r.get(tc),b,a))
    z=[x for x in samples if abs(x[2]-x[1])<=1e-12]
    # Without guaranteed timestamp unit, count eligible observations and report conservative upper bound from 3s lock only if parseable epoch seconds.
    times=[]
    for t,_,_ in z:
        try:
            v=float(t); v=v/1e9 if v>1e15 else (v/1e3 if v>1e12 else v); times.append(v)
        except: pass
    times=sorted(times)
    cycles=0; last=-1e100
    for t in times:
        if t-last>=CT_LOCK_SEC:
            cycles+=1; last=t
    rep={'status':'MEASURED','samples':len(samples),'zero_spread_samples':len(z),'zero_spread_ratio':len(z)/len(samples) if samples else 0,'ct_lock_sec':CT_LOCK_SEC,'max_cycles_3s_from_parseable_zero_spread_timestamps':cycles if times else None,'timestamp_parseable_zero_spread':len(times),'latency_scenarios_ms':LATENCY_MS,'note':'Latency survival requires sub-tick sequence or dual-feed timestamp semantics; not fabricated from bar data.'}
    out.write_text(json.dumps(rep,indent=2)); print(out.read_text()); return 0
if __name__=='__main__': raise SystemExit(main())