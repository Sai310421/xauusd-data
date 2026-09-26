#!/usr/bin/env python3
import argparse,csv,json
from collections import deque
from pathlib import Path
CLOCKS=[("C050",50),("C100",100),("C250",250),("C500",500),("C1000",1000),("C2000",2000),("C3000",3000),("C5000",5000),("C10000",10000),("TRUE_TICK",0)]
def pf(p):
 g=sum(x for x in p if x>0); l=-sum(x for x in p if x<0); return g/l if l else (999.0 if g else 0.0)
def dd(e):
 peak=e[0]; m=0
 for x in e: peak=max(peak,x); m=max(m,(peak-x)/peak*100 if peak else 0)
 return m
def ticks(path):
 with open(path,newline="",encoding="utf-8-sig") as f:
  r=csv.DictReader(f)
  if not {"timestamp","bid","ask"}.issubset(r.fieldnames or []): raise SystemExit("FAIL-CLOSED: raw timestamp,bid,ask required")
  for x in r:
   t,b,a=float(x["timestamp"]),float(x["bid"]),float(x["ask"])
   if a>=b>0: yield t,b,a
def run(path,ms):
 b=deque(maxlen=30); pos=None; p=[]; e=[1000.0]; last=-1e30; nt=0
 for t,bid,ask in ticks(path):
  nt+=1; mid=(bid+ask)/2; b.append((t,mid))
  if pos:
   side,ent,t0=pos; mark=bid if side>0 else ask; q=(mark-ent)*side*100
   if q>0 or (t-t0)*1000>=900: p.append(q); e.append(e[-1]+q); pos=None
  if pos or len(b)<10 or ask-bid>0.60 or (ms and (t-last)*1000<ms): continue
  last=t; w=list(b)[-10:]; mom=w[-1][1]-w[0][1]
  ds=[1 if w[i][1]>w[i-1][1] else -1 if w[i][1]<w[i-1][1] else 0 for i in range(1,len(w))]
  tail=[d for d in ds[-3:] if d]
  if len(tail)<3 or abs(sum(tail))!=3 or abs(mom)<0.03: continue
  side=1 if mom>0 else -1
  if tail[-1]!=side: continue
  pos=(side,ask if side>0 else bid,t)
 wins=sum(x>0 for x in p)
 return {"N":len(p),"WR_pct":100*wins/len(p) if p else 0,"PF":pf(p),"EV":sum(p)/len(p) if p else 0,"Net":sum(p),"Return_pct":(e[-1]/e[0]-1)*100,"MaxDD_pct":dd(e),"ticks":nt}
def main():
 a=argparse.ArgumentParser(); a.add_argument("--ticks",required=True); a.add_argument("--out",default="results/tickscalper-sub10s-clock-lab"); z=a.parse_args(); o=Path(z.out); o.mkdir(parents=True,exist_ok=True)
 rows=[]
 for cid,ms in CLOCKS:
  x=run(z.ticks,ms); x.update(clock_id=cid,entry_clock_ms=ms); rows.append(x)
 with open(o/"kpi.csv","w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=["clock_id","entry_clock_ms","N","WR_pct","PF","EV","Net","Return_pct","MaxDD_pct","ticks"]); w.writeheader(); w.writerows(rows)
 (o/"summary.json").write_text(json.dumps(rows,indent=2))
 print(json.dumps(rows,indent=2))
if __name__=="__main__": main()
