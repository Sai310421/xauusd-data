#!/usr/bin/env python3
import csv, json, math, sys
from pathlib import Path

BASE={"N":176483,"BUY":88223,"SELL":88260,"WR":72.71,"PF":1.74,"MaxDD":3.97}

def pct_ratio(x,b): return (x/b) if b else 0.0

def main():
    if len(sys.argv)<2:
        print("usage: parity_gate.py trades.csv [out.json]"); return 2
    p=Path(sys.argv[1]); out=Path(sys.argv[2]) if len(sys.argv)>2 else Path("parity_report.json")
    if not p.exists():
        out.write_text(json.dumps({"status":"BLOCKED_SIGNAL","reason":"trade/signal replay file missing","baseline":BASE},indent=2))
        print(out.read_text()); return 3
    rows=list(csv.DictReader(p.open(encoding="utf-8")))
    if not rows:
        out.write_text(json.dumps({"status":"BLOCKED_SIGNAL","reason":"empty replay","baseline":BASE},indent=2)); print(out.read_text()); return 3
    n=len(rows); buy=sum(str(r.get("side","")).upper()=="BUY" for r in rows); sell=sum(str(r.get("side","")).upper()=="SELL" for r in rows)
    pnls=[]
    for r in rows:
        try: pnls.append(float(r.get("pnl",0)))
        except: pnls.append(0.0)
    wins=sum(x>0 for x in pnls); wr=100*wins/n if n else 0
    gp=sum(x for x in pnls if x>0); gl=-sum(x for x in pnls if x<0); pf=gp/gl if gl>0 else (float('inf') if gp>0 else 0.0)
    eq=0; peak=0; mdd=0
    for x in pnls:
        eq+=x; peak=max(peak,eq); dd=peak-eq; mdd=max(mdd,dd)
    # MaxDD percentage requires equity base; use supplied equity_pct_dd if available, otherwise mark unavailable.
    dd_vals=[]
    for r in rows:
        try:
            if r.get("dd_pct") not in (None,""): dd_vals.append(float(r["dd_pct"]))
        except: pass
    maxdd=max(dd_vals) if dd_vals else None
    nret=pct_ratio(n,BASE["N"])
    side_balance=1-abs(buy-sell)/max(1,n)
    score_parts={
      "N":max(0,1-abs(n-BASE["N"])/BASE["N"]),
      "side":side_balance,
      "WR":max(0,1-abs(wr-BASE["WR"])/max(BASE["WR"],1e-9)),
      "PF":max(0,1-abs(pf-BASE["PF"])/BASE["PF"]) if math.isfinite(pf) else 0,
      "DD":None if maxdd is None else max(0,1-abs(maxdd-BASE["MaxDD"])/BASE["MaxDD"]),
    }
    usable=[v for v in score_parts.values() if v is not None]
    score=100*sum(usable)/len(usable)
    status="PASS" if (0.99<=nret<=1.01 and abs(buy-BASE['BUY'])/BASE['BUY']<=0.01 and abs(sell-BASE['SELL'])/BASE['SELL']<=0.01) else "FAIL"
    rep={"status":status,"baseline":BASE,"actual":{"N":n,"BUY":buy,"SELL":sell,"WR":wr,"PF":pf,"MaxDD":maxdd},"N_retention":nret,"parity_score_pct":score,"score_parts":score_parts,"rule":"N/BUY/SELL structural parity first; economics only after structural pass"}
    out.write_text(json.dumps(rep,indent=2,ensure_ascii=False)); print(out.read_text()); return 0 if status=="PASS" else 4
if __name__=="__main__": raise SystemExit(main())