#!/usr/bin/env python3
"""Parity calibration for NOTE-HFT reconstructed alpha.

Input CSV: timestamp,ask,bid
Optional columns: broker_ask,broker_bid (used only by execution/reality layers).

This tool does NOT tune the frozen execution logic. It evaluates only the
missing alpha window length (2..9 by default) and reports structural parity
against the documented baseline fingerprint.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from reconstructed_alpha import ReconstructedDirectionalAlpha

BASE = {"N":176483,"BUY":88223,"SELL":88260,"WR":72.71,"PF":1.74,"MaxDD":3.97}

def load(path):
    rows=[]
    with open(path,newline='',encoding='utf-8-sig') as f:
        r=csv.DictReader(f)
        need={'timestamp','ask','bid'}
        if not need.issubset(r.fieldnames or []):
            raise ValueError(f"required columns: {sorted(need)}")
        for x in r: rows.append(x)
    return rows

def evaluate(rows,window):
    a=ReconstructedDirectionalAlpha(window=window)
    buy=sell=0
    for r in rows:
        s=a.update(r['ask'],r['bid']).signal
        buy += s==1; sell += s==-1
    n=buy+sell
    nerr=abs(n-BASE['N'])/BASE['N']
    berr=abs(buy-BASE['BUY'])/BASE['BUY']
    serr=abs(sell-BASE['SELL'])/BASE['SELL']
    score=max(0.0,1.0-(0.50*nerr+0.25*berr+0.25*serr))
    return {"window":window,"N":n,"BUY":buy,"SELL":sell,"N_retention":n/BASE['N'],"structural_parity_score":score,
            "N_error_pct":100*nerr,"BUY_error_pct":100*berr,"SELL_error_pct":100*serr}

def main():
    p=argparse.ArgumentParser(); p.add_argument('csv'); p.add_argument('-o','--out',default='parity_v3.json')
    p.add_argument('--min-window',type=int,default=2); p.add_argument('--max-window',type=int,default=9)
    a=p.parse_args(); path=Path(a.csv)
    if not path.exists():
        Path(a.out).write_text(json.dumps({"status":"BLOCKED_DATA_ACCESS","reason":"UDP/reference BidAsk CSV missing","baseline":BASE},indent=2))
        return 3
    rows=load(path)
    results=[evaluate(rows,w) for w in range(a.min_window,a.max_window+1)]
    results.sort(key=lambda x:(-x['structural_parity_score'],abs(x['window']-5)))
    best=results[0]
    status='STRUCTURAL_PARITY_PASS' if best['N_error_pct']<=1 and best['BUY_error_pct']<=1 and best['SELL_error_pct']<=1 else 'STRUCTURAL_PARITY_FAIL'
    out={"status":status,"baseline":BASE,"default_reconstruction_window":5,"best":best,"candidates":results,
         "note":"Only missing alpha window is compared; frozen execution conditions are not tuned."}
    Path(a.out).write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2)); return 0 if status.endswith('PASS') else 2
if __name__=='__main__': raise SystemExit(main())
