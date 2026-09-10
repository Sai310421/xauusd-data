from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog

TRIGGERS=[0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.20]
ADDS=[0.01,0.015,0.02,0.025,0.03,0.04,0.05]
REVERSALS=[0.06,0.08,0.10,0.12,0.16,0.20,0.24,0.30]
MAX_LAYERS=10
INITIAL=1000.0
COST_PER_LAYER=0.0  # raw spread is embedded in Bid/Ask path; broker commission/slippage added in next gate

@dataclass
class Result:
    trigger: float; add: float; reversal: float; n: int; wr: float; pf: float; net: float; maxdd: float; ev: float; score: float

def fpx(x):
    return float(x.as_double()) if hasattr(x,'as_double') else float(x)

def run_combo(ticks: Iterable[QuoteTick], trigger: float, add: float, reversal: float) -> Result:
    ticks = ticks if isinstance(ticks, list) else list(ticks)
    anchor=None; side=0; entries=[]; last_add=None; extreme=None
    pnls=[]; equity=INITIAL; peak=INITIAL; maxdd=0.0
    for t in ticks:
        bid=fpx(t.bid_price); ask=fpx(t.ask_price); mid=(bid+ask)/2.0
        if anchor is None:
            anchor=mid; continue
        if side==0:
            if mid>=anchor+trigger:
                side=1; entries=[ask]; last_add=ask; extreme=bid
            elif mid<=anchor-trigger:
                side=-1; entries=[bid]; last_add=bid; extreme=ask
            else:
                continue
        px=bid if side>0 else ask
        extreme=max(extreme,px) if side>0 else min(extreme,px)
        while len(entries)<MAX_LAYERS:
            target=last_add+side*add
            if (side>0 and px<target) or (side<0 and px>target): break
            entries.append(ask if side>0 else bid); last_add=target
        rev=(px<=extreme-reversal) if side>0 else (px>=extreme+reversal)
        if rev:
            pnl=sum((px-e)*side for e in entries)-COST_PER_LAYER*len(entries)
            pnls.append(pnl); equity+=pnl; peak=max(peak,equity); maxdd=max(maxdd,(peak-equity)/max(peak,1e-9)*100)
            anchor=mid; side=0; entries=[]; last_add=None; extreme=None
    if side!=0 and ticks:
        t=ticks[-1]
        bid=fpx(t.bid_price); ask=fpx(t.ask_price); px=bid if side>0 else ask
        pnl=sum((px-e)*side for e in entries)-COST_PER_LAYER*len(entries)
        pnls.append(pnl)
    a=np.asarray(pnls,float)
    n=len(a)
    if n==0:return Result(trigger,add,reversal,0,0,0,0,0,float('-inf'),float('-inf'))
    wins=a[a>0]; losses=a[a<0]
    pf=float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else (math.inf if len(wins) else 0.0)
    wr=float((a>0).mean()*100); ev=float(a.mean()); net=float(a.sum())
    score=ev*math.sqrt(n)/(1.0+maxdd/5.0)
    return Result(trigger,add,reversal,n,wr,pf,net,maxdd,ev,score)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True)
    ap.add_argument('--raw-bidask-only',action='store_true'); ap.add_argument('--symbol',default='XAUUSD')
    ap.add_argument('--trigger',type=float,default=None,help='Run one trigger shard only')
    a=ap.parse_args()
    if not a.raw_bidask_only: raise SystemExit('RAW_BIDASK_ONLY_REQUIRED')
    cp=Path(a.catalog); man=json.loads((cp/'catalog_manifest.json').read_text())
    cat=ParquetDataCatalog(str(cp)); inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')==a.symbol)
    ticks=list(cat.query(data_cls=QuoteTick,identifiers=[inst.id.value]))
    if not ticks: raise SystemExit('NO_RAW_TICKS')
    triggers=TRIGGERS if a.trigger is None else [a.trigger]
    if any(all(abs(tr-x)>1e-12 for x in TRIGGERS) for tr in triggers): raise SystemExit('INVALID_TRIGGER_SHARD')
    rows=[]
    for tr in triggers:
        for ad in ADDS:
            for rv in REVERSALS:
                rows.append(run_combo(ticks,tr,ad,rv).__dict__)
    df=pd.DataFrame(rows)
    df['positive_ev']=df['ev']>0
    df['eligible']=(df['n']>=100)&(df['ev']>0)&(df['pf']>1.0)&(df['maxdd']<=20.0)
    ranked=df.sort_values(['eligible','score','ev','pf'],ascending=[False,False,False,False])
    out=Path('results/ae-bt')/a.experiment_id; out.mkdir(parents=True,exist_ok=True)
    ranked.to_csv(out/'ev_g75_surface.csv',index=False)
    top=ranked.head(25).to_dict(orient='records')
    summary={'verification_level':'RAW_BIDASK_EV_SURFACE_RESEARCH','symbol':a.symbol,'raw_ticks':len(ticks),'period':man,'grid':{'trigger':triggers,'add':ADDS,'reversal':REVERSALS,'combinations':len(rows)},'selection_rule':'n>=100, EV>0, PF>1, MaxDD<=20; rank by EV*sqrt(N)/(1+DD/5)','top25':top,'note':'Trigger-sharded deterministic price-distance expectancy surface. Commission/probabilistic slippage and OOS split required before promotion.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__': main()
