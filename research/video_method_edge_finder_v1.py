#!/usr/bin/env python3
"""Real Raw Bid/Ask EDGE Finder following the uploaded video's procedure.

Video-supported procedure:
  numerical sweep -> lane metrics (median profit / median RF / no-loss%)
  -> filter -> inspect distribution -> EV -> Monte Carlo risk sizing.

The uploaded video labels its own sweep "SYNTHETIC" and does not define P0-P3.
Therefore this implementation does NOT pretend to recover those hidden rules.
AMOS v1 defines one explicit numerical hypothesis family:
  post-impulse continuation, P0..P3 = train-only absolute-return quantiles
  50/65/80/90%, fixed 3-bar hold.

Discovery is chronological 70%; OOS is final 30%. Raw QuoteTick only.
"""
from __future__ import annotations
import argparse,csv,json,math,random
import numpy as np
from pathlib import Path
from statistics import mean,median,pstdev
import nautilus_trader
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from research.nautilus_catalog_compat import query_quote_ticks_compat

SCHEMA="AMOS.VideoMethodEdgeFinder.RealRaw.v1"
P_QUANTILES={0:.50,1:.65,2:.80,3:.90}

def _f(x): return float(x.as_double()) if hasattr(x,"as_double") else float(x)
def _q(xs,q):
    s=sorted(float(x) for x in xs); p=(len(s)-1)*q; lo=int(p); hi=min(lo+1,len(s)-1); w=p-lo
    return s[lo]*(1-w)+s[hi]*w
def _maxdd(pnls):
    eq=0.; peak=0.; worst=0.
    for x in pnls:
        eq+=x; peak=max(peak,eq); worst=max(worst,peak-eq)
    return worst
def _rf(pnls):
    net=sum(pnls); dd=_maxdd(pnls)
    return net/dd if dd>1e-12 else (999.0 if net>0 else 0.0)
def _bars(ticks,minutes=5):
    ns=minutes*60*1_000_000_000; close={}; spread={}
    for t in ticks:
        k=(int(t.ts_event)//ns)*ns; bid=_f(t.bid_price); ask=_f(t.ask_price)
        close[k]=(bid+ask)/2; spread[k]=ask-bid
    ks=sorted(close); prices=[close[k] for k in ks]
    rets=[prices[i]/prices[i-1]-1 for i in range(1,len(prices))]
    return ks[1:],prices[1:],rets,mean(spread.values()) if spread else 0.

def _trades(rets,threshold,direction,hold=3):
    sign=1. if direction=="Long" else -1.
    out=[]
    # Numeric-only hypothesis: impulse in trade direction, then hold fixed horizon.
    for i in range(1,len(rets)-hold):
        trigger=rets[i-1]
        if sign*trigger < threshold: continue
        fwd=1.
        for j in range(i,i+hold): fwd*=1+rets[j]
        out.append(sign*(fwd-1))
    return out

def _bootstrap_metrics(trades,paths=2000,ntrades=400,seed=11,unit_dollars=1000.):
    if len(trades)<30: return None
    # Vectorized bootstrap preserves the video's 2,000 x 400 design.
    base=np.asarray(trades,dtype=float)*float(unit_dollars)
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(base),size=(paths,ntrades))
    samples=base[idx]
    profits=samples.sum(axis=1)
    equity=np.cumsum(samples,axis=1)
    peaks=np.maximum.accumulate(np.maximum(equity,0.0),axis=1)
    dd=np.max(peaks-equity,axis=1)
    rfs=np.where(dd>1e-12,profits/dd,np.where(profits>0,999.0,0.0))
    wins=base[base>0]; losses=base[base<=0]
    return {
      "MedProfit":float(np.median(profits)),"P90Profit":float(np.quantile(profits,.90)),"MedRF":float(np.median(rfs)),
      "NoLossPct":float(100*np.mean(profits<=0)),
      "WinRatePct":float(100*np.mean(base>0)),
      "AvgWin":float(wins.mean()) if len(wins) else 0.,"AvgLoss":float(losses.mean()) if len(losses) else 0.,
      "EV":float(base.mean()),"Trades":int(len(base)),
    }

def _lane(symbol,direction,p,train_rets,oos_rets,paths,ntrades):
    q=P_QUANTILES[p]; abs_train=[abs(x) for x in train_rets]; threshold=_q(abs_train,q)
    train=_trades(train_rets,threshold,direction); oos=_trades(oos_rets,threshold,direction)
    a=_bootstrap_metrics(train,paths,ntrades,seed=11+p)
    b=_bootstrap_metrics(oos,paths,ntrades,seed=111+p)
    if a is None: return None
    row={"Pair":symbol,"Dir":direction,"P":p,"Quantile":q,"ThresholdReturn":threshold,
         **{f"IS_{k}":v for k,v in a.items()}}
    if b:
        row.update({f"OOS_{k}":v for k,v in b.items()})
    else:
        for k in a: row[f"OOS_{k}"]=None
    # Video filter copied exactly, but dollar threshold is NOT applied because
    # source sizing/contract definition is absent. We preserve it separately.
    row["Video_RF_Filter"]=a["MedRF"]>=1.5
    row["Video_NoLoss_Filter"]=a["NoLossPct"]<=3.0
    row["Video_MedProfit300_Filter_NotComparable"]=a["MedProfit"]>=300
    row["DiscoveryCandidate"]=bool(a["MedRF"]>=1.5 and a["NoLossPct"]<=3.0 and a["EV"]>0)
    # Fail closed: OOS may confirm only a lane selected without seeing OOS.
    row["OOSConfirmed"]=bool(row["DiscoveryCandidate"] and b and b["MedRF"]>=1.0 and b["NoLossPct"]<=10.0 and b["EV"]>0)
    return row

def _risk_mc(trades_r, risk_pct, paths=2000,ntrades=400,seed=11):
    arr=np.asarray(trades_r,dtype=float); sd=float(arr.std()) or 1e-12
    rr=np.clip(arr/sd,-1.0,3.0); rng=np.random.default_rng(seed)
    samples=rr[rng.integers(0,len(rr),size=(paths,ntrades))]
    factors=np.maximum(0.0,1.0+risk_pct*samples)
    equity=np.cumprod(factors,axis=1)
    peaks=np.maximum.accumulate(np.maximum(equity,1.0),axis=1)
    dds=np.max((peaks-equity)/peaks,axis=1)
    ruined=np.any(equity<=.5,axis=1)
    return {"risk_pct":100*risk_pct,"median_maxdd_pct":float(100*np.median(dds)),
            "bad10_maxdd_pct":float(100*np.quantile(dds,.90)),"ruin50_pct":float(100*np.mean(ruined))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--catalog",required=True); ap.add_argument("--symbols",nargs="+",required=True)
    ap.add_argument("--experiment-id",required=True); ap.add_argument("--paths",type=int,default=2000); ap.add_argument("--path-trades",type=int,default=400)
    ap.add_argument("--raw-bidask-only",action="store_true"); args=ap.parse_args()
    if not args.raw_bidask_only: raise SystemExit("RAW_BIDASK_ONLY_REQUIRED")
    cat=ParquetDataCatalog(args.catalog); inst={x.id.symbol.value.replace("/",""):x for x in cat.instruments()}
    lanes=[]; raw_counts={}; oos_trade_map={}
    for symbol in args.symbols:
        if symbol not in inst: raise SystemExit(f"MISSING:{symbol}")
        ticks=query_quote_ticks_compat(cat,identifiers=[inst[symbol].id.value]); raw_counts[symbol]=len(ticks)
        _,_,rets,_spread=_bars(ticks,5); cut=int(len(rets)*.70); train=rets[:cut]; oos=rets[cut:]
        for direction in ("Long","Short"):
            for p in range(4):
                row=_lane(symbol,direction,p,train,oos,args.paths,args.path_trades)
                if row:
                    lanes.append(row)
                    if row["OOSConfirmed"]:
                        th=row["ThresholdReturn"]; oos_trade_map[(symbol,direction,p)]=_trades(oos,th,direction)
    lanes.sort(key=lambda r:(not r["OOSConfirmed"],-float(r.get("OOS_EV") or -1e99),-float(r["IS_MedRF"])))
    candidates=[r for r in lanes if r["DiscoveryCandidate"]]; confirmed=[r for r in lanes if r["OOSConfirmed"]]
    risk=[]
    for r in confirmed[:5]:
        key=(r["Pair"],r["Dir"],r["P"]); tr=oos_trade_map[key]
        if len(tr)>=30:
            risk.append({"Pair":r["Pair"],"Dir":r["Dir"],"P":r["P"],
                         "mc":[_risk_mc(tr,x,args.paths,args.path_trades,seed=11+int(x*1000)) for x in (.005,.01,.02,.05,.10)]})
    out=Path("results/edge-finder")/args.experiment_id; out.mkdir(parents=True,exist_ok=True)
    fields=list(lanes[0]) if lanes else []
    with (out/"sweep.csv").open("w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(lanes)
    summary={"schema":SCHEMA,"source_method":"UPLOADED_VIDEO_PROCEDURE","source_video_sweep_label":"SYNTHETIC",
      "hidden_source_parameter_definition_recovered":False,
      "amos_hypothesis":{"family":"POST_IMPULSE_CONTINUATION","timeframe":"M5","hold_bars":3,"P_quantiles":P_QUANTILES},
      "lane_count":len(lanes),"discovery_candidates":len(candidates),"oos_confirmed":len(confirmed),
      "raw_tick_counts":raw_counts,"top_oos":confirmed[:10],"risk_mc":risk,
      "video_filter":{"MedRF_gte":1.5,"NoLossPct_lte":3.0,"MedProfit_gte_dollars":300,
        "medprofit_filter_applied":False,"reason":"video does not disclose sizing/contract basis; dollar normalization is not comparable"},
      "selection":{"discovery":"first chronological 70%","oos":"last chronological 30%"},
      "monte_carlo":{"paths":args.paths,"trades_per_path":args.path_trades,"seed_family":"11+"},
      "execution_allowed":False,"production_weighting_allowed":False,
      "limitations":["P0-P3 are AMOS-defined because the video does not disclose their meaning.",
        "This first search tests one numeric hypothesis family, not all possible market EDGE.",
        "Dollar MedProfit uses an explicit normalization and is not broker PnL.",
        "Risk Monte Carlo uses a volatility-normalized clipped R proxy because the hypothesis has no source-defined stop."]}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
