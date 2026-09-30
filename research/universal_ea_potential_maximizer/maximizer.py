#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, json, math
from typing import Any, Dict, List

EVIDENCE={"PREDICTED":0,"PROXY":1,"NAUTILUS_IS":2,"NAUTILUS_OOS":3,"NAUTILUS_STRESS":4,"VALIDATED":5}

def loadj(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def num(d,k,default=0.0):
    try: return float(d.get(k,default))
    except Exception: return float(default)

def canonical_kpi(raw:Dict[str,Any])->Dict[str,Any]:
    aliases={
      "Return":["Return","return_pct","return","net_return_pct","sum_R"],
      "MaxFloatingDD":["MaxFloatingDD","max_floating_dd_pct","maxDD_pct","maxDD_R"],
      "MaxClosedDD":["MaxClosedDD","max_closed_dd_pct"],
      "PF":["PF","pf"],
      "RF":["RF","recovery_factor"],
      "WR":["WR","win_rate"],
      "EV_per_trade":["EV_per_trade","mean_R","expectancy"],
      "N":["N","trades","trade_count"],
      "PayoffRatio":["PayoffRatio","payoff_ratio"],
      "MaxConsecutiveLoss":["MaxConsecutiveLoss","max_consecutive_loss"],
      "MinMarginLevel":["MinMarginLevel","min_margin_level_pct"],
      "MaxConcurrentLot":["MaxConcurrentLot","max_concurrent_lot"],
      "RuinProbability":["RuinProbability","ruin_probability"],
      "CVaR":["CVaR","cvar"],
      "SpreadCost":["SpreadCost","spread_cost"],
      "SlippageCost":["SlippageCost","slippage_cost"],
      "Commission":["Commission","commission"],
      "TotalCost":["TotalCost","total_cost"]
    }
    out={}
    for ck,ks in aliases.items():
        out[ck]=None
        for k in ks:
            if k in raw and raw[k] is not None:
                out[ck]=raw[k]; break
    if out["TotalCost"] is None:
        vals=[out[x] for x in ("SpreadCost","SlippageCost","Commission") if isinstance(out[x],(int,float))]
        if vals: out["TotalCost"]=sum(vals)
    return out

def objective(k:Dict[str,Any],cfg:Dict[str,Any])->float:
    o=cfg["objective"]
    growth=num(k,"Return",0.0)
    dd=num(k,"MaxFloatingDD",0.0)
    cvar=abs(num(k,"CVaR",0.0))
    ruin=num(k,"RuinProbability",0.0)
    cost=num(k,"TotalCost",0.0)
    instability=num(k,"Instability",0.0)
    return growth-o["lambda_D"]*dd-o["lambda_T"]*cvar-o["lambda_R"]*ruin-o["lambda_C"]*cost-o["lambda_U"]*instability

def evidence_ok(label:str, minimum:str)->bool:
    return EVIDENCE.get(label,-1)>=EVIDENCE.get(minimum,999)

def gates(k:Dict[str,Any],cfg:Dict[str,Any],label:str)->Dict[str,Any]:
    g=cfg["gates"]; reasons=[]
    pf=num(k,"PF",-999); ev=num(k,"EV_per_trade",-999); n=int(num(k,"N",0))
    dd=num(k,"MaxFloatingDD",999); ruin=num(k,"RuinProbability",999); ml=num(k,"MinMarginLevel",-999)
    if pf<g["pf_oos_min"]: reasons.append(f"PF<{g['pf_oos_min']}")
    if ev<=g["ev_oos_min"]: reasons.append(f"EV<={g['ev_oos_min']}")
    if n<g["min_trades"]: reasons.append(f"N<{g['min_trades']}")
    if dd>g["dd_limit_pct"]: reasons.append(f"DD>{g['dd_limit_pct']}")
    if ruin>g["ruin_limit"]: reasons.append(f"Ruin>{g['ruin_limit']}")
    if ml<g["min_margin_level_pct"]: reasons.append(f"MinMarginLevel<{g['min_margin_level_pct']}")
    if not evidence_ok(label,"NAUTILUS_OOS"): reasons.append("Evidence<NAUTILUS_OOS")
    return {"pass":len(reasons)==0,"reasons":reasons}

def classify(original,parameter,a17,margin_evidence:bool)->str:
    oe=num(original,"EV_per_trade",-999)
    pj=num(parameter,"J",-1e99); oj=num(original,"J",-1e99); aj=num(a17,"J",-1e99)
    if oe<=0 and not margin_evidence: return "NO_RECOVERABLE_EDGE"
    if pj>oj and num(parameter,"PF",0)>num(original,"PF",0): return "PARAMETER_POTENTIAL"
    if aj>pj and num(a17,"PF",0)>=num(parameter,"PF",0) and num(a17,"MaxFloatingDD",999)<num(parameter,"MaxFloatingDD",999):
        return "CONTROLLER_POTENTIAL"
    if num(a17,"EV_per_trade",-999)>max(0.0,num(parameter,"EV_per_trade",-999)): return "ENTRY_POTENTIAL"
    return "ROBUST_MAX_CANDIDATE" if a17.get("GatePass") else "PARAMETER_POTENTIAL"

def stable_region(rows:List[Dict[str,Any]],eps:float)->List[Dict[str,Any]]:
    if not rows: return []
    m=max(num(r,"J",-1e99) for r in rows)
    floor=(1-eps)*m if m>=0 else m-abs(m)*eps
    return [r for r in rows if num(r,"J",-1e99)>=floor]

def marginal(base:Dict[str,Any],variant:Dict[str,Any])->Dict[str,Any]:
    dd0=num(base,"MaxFloatingDD",0); ddi=num(variant,"MaxFloatingDD",0); ret0=num(base,"Return",0)
    return {
      "ME":num(variant,"J",0)-num(base,"J",0),
      "DeltaReturn":num(variant,"Return",0)-num(base,"Return",0),
      "DDReduction":((dd0-ddi)/dd0 if dd0 else None),
      "ReturnRetention":(num(variant,"Return",0)/ret0 if ret0 else None)
    }

def enrich(name:str,raw:Dict[str,Any],cfg,label)->Dict[str,Any]:
    k=canonical_kpi(raw); k["EvidenceLabel"]=label; k["Variant"]=name
    k["J"]=objective(k,cfg); gate=gates(k,cfg,label); k["GatePass"]=gate["pass"]; k["GateReasons"]=gate["reasons"]
    return k

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True); ap.add_argument("--subject",required=True); ap.add_argument("--out",required=True)
    a=ap.parse_args(); cfg=loadj(a.config); sub=loadj(a.subject)
    variants=sub.get("variants",{})
    outrows={}
    for name in ("ORIGINAL","PARAMETER_MAX","A17_ROBUST_MAX"):
        v=variants.get(name,{})
        outrows[name]=enrich(name,v.get("kpi",v),cfg,v.get("evidence_label","PROXY"))
    singleton_results=[]
    for x in sub.get("a17_singletons",[]):
        row=enrich(x.get("name","A?"),x.get("kpi",x),cfg,x.get("evidence_label","PROXY"))
        row["MarginalEDGE"]=marginal(outrows["PARAMETER_MAX"],row)
        singleton_results.append(row)
    passing_singletons=[x for x in singleton_results if x["MarginalEDGE"]["ME"]>0 and num(x,"EV_per_trade",-999)>0]
    combos_allowed=len(passing_singletons)>0
    parameter_candidates=[]
    for x in sub.get("parameter_candidates",[]):
        row=enrich(x.get("name","candidate"),x.get("kpi",x),cfg,x.get("evidence_label","PROXY"))
        parameter_candidates.append(row)
    stable=stable_region(parameter_candidates,cfg["gates"]["stable_region_epsilon"])
    margin_evidence=any(x["MarginalEDGE"]["ME"]>0 for x in singleton_results)
    cls=classify(outrows["ORIGINAL"],outrows["PARAMETER_MAX"],outrows["A17_ROBUST_MAX"],margin_evidence)
    final={
      "framework":"AE Universal EA Potential Maximizer Nautilus v1.0",
      "subject":sub.get("name","UNKNOWN"),
      "official_kpi_policy":"Only Nautilus Raw Tick can become official KPI.",
      "variants":outrows,
      "parameter_stable_region":stable,
      "a17_singletons":singleton_results,
      "a17_passing_singletons":[x["Variant"] for x in passing_singletons],
      "pair_triple_search_allowed":combos_allowed,
      "potential_classification":cls,
      "final_status":"VALIDATED" if outrows["A17_ROBUST_MAX"]["GatePass"] and outrows["A17_ROBUST_MAX"]["EvidenceLabel"]=="VALIDATED" else "NOT_VALIDATED",
      "audit":{
        "failed_or_invalid_runs_retained":True,
        "final_oos_used_for_selection":False,
        "raw_tick_required_for_final":True
      }
    }
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(final,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(final,indent=2,ensure_ascii=False))
if __name__=="__main__": main()
