from __future__ import annotations

import argparse, json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

@dataclass
class KPI:
    variant: str
    pf: float
    dd: float
    expectancy: float
    n: int
    wr: float = 0.0
    net: float = 0.0
    params: dict | None = None


def load_cells(paths: Iterable[Path]) -> list[KPI]:
    out=[]
    for p in paths:
        try:
            x=json.loads(p.read_text(encoding='utf-8'))
            out.append(KPI(
                variant=str(x.get('variant',p.stem)),
                pf=float(x.get('PF',0.0)),
                dd=float(x.get('max_DD_pct',999.0)),
                expectancy=float(x.get('expectancy',0.0)),
                n=int(x.get('N',0)),
                wr=float(x.get('WR_pct',0.0)),
                net=float(x.get('net_virtual',0.0)),
                params=x.get('params',{}),
            ))
        except Exception:
            continue
    return out


def dominates(a: KPI,b: KPI)->bool:
    better_or_equal=(a.pf>=b.pf and a.expectancy>=b.expectancy and a.dd<=b.dd and a.n>=b.n)
    strictly=(a.pf>b.pf or a.expectancy>b.expectancy or a.dd<b.dd or a.n>b.n)
    return better_or_equal and strictly


def pareto(rows:list[KPI])->list[KPI]:
    return [r for r in rows if not any(dominates(o,r) for o in rows if o is not r)]


def score(r:KPI,target_pf:float,target_dd:float,min_n:int)->float:
    # Hard constraint penalties first, then maximize expectancy/net quality.
    p=0.0
    if r.pf<target_pf: p += (target_pf-r.pf)*8.0
    if r.dd>target_dd: p += (r.dd-target_dd)*1.5
    if r.n<min_n: p += (min_n-r.n)/max(min_n,1)*4.0
    # positive terms
    return r.expectancy + 0.35*r.pf - 0.08*r.dd + 0.001*r.n - p


def finite_sensitivity(rows:list[KPI], key:str, metric:str):
    pts=[]
    for r in rows:
        if not r.params or key not in r.params: continue
        try: pts.append((float(r.params[key]), float(getattr(r,metric))))
        except Exception: pass
    pts=sorted(set(pts))
    res=[]
    for (x1,y1),(x2,y2) in zip(pts,pts[1:]):
        if x2!=x1:
            res.append({'from':x1,'to':x2,'slope':(y2-y1)/(x2-x1)})
    return res


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True,help='Directory containing APEX cell json files')
    ap.add_argument('--target-pf',type=float,default=1.5)
    ap.add_argument('--target-dd',type=float,default=5.0)
    ap.add_argument('--min-n',type=int,default=80)
    ap.add_argument('--output',default='results/armada-kpi-controller.json')
    a=ap.parse_args()

    rows=load_cells(Path(a.input).rglob('*.json'))
    if not rows: raise SystemExit('no valid KPI json cells')
    ranked=sorted(rows,key=lambda r:score(r,a.target_pf,a.target_dd,a.min_n),reverse=True)
    feasible=[r for r in rows if r.pf>=a.target_pf and r.dd<=a.target_dd and r.n>=a.min_n and r.expectancy>0]
    obj={
      'target':{'PF_min':a.target_pf,'DD_max_pct':a.target_dd,'N_min':a.min_n,'expectancy_min':0},
      'feasible_count':len(feasible),
      'best_feasible':feasible[0].__dict__ if feasible else None,
      'best_ranked':ranked[0].__dict__,
      'pareto_frontier':[r.__dict__ for r in pareto(rows)],
      'ranked':[{'variant':r.variant,'score':score(r,a.target_pf,a.target_dd,a.min_n),**r.__dict__} for r in ranked],
      'sensitivity':{
        k:{m:finite_sensitivity(rows,k,m) for m in ('pf','dd','expectancy','n')}
        for k in ('trail_atr','protect_atr','max_hold_minutes','adverse_stop_atr','min_atr')
      },
      'control_level':(
        'TARGET_HIT' if feasible else
        'PARETO_ONLY'
      ),
      'note':'TARGET_HIT means at least one tested configuration satisfies the requested hard KPI constraints. It does not prove out-of-sample control.'
    }
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(obj,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
