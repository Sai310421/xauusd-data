#!/usr/bin/env python3
"""
AMOS Inverse ML Selector v1.8
Tests the v1.7 observation that LOW model scores (previously rejected candidates)
may contain the edge.

Fixed framework is unchanged:
M15 context -> Sweep -> MSS -> Volume -> Equilibrium -> Pullback.
ML does not alter gates. It is only used as an inverse post-gate ranker.

Leakage control:
- Expanding chronological walk-forward.
- Model fit on past data only.
- Inverse threshold = median TRAIN score only.
- Each next block is untouched OOS.
"""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FEATURES=["rr_to_session_target","acc_eff","acc_range_atr","bars_from_sweep","hour_sin","hour_cos",
"m15_body_atr","m15_range_atr","m15_close_pos","m15_efficiency","m15_prev4_eff",
"m15_prev4_range_atr","amd_accumulation","sweep_depth_atr","mss_body_atr","volume_ratio","eq_pos"]

def pf(r):
 r=np.asarray(r,float);gp=r[r>0].sum();gl=-r[r<0].sum()
 return float(gp/gl) if gl>0 else (float("inf") if gp>0 else 0.)
def kpi(x):
 if len(x)==0:return {"N":0,"WR_pct":0.,"PF_R":0.,"sum_R":0.}
 return {"N":int(len(x)),"WR_pct":float(100*(x.outcome_R>0).mean()),"PF_R":pf(x.outcome_R),
         "sum_R":float(x.outcome_R.sum()),"avg_R":float(x.outcome_R.mean())}
def mdl():
 return Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
 ("clf",LogisticRegression(C=.5,class_weight="balanced",max_iter=3000,random_state=42))])

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--data",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);base=out/"candidate_build"
 subprocess.run([sys.executable,"research/amos_gate_candidate_learning_v1_6.py","--data",a.data,
                 "--out",str(base),"--target-r","1.5","--horizon","60"],check=True)
 c=pd.read_csv(base/"all_candidates.csv");c["time"]=pd.to_datetime(c.time);c=c.sort_values("time").reset_index(drop=True)
 n=len(c);starts=[int(n*.40),int(n*.55),int(n*.70),int(n*.85)]
 rows=[];parts=[]
 for fi,start in enumerate(starts,1):
  end=starts[fi] if fi<len(starts) else n
  tr=c.iloc[:start].copy();te=c.iloc[start:end].copy()
  if len(te)==0 or tr.label.nunique()<2:continue
  m=mdl();m.fit(tr[FEATURES],tr.label)
  trscore=m.predict_proba(tr[FEATURES])[:,1];te["score"]=m.predict_proba(te[FEATURES])[:,1]
  th=float(np.quantile(trscore,.50))
  # INVERSE: low score is accepted; exact complement of v1.7 selection apart from equality.
  te["inverse_selected"]=te.score<th
  te["fold"]=fi
  allk=kpi(te);inv=kpi(te[te.inverse_selected]);high=kpi(te[~te.inverse_selected])
  amd_inv=kpi(te[(te.inverse_selected)&(te.amd_accumulation==1)])
  nonamd_inv=kpi(te[(te.inverse_selected)&(te.amd_accumulation==0)])
  rows.append({"fold":fi,"train_n":len(tr),"oos_n":len(te),"threshold":th,
    "all_PF":allk["PF_R"],"inverse_N":inv["N"],"inverse_WR":inv["WR_pct"],
    "inverse_PF":inv["PF_R"],"inverse_sumR":inv["sum_R"],
    "high_N":high["N"],"high_PF":high["PF_R"],
    "inverse_AMD_N":amd_inv["N"],"inverse_AMD_PF":amd_inv["PF_R"],
    "inverse_nonAMD_N":nonamd_inv["N"],"inverse_nonAMD_PF":nonamd_inv["PF_R"]})
  parts.append(te)
 oos=pd.concat(parts,ignore_index=True)
 inv=oos[oos.inverse_selected];high=oos[~oos.inverse_selected]
 # Stability checks: no promotion from one lucky fold.
 fold_positive=sum(1 for r in rows if r["inverse_PF"]>1)
 fold_beats_high=sum(1 for r in rows if r["inverse_PF"]>r["high_PF"])
 result={
  "version":"v1.8","mode":"INVERSE_LOW_SCORE",
  "fixed_gate_order":["Liquidity Sweep","MSS","Volume Influx","Equilibrium","Pullback"],
  "candidate_N":n,"walk_forward_folds":rows,
  "pooled_oos_all":kpi(oos),"pooled_inverse_selected":kpi(inv),"pooled_high_score":kpi(high),
  "pooled_inverse_AMD":kpi(inv[inv.amd_accumulation==1]),
  "pooled_inverse_nonAMD":kpi(inv[inv.amd_accumulation==0]),
  "stability":{"folds_pf_gt_1":fold_positive,"folds_beating_high_score":fold_beats_high,"fold_count":len(rows)},
  "promotion_gate":{"min_pooled_PF":1.2,"min_N":10,"require_pf_gt_1_folds":3,
    "pass":bool(kpi(inv)["PF_R"]>=1.2 and len(inv)>=10 and fold_positive>=3)}
 }
 pd.DataFrame(rows).to_csv(out/"inverse_walkforward_folds.csv",index=False)
 oos.to_csv(out/"inverse_oos.csv",index=False)
 (out/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
