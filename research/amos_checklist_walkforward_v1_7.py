#!/usr/bin/env python3
"""
AMOS Checklist Walk-Forward v1.7
Robustness audit only. Does not alter the fixed video gates or live entry logic.
Runs v1.6 candidate builder, then expanding-window chronological folds.
ML may rank only after all fixed gates pass.
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
 return {"N":int(len(x)),"WR_pct":float(100*(x.outcome_R>0).mean()),"PF_R":pf(x.outcome_R),"sum_R":float(x.outcome_R.sum())}
def model():
 return Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
 ("clf",LogisticRegression(C=.5,class_weight="balanced",max_iter=3000,random_state=42))])
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--data",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);base=out/"candidate_build"
 subprocess.run([sys.executable,"research/amos_gate_candidate_learning_v1_6.py","--data",a.data,
                 "--out",str(base),"--target-r","1.5","--horizon","60"],check=True)
 c=pd.read_csv(base/"all_candidates.csv");c["time"]=pd.to_datetime(c.time);c=c.sort_values("time").reset_index(drop=True)
 n=len(c)
 # Expanding train, next chronological block OOS. Threshold always train median; never tuned on fold OOS.
 starts=[int(n*.40),int(n*.55),int(n*.70),int(n*.85)]
 rows=[];oos_parts=[]
 for fi,start in enumerate(starts,1):
  end=starts[fi] if fi<len(starts) else n
  tr=c.iloc[:start].copy();te=c.iloc[start:end].copy()
  if len(te)==0 or tr.label.nunique()<2:continue
  m=model();m.fit(tr[FEATURES],tr.label)
  trscore=m.predict_proba(tr[FEATURES])[:,1];tescore=m.predict_proba(te[FEATURES])[:,1]
  th=float(np.quantile(trscore,.50));te["score"]=tescore;te["selected"]=te.score>=th;te["fold"]=fi
  ka=kpi(te);ks=kpi(te[te.selected]);kr=kpi(te[~te.selected])
  rows.append({"fold":fi,"train_n":len(tr),"oos_n":len(te),"threshold":th,
               "all_PF":ka["PF_R"],"all_WR":ka["WR_pct"],"selected_N":ks["N"],
               "selected_PF":ks["PF_R"],"selected_WR":ks["WR_pct"],"selected_sumR":ks["sum_R"],
               "rejected_N":kr["N"],"rejected_PF":kr["PF_R"]})
  oos_parts.append(te)
 wf=pd.DataFrame(rows);oos=pd.concat(oos_parts,ignore_index=True) if oos_parts else pd.DataFrame()
 result={"purpose":"robustness audit; no gate/live logic changes",
 "fixed_gate_order":["Liquidity Sweep","MSS","Volume Influx","Equilibrium","Pullback"],
 "candidate_N":n,"folds":rows,
 "pooled_oos_all":kpi(oos) if len(oos) else {},
 "pooled_oos_selected":kpi(oos[oos.selected]) if len(oos) else {},
 "pooled_oos_rejected":kpi(oos[~oos.selected]) if len(oos) else {},
 "promotion_rule":"Do not promote ML unless pooled selected PF improves vs pooled all, N remains material, and fold direction is stable."}
 wf.to_csv(out/"walk_forward_folds.csv",index=False)
 if len(oos):oos.to_csv(out/"walk_forward_oos.csv",index=False)
 (out/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
