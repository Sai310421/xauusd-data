#!/usr/bin/env python3
"""
AMOS Inverse Selector v1.9 - External Raw Bid/Ask OOS
Train: historical Feb-May checklist candidates only.
Test: cached Dukascopy XAUUSD Raw QuoteTicks (future July-Aug period).
Fixed gates unchanged: Sweep -> MSS -> Volume -> Equilibrium -> Pullback.
Inverse ML = accept score below TRAIN median only.
Raw test labels use first raw quote after candidate and Bid/Ask executable side.
"""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from nautilus_trader.persistence.catalog import ParquetDataCatalog

FEATURES=["rr_to_session_target","acc_eff","acc_range_atr","bars_from_sweep","hour_sin","hour_cos",
"m15_body_atr","m15_range_atr","m15_close_pos","m15_efficiency","m15_prev4_eff",
"m15_prev4_range_atr","amd_accumulation","sweep_depth_atr","mss_body_atr","volume_ratio","eq_pos"]

def fpx(x): return float(x.as_double()) if hasattr(x,"as_double") else float(x)
def pf(r):
 r=np.asarray(r,float);gp=r[r>0].sum();gl=-r[r<0].sum()
 return float(gp/gl) if gl>0 else (float("inf") if gp>0 else 0.)
def kpi(x):
 if len(x)==0:return {"N":0,"WR_pct":0.,"PF_R":0.,"sum_R":0.}
 return {"N":int(len(x)),"WR_pct":float(100*(x.raw_R>0).mean()),"PF_R":pf(x.raw_R),
         "sum_R":float(x.raw_R.sum()),"avg_R":float(x.raw_R.mean())}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--history",required=True);ap.add_argument("--out",required=True)
 a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 cat=ParquetDataCatalog(a.catalog)
 inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
 ticks=cat.query_quote_ticks(identifiers=[inst.id.value])
 if not ticks:raise SystemExit("no XAUUSD raw quote ticks")
 ts=np.fromiter((int(x.ts_event) for x in ticks),dtype=np.int64,count=len(ticks))
 bid=np.fromiter((fpx(x.bid_price) for x in ticks),dtype=np.float64,count=len(ticks))
 ask=np.fromiter((fpx(x.ask_price) for x in ticks),dtype=np.float64,count=len(ticks))
 start=pd.Timestamp(ts[0],unit="ns");end=pd.Timestamp(ts[-1],unit="ns")
 # Causal M1 BID bars built directly from raw quotes; volume = raw quote count.
 raw=pd.DataFrame({"datetime":pd.to_datetime(ts,unit="ns"),"bid":bid})
 g=raw.set_index("datetime").bid.resample("1min")
 bars=pd.DataFrame({"open":g.first(),"high":g.max(),"low":g.min(),"close":g.last(),"volume":g.count()}).dropna().reset_index()
 bars.to_csv(out/"raw_m1_bid_bars.csv",index=False)

 hist=pd.read_csv(a.history);hist.columns=[c.lower() for c in hist.columns];hist["datetime"]=pd.to_datetime(hist.datetime)
 hist=hist[hist.datetime<start].copy()
 cols=["datetime","open","high","low","close","volume"]
 combo=pd.concat([hist[cols],bars[cols]],ignore_index=True).sort_values("datetime")
 combo.to_csv(out/"combined_for_candidate_detection.csv",index=False)

 # Reuse frozen v1.6 candidate detector; its internal split is ignored here.
 base=out/"candidate_detection"
 subprocess.run([sys.executable,"research/amos_gate_candidate_learning_v1_6.py","--data",str(out/"combined_for_candidate_detection.csv"),
                 "--out",str(base),"--target-r","1.5","--horizon","60"],check=True)
 c=pd.read_csv(base/"all_candidates.csv");c["time"]=pd.to_datetime(c.time)
 tr=c[c.time<start].copy();te=c[(c.time>=start)&(c.time<=end)].copy()
 if len(tr)<30 or tr.label.nunique()<2:raise SystemExit(f"insufficient historical train candidates {len(tr)}")
 if len(te)==0:raise SystemExit("zero future raw candidates")

 m=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
             ("clf",LogisticRegression(C=.5,class_weight="balanced",max_iter=3000,random_state=42))])
 m.fit(tr[FEATURES],tr.label);trscore=m.predict_proba(tr[FEATURES])[:,1]
 threshold=float(np.quantile(trscore,.50));te["score"]=m.predict_proba(te[FEATURES])[:,1]
 te["inverse_selected"]=te.score<threshold

 # Re-label future candidates on actual raw executable Bid/Ask quotes.
 rawR=[];rawEntry=[];rawExit=[];rawReason=[];rawMFE=[];rawMAE=[]
 for _,q in te.iterrows():
  tns=int(pd.Timestamp(q.time).value);j=int(np.searchsorted(ts,tns,side="left"))
  if j>=len(ts): rawR.append(0.);rawEntry.append(np.nan);rawExit.append(np.nan);rawReason.append("NOQUOTE");rawMFE.append(0.);rawMAE.append(0.);continue
  di=int(q["dir"]); entry=ask[j] if di>0 else bid[j]; stop=float(q.stop)
  risk=abs(entry-stop)
  if risk<=0: rawR.append(0.);rawEntry.append(entry);rawExit.append(entry);rawReason.append("BADRISK");rawMFE.append(0.);rawMAE.append(0.);continue
  target=entry+di*1.5*risk;deadline=tns+60*60*1_000_000_000;k=j;res=0.;reason="TIMEOUT";xp=bid[j] if di>0 else ask[j];mfe=mae=0.
  while k<len(ts) and ts[k]<=deadline:
   ex=bid[k] if di>0 else ask[k]
   fav=(ex-entry)*di/risk;adv=(entry-ex)*di/risk;mfe=max(mfe,fav);mae=max(mae,adv)
   stophit=(ex<=stop) if di>0 else (ex>=stop);tphit=(ex>=target) if di>0 else (ex<=target)
   if stophit:res=-1.;reason="SL";xp=ex;break
   if tphit:res=1.5;reason="TP";xp=ex;break
   xp=ex;k+=1
  rawR.append(res);rawEntry.append(entry);rawExit.append(xp);rawReason.append(reason);rawMFE.append(mfe);rawMAE.append(mae)
 te["raw_entry"]=rawEntry;te["raw_exit"]=rawExit;te["raw_R"]=rawR;te["raw_reason"]=rawReason;te["raw_MFE_R"]=rawMFE;te["raw_MAE_R"]=rawMAE
 inv=te[te.inverse_selected];high=te[~te.inverse_selected]
 amd=inv[inv.amd_accumulation==1];non=inv[inv.amd_accumulation==0]
 result={"version":"v1.9","verification_level":"EXTERNAL_FUTURE_RAW_BIDASK_OOS",
  "engine":"NautilusTrader ParquetDataCatalog + causal M1 BID bars + raw Bid/Ask execution labels",
  "fixed_gate_order":["Liquidity Sweep","MSS","Volume Influx","Equilibrium","Pullback"],
  "train_period":{"end_exclusive":str(start),"N":len(tr)},"raw_test_period":{"start":str(start),"end":str(end),"raw_ticks":len(ticks),"m1_bars":len(bars)},
  "threshold":"historical TRAIN median only","threshold_value":threshold,
  "raw_oos_all":kpi(te),"raw_oos_inverse":kpi(inv),"raw_oos_high":kpi(high),
  "raw_oos_inverse_AMD":kpi(amd),"raw_oos_inverse_nonAMD":kpi(non),
  "promotion_gate":{"min_PF":1.2,"min_N":10,"pass":bool(len(inv)>=10 and kpi(inv)["PF_R"]>=1.2)},
  "limitations":["Signal features use causal M1 BID bars built from raw QuoteTicks; execution outcome uses raw Bid/Ask.","Historical training labels are M1-bar labels from the frozen v1.6 research screen; raw future test is not used for fitting or threshold selection.","No explicit commission/slippage beyond observed raw spread in this gate."]}
 te.to_csv(out/"raw_oos_candidates.csv",index=False)
 (out/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
