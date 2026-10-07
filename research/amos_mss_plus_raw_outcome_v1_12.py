#!/usr/bin/env python3
"""AMOS MSS+ Raw Bid/Ask outcome audit v1.12.
Consumes v1.11 completed candidates. Entry is first raw quote at/after the
next M1 bar boundary (causal). Buy enters Ask/exits Bid; sell enters Bid/exits Ask.
Frozen evaluation: -1R SL, +1.5R TP, 60 minute horizon. No threshold optimization.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick

def fpx(x):return float(x.as_double()) if hasattr(x,"as_double") else float(x)
def metrics(x):
 r=np.asarray(x.raw_R,dtype=float);pos=r[r>0].sum();neg=-r[r<0].sum()
 pf=float(pos/neg) if neg>0 else (float("inf") if pos>0 else 0.)
 eq=np.cumsum(r);peak=np.maximum.accumulate(np.r_[0.,eq])[:-1];dd=peak-eq
 # Fixed 0.35% risk/trade diagnostic equity, chronological closed-signal sequence.
 bal=1000.;pk=1000.;mdd=0.
 for v in r:
  bal*=1.+.0035*v;pk=max(pk,bal);mdd=max(mdd,(pk-bal)/pk*100)
 return {"N":int(len(r)),"wins":int((r>0).sum()),"losses":int((r<0).sum()),"timeouts":int((r==0).sum()),
         "WR_pct":float(100*(r>0).mean()) if len(r) else 0.,"PF_R":pf,"sum_R":float(r.sum()),
         "avg_R":float(r.mean()) if len(r) else 0.,"max_closed_sequence_DD_R":float(dd.max()) if len(dd) else 0.,
         "fixed_risk_0_35pct_return_pct":float((bal/1000.-1)*100),"fixed_risk_0_35pct_maxDD_pct":float(mdd)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--candidates",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);c=pd.read_csv(a.candidates);c["entry_time"]=pd.to_datetime(c.entry_time)
 cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
 ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
 ts=np.fromiter((int(x.ts_event) for x in ticks),dtype=np.int64,count=len(ticks))
 bid=np.fromiter((fpx(x.bid_price) for x in ticks),dtype=float,count=len(ticks));ask=np.fromiter((fpx(x.ask_price) for x in ticks),dtype=float,count=len(ticks))
 rows=[]
 for _,q in c.sort_values("entry_time").iterrows():
  tns=int(pd.Timestamp(q.entry_time).value);j=int(np.searchsorted(ts,tns,side="left"));di=int(q["dir"]);stop=float(q.stop)
  if j>=len(ts):continue
  entry=ask[j] if di>0 else bid[j];risk=(entry-stop)*di
  if risk<=0:
   rows.append({**q.to_dict(),"raw_entry":entry,"raw_exit":entry,"raw_R":0.,"reason":"BADRISK","risk_price":risk});continue
  target=entry+di*1.5*risk;deadline=tns+60*60*1_000_000_000;k=j;res=0.;reason="TIMEOUT";xp=bid[j] if di>0 else ask[j]
  mfe=mae=0.
  while k<len(ts) and ts[k]<=deadline:
   ex=bid[k] if di>0 else ask[k];fav=(ex-entry)*di/risk;adv=(entry-ex)*di/risk;mfe=max(mfe,fav);mae=max(mae,adv)
   if (ex<=stop if di>0 else ex>=stop):res=-1.;reason="SL";xp=ex;break
   if (ex>=target if di>0 else ex<=target):res=1.5;reason="TP";xp=ex;break
   xp=ex;k+=1
  rows.append({**q.to_dict(),"raw_entry":entry,"raw_exit":xp,"target":target,"risk_price":risk,"raw_R":res,"reason":reason,"MFE_R":mfe,"MAE_R":mae})
 x=pd.DataFrame(rows);x.to_csv(out/"raw_outcomes.csv",index=False)
 bypoi={str(k):metrics(v) for k,v in x.groupby("poi_type")} if len(x) else {}
 result={"version":"v1.12","verification":"RAW_BIDASK_SIGNAL_OUTCOME_AUDIT","gate_source":"v1.11 corrected prior MSS+ composite",
  "evaluation":{"entry":"first raw quote >= completed pullback bar close","buy":"Ask entry / Bid exit","sell":"Bid entry / Ask exit","TP_R":1.5,"SL_R":-1.0,"horizon_minutes":60,
                "explicit_commission_slippage":"none beyond observed raw spread"},
  "overall":metrics(x) if len(x) else metrics(pd.DataFrame({"raw_R":[]})),"by_POI":bypoi,
  "limitations":["DD is closed-signal sequence DD, not a concurrent-position Nautilus portfolio DD.","Fixed 0.35% risk figures are diagnostic compounding, not broker margin simulation.","This validates the current M1-based MSS+ approximation; M15 AMD -> M5 checklist -> M1 refinement remains a separate architecture integration."]}
 (out/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
