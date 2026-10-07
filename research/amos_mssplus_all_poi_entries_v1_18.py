#!/usr/bin/env python3
"""v1.18: enter EVERY v1.16 MSS+-linked destination POI revisit independently.
No Close-reaction confirmation and no one-entry-per-MSS+ suppression.
This is a measurement branch, not a promotion rule.
"""
from __future__ import annotations
import argparse,json,importlib.util,sys,math
from collections import Counter
from pathlib import Path
import numpy as np,pandas as pd
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick
spec=importlib.util.spec_from_file_location("v16",Path(__file__).with_name("amos_mssplus_destination_poi_v1_16.py"))
m=importlib.util.module_from_spec(spec);sys.modules["v16"]=m;spec.loader.exec_module(m)

def all_touch_entries(ctx,st,z,raw,tf):
 stats=Counter();out=[];raw_ns=raw.ns.to_numpy();max_wait=pd.Timedelta(minutes=60 if tf==1 else 240)
 for cid,cx in ctx.reset_index(drop=True).iterrows():
  t0=pd.Timestamp(cx.mssplus_time);t1=t0+max_wait;travel=int(cx.travel_dir);di=-travel
  cand=st[(st.dir==di)&(st.signal_time>=t0)&(st.signal_time<=t1)].sort_values("signal_time")
  for sid,s in cand.iterrows():
   ahead=(float(s.poi)<=float(cx.mssplus_close)) if travel<0 else (float(s.poi)>=float(cx.mssplus_close))
   if not ahead:stats["reject_off_path"]+=1;continue
   stats["linked_destination_poi"]+=1
   start=pd.Timestamp(s.signal_time);expiry=min(t1,start+pd.Timedelta(minutes=180 if tf==1 else 720))
   q=z[(z.datetime>=start)&(z.datetime<=expiry)]
   for k in range(1,len(q)):
    b=q.iloc[k];tol=.10*float(s.atr)
    invalid=(float(b.low)<=float(s.sl)) if di>0 else (float(b.high)>=float(s.sl))
    if invalid:stats["invalid_before_touch"]+=1;break
    touch=(float(b.low)<=float(s.poi)+tol) if di>0 else (float(b.high)>=float(s.poi)-tol)
    if not touch:continue
    stats["poi_revisit"]+=1
    # Preserve v1.16 causal touch semantics: touch is known at this TF bar close.
    touch_known=pd.Timestamp(b.datetime)+pd.Timedelta(minutes=tf)
    j=int(np.searchsorted(raw_ns,int(touch_known.value),side="left"))
    if j>=len(raw):stats["no_raw_quote"]+=1;break
    first=raw.iloc[j];entry=float(first.ask if di>0 else first.bid);risk=(entry-float(s.sl))*di
    if risk<=0:stats["bad_risk"]+=1;break
    target=float(s.target)
    if not np.isfinite(target) or (target-entry)*di<=0:target=entry+di*2*risk
    out.append({"tf":tf,"context_id":cid,"setup_id":int(sid),"mssplus_time":t0,"travel_dir":travel,
      "pattern":str(s.pattern),"entry_dir":di,"poi":float(s.poi),"poi_signal_time":pd.Timestamp(s.signal_time),
      "touch_bar_time":pd.Timestamp(b.datetime),"entry_ns":int(first.ns),"entry_time":first.datetime,
      "entry":entry,"sl":float(s.sl),"tp":float(target),"risk":float(risk)})
    stats["entries"]+=1;stats["pattern_"+str(s.pattern)]+=1
    break
 return pd.DataFrame(out),stats

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--out",required=True);ap.add_argument("--tf",type=int,choices=[1,15],required=True);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
 ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value]);ticks.sort(key=lambda x:int(x.ts_event))
 ts=np.fromiter((int(x.ts_event) for x in ticks),dtype=np.int64,count=len(ticks));bid=np.fromiter((m.v14.fpx(x.bid_price) for x in ticks),float,count=len(ticks));ask=np.fromiter((m.v14.fpx(x.ask_price) for x in ticks),float,count=len(ticks))
 raw=pd.DataFrame({"datetime":pd.to_datetime(ts,unit="ns"),"ns":ts,"bid":bid,"ask":ask});z=m.v14.feature(m.v14.bars(raw,a.tf));ctx,counts=m.mssplus_context(z,a.tf);st=m.add_targets(m.line_setups(z),z,a.tf)
 e,stats=all_touch_entries(ctx,st,z,raw,a.tf);tr=m.simulate(e,raw,a.tf)
 e.to_csv(out/"all_touch_entries.csv",index=False);tr.to_csv(out/"all_touch_trades.csv",index=False)
 overall=m.metrics(tr);by={k:m.metrics(v) for k,v in tr.groupby("pattern")} if len(tr) else {}
 result={"version":"v1.18","mode":"ALL_LINKED_POI_REVISITS_ENTER","logic":"same v1.16 MSS+ and destination-POI linkage; every qualifying POI revisit is an independent entry; Close confirmation removed; multiple entries per MSS+ allowed",
 "tf":a.tf,"raw_ticks":len(raw),"mssplus_counts":counts,"line_setups":len(st),"stats":dict(stats),**overall,"pattern_metrics":by,
 "measurement_note":"This branch measures all v1.16-style POI revisits. Touch becomes causal at the completed TF bar; execution is first subsequent raw Bid/Ask quote."}
 (out/"result.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8");print(json.dumps(result,indent=2,default=str))
if __name__=="__main__":main()
