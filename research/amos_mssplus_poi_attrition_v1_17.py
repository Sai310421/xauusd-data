#!/usr/bin/env python3
"""v1.17 diagnostic only: explain POI-revisit -> entry attrition without changing v1.16 rules."""
from __future__ import annotations
import argparse,json,importlib.util,sys
from collections import Counter
from pathlib import Path
import numpy as np,pandas as pd
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick
spec=importlib.util.spec_from_file_location("v16",Path(__file__).with_name("amos_mssplus_destination_poi_v1_16.py"))
m=importlib.util.module_from_spec(spec);sys.modules["v16"]=m;spec.loader.exec_module(m)

def audit(ctx,st,z,raw,tf):
 s=Counter();rows=[];raw_ns=raw.ns.to_numpy();max_wait=pd.Timedelta(minutes=60 if tf==1 else 240)
 for cid,cx in ctx.reset_index(drop=True).iterrows():
  t0=pd.Timestamp(cx.mssplus_time);t1=t0+max_wait;travel=int(cx.travel_dir);entry_dir=-travel
  cand=st[(st.dir==entry_dir)&(st.signal_time>=t0)&(st.signal_time<=t1)].sort_values("signal_time")
  cs=Counter();entered=False
  for _,p in cand.iterrows():
   ahead=(float(p.poi)<=float(cx.mssplus_close)) if travel<0 else (float(p.poi)>=float(cx.mssplus_close))
   if not ahead:cs["off_path"]+=1;continue
   cs["linked"]+=1;start=pd.Timestamp(p.signal_time);expiry=min(t1,start+pd.Timedelta(minutes=180 if tf==1 else 720))
   q=z[(z.datetime>=start)&(z.datetime<=expiry)];touched=False;terminal=None
   for k in range(1,len(q)):
    b=q.iloc[k];prev=q.iloc[k-1];tol=.10*float(p.atr)
    invalid=(float(b.low)<=float(p.sl)) if entry_dir>0 else (float(b.high)>=float(p.sl))
    if invalid:
     terminal="invalid_after_revisit" if touched else "invalid_before_revisit";break
    if not touched:
     touch=(float(b.low)<=float(p.poi)+tol) if entry_dir>0 else (float(b.high)>=float(p.poi)-tol)
     if touch:touched=True;cs["revisit"]+=1
    if touched:
     confirm=(float(b.close)>float(p.poi) and float(b.close)>float(prev.close)) if entry_dir>0 else (float(b.close)<float(p.poi) and float(b.close)<float(prev.close))
     if confirm:
      ce=pd.Timestamp(b.datetime)+pd.Timedelta(minutes=tf);j=int(np.searchsorted(raw_ns,int(ce.value),side="left"))
      if j>=len(raw):terminal="no_raw_quote";break
      first=raw.iloc[j];entry=float(first.ask if entry_dir>0 else first.bid);risk=(entry-float(p.sl))*entry_dir
      if risk<=0:terminal="bad_risk";break
      terminal="entry";entered=True;cs["entry"]+=1;break
   if terminal is None:terminal="revisit_no_close_confirm" if touched else "no_revisit_before_expiry"
   cs[terminal]+=1
   if entered:break
  if entered:s["contexts_with_entry"]+=1
  else:s["contexts_without_entry"]+=1
  if cs["linked"]>0:s["contexts_with_linked_poi"]+=1
  if cs["revisit"]>0:s["contexts_with_any_revisit"]+=1
  if cs["revisit"]>1:s["contexts_with_multiple_revisits"]+=1
  for k,v in cs.items():s["candidate_"+k]+=v
  rows.append({"context_id":cid,"mssplus_time":t0,**dict(cs)})
 return s,pd.DataFrame(rows)

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--out",required=True);ap.add_argument("--tf",type=int,choices=[1,15],default=1);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
 ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value]);ticks.sort(key=lambda x:int(x.ts_event))
 ts=np.fromiter((int(x.ts_event) for x in ticks),dtype=np.int64,count=len(ticks));bid=np.fromiter((m.v14.fpx(x.bid_price) for x in ticks),float,count=len(ticks));ask=np.fromiter((m.v14.fpx(x.ask_price) for x in ticks),float,count=len(ticks))
 raw=pd.DataFrame({"datetime":pd.to_datetime(ts,unit="ns"),"ns":ts,"bid":bid,"ask":ask});z=m.v14.feature(m.v14.bars(raw,a.tf));ctx,_=m.mssplus_context(z,a.tf);st=m.add_targets(m.line_setups(z),z,a.tf)
 stats,detail=audit(ctx,st,z,raw,a.tf);detail.to_csv(out/"context_attrition.csv",index=False)
 result={"version":"v1.17","diagnostic_only":True,"logic_change":False,"tf":a.tf,"mssplus_contexts":len(ctx),"stats":dict(stats),
 "note":"Counts separate candidate-level POI revisits from unique MSS+ contexts. v1.16 entry rules are unchanged."}
 (out/"result.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8");print(json.dumps(result,indent=2,default=str))
if __name__=="__main__":main()
