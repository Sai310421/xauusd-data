#!/usr/bin/env python3
"""v1.19 corrected MSS+ -> POI direction.
Bullish MSS+ => BUY setup, destination POI BELOW MSS+ close, buy on revisit.
Bearish MSS+ => SELL setup, destination POI ABOVE MSS+ close, sell on revisit.
Runs BOTH all-touch and Close-confirm variants from the same corrected population.
"""
from __future__ import annotations
import argparse,json,importlib.util,sys
from collections import Counter
from pathlib import Path
import numpy as np,pandas as pd
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick
spec=importlib.util.spec_from_file_location("v16",Path(__file__).with_name("amos_mssplus_destination_poi_v1_16.py"))
m=importlib.util.module_from_spec(spec);sys.modules["v16"]=m;spec.loader.exec_module(m)

def entries(ctx,st,z,raw,tf,confirm):
 out=[];stats=Counter();raw_ns=raw.ns.to_numpy();max_wait=pd.Timedelta(minutes=60 if tf==1 else 240)
 for cid,cx in ctx.reset_index(drop=True).iterrows():
  t0=pd.Timestamp(cx.mssplus_time);t1=t0+max_wait;di=int(cx.travel_dir) # CORRECT: MSS+ direction == entry direction
  cand=st[(st.dir==di)&(st.signal_time>=t0)&(st.signal_time<=t1)].sort_values("signal_time")
  for sid,s in cand.iterrows():
   # CORRECT destination: bullish MSS+ retraces DOWN to BUY POI; bearish retraces UP to SELL POI.
   in_path=(float(s.poi)<=float(cx.mssplus_close)) if di>0 else (float(s.poi)>=float(cx.mssplus_close))
   if not in_path:stats["reject_poi_wrong_side"]+=1;continue
   stats["linked_destination_poi"]+=1
   start=pd.Timestamp(s.signal_time);expiry=min(t1,start+pd.Timedelta(minutes=180 if tf==1 else 720))
   q=z[(z.datetime>=start)&(z.datetime<=expiry)];touched=False
   for k in range(1,len(q)):
    b=q.iloc[k];prev=q.iloc[k-1];tol=.10*float(s.atr)
    invalid=(float(b.low)<=float(s.sl)) if di>0 else (float(b.high)>=float(s.sl))
    if invalid:stats["invalid_before_entry"]+=1;break
    if not touched:
     touch=(float(b.low)<=float(s.poi)+tol) if di>0 else (float(b.high)>=float(s.poi)-tol)
     if not touch:continue
     touched=True;stats["poi_revisit"]+=1
     if not confirm: trigger=b; trigger_time=pd.Timestamp(b.datetime)+pd.Timedelta(minutes=tf)
    if touched and confirm:
     ok=(float(b.close)>float(s.poi) and float(b.close)>float(prev.close)) if di>0 else (float(b.close)<float(s.poi) and float(b.close)<float(prev.close))
     if not ok:continue
     trigger=b;trigger_time=pd.Timestamp(b.datetime)+pd.Timedelta(minutes=tf)
    if touched and (not confirm or ok):
     j=int(np.searchsorted(raw_ns,int(trigger_time.value),side="left"))
     if j>=len(raw):stats["no_raw_quote"]+=1;break
     first=raw.iloc[j];ep=float(first.ask if di>0 else first.bid);risk=(ep-float(s.sl))*di
     if risk<=0:stats["bad_risk"]+=1;break
     tp=float(s.target)
     if not np.isfinite(tp) or (tp-ep)*di<=0:tp=ep+di*2*risk
     out.append({"tf":tf,"context_id":cid,"setup_id":int(sid),"mssplus_time":t0,"mssplus_dir":di,
       "pattern":str(s.pattern),"entry_dir":di,"poi":float(s.poi),"poi_signal_time":pd.Timestamp(s.signal_time),
       "entry_ns":int(first.ns),"entry_time":first.datetime,"entry":ep,"sl":float(s.sl),"tp":tp,"risk":risk})
     stats["entries"]+=1;stats["pattern_"+str(s.pattern)]+=1
     break
 return pd.DataFrame(out),stats

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--out",required=True);ap.add_argument("--tf",type=int,choices=[1,15],required=True);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
 ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value]);ticks.sort(key=lambda x:int(x.ts_event))
 ts=np.fromiter((int(x.ts_event) for x in ticks),dtype=np.int64,count=len(ticks));bid=np.fromiter((m.v14.fpx(x.bid_price) for x in ticks),float,count=len(ticks));ask=np.fromiter((m.v14.fpx(x.ask_price) for x in ticks),float,count=len(ticks))
 raw=pd.DataFrame({"datetime":pd.to_datetime(ts,unit="ns"),"ns":ts,"bid":bid,"ask":ask});z=m.v14.feature(m.v14.bars(raw,a.tf));ctx,counts=m.mssplus_context(z,a.tf);st=m.add_targets(m.line_setups(z),z,a.tf)
 result={"version":"v1.19","correction":"MSS+ direction equals entry direction. Bullish -> lower BUY POI -> BUY; bearish -> upper SELL POI -> SELL.","tf":a.tf,"raw_ticks":len(raw),"mssplus_counts":counts}
 for mode,confirm in [("ALL_TOUCH",False),("CLOSE_CONFIRM",True)]:
  e,s=entries(ctx,st,z,raw,a.tf,confirm);tr=m.simulate(e,raw,a.tf);e.to_csv(out/f"{mode.lower()}_entries.csv",index=False);tr.to_csv(out/f"{mode.lower()}_trades.csv",index=False)
  result[mode]={"stats":dict(s),**m.metrics(tr),"pattern_metrics":{k:m.metrics(v) for k,v in tr.groupby("pattern")} if len(tr) else {}}
 (out/"result.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8");print(json.dumps(result,indent=2,default=str))
if __name__=="__main__":main()
