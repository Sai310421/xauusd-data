#!/usr/bin/env python3
"""AMOS MSS+ -> destination POI standalone v1.16 (M1/M15).
Correction: POI is NOT detected as an independent entry engine.
Order is:
  TF-local Sweep -> CISD -> MSS -> Displacement -> MSS+
  -> MSS+ travels toward a line-structure POI
  -> destination V/A/QM/OCL reversal structure
  -> POI revisit -> favorable Close reaction -> first subsequent raw Bid/Ask entry.

The MSS+ travel direction is opposite the reversal entry direction:
bearish MSS+ -> BUY POI below; bullish MSS+ -> SELL POI above.
M5 is not used.
"""
from __future__ import annotations
import argparse,json,math,importlib.util,sys
from collections import Counter
from pathlib import Path
import numpy as np,pandas as pd
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick

spec=importlib.util.spec_from_file_location("v14",Path(__file__).with_name("amos_m5_mssplus_m1_line_entry_v1_14.py"))
v14=importlib.util.module_from_spec(spec);sys.modules["v14"]=v14;spec.loader.exec_module(v14)
INITIAL=1000.;QTY_OZ=1.;COMMISSION_RT_PER_LOT=7.;CASHBACK_RT_PER_LOT=6.

def mssplus_context(z,tf):
 c=Counter();states={1:None,-1:None};out=[]
 for i in range(30,len(z)):
  r=z.iloc[i];p=z.iloc[i-1]
  if not np.isfinite(r.atr) or r.atr<=0:continue
  for side in (1,-1):
   s=states[side]
   if s and i-s["sweep_i"]>28:states[side]=None
  bull=bool(r.low<r.swing_lo and r.close>r.swing_lo);bear=bool(r.high>r.swing_hi and r.close<r.swing_hi)
  if bull:
   states[1]={"sweep_i":i,"extreme":float(r.low),"mss_ref":float(z.high.iloc[max(0,i-3):i].max()),"cisd_i":None,"mss_i":None,"disp_seen":False};c["sweep"]+=1
  if bear:
   states[-1]={"sweep_i":i,"extreme":float(r.high),"mss_ref":float(z.low.iloc[max(0,i-3):i].min()),"cisd_i":None,"mss_i":None,"disp_seen":False};c["sweep"]+=1
  for side in (1,-1):
   s=states[side]
   if not s:continue
   if s["cisd_i"] is None:
    if i>s["sweep_i"] and i-s["sweep_i"]<=8:
     hit=(r.close>p.open and r.close>p.close) if side==1 else (r.close<p.open and r.close<p.close)
     if hit:s["cisd_i"]=i;c["cisd"]+=1
    elif i-s["sweep_i"]>8:states[side]=None
    continue
   if s["mss_i"] is None:
    if i-s["cisd_i"]>12:states[side]=None;continue
    if i<=s["cisd_i"]:continue
    hit=(r.close>s["mss_ref"]) if side==1 else (r.close<s["mss_ref"])
    if hit:s["mss_i"]=i;c["mss"]+=1
    else:continue
   if i<s["mss_i"] or i-s["mss_i"]>4:
    if i-s["mss_i"]>4:states[side]=None
    continue
   if float(r.disp)<.65:continue
   if not s["disp_seen"]:c["displacement"]+=1;s["disp_seen"]=True
   structural_poi=v14.poi_zone(z,i,side)
   if structural_poi is None:continue
   c["mss_plus"]+=1
   out.append({"mssplus_time":pd.Timestamp(r.datetime)+pd.Timedelta(minutes=tf),"travel_dir":int(side),
               "mssplus_close":float(r.close),"sweep_extreme":float(s["extreme"]),
               "mssplus_poi_type":str(structural_poi[2]),"mssplus_i":i})
   states[side]=None
 return pd.DataFrame(out),dict(c)

def line_setups(z):
 return v14.m1_setups(z)

def linked_entries(ctx,st,z,raw,tf):
 stats=Counter();out=[]
 if ctx.empty or st.empty:return pd.DataFrame(),stats
 raw_ns=raw.ns.to_numpy()
 max_wait=pd.Timedelta(minutes=60 if tf==1 else 240)
 setup_look=pd.Timedelta(minutes=0) # destination structure must appear after MSS+
 for _,cx in ctx.iterrows():
  t0=pd.Timestamp(cx.mssplus_time);t1=t0+max_wait;travel=int(cx.travel_dir);entry_dir=-travel
  cand=st[(st.dir==entry_dir)&(st.signal_time>=t0-setup_look)&(st.signal_time<=t1)].copy()
  stats["mssplus_context"]+=1
  for _,s in cand.sort_values("signal_time").iterrows():
   # Destination geometry: bearish MSS+ must travel down into BUY POI; bullish MSS+ up into SELL POI.
   ahead=(float(s.poi)<=float(cx.mssplus_close)) if travel<0 else (float(s.poi)>=float(cx.mssplus_close))
   if not ahead:
    stats["reject_poi_not_in_mssplus_path"]+=1;continue
   stats["linked_destination_poi"]+=1
   start=pd.Timestamp(s.signal_time);expiry=min(t1,start+pd.Timedelta(minutes=180 if tf==1 else 720))
   q=z[(z.datetime>=start)&(z.datetime<=expiry)]
   touched=False;touch_time=None
   for k in range(1,len(q)):
    b=q.iloc[k];prev=q.iloc[k-1];tol=.10*float(s.atr)
    invalid=(float(b.low)<=float(s.sl)) if entry_dir>0 else (float(b.high)>=float(s.sl))
    if invalid:stats["invalid_before_entry"]+=1;break
    if not touched:
     touch=(float(b.low)<=float(s.poi)+tol) if entry_dir>0 else (float(b.high)>=float(s.poi)-tol)
     if touch:touched=True;touch_time=pd.Timestamp(b.datetime);stats["poi_revisit"]+=1
    if touched:
     confirm=(float(b.close)>float(s.poi) and float(b.close)>float(prev.close)) if entry_dir>0 else (float(b.close)<float(s.poi) and float(b.close)<float(prev.close))
     if confirm:
      confirm_end=pd.Timestamp(b.datetime)+pd.Timedelta(minutes=tf);j=int(np.searchsorted(raw_ns,int(confirm_end.value),side="left"))
      if j>=len(raw):break
      first=raw.iloc[j];entry=float(first.ask if entry_dir>0 else first.bid);risk=(entry-float(s.sl))*entry_dir
      if risk<=0:stats["bad_risk"]+=1;break
      target=float(s.get("target",np.nan)) if "target" in s.index else np.nan
      if not np.isfinite(target) or (target-entry)*entry_dir<=0:target=entry+entry_dir*2*risk
      out.append({"tf":tf,"mssplus_time":t0,"travel_dir":travel,"mssplus_close":float(cx.mssplus_close),
       "pattern":str(s.pattern),"entry_dir":entry_dir,"poi":float(s.poi),"poi_signal_time":pd.Timestamp(s.signal_time),
       "poi_revisit_time":touch_time,"confirm_time":confirm_end,"entry_ns":int(first.ns),"entry_time":first.datetime,
       "entry":entry,"sl":float(s.sl),"tp":float(target),"risk":float(risk),"mssplus_poi_type":cx.mssplus_poi_type})
      stats["entries"]+=1;stats["pattern_"+str(s.pattern)]+=1
      break
   if stats["entries"] and len(out) and out[-1]["mssplus_time"]==t0:break
  else:stats["no_linked_entry"]+=1
 return pd.DataFrame(out),stats

def add_targets(st,z,tf):
 # Restore v19 target semantics: prior opposite Close pivot where available, otherwise 2 ATR fallback.
 # m1_setups did not retain target, so reconstruct conservatively from preceding close extrema.
 if st.empty:return st
 st=st.copy();targets=[]
 for _,s in st.iterrows():
  i=int(np.searchsorted(z.datetime.to_numpy(),np.datetime64(s.signal_time),side="right")-1)
  look=z.iloc[max(0,i-40):i]
  if int(s.dir)>0:
   opp=float(look.close.max()) if len(look) else float(s.poi)+2*float(s.atr)
   targets.append(max(opp,float(s.poi)+2*float(s.atr)))
  else:
   opp=float(look.close.min()) if len(look) else float(s.poi)-2*float(s.atr)
   targets.append(min(opp,float(s.poi)-2*float(s.atr)))
 st["target"]=targets;return st

def simulate(entries,raw,tf):
 if entries.empty:return pd.DataFrame()
 ns=raw.ns.to_numpy();rows=[];timeout_ns=int((240 if tf==1 else 960)*60*1e9)
 for _,e in entries.sort_values("entry_ns").iterrows():
  j=int(np.searchsorted(ns,int(e.entry_ns),side="left"));di=int(e.entry_dir);entry=float(e.entry);sl=float(e.sl);tp=float(e.tp)
  deadline=int(e.entry_ns)+timeout_ns;k=j;xp=entry;reason="TIMEOUT"
  while k<len(raw) and ns[k]<=deadline:
   px=float(raw.bid.iat[k] if di>0 else raw.ask.iat[k])
   if (px<=sl if di>0 else px>=sl):xp=px;reason="SL";break
   if (px>=tp if di>0 else px<=tp):xp=px;reason="TP";break
   xp=px;k+=1
  exit_ns=int(ns[min(k,len(raw)-1)])
  gross=(xp-entry)*di*QTY_OZ;cost=QTY_OZ/100*COMMISSION_RT_PER_LOT;cb=QTY_OZ/100*CASHBACK_RT_PER_LOT
  rows.append({**e.to_dict(),"exit_ns":exit_ns,"exit_time":pd.Timestamp(exit_ns,unit="ns"),"exit":xp,"reason":reason,"pnl":gross-cost+cb})
 return pd.DataFrame(rows)

def metrics(x):
 if x.empty:return {"N":0,"WR_pct":0.,"PF":0.,"Net_USD":0.,"Return_pct":0.,"MaxClosedDD_USD":0.,"RF":None}
 p=x.pnl.to_numpy(float);gp=float(p[p>0].sum());gl=float(-p[p<0].sum());eq=INITIAL;pk=INITIAL;dd=0.
 for v in p: eq+=v;pk=max(pk,eq);dd=max(dd,pk-eq)
 net=float(p.sum());return {"N":len(p),"WR_pct":float((p>0).mean()*100),"PF":gp/gl if gl>0 else (math.inf if gp>0 else 0.),
  "Net_USD":net,"Return_pct":net/INITIAL*100,"MaxClosedDD_USD":dd,"MaxClosedDD_pct_initial":dd/INITIAL*100,"RF":net/dd if dd>0 else None}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--out",required=True);ap.add_argument("--tf",type=int,choices=[1,15],required=True);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
 ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value]);ticks.sort(key=lambda x:int(x.ts_event))
 ts=np.fromiter((int(x.ts_event) for x in ticks),dtype=np.int64,count=len(ticks));bid=np.fromiter((v14.fpx(x.bid_price) for x in ticks),float,count=len(ticks));ask=np.fromiter((v14.fpx(x.ask_price) for x in ticks),float,count=len(ticks))
 raw=pd.DataFrame({"datetime":pd.to_datetime(ts,unit="ns"),"ns":ts,"bid":bid,"ask":ask})
 z=v14.feature(v14.bars(raw,a.tf));ctx,counts=mssplus_context(z,a.tf);st=add_targets(line_setups(z),z,a.tf);entries,stats=linked_entries(ctx,st,z,raw,a.tf);tr=simulate(entries,raw,a.tf)
 ctx.to_csv(out/"mssplus_contexts.csv",index=False);st.to_csv(out/"line_poi_setups.csv",index=False);entries.to_csv(out/"linked_entries.csv",index=False);tr.to_csv(out/"trades.csv",index=False)
 m=metrics(tr);by={k:metrics(v) for k,v in tr.groupby("pattern")} if len(tr) else {}
 res={"version":"v1.16","verification":"TF_STANDALONE_RAW_BIDASK","timeframe_min":a.tf,"m5_used":False,
  "raw_ticks":len(raw),"bars":len(z),"mssplus_counts":counts,"line_setups":len(st),"link_stats":dict(stats),
  "architecture":"TF-local MSS+ travel -> destination V/A/QM/OCL POI -> POI revisit -> reversal Close -> raw Bid/Ask entry",
  "direction_rule":"MSS+ travel direction is opposite the reversal entry direction; POI must lie in the MSS+ travel path.",
  "entry_rule":"POI cannot independently trigger a trade; an active prior MSS+ link is mandatory.",
  "cost_assumption":{"commission_rt_per_lot":COMMISSION_RT_PER_LOT,"cashback_rt_per_lot":CASHBACK_RT_PER_LOT},
  **m,"pattern_metrics":by}
 (out/"result.json").write_text(json.dumps(res,indent=2,default=str),encoding="utf-8");print(json.dumps(res,indent=2,default=str))
if __name__=="__main__":main()
