#!/usr/bin/env python3
"""AMOS M15 AMD standalone raw Bid/Ask v1.15.
M15 only. No M1, no M5, no G75.
Ports the existing v1.4 AMD state machine to causal M15 bars built directly from raw BID QuoteTicks.
Entry is first raw quote after the completed M15 trigger bar; exits use executable Bid/Ask.
No OHLC input/fallback and no parameter optimization.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np,pandas as pd
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick

INITIAL=1000.0
QTY_OZ=1.0
COMMISSION_RT_PER_LOT=7.0
CASHBACK_RT_PER_LOT=6.0
ROUTES={
 "AMD_ASIA":{"activate":(9,10),"acc":"PREV_NY"},
 "AMD_LONDON":{"activate":(16,18),"acc":"ASIA"},
 "AMD_NY":{"activate":(23,24),"acc":"LONDON"},
}
def fpx(x):return float(x.as_double()) if hasattr(x,"as_double") else float(x)
def atr(d,n=14):
 p=d.close.shift(1);tr=pd.concat([(d.high-d.low).abs(),(d.high-p).abs(),(d.low-p).abs()],axis=1).max(axis=1)
 return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def m15(raw):
 g=raw.set_index("datetime").bid.resample("15min")
 y=pd.DataFrame({"open":g.first(),"high":g.max(),"low":g.min(),"close":g.last(),"volume":g.count()}).dropna().reset_index()
 y["atr"]=atr(y);y["jst"]=y.datetime+pd.Timedelta(hours=9);y["jst_date"]=y.jst.dt.normalize();y["jst_min"]=y.jst.dt.hour*60+y.jst.dt.minute
 return y
def ranges(m):
 out={}
 for day in sorted(m.jst_date.unique()):
  day=pd.Timestamp(day);q=m[m.jst_date==day]
  def rg(a,b):
   w=q[(q.jst_min>=a)&(q.jst_min<b)]
   return None if w.empty else (float(w.high.max()),float(w.low.min()))
  out[(day,"ASIA")]=rg(9*60,16*60);out[(day,"LONDON")]=rg(16*60,23*60)
  prev=day-pd.Timedelta(days=1);q1=m[(m.jst_date==prev)&(m.jst_min>=23*60)];q2=m[(m.jst_date==day)&(m.jst_min<6*60)]
  w=pd.concat([q1,q2]).sort_values("jst");out[(day,"PREV_NY")]=None if w.empty else (float(w.high.max()),float(w.low.min()))
 return out
class S:
 def __init__(self,r):self.route=r;self.reset()
 def reset(self):
  self.state=0;self.age=0;self.dir=0;self.acc_hi=np.nan;self.acc_lo=np.nan;self.sweep_ext=np.nan;self.disp_ext=np.nan
  self.fvg=None;self.sweep_time=None;self.confirm_time=None
 def tick(self):
  if self.state>0:self.age+=1
def raw_exit(raw,start_idx,di,sl,tp):
 bid=raw.bid.to_numpy();ask=raw.ask.to_numpy()
 for k in range(start_idx,len(raw)):
  if di>0:
   if bid[k]<=sl:return k,float(bid[k]),"SL"
   if bid[k]>=tp:return k,float(bid[k]),"TP"
  else:
   if ask[k]>=sl:return k,float(ask[k]),"SL"
   if ask[k]<=tp:return k,float(ask[k]),"TP"
 k=len(raw)-1;return k,float(bid[k] if di>0 else ask[k]),"EOD"
def metrics(rows):
 if not rows:return {"N":0,"WR_pct":0.,"PF":0.,"Net_USD":0.,"Return_pct":0.,"MaxRealizedDD_USD":0.,"RF":None}
 p=np.array([x["pnl"] for x in rows],float);gp=float(p[p>0].sum());gl=float(-p[p<0].sum())
 eq=INITIAL;pk=INITIAL;dd=0.
 for x in sorted(rows,key=lambda q:q["exit_ns"]):
  eq+=x["pnl"];pk=max(pk,eq);dd=max(dd,pk-eq)
 net=float(p.sum())
 return {"N":len(rows),"WR_pct":float((p>0).mean()*100),"PF":gp/gl if gl>0 else (math.inf if gp>0 else 0.),
         "Net_USD":net,"Return_pct":net/INITIAL*100,"MaxRealizedDD_USD":dd,"MaxRealizedDD_pct_initial":dd/INITIAL*100,
         "RF":net/dd if dd>0 else None}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--out",required=True)
 ap.add_argument("--sweep-ttl",type=int,default=8);ap.add_argument("--confirm-ttl",type=int,default=8);ap.add_argument("--pullback-ttl",type=int,default=12);ap.add_argument("--min-rr",type=float,default=1.25);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
 ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value]);ticks.sort(key=lambda x:int(x.ts_event))
 ts=np.fromiter((int(x.ts_event) for x in ticks),dtype=np.int64,count=len(ticks));bid=np.fromiter((fpx(x.bid_price) for x in ticks),float,count=len(ticks));ask=np.fromiter((fpx(x.ask_price) for x in ticks),float,count=len(ticks))
 raw=pd.DataFrame({"datetime":pd.to_datetime(ts,unit="ns"),"ns":ts,"bid":bid,"ask":ask});m=m15(raw);rg=ranges(m)
 states={r:S(r) for r in ROUTES};active_until={r:-1 for r in ROUTES};trades=[];evidence=[]
 raw_ns=raw.ns.to_numpy()
 for i in range(40,len(m)):
  b=m.iloc[i]
  if not np.isfinite(b.atr) or b.atr<=0:continue
  for route,s in states.items():
   s.tick();a0,a1=ROUTES[route]["activate"];h=int(b.jst.hour)
   if s.state==0:
    if not (a0<=h<a1):continue
    acc=rg.get((pd.Timestamp(b.jst_date),ROUTES[route]["acc"]))
    if acc is None:continue
    s.acc_hi,s.acc_lo=acc;s.state=1;s.age=0
    evidence.append({"time":b.datetime,"route":route,"state":"ACCUMULATION_LOCK","dir":0})
    continue
   if s.state==1:
    up=b.high>s.acc_hi+b.atr*.05 and b.close<s.acc_hi;dn=b.low<s.acc_lo-b.atr*.05 and b.close>s.acc_lo
    if up or dn:
     s.dir=-1 if up else 1;s.sweep_ext=float(b.high if up else b.low);s.sweep_time=str(b.datetime);s.state=2;s.age=0
     evidence.append({"time":b.datetime,"route":route,"state":"MANIPULATION_SWEEP","dir":s.dir})
    elif s.age>a.sweep_ttl:s.reset()
    continue
   if s.state==2:
    prior=m.iloc[max(0,i-4):i];struct=float(prior.low.min()) if s.dir<0 else float(prior.high.max())
    disp=abs(float(b.close-b.open))>=float(b.atr)*.45;cisd=(b.close<struct-b.atr*.01) if s.dir<0 else (b.close>struct+b.atr*.01)
    if disp and cisd:
     s.disp_ext=float(b.low if s.dir<0 else b.high);s.confirm_time=str(b.datetime);s.fvg=None
     if i>=2:
      if s.dir<0 and m.low.iat[i-2]>b.high:s.fvg=(float(b.high),float(m.low.iat[i-2]))
      elif s.dir>0 and m.high.iat[i-2]<b.low:s.fvg=(float(m.high.iat[i-2]),float(b.low))
     s.state=3;s.age=0;evidence.append({"time":b.datetime,"route":route,"state":"CISD_MSS_CONFIRMED","dir":s.dir})
    elif s.age>a.confirm_ttl:s.reset()
    continue
   if s.state==3:
    if s.dir<0:leg=s.sweep_ext-s.disp_ext;z1=s.disp_ext+leg*.62;z2=s.disp_ext+leg*.79
    else:leg=s.disp_ext-s.sweep_ext;z1=s.disp_ext-leg*.79;z2=s.disp_ext-leg*.62
    if leg<=0:s.reset();continue
    lo,hi=sorted((z1,z2));ote=b.high>=lo and b.low<=hi;fvg=False
    if s.fvg:
     fl,fh=sorted(s.fvg);fvg=b.high>=fl and b.low<=fh
    if ote or fvg:
     rng=max(float(b.high-b.low),1e-9);trigger=(b.close<b.open and (b.high-b.close)/rng>=.45) if s.dir<0 else (b.close>b.open and (b.close-b.low)/rng>=.45)
     close_ns=int((pd.Timestamp(b.datetime)+pd.Timedelta(minutes=15)).value);k=int(np.searchsorted(raw_ns,close_ns,side="left"))
     if trigger and k<len(raw) and raw_ns[k]>=active_until[route]:
      entry=float(raw.ask.iat[k] if s.dir>0 else raw.bid.iat[k]);sl=s.sweep_ext+b.atr*.08 if s.dir<0 else s.sweep_ext-b.atr*.08
      risk=(entry-sl)*s.dir
      if risk>0:
       target=s.acc_lo if s.dir<0 else s.acc_hi;reward=(entry-target) if s.dir<0 else (target-entry);rr=reward/risk
       if rr<a.min_rr:rr=2.;target=entry-risk*rr if s.dir<0 else entry+risk*rr
       ek,exitpx,reason=raw_exit(raw,k,s.dir,float(sl),float(target));gross=(exitpx-entry)*s.dir*QTY_OZ;cost=(QTY_OZ/100)*COMMISSION_RT_PER_LOT;cb=(QTY_OZ/100)*CASHBACK_RT_PER_LOT;pnl=gross-cost+cb
       row={"route":route,"dir":s.dir,"entry_time":raw.datetime.iat[k],"entry_ns":int(raw_ns[k]),"entry":entry,"sl":float(sl),"tp":float(target),"rr_initial":float(rr),
            "exit_time":raw.datetime.iat[ek],"exit_ns":int(raw_ns[ek]),"exit":exitpx,"reason":reason,"pnl":pnl,"used_ote":int(ote),"used_fvg":int(fvg)}
       trades.append(row);active_until[route]=int(raw_ns[ek]);evidence.append({"time":raw.datetime.iat[k],"route":route,"state":"AMD_ENTRY","dir":s.dir});s.reset()
    elif s.age>a.pullback_ttl:s.reset()
 pd.DataFrame(trades).to_csv(out/"trades.csv",index=False);pd.DataFrame(evidence).to_csv(out/"sequence.csv",index=False)
 overall=metrics(trades);by={r:metrics([x for x in trades if x["route"]==r]) for r in ROUTES}
 start=pd.Timestamp(raw.datetime.iat[0]).date();end=pd.Timestamp(raw.datetime.iat[-1]).date();bd=max(1,len(pd.bdate_range(start,end)));scale=21/bd
 res={"version":"v1.15","verification_level":"M15_AMD_STANDALONE_NAUTILUS_RAW_BIDASK_REPLAY","m1_used":False,"m5_used":False,"g75_used":False,"ohlc_input_used":False,
      "raw_ticks":len(raw),"m15_bars":len(m),"data_start":str(raw.datetime.iat[0]),"data_end":str(raw.datetime.iat[-1]),"BusinessDays":bd,
      "architecture":"M15 AMD only: Accumulation -> Manipulation Sweep -> CISD/MSS+displacement -> OTE/FVG pullback -> M15 rejection -> first subsequent raw Bid/Ask",
      "routes":ROUTES,"cost_assumption":{"commission_rt_per_lot":COMMISSION_RT_PER_LOT,"cashback_rt_per_lot":CASHBACK_RT_PER_LOT},
      **overall,"N21_linearized":overall["N"]*scale,"Monthly21_pct_linearized":overall["Return_pct"]*scale,"route_metrics":by,
      "note":"Same v1.4 AMD thresholds/session definitions; execution upgraded to raw Bid/Ask. No optimization."}
 (out/"result.json").write_text(json.dumps(res,indent=2,default=str),encoding="utf-8");print(json.dumps(res,indent=2,default=str))
if __name__=="__main__":main()
