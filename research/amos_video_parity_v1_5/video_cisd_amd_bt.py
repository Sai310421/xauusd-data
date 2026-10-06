#!/usr/bin/env python3
# AMOS Video-Parity CISD/AMD state engine v1.5
# Video checklist is treated as the specification:
# Liquidity Sweep, HTF PDA Delivery, CISD Candle, Macro Window,
# CISD Confirmed, Volume Influx, CISD Entry, Clear Targets.
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

def atr(d,n=14):
 p=d.close.shift(1);tr=pd.concat([(d.high-d.low).abs(),(d.high-p).abs(),(d.low-p).abs()],axis=1).max(axis=1)
 return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def m15(d):
 x=d.set_index("datetime");y=pd.DataFrame({"open":x.open.resample("15min").first(),"high":x.high.resample("15min").max(),"low":x.low.resample("15min").min(),"close":x.close.resample("15min").last(),"volume":x.volume.resample("15min").sum()}).dropna().reset_index()
 y["atr"]=atr(y);y["vma"]=y.volume.shift(1).rolling(20).mean();y["jst"]=y.datetime+pd.Timedelta(hours=9);y["day"]=y.jst.dt.normalize();y["min"]=y.jst.dt.hour*60+y.jst.dt.minute;return y
def ranges(m):
 R={}
 for day,q in m.groupby("day"):
  def rg(a,b):
   z=q[(q["min"]>=a)&(q["min"]<b)];return None if z.empty else (float(z.high.max()),float(z.low.min()))
  R[(day,"ASIA")]=rg(9*60,16*60);R[(day,"LONDON")]=rg(16*60,23*60)
 return R
ROUTES={"AMD_LONDON":{"h":(16,18),"src":"ASIA"},"AMD_NY":{"h":(23,24),"src":"LONDON"}}
class S:
 def __init__(self,r):self.r=r;self.reset()
 def reset(self):self.k=0;self.age=0;self.di=0;self.h=np.nan;self.l=np.nan;self.sx=np.nan;self.cisd_level=np.nan;self.disp=np.nan;self.fvg=None;self.flags={};self.times={}
 def tick(self):
  if self.k:self.age+=1
class P:
 def __init__(self):self.a=None;self.t=[]
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--data",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);d=pd.read_csv(a.data);d.columns=[c.lower() for c in d.columns];d.datetime=pd.to_datetime(d.datetime)
 for c in ["open","high","low","close","volume"]:d[c]=pd.to_numeric(d[c],errors="coerce")
 d=d.dropna().sort_values("datetime").reset_index(drop=True);x=m15(d);R=ranges(x);states={r:S(r) for r in ROUTES};pos={r:P() for r in ROUTES};ev=[]
 for i in range(40,len(x)):
  b=x.iloc[i];t=b.datetime
  # exits: fixed 1.5R for parity screening, SL-first same-bar
  for r,p in pos.items():
   if p.a:
    q=p.a;hit=None
    if q["di"]>0:
     if b.low<=q["sl"]:hit=-1
     elif b.high>=q["tp"]:hit=1.5
    else:
     if b.high>=q["sl"]:hit=-1
     elif b.low<=q["tp"]:hit=1.5
    if hit is not None:p.t.append({**q,"exit":str(t),"R":hit,"win":int(hit>0)});p.a=None
  for r,cfg in ROUTES.items():
   s=states[r];s.tick();hj=int(b.jst.hour)
   if s.k==0:
    if not(cfg["h"][0]<=hj<cfg["h"][1]):continue
    z=R.get((b.day,cfg["src"]))
    if not z:continue
    s.h,s.l=z;s.k=1;s.age=0;s.flags={"liquidity_sweep":0,"htf_pda":0,"cisd_candle":0,"macro_window":1,"cisd_confirmed":0,"volume_influx":0,"cisd_entry":0,"clear_targets":0}
    ev.append({"time":t,"route":r,"stage":"ACCUMULATION_LOCK",**s.flags});continue
   if s.k==1:
    # Manipulation: external-liquidity raid and close back through session boundary.
    up=b.high>s.h+b.atr*.03 and b.close<s.h;dn=b.low<s.l-b.atr*.03 and b.close>s.l
    if up or dn:
     s.di=-1 if up else 1;s.sx=float(b.high if up else b.low);s.flags["liquidity_sweep"]=1
     # PDA delivery is a quality flag, not mandatory: sweep occurs in outer 25% of previous range extension context.
     s.flags["htf_pda"]=int(abs(float(b.close-(s.h+s.l)/2))>=.25*(s.h-s.l))
     s.times["sweep"]=str(t);s.k=2;s.age=0;ev.append({"time":t,"route":r,"stage":"LIQUIDITY_SWEEP","dir":s.di,**s.flags})
    elif s.age>8:s.reset()
    continue
   if s.k==2:
    # CISD candle: first strong opposite delivery candle after the raid.
    body=abs(float(b.close-b.open));opp=(b.close<b.open) if s.di<0 else (b.close>b.open)
    vol=bool(np.isfinite(b.vma) and b.vma>0 and b.volume>=b.vma*1.10)
    if opp and body>=b.atr*.35:
     s.flags["cisd_candle"]=1;s.flags["volume_influx"]=int(vol);s.cisd_level=float(b.open);s.disp=float(b.low if s.di<0 else b.high);s.times["cisd_candle"]=str(t);s.k=3;s.age=0
     ev.append({"time":t,"route":r,"stage":"CISD_CANDLE","dir":s.di,"cisd_level":s.cisd_level,**s.flags})
    elif s.age>8:s.reset()
    continue
   if s.k==3:
    # CISD confirmation is distinct: a later close through the CISD candle open.
    ok=(b.close<s.cisd_level) if s.di<0 else (b.close>s.cisd_level)
    if ok:
     s.flags["cisd_confirmed"]=1;s.times["confirmed"]=str(t)
     # detect FVG around confirmation; optional confluence, not checklist replacement
     if i>=2:
      if s.di<0 and x.low.iat[i-2]>b.high:s.fvg=(float(b.high),float(x.low.iat[i-2]))
      elif s.di>0 and x.high.iat[i-2]<b.low:s.fvg=(float(x.high.iat[i-2]),float(b.low))
     s.flags["clear_targets"]=1 # opposite side of prior session range exists
     s.k=4;s.age=0;ev.append({"time":t,"route":r,"stage":"CISD_CONFIRMED","dir":s.di,**s.flags})
    elif s.age>8:s.reset()
    continue
   if s.k==4:
    # CISD Entry: retrace to CISD open / FVG then reject in distribution direction.
    touch= b.high>=s.cisd_level>=b.low
    if s.fvg:
     lo,hi=sorted(s.fvg);touch=touch or (b.high>=lo and b.low<=hi)
    reject=(b.close<b.open) if s.di<0 else (b.close>b.open)
    if touch and reject and pos[r].a is None:
     s.flags["cisd_entry"]=1;entry=float(b.close);sl=s.sx+b.atr*.08 if s.di<0 else s.sx-b.atr*.08;risk=abs(entry-sl)
     if risk>0:
      tp=entry-1.5*risk if s.di<0 else entry+1.5*risk
      pos[r].a={"entry":str(t),"route":r,"di":s.di,"price":entry,"sl":sl,"tp":tp,
                "pda":s.flags["htf_pda"],"volume":s.flags["volume_influx"],"macro":s.flags["macro_window"],"target":s.flags["clear_targets"]}
      ev.append({"time":t,"route":r,"stage":"CISD_ENTRY","dir":s.di,**s.flags})
     s.reset()
    elif s.age>12:s.reset()
 trades=[]
 for p in pos.values():trades+=p.t
 td=pd.DataFrame(trades);pd.DataFrame(ev).to_csv(out/"video_gate_sequence.csv",index=False);td.to_csv(out/"trades.csv",index=False)
 rows=[]
 if not td.empty:
  for r,g in td.groupby("route"):
   gp=g.loc[g.R>0,"R"].sum();gl=-g.loc[g.R<0,"R"].sum();rows.append({"route":r,"N":len(g),"wins":int(g.win.sum()),"WR_pct":100*g.win.mean(),"PF":gp/gl if gl else np.inf,"sum_R":g.R.sum(),"PDA_N":int(g.pda.sum()),"Volume_N":int(g.volume.sum())})
 pd.DataFrame(rows).to_csv(out/"route_kpi.csv",index=False)
 manifest={"spec":"VIDEO_CHECKLIST_PARITY","timeframe":"M15","checklist":["Liquidity Sweep","HTF PDA Delivery","CISD Candle","Macro Window","CISD Confirmed","Volume Influx","CISD Entry","Clear Targets"],"mandatory_order":["Liquidity Sweep","CISD Candle","CISD Confirmed","CISD Entry"],"quality_flags":["HTF PDA Delivery","Macro Window","Volume Influx","Clear Targets"],"note":"Independent behavioral reconstruction; semantics are operational hypotheses where video does not expose source code."}
 (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8");print(json.dumps(manifest,indent=2));print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__":main()
