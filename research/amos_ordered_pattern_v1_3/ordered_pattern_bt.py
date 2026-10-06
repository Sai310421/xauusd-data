#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

PATS=["P01_MSS_EQ","P02_MSS_IFVG","P03_IFVG_MSS_EQ","P04_MSS_EQ_MACRO","P05_MSS_IFVG_MACRO","P06_MSS_ANY_RETRACE"]
def W(t,a,b):
 m=t.hour*60+t.minute; A=int(a[:2])*60+int(a[3:]);B=int(b[:2])*60+int(b[3:]);return A<=m<B
def ATR(d,n=14):
 p=d.close.shift();tr=pd.concat([(d.high-d.low).abs(),(d.high-p).abs(),(d.low-p).abs()],axis=1).max(axis=1);return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def REF(d,i):
 t=d.datetime.iat[i];m=t.hour*60+t.minute;day=t.normalize()
 def rg(a,b):
  x=d[(d.datetime>=day+pd.Timedelta(minutes=a))&(d.datetime<day+pd.Timedelta(minutes=b))]
  return None if x.empty else (float(x.high.max()),float(x.low.min()))
 if 420<=m<600:
  r=rg(0,360);return (*r,"ASIA") if r else None
 if 810<=m<990:
  r=rg(420,600) or rg(0,360);return (*r,"LONDON") if r else None
 return None
def FVG(d,i,di,lb=35):
 for k in range(i-2,max(2,i-lb)-1,-1):
  if di<0 and d.low.iat[k]>d.high.iat[k-2]:return True,float(d.high.iat[k-2]),float(d.low.iat[k]),k
  if di>0 and d.high.iat[k]<d.low.iat[k-2]:return True,float(d.high.iat[k]),float(d.low.iat[k-2]),k
 return False,0.,0.,-1
def EQ(di,b,se,de):
 if di<0:
  z=se-de
  if z<=0:return False
  lo,hi=de+z*.50,de+z*.79;return b.high>=lo and b.low<=hi
 z=de-se
 if z<=0:return False
 lo,hi=de-z*.79,de-z*.50;return b.high>=lo and b.low<=hi
def TARGET(di,e,sl,H,L):
 R=abs(e-sl)
 if R<=0:return False,0,0
 tp=H if di>0 else L;rw=(tp-e) if di>0 else (e-tp);rr=rw/R
 return rw>0 and rr>=1.25,tp,rr
class S:
 def __init__(x):x.eq=x.pk=100.;x.dd=x.gw=x.gl=x.sr=0.;x.w=x.l=0;x.pos=None
 def upd(x,b,risk):
  if not x.pos:return
  p=x.pos;hit=None
  if p["di"]>0:
   if b.low<=p["sl"]:hit=-1.
   elif b.high>=p["tp"]:hit=p["rr"]
  else:
   if b.high>=p["sl"]:hit=-1.
   elif b.low<=p["tp"]:hit=p["rr"]
  if hit is None:return
  if hit>0:x.w+=1;x.gw+=hit
  else:x.l+=1;x.gl+=1
  x.sr+=hit;x.eq*=max(.0001,1+risk/100*hit);x.pk=max(x.pk,x.eq);x.dd=max(x.dd,100*(x.pk-x.eq)/x.pk);x.pos=None
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--data",required=True);ap.add_argument("--out",required=True);ap.add_argument("--risk-pct",type=float,default=.35);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);d=pd.read_csv(a.data);d.columns=[c.lower() for c in d.columns];d["datetime"]=pd.to_datetime(d.datetime)
 for c in ["open","high","low","close","volume"]:d[c]=pd.to_numeric(d[c],errors="coerce")
 d=d.dropna().sort_values("datetime").reset_index(drop=True);d["atr"]=ATR(d);st={p:S() for p in PATS};ev=[];q=None;sid=0
 for i in range(60,len(d)):
  b=d.iloc[i]
  for x in st.values():x.upd(b,a.risk_pct)
  if not np.isfinite(b.atr) or b.atr<=0:continue
  rr0=REF(d,i)
  if q is None:
   if not rr0:continue
   H,L,rn=rr0;mn=b.atr*.03;mx=b.atr*.80;sh=b.high>H+mn and b.close<H and b.high-H<=mx;sl=b.low<L-mn and b.close>L and L-b.low<=mx
   if not(sh or sl):continue
   sid+=1;di=-1 if sh else 1
   q={"id":sid,"di":di,"si":i,"se":float(b.high if sh else b.low),"str":float(d.low.iloc[i-8:i].min() if sh else d.high.iloc[i-8:i].max()),"mi":None,"fi":None,"de":None,"zone":None,"age":0,"fired":set(),"H":H,"L":L}
   ev.append([sid,b.datetime,"SWEEP",di,""]);continue
  q["age"]+=1
  if q["age"]>24:q=None;continue
  di=q["di"];q["se"]=max(q["se"],float(b.high)) if di<0 else min(q["se"],float(b.low));disp=abs(b.close-b.open)>=b.atr*.55
  mss=(b.close<q["str"]-b.atr*.02 and disp) if di<0 else (b.close>q["str"]+b.atr*.02 and disp)
  f,zl,zh,fi=FVG(d,i,di)
  if f and q["fi"] is None and fi>=q["si"]:q["fi"]=i;q["zone"]=(zl,zh);ev.append([q["id"],b.datetime,"IFVG",di,""])
  if mss and q["mi"] is None:q["mi"]=i;q["de"]=float(b.low if di<0 else b.high);ev.append([q["id"],b.datetime,"MSS",di,""]);continue
  if q["mi"] is None:continue
  q["de"]=min(q["de"],float(b.low)) if di<0 else max(q["de"],float(b.high));eq=EQ(di,b,q["se"],q["de"]);it=bool(q["zone"] and i>q["mi"] and b.high>=q["zone"][0] and b.low<=q["zone"][1]);macro=W(b.datetime,"07:00","10:00") or W(b.datetime,"13:30","16:00")
  e=float(b.close);slp=q["se"]-b.atr*.08 if di>0 else q["se"]+b.atr*.08;ok,tp,tr=TARGET(di,e,slp,q["H"],q["L"])
  if not ok:continue
  cond={"P01_MSS_EQ":eq,"P02_MSS_IFVG":it,"P03_IFVG_MSS_EQ":eq and q["fi"] is not None and q["fi"]<q["mi"],"P04_MSS_EQ_MACRO":eq and macro,"P05_MSS_IFVG_MACRO":it and macro,"P06_MSS_ANY_RETRACE":eq or it}
  for p,v in cond.items():
   if not v or p in q["fired"] or st[p].pos is not None:continue
   st[p].pos={"di":di,"sl":slp,"tp":tp,"rr":tr};q["fired"].add(p);ev.append([q["id"],b.datetime,"ENTRY",di,p])
 rows=[]
 for p,x in st.items():
  n=x.w+x.l;pf=x.gw/x.gl if x.gl else (999 if x.gw else 0)
  rows.append({"pattern":p,"N":n,"wins":x.w,"losses":x.l,"WR_pct":100*x.w/n if n else 0,"PF_R":pf,"sum_R":x.sr,"Return_pct_risk_compound":x.eq-100,"MaxDD_pct":x.dd,"avg_R":x.sr/n if n else 0})
 r=pd.DataFrame(rows).sort_values(["PF_R","N"],ascending=False);r.to_csv(out/"pattern_kpi.csv",index=False);pd.DataFrame(ev,columns=["setup","time","event","dir","pattern"]).to_csv(out/"sequence_events.csv",index=False)
 meta={"verification_level":"M1_OHLC_SEQUENCE_SCREEN","start":str(d.datetime.iloc[0]),"end":str(d.datetime.iloc[-1]),"rows":len(d),"rule":"Sweep -> MSS -> post-MSS EQ/IFVG retrace -> ClearTarget","limitations":["M1 OHLC screening","no bid/ask costs","Raw Tick/Nautilus required"]};(out/"manifest.json").write_text(json.dumps(meta,indent=2))
 print(r.to_string(index=False));print(json.dumps(meta))
if __name__=="__main__":main()
