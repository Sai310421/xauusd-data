#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd

PATTERNS={
 "P1_SWEEP_MSS_EQ_PULLBACK":{"seq":["sweep","mss","equilibrium","pullback"],"ttl":[0,8,12,8]},
 "P2_SWEEP_MSS_IFVG_RETEST":{"seq":["sweep","mss","ifvg","ifvg_retest"],"ttl":[0,8,12,8]},
 "P3_SWEEP_IFVG_INV_RETEST":{"seq":["sweep","ifvg","ifvg_retest"],"ttl":[0,12,8]},
 "P4_HTFPDA_SWEEP_MSS_EQ":{"seq":["htf_pda","sweep","mss","equilibrium","pullback"],"ttl":[0,12,8,12,8]},
 "P5_SWEEP_MSS_VOL_EQ":{"seq":["sweep","mss","volume","equilibrium","pullback"],"ttl":[0,8,8,12,8]},
}
def win(t,a,b):
 m=t.hour*60+t.minute;x,y=map(lambda s:int(s[:2])*60+int(s[3:]),(a,b));return x<=m<y
def atr(d,n=14):
 p=d.close.shift(1);tr=pd.concat([(d.high-d.low).abs(),(d.high-p).abs(),(d.low-p).abs()],axis=1).max(axis=1)
 return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def ranges(d):
 out={};mins=d.datetime.dt.hour*60+d.datetime.dt.minute
 for day,idx in d.groupby(d.datetime.dt.normalize()).groups.items():
  q=d.loc[idx];qm=mins.loc[idx]
  def rg(a,b):
   w=q[(qm>=a)&(qm<b)];return None if w.empty else (float(w.high.max()),float(w.low.min()))
  out[pd.Timestamp(day)]={"asia":rg(0,360),"london":rg(420,600)}
 return out
def ref(R,t):
 m=t.hour*60+t.minute;r=R.get(t.normalize(),{})
 if 420<=m<600 and r.get("asia"):return (*r["asia"],"ASIA")
 if 810<=m<990:
  x=r.get("london") or r.get("asia")
  if x:return (*x,"LONDON")
 return None
def m15(d):
 x=d.set_index("datetime");return pd.DataFrame({"open":x.open.resample("15min").first(),"high":x.high.resample("15min").max(),"low":x.low.resample("15min").min(),"close":x.close.resample("15min").last()}).dropna().reset_index()
def pda(h,t,di,px):
 j=h.datetime.searchsorted(t,side="right")-2
 if j<40:return False
 w=h.iloc[j-39:j+1];eq=(w.high.max()+w.low.min())/2;return px>=eq if di<0 else px<=eq
def ifvg(d,i,di,lb=35):
 for k in range(i-2,max(1,i-lb)-1,-1):
  if di<0 and d.low.iat[k]>d.high.iat[k-2]:
   lo,hi=d.high.iat[k-2],d.low.iat[k]
   if (d.close.iloc[k+1:i+1]<lo).any():return True,float(lo),float(hi)
  if di>0 and d.high.iat[k]<d.low.iat[k-2]:
   lo,hi=d.high.iat[k],d.low.iat[k-2]
   if (d.close.iloc[k+1:i+1]>hi).any():return True,float(lo),float(hi)
 return False,0.,0.
def eqzone(di,se,de):
 if di<0:
  z=se-de;return (de+z*.50,de+z*.79) if z>0 else None
 z=de-se;return (de-z*.79,de-z*.50) if z>0 else None
class Stat:
 def __init__(self):self.eq=100.;self.pk=100.;self.dd=0.;self.w=0;self.l=0;self.r=0.;self.gw=0.;self.gl=0.;self.a=None
 def close(self,x,risk):
  if x>0:self.w+=1;self.gw+=x
  else:self.l+=1;self.gl+=1
  self.r+=x;self.eq*=max(.0001,1+risk*x/100);self.pk=max(self.pk,self.eq);self.dd=max(self.dd,100*(self.pk-self.eq)/self.pk);self.a=None
 def pf(self):return self.gw/self.gl if self.gl else 0
class Track:
 def __init__(self,name):self.name=name;self.step=0;self.age=0;self.dir=0;self.se=0.;self.de=0.;self.zone=None;self.fvg=None;self.setup=0
 def reset(self):self.__init__(self.name)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--data",required=True);ap.add_argument("--out",required=True);ap.add_argument("--risk-pct",type=float,default=.35);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);d=pd.read_csv(a.data);d.columns=[c.lower() for c in d.columns];d.datetime=pd.to_datetime(d.datetime)
 for c in ["open","high","low","close","volume"]:d[c]=pd.to_numeric(d[c],errors="coerce")
 d=d.dropna().sort_values("datetime").reset_index(drop=True);d["atr"]=atr(d);d["vma"]=d.volume.shift(1).rolling(20).mean();R=ranges(d);H=m15(d)
 S={k:Stat() for k in PATTERNS};T={k:Track(k) for k in PATTERNS};evidence=[];sid=0
 for i in range(60,len(d)):
  b=d.iloc[i];t=b.datetime
  if not np.isfinite(b.atr) or b.atr<=0:continue
  for st in S.values():
   if st.a:
    q=st.a;hit=None
    if q["d"]>0:
     if b.low<=q["sl"]:hit=-1.
     elif b.high>=q["tp"]:hit=q["rr"]
    else:
     if b.high>=q["sl"]:hit=-1.
     elif b.low<=q["tp"]:hit=q["rr"]
    if hit is not None:st.close(hit,a.risk_pct)
  rr=ref(R,t)
  if not rr:continue
  rh,rl,_=rr;mn=b.atr*.03;mx=b.atr*.80
  sh=b.high>rh+mn and b.close<rh and b.high-rh<=mx;sl=b.low<rl-mn and b.close>rl and rl-b.low<=mx
  for name,p in PATTERNS.items():
   tr=T[name];seq=p["seq"]
   if tr.step>0:
    tr.age+=1
    ttl=p["ttl"][tr.step] if tr.step<len(p["ttl"]) else 8
    if tr.age>ttl:tr.reset()
   want=seq[tr.step]
   event=False
   if want=="htf_pda":
    # precondition only; latch direction once a later sweep supplies it
    event=True
   elif want=="sweep":
    if sh or sl:
     tr.dir=-1 if sh else 1;tr.se=float(b.high if sh else b.low);tr.de=float(b.low if sh else b.high);event=True;sid+=1;tr.setup=sid
     if name=="P4_HTFPDA_SWEEP_MSS_EQ" and not pda(H,t,tr.dir,float(b.close)):event=False
   elif tr.dir:
    tr.se=max(tr.se,float(b.high)) if tr.dir<0 else min(tr.se,float(b.low));tr.de=min(tr.de,float(b.low)) if tr.dir<0 else max(tr.de,float(b.high))
    struct=float(d.low.iloc[i-8:i].min()) if tr.dir<0 else float(d.high.iloc[i-8:i].max())
    disp=abs(b.close-b.open)>=b.atr*.55
    if want=="mss":event=(b.close<struct-b.atr*.02 and disp) if tr.dir<0 else (b.close>struct+b.atr*.02 and disp)
    elif want=="volume":event=bool(np.isfinite(b.vma) and b.vma>0 and b.volume>=b.vma*1.10)
    elif want=="equilibrium":
     tr.zone=eqzone(tr.dir,tr.se,tr.de);event=tr.zone is not None
    elif want=="pullback":
     if tr.zone:
      lo,hi=sorted(tr.zone);event=b.high>=lo and b.low<=hi
    elif want=="ifvg":
     ok,lo,hi=ifvg(d,i,tr.dir);event=ok
     if ok:tr.fvg=(lo,hi)
    elif want=="ifvg_retest":
     if tr.fvg:
      lo,hi=tr.fvg;event=b.high>=lo and b.low<=hi
   if not event:continue
   evidence.append({"time":t,"pattern":name,"setup":tr.setup,"step":tr.step+1,"gate":want,"dir":tr.dir,"price":b.close})
   tr.step+=1;tr.age=0
   if tr.step>=len(seq):
    st=S[name]
    if st.a is None and tr.dir:
     e=float(b.close);stop=tr.se-b.atr*.08 if tr.dir>0 else tr.se+b.atr*.08;rd=abs(e-stop);target=rh if tr.dir>0 else rl;reward=(target-e) if tr.dir>0 else (e-target)
     rrn=reward/rd if rd>0 else 0
     if rrn>=1.25:st.a={"d":tr.dir,"e":e,"sl":stop,"tp":target,"rr":rrn}
    tr.reset()
 rows=[]
 for n,st in S.items():
  N=st.w+st.l;rows.append({"pattern":n,"sequence":">".join(PATTERNS[n]["seq"]),"N":N,"wins":st.w,"losses":st.l,"WR_pct":100*st.w/N if N else 0,"PF_R":st.pf(),"sum_R":st.r,"Return_pct":st.eq-100,"MaxDD_pct":st.dd})
 pd.DataFrame(rows).to_csv(out/"pattern_kpi.csv",index=False);pd.DataFrame(evidence).to_csv(out/"sequence_evidence.csv",index=False)
 (out/"manifest.json").write_text(json.dumps({"level":"M1_OHLC_SEQUENCE_SCREEN","start":str(d.datetime.iloc[0]),"end":str(d.datetime.iloc[-1]),"patterns":PATTERNS},indent=2),encoding="utf-8")
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__":main()
