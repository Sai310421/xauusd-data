#!/usr/bin/env python3
"""
AMOS Gate Candidate Learning v1.6
Purpose:
  Increase the learning population by recording EVERY completed P5 gate sequence,
  not only entries that passed the old RR/one-position filters.
Architecture:
  M15 AMD/context features -> fixed P5 order
  Sweep -> MSS -> Volume -> Equilibrium -> Pullback
  -> candidate snapshot -> future label (1.5R before 1R stop within 60 M1 bars)
Leakage control:
  Features are causal at candidate time. Future bars are used only for labels.
  Chronological 60/20/20 train/validation/test. Test is untouched.
"""
import argparse,json,math
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SEQ=["sweep","mss","volume","equilibrium","pullback"]
TTL={"mss":8,"volume":8,"equilibrium":12,"pullback":8}
FEATURES=[
 "rr_to_session_target","acc_eff","acc_range_atr","bars_from_sweep",
 "hour_sin","hour_cos","m15_body_atr","m15_range_atr","m15_close_pos","m15_efficiency",
 "m15_prev4_eff","m15_prev4_range_atr","amd_accumulation","sweep_depth_atr","mss_body_atr",
 "volume_ratio","eq_pos"
]
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
def eqzone(di,se,de):
 if di<0:
  z=se-de;return (de+z*.50,de+z*.79) if z>0 else None
 z=de-se;return (de-z*.79,de-z*.50) if z>0 else None
def pf(r):
 r=np.asarray(r,float);gp=r[r>0].sum();gl=-r[r<0].sum()
 return float(gp/gl) if gl>0 else (float("inf") if gp>0 else 0.)
def kpi(x):
 if len(x)==0:return {"N":0,"WR_pct":0.,"PF_R":0.,"sum_R":0.}
 return {"N":int(len(x)),"WR_pct":float(100*(x.label>0).mean()),"PF_R":pf(x.outcome_R),"sum_R":float(x.outcome_R.sum())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--data",required=True);ap.add_argument("--out",required=True)
 ap.add_argument("--target-r",type=float,default=1.5);ap.add_argument("--horizon",type=int,default=60)
 a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(a.data);d.columns=[c.lower() for c in d.columns];d.datetime=pd.to_datetime(d.datetime)
 for c in ["open","high","low","close","volume"]:d[c]=pd.to_numeric(d[c],errors="coerce")
 d=d.dropna().sort_values("datetime").reset_index(drop=True);d["atr"]=atr(d);d["vma"]=d.volume.shift(1).rolling(20).mean()
 R=ranges(d)
 m=d.set_index("datetime").resample("15min").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
 m["atr"]=atr(m);m["body_atr"]=(m.close-m.open).abs()/m.atr;m["range_atr"]=(m.high-m.low)/m.atr
 m["close_pos"]=(m.close-m.low)/(m.high-m.low).replace(0,np.nan);m["efficiency"]=(m.close-m.open).abs()/(m.high-m.low).replace(0,np.nan)
 m["prev4_range"]=(m.high.shift(1).rolling(4).max()-m.low.shift(1).rolling(4).min())
 m["prev4_range_atr"]=m.prev4_range/m.atr
 m["prev4_eff"]=(m.close.shift(1)-m.open.shift(4)).abs()/m.prev4_range.replace(0,np.nan)
 # M15 accumulation is context only, not a mandatory gate.
 m["amd_accumulation"]=((m.prev4_eff<=.35)&(m.prev4_range_atr<=3.5)).astype(int)

 cand=[];step=0;age=0;di=0;se=de=0.;zone=None;sweep_i=-1;sweep_time=None
 acc_eff=acc_range_atr=np.nan;sweep_depth=np.nan;mss_body=np.nan;vol_ratio=np.nan
 for i in range(60,len(d)-a.horizon-1):
  b=d.iloc[i];t=b.datetime
  if not np.isfinite(b.atr) or b.atr<=0:continue
  rr=ref(R,t)
  if not rr:
   if step: step=0;age=0
   continue
  rh,rl,session=rr;mn=b.atr*.03;mx=b.atr*.80
  sh=b.high>rh+mn and b.close<rh and b.high-rh<=mx
  sl=b.low<rl-mn and b.close>rl and rl-b.low<=mx
  if step>0:
   age+=1;want=SEQ[step]
   if age>TTL.get(want,8):
    step=0;age=0;di=0;zone=None
  want=SEQ[step];event=False
  if want=="sweep":
   if sh or sl:
    di=-1 if sh else 1;se=float(b.high if sh else b.low);de=float(b.low if sh else b.high)
    sweep_depth=float((b.high-rh)/b.atr if sh else (rl-b.low)/b.atr)
    sweep_i=i;sweep_time=t;pre=d.iloc[max(0,i-30):i]
    ar=float(pre.high.max()-pre.low.min()) if len(pre)>=10 else np.nan
    acc_range_atr=ar/float(b.atr) if np.isfinite(ar) and b.atr>0 else np.nan
    acc_eff=abs(float(pre.close.iloc[-1]-pre.open.iloc[0]))/ar if np.isfinite(ar) and ar>0 else np.nan
    event=True
  elif di:
   se=max(se,float(b.high)) if di<0 else min(se,float(b.low))
   de=min(de,float(b.low)) if di<0 else max(de,float(b.high))
   struct=float(d.low.iloc[i-8:i].min()) if di<0 else float(d.high.iloc[i-8:i].max())
   disp=abs(b.close-b.open)>=b.atr*.55
   if want=="mss":
    event=(b.close<struct-b.atr*.02 and disp) if di<0 else (b.close>struct+b.atr*.02 and disp)
    if event:mss_body=float(abs(b.close-b.open)/b.atr)
   elif want=="volume":
    vol_ratio=float(b.volume/b.vma) if np.isfinite(b.vma) and b.vma>0 else np.nan
    event=bool(np.isfinite(vol_ratio) and vol_ratio>=1.10)
   elif want=="equilibrium":
    zone=eqzone(di,se,de);event=zone is not None
   elif want=="pullback" and zone:
    lo,hi=sorted(zone);event=b.high>=lo and b.low<=hi
  if not event:continue
  step+=1;age=0
  if step<len(SEQ):continue

  entry=float(b.close);stop=se-b.atr*.08 if di>0 else se+b.atr*.08;risk=abs(entry-stop)
  session_target=rh if di>0 else rl;reward=(session_target-entry)*di
  rrn=reward/risk if risk>0 else np.nan
  if risk>0:
   target=entry+di*a.target_r*risk
   outcome=0.;exit_reason="TIMEOUT";mfe=0.;mae=0.
   for j in range(i+1,min(len(d),i+1+a.horizon)):
    x=d.iloc[j]
    fav=((x.high-entry) if di>0 else (entry-x.low))/risk
    adv=((entry-x.low) if di>0 else (x.high-entry))/risk
    mfe=max(mfe,float(fav));mae=max(mae,float(adv))
    # Conservative same-bar ordering: stop first.
    stop_hit=(x.low<=stop) if di>0 else (x.high>=stop)
    tp_hit=(x.high>=target) if di>0 else (x.low<=target)
    if stop_hit:outcome=-1.;exit_reason="SL";break
    if tp_hit:outcome=a.target_r;exit_reason="TP";break
   mi=m.index.searchsorted(t,side="right")-1
   if mi>=0:
    z=m.iloc[mi];h=t.hour+t.minute/60
    eqmid=np.mean(zone) if zone else entry
    cand.append({
      "time":t,"session":session,"dir":di,"entry":entry,"stop":stop,"risk":risk,
      "rr_to_session_target":rrn,"acc_eff":acc_eff,"acc_range_atr":acc_range_atr,
      "bars_from_sweep":i-sweep_i,"hour_sin":np.sin(2*np.pi*h/24),"hour_cos":np.cos(2*np.pi*h/24),
      "m15_body_atr":z.body_atr,"m15_range_atr":z.range_atr,"m15_close_pos":z.close_pos,
      "m15_efficiency":z.efficiency,"m15_prev4_eff":z.prev4_eff,"m15_prev4_range_atr":z.prev4_range_atr,
      "amd_accumulation":z.amd_accumulation,"sweep_depth_atr":sweep_depth,"mss_body_atr":mss_body,
      "volume_ratio":vol_ratio,"eq_pos":abs(entry-eqmid)/b.atr,
      "label":int(outcome>0),"outcome_R":outcome,"exit_reason":exit_reason,"MFE_R":mfe,"MAE_R":mae
    })
  step=0;age=0;di=0;zone=None

 c=pd.DataFrame(cand).sort_values("time").reset_index(drop=True)
 if len(c)<30:raise SystemExit(f"candidate population too small: {len(c)}")
 n=len(c);i1=int(n*.60);i2=int(n*.80);tr=c.iloc[:i1].copy();va=c.iloc[i1:i2].copy();te=c.iloc[i2:].copy()
 model=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                 ("clf",LogisticRegression(C=.5,class_weight="balanced",max_iter=3000,random_state=42))])
 model.fit(tr[FEATURES],tr.label)
 for x in (tr,va,te):x["score"]=model.predict_proba(x[FEATURES])[:,1]
 # Choose threshold on validation only; require at least max(5,25%) retained.
 qs=[.35,.45,.55,.65,.75]
 choices=[]
 for q in qs:
  th=float(tr.score.quantile(q));sel=va[va.score>=th]
  if len(sel)>=max(5,int(len(va)*.25)):choices.append((pf(sel.outcome_R),len(sel),q,th))
 if not choices:th=float(tr.score.quantile(.5));best=(0,0,.5,th)
 else:best=max(choices,key=lambda x:(x[0],x[1]))
 th=best[3]
 for x in (tr,va,te):x["selected"]=x.score>=th
 result={
  "architecture":"M15 AMD/context -> fixed P5 gates -> candidate -> ML rank",
  "fixed_gate_order":["Liquidity Sweep","MSS","Volume Influx","Equilibrium","Pullback"],
  "label":f"{a.target_r}R target before 1R structural stop within {a.horizon} M1 bars; timeout=0R",
  "candidate_population":n,"split":{"train":len(tr),"validation":len(va),"test_oos":len(te)},
  "threshold_source":"validation selection among train-score quantiles; OOS untouched",
  "threshold":th,"chosen_train_quantile":best[2],
  "train_all":kpi(tr),"train_selected":kpi(tr[tr.selected]),
  "validation_all":kpi(va),"validation_selected":kpi(va[va.selected]),
  "oos_all":kpi(te),"oos_selected":kpi(te[te.selected]),"oos_rejected":kpi(te[~te.selected]),
  "amd_oos":kpi(te[te.amd_accumulation==1]),"amd_oos_selected":kpi(te[(te.amd_accumulation==1)&te.selected])
 }
 c.to_csv(out/"all_candidates.csv",index=False);tr.to_csv(out/"train.csv",index=False);va.to_csv(out/"validation.csv",index=False);te.to_csv(out/"oos.csv",index=False)
 (out/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
