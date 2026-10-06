#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

ROUTES={
 "AMD_ASIA":{"activate":(9,10),"acc":"PREV_NY"},
 "AMD_LONDON":{"activate":(16,18),"acc":"ASIA"},
 "AMD_NY":{"activate":(23,1),"acc":"LONDON"},
}
VARIANTS=("CORE","MACRO","PDA_VOL","VIDEO_OR")

def atr(d,n=14):
 p=d.close.shift(1)
 tr=pd.concat([(d.high-d.low).abs(),(d.high-p).abs(),(d.low-p).abs()],axis=1).max(axis=1)
 return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def resample(d,tf):
 x=d.set_index("datetime")
 y=pd.DataFrame({"open":x.open.resample(tf).first(),"high":x.high.resample(tf).max(),"low":x.low.resample(tf).min(),"close":x.close.resample(tf).last(),"volume":x.volume.resample(tf).sum()}).dropna().reset_index()
 return y

def build_m15(d):
 m=resample(d,"15min");m["atr"]=atr(m);m["vmed"]=m.volume.shift(1).rolling(20).median()
 m["jst"]=m.datetime+pd.Timedelta(hours=9);m["jst_date"]=m.jst.dt.normalize();m["jst_min"]=m.jst.dt.hour*60+m.jst.dt.minute
 return m

def inside_hour(h,a,b):
 return (a<=h<b) if a<b else (h>=a or h<b)

def activation(route,row):
 a,b=ROUTES[route]["activate"];return inside_hour(int(row.jst.hour),a,b)

def macro_window(route,row):
 # Narrow timing gate inside the broader AMD activation window. Configurable hypothesis; video proves Macro is optional.
 h=int(row.jst.hour);m=int(row.jst.minute)
 start=ROUTES[route]["activate"][0]
 return h==start and m<30

def session_ranges(m):
 out={}
 for day in sorted(m.jst_date.unique()):
  day=pd.Timestamp(day);q=m[m.jst_date==day]
  def rg(a,b):
   w=q[(q.jst_min>=a)&(q.jst_min<b)]
   return None if w.empty else (float(w.high.max()),float(w.low.min()))
  out[(day,"ASIA")]=rg(9*60,16*60);out[(day,"LONDON")]=rg(16*60,23*60)
  prev=day-pd.Timedelta(days=1)
  q1=m[(m.jst_date==prev)&(m.jst_min>=23*60)];q2=m[(m.jst_date==day)&(m.jst_min<6*60)]
  w=pd.concat([q1,q2]).sort_values("jst")
  out[(day,"PREV_NY")]=None if w.empty else (float(w.high.max()),float(w.low.min()))
 return out

def find_cisd_origin(m,i,di,look=8):
 # Video/common ICT rule: mark the OPEN of the first same-delivery candle series that led into the sweep.
 j=i
 # sweep candle can already be reversal-colored, so step back to the delivery series
 want=lambda k: (m.close.iat[k]>m.open.iat[k]) if di<0 else (m.close.iat[k]<m.open.iat[k])
 if not want(j): j-=1
 if j<1 or not want(j): return None,None
 end=j;start=j
 while start-1>=max(0,i-look) and want(start-1): start-=1
 level=float(m.open.iat[start])
 return level,start

def htf_pda(m30,t,di,px):
 j=m30.datetime.searchsorted(t,side="right")-2
 if j<24:return False
 w=m30.iloc[j-23:j+1];mid=(float(w.high.max())+float(w.low.min()))/2
 return px>=mid if di<0 else px<=mid

def recent_fvg_targets(m30,t,di,entry,look=40):
 j=m30.datetime.searchsorted(t,side="right")-2;out=[]
 if j<3:return out
 for k in range(max(2,j-look),j+1):
  if m30.low.iat[k]>m30.high.iat[k-2]: # bullish FVG
   lo=float(m30.high.iat[k-2]);hi=float(m30.low.iat[k])
   if di>0 and lo>entry:out.extend([lo,hi])
  if m30.high.iat[k]<m30.low.iat[k-2]: # bearish FVG
   lo=float(m30.high.iat[k]);hi=float(m30.low.iat[k-2])
   if di<0 and hi<entry:out.extend([hi,lo])
 return out

def irl_targets(m,i,di,entry,look=24):
 out=[]
 for k in range(max(2,i-look),i-1):
  if di<0 and m.low.iat[k]<m.low.iat[k-1] and m.low.iat[k]<m.low.iat[k+1] and m.low.iat[k]<entry:out.append(float(m.low.iat[k]))
  if di>0 and m.high.iat[k]>m.high.iat[k-1] and m.high.iat[k]>m.high.iat[k+1] and m.high.iat[k]>entry:out.append(float(m.high.iat[k]))
 return out

def choose_target(m,m30,i,t,di,entry,sl,acc_hi,acc_lo,min_rr=.8):
 risk=abs(entry-sl)
 if risk<=0:return None,None,None
 cands=irl_targets(m,i,di,entry)+recent_fvg_targets(m30,t,di,entry)
 cands.append(acc_lo if di<0 else acc_hi)
 good=[]
 for x in cands:
  rew=(entry-x) if di<0 else (x-entry)
  rr=rew/risk
  if rew>0 and rr>=min_rr:good.append((rew,x,rr))
 if not good:return None,None,None
 good.sort(key=lambda z:z[0])
 return good[0][1],good[0][2],"NEAREST_IRL_FVG"

class S:
 def __init__(self):self.eq=100.;self.pk=100.;self.dd=0.;self.w=0;self.l=0;self.gw=0.;self.gl=0.;self.r=0.;self.active=None;self.trades=[]
 def close(self,r,risk,t):
  q=self.active;self.trades.append({**q,"exit_time":str(t),"R":r,"win":int(r>0)})
  if r>0:self.w+=1;self.gw+=r
  else:self.l+=1;self.gl+=1
  self.r+=r;self.eq*=max(.0001,1+risk*r/100);self.pk=max(self.pk,self.eq);self.dd=max(self.dd,100*(self.pk-self.eq)/self.pk);self.active=None
 def pf(self):return self.gw/self.gl if self.gl else (np.inf if self.gw else 0.)

class State:
 def __init__(self,route):self.route=route;self.reset()
 def reset(self):
  self.phase=0;self.age=0;self.di=0;self.acc_hi=np.nan;self.acc_lo=np.nan;self.sweep_ext=np.nan;self.sweep_i=-1
  self.cisd_level=np.nan;self.cisd_i=-1;self.confirm_i=-1;self.pda=False;self.vol=False;self.macro=False
 def tick(self):
  if self.phase:self.age+=1

def pass_variant(v,s):
 if v=="CORE":return True
 if v=="MACRO":return s.macro
 if v=="PDA_VOL":return s.pda and s.vol
 if v=="VIDEO_OR":return s.macro or (s.pda and s.vol)
 return False

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--data",required=True);ap.add_argument("--out",required=True);ap.add_argument("--risk-pct",type=float,default=.35)
 ap.add_argument("--sweep-ttl",type=int,default=8);ap.add_argument("--confirm-ttl",type=int,default=8);ap.add_argument("--entry-ttl",type=int,default=6);ap.add_argument("--min-target-rr",type=float,default=.8)
 a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(a.data);d.columns=[c.lower() for c in d.columns];d.datetime=pd.to_datetime(d.datetime)
 for c in ["open","high","low","close","volume"]:d[c]=pd.to_numeric(d[c],errors="coerce")
 d=d.dropna().sort_values("datetime").reset_index(drop=True)
 m=build_m15(d);m30=resample(d,"30min");R=session_ranges(m)
 stats={(r,v):S() for r in ROUTES for v in VARIANTS};states={r:State(r) for r in ROUTES};ev=[]

 for i in range(40,len(m)):
  b=m.iloc[i];t=b.datetime
  if not np.isfinite(b.atr) or b.atr<=0:continue
  for st in stats.values():
   if st.active:
    q=st.active;hit=None
    if q["dir"]>0:
     if b.low<=q["sl"]:hit=-1.
     elif b.high>=q["tp"]:hit=q["rr"]
    else:
     if b.high>=q["sl"]:hit=-1.
     elif b.low<=q["tp"]:hit=q["rr"]
    if hit is not None:st.close(hit,a.risk_pct,t)

  for route,s in states.items():
   s.tick()
   if s.phase==0:
    if not activation(route,b):continue
    acc=R.get((pd.Timestamp(b.jst_date),ROUTES[route]["acc"]))
    if acc is None:continue
    s.acc_hi,s.acc_lo=acc;s.phase=1;s.age=0
    ev.append({"time":t,"route":route,"phase":"ACCUMULATION","acc_hi":s.acc_hi,"acc_lo":s.acc_lo})
    continue

   if s.phase==1:
    up=b.high>s.acc_hi+b.atr*.03 and b.close<s.acc_hi
    dn=b.low<s.acc_lo-b.atr*.03 and b.close>s.acc_lo
    if up or dn:
     s.di=-1 if up else 1;s.sweep_ext=float(b.high if up else b.low);s.sweep_i=i
     level,ci=find_cisd_origin(m,i,s.di)
     if level is None:
      s.reset();continue
     s.cisd_level=level;s.cisd_i=ci;s.pda=htf_pda(m30,t,s.di,float(b.close));s.macro=macro_window(route,b)
     s.phase=2;s.age=0
     ev.append({"time":t,"route":route,"phase":"LIQUIDITY_SWEEP+CISD_CANDLE","dir":s.di,"cisd_level":level,"pda":s.pda,"macro":s.macro})
    elif s.age>a.sweep_ttl:s.reset()
    continue

   if s.phase==2:
    # CISD Confirmed = BODY CLOSE through origin of the delivery sequence, after the sweep.
    confirmed=(b.close<s.cisd_level) if s.di<0 else (b.close>s.cisd_level)
    if confirmed:
     s.confirm_i=i;s.vol=bool(np.isfinite(b.vmed) and b.vmed>0 and b.volume>=b.vmed*1.15);s.phase=3;s.age=0
     ev.append({"time":t,"route":route,"phase":"CISD_CONFIRMED","dir":s.di,"cisd_level":s.cisd_level,"volume":s.vol})
    elif s.age>a.confirm_ttl:s.reset()
    continue

   if s.phase==3:
    # Separate CISD Entry gate: retest of confirmed CISD level, then HOLD on the distribution side.
    touched=(b.high>=s.cisd_level) if s.di<0 else (b.low<=s.cisd_level)
    held=(b.close<=s.cisd_level) if s.di<0 else (b.close>=s.cisd_level)
    if touched and held and i>s.confirm_i:
     entry=float(s.cisd_level);sl=s.sweep_ext+b.atr*.05 if s.di<0 else s.sweep_ext-b.atr*.05
     tp,rr,target_kind=choose_target(m,m30,i,t,s.di,entry,sl,s.acc_hi,s.acc_lo,a.min_target_rr)
     if tp is not None:
      for v in VARIANTS:
       st=stats[(route,v)]
       if st.active is None and pass_variant(v,s):
        st.active={"entry_time":str(t),"route":route,"variant":v,"dir":s.di,"entry":entry,"sl":sl,"tp":tp,"rr":rr,
                   "target_kind":target_kind,"pda":int(s.pda),"macro":int(s.macro),"volume":int(s.vol),
                   "cisd_level":s.cisd_level,"sweep_index":s.sweep_i,"confirm_index":s.confirm_i}
      ev.append({"time":t,"route":route,"phase":"CISD_ENTRY","dir":s.di,"entry":entry,"tp":tp,"rr":rr,"pda":s.pda,"macro":s.macro,"volume":s.vol})
     s.reset()
    elif s.age>a.entry_ttl:s.reset()

 rows=[];alltr=[]
 for (r,v),st in stats.items():
  N=st.w+st.l
  rows.append({"route":r,"variant":v,"N":N,"wins":st.w,"losses":st.l,"WR_pct":100*st.w/N if N else 0.,"PF_R":st.pf(),"sum_R":st.r,"Return_pct":st.eq-100,"MaxDD_pct":st.dd})
  alltr.extend(st.trades)
 k=pd.DataFrame(rows);k.to_csv(out/"video_parity_kpi.csv",index=False);pd.DataFrame(ev).to_csv(out/"video_sequence_evidence.csv",index=False);pd.DataFrame(alltr).to_csv(out/"video_trades.csv",index=False)
 print(k.to_string(index=False))
 (out/"manifest.json").write_text(json.dumps({"model":"VIDEO_PARITY_CISD_V1_5","common_order":["Liquidity Sweep","CISD Candle","CISD Confirmed","CISD Entry"],"context_variants":{"MACRO":"Macro Window","PDA_VOL":"HTF PDA + Volume Influx","VIDEO_OR":"Macro OR (HTF PDA + Volume)"},"target":"nearest IRL / M30 FVG / opposite accumulation bound","timeframe":"M15","data_start":str(m.datetime.iloc[0]),"data_end":str(m.datetime.iloc[-1])},indent=2),encoding="utf-8")

if __name__=="__main__":main()
