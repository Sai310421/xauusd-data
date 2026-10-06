#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

# JST AMD windows are configurable. Defaults reflect the known session-transition areas used in the current AMOS design.
DEFAULT_WINDOWS=[(9,10),(16,18),(23,1)]

def atr(d,n=14):
    p=d.close.shift(1)
    tr=pd.concat([(d.high-d.low).abs(),(d.high-p).abs(),(d.low-p).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def m15_from_m1(d):
    x=d.set_index("datetime")
    y=pd.DataFrame({
        "open":x.open.resample("15min").first(),
        "high":x.high.resample("15min").max(),
        "low":x.low.resample("15min").min(),
        "close":x.close.resample("15min").last(),
        "volume":x.volume.resample("15min").sum(),
    }).dropna().reset_index()
    y["atr"]=atr(y,14)
    y["vma"]=y.volume.shift(1).rolling(20).mean()
    return y

def in_window_jst(ts, windows):
    h=(ts.hour+9)%24
    for a,b in windows:
        if a<b and a<=h<b: return True
        if a>b and (h>=a or h<b): return True
    return False

def parse_windows(s):
    out=[]
    for z in s.split(","):
        a,b=z.split("-");out.append((int(a),int(b)))
    return out

class Stat:
    def __init__(self):
        self.eq=100.;self.pk=100.;self.dd=0.;self.w=0;self.l=0;self.gw=0.;self.gl=0.;self.r=0.;self.trades=[];self.active=None
    def close(self,r,risk,exit_time):
        q=self.active
        self.trades.append({**q,"exit_time":str(exit_time),"R":r,"win":int(r>0)})
        if r>0:self.w+=1;self.gw+=r
        else:self.l+=1;self.gl+=1.
        self.r+=r;self.eq*=max(.0001,1+risk*r/100);self.pk=max(self.pk,self.eq);self.dd=max(self.dd,100*(self.pk-self.eq)/self.pk);self.active=None
    def pf(self):return self.gw/self.gl if self.gl>0 else (np.inf if self.gw>0 else 0.)

class AMDState:
    def __init__(self):self.reset()
    def reset(self):
        self.state=0;self.age=0;self.dir=0;self.acc_hi=np.nan;self.acc_lo=np.nan;self.sweep_ext=np.nan;self.disp_ext=np.nan;self.fvg=None;self.start=None
    def step_age(self):
        if self.state>0:self.age+=1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True);ap.add_argument("--out",required=True)
    ap.add_argument("--risk-pct",type=float,default=.35)
    ap.add_argument("--windows-jst",default="09-10,16-18,23-01")
    ap.add_argument("--acc-bars",type=int,default=8) # 2h on M15
    ap.add_argument("--sweep-ttl",type=int,default=4)
    ap.add_argument("--confirm-ttl",type=int,default=4)
    ap.add_argument("--pullback-ttl",type=int,default=6)
    ap.add_argument("--min-rr",type=float,default=1.25)
    a=ap.parse_args();windows=parse_windows(a.windows_jst)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.data);d.columns=[c.lower() for c in d.columns];d["datetime"]=pd.to_datetime(d.datetime)
    for c in ["open","high","low","close","volume"]:d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna().sort_values("datetime").reset_index(drop=True)
    m=m15_from_m1(d);st=Stat();s=AMDState();evidence=[]

    for i in range(max(30,a.acc_bars+2),len(m)):
        b=m.iloc[i];t=b.datetime
        if st.active:
            q=st.active;hit=None
            if q["dir"]>0:
                if b.low<=q["sl"]:hit=-1.
                elif b.high>=q["tp"]:hit=q["rr"]
            else:
                if b.high>=q["sl"]:hit=-1.
                elif b.low<=q["tp"]:hit=q["rr"]
            if hit is not None:st.close(hit,a.risk_pct,t)

        if not in_window_jst(t,windows):
            if s.state>0:s.reset()
            continue
        if not np.isfinite(b.atr) or b.atr<=0:continue

        s.step_age()
        # STATE 0: lock prior accumulation range at AMD-window entry
        if s.state==0:
            prev=m.iloc[i-a.acc-bars:i] if False else m.iloc[i-a.acc_bars:i]
            s.acc_hi=float(prev.high.max());s.acc_lo=float(prev.low.min());s.state=1;s.age=0;s.start=str(t)
            evidence.append({"time":t,"state":"ACCUMULATION_LOCK","dir":0,"price":b.close,"acc_hi":s.acc_hi,"acc_lo":s.acc_lo})
            continue

        # STATE 1: manipulation sweep outside accumulation, close back inside
        if s.state==1:
            up=b.high>s.acc_hi+b.atr*.05 and b.close<s.acc_hi
            dn=b.low<s.acc_lo-b.atr*.05 and b.close>s.acc_lo
            if up or dn:
                s.dir=-1 if up else 1;s.sweep_ext=float(b.high if up else b.low);s.state=2;s.age=0
                evidence.append({"time":t,"state":"MANIPULATION_SWEEP","dir":s.dir,"price":b.close,"acc_hi":s.acc_hi,"acc_lo":s.acc_lo})
            elif s.age>a.sweep_ttl:s.reset()
            continue

        # STATE 2: CISD/MSS + displacement after sweep
        if s.state==2:
            prior=m.iloc[max(0,i-4):i]
            struct=float(prior.low.min()) if s.dir<0 else float(prior.high.max())
            disp=abs(b.close-b.open)>=b.atr*.55
            cisd=(b.close<struct-b.atr*.02) if s.dir<0 else (b.close>struct+b.atr*.02)
            if disp and cisd:
                s.disp_ext=float(b.low if s.dir<0 else b.high)
                # simple 3-candle FVG generated with displacement
                if i>=2:
                    if s.dir<0 and m.low.iat[i-2]>b.high:s.fvg=(float(b.high),float(m.low.iat[i-2]))
                    elif s.dir>0 and m.high.iat[i-2]<b.low:s.fvg=(float(m.high.iat[i-2]),float(b.low))
                s.state=3;s.age=0
                evidence.append({"time":t,"state":"CISD_CONFIRMED","dir":s.dir,"price":b.close,"fvg":str(s.fvg)})
            elif s.age>a.confirm_ttl:s.reset()
            continue

        # STATE 3: wait for retrace into FVG or 62-79% OTE of manipulation->displacement leg
        if s.state==3:
            if s.dir<0:
                leg=s.sweep_ext-s.disp_ext
                z1=s.disp_ext+leg*.62;z2=s.disp_ext+leg*.79
            else:
                leg=s.disp_ext-s.sweep_ext
                z1=s.disp_ext-leg*.79;z2=s.disp_ext-leg*.62
            lo,hi=sorted((z1,z2))
            ote_touch=(b.high>=lo and b.low<=hi)
            fvg_touch=False
            if s.fvg:
                fl,fh=sorted(s.fvg);fvg_touch=(b.high>=fl and b.low<=fh)
            if ote_touch or fvg_touch:
                # final trigger on M15 close back in distribution direction
                trig=(b.close<b.open) if s.dir<0 else (b.close>b.open)
                if trig and st.active is None:
                    entry=float(b.close)
                    sl=s.sweep_ext+b.atr*.08 if s.dir<0 else s.sweep_ext-b.atr*.08
                    risk=abs(entry-sl)
                    # target first opposing side of accumulation; fallback 2R
                    target=s.acc_lo if s.dir<0 else s.acc_hi
                    reward=(entry-target) if s.dir<0 else (target-entry)
                    rr=reward/risk if risk>0 else 0.
                    if rr<a.min_rr:
                        rr=2.0;target=entry-risk*rr if s.dir<0 else entry+risk*rr
                    st.active={"entry_time":str(t),"dir":s.dir,"entry":entry,"sl":sl,"tp":target,"rr":rr,
                               "route":"AMD_M15","window_jst":f"{((t.hour+9)%24):02d}:00",
                               "used_ote":int(ote_touch),"used_fvg":int(fvg_touch)}
                    evidence.append({"time":t,"state":"AMD_ENTRY","dir":s.dir,"price":entry,"rr":rr,"ote":ote_touch,"fvg":fvg_touch})
                    s.reset()
            elif s.age>a.pullback_ttl:s.reset()

    N=st.w+st.l
    summary={"route":"AMD_M15","N":N,"wins":st.w,"losses":st.l,"WR_pct":100*st.w/N if N else 0.,
             "PF_R":st.pf(),"sum_R":st.r,"Return_pct":st.eq-100,"MaxDD_pct":st.dd,
             "windows_jst":windows,"timeframe":"M15","data_start":str(m.datetime.iloc[0]),"data_end":str(m.datetime.iloc[-1])}
    pd.DataFrame(st.trades).to_csv(out/"amd_m15_trades.csv",index=False)
    pd.DataFrame(evidence).to_csv(out/"amd_m15_sequence.csv",index=False)
    if st.trades:
        td=pd.DataFrame(st.trades);td["entry_time"]=pd.to_datetime(td.entry_time);td["hour_jst"]=(td.entry_time.dt.hour+9)%24
        def agg(g):
            gp=g.loc[g.R>0,"R"].sum();gl=-g.loc[g.R<0,"R"].sum()
            return pd.Series({"N":len(g),"wins":int(g.win.sum()),"WR_pct":100*g.win.mean(),"PF_R":gp/gl if gl>0 else np.inf,"sum_R":g.R.sum(),"avg_R":g.R.mean()})
        td.groupby("hour_jst").apply(agg,include_groups=False).reset_index().to_csv(out/"amd_m15_hourly.csv",index=False)
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    if st.trades:
        print(pd.read_csv(out/"amd_m15_hourly.csv").to_string(index=False))

if __name__=="__main__":main()
