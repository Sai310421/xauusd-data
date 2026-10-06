#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

ROUTES={
    "AMD_ASIA":{"activate":(9,10),"acc":"PREV_NY"},
    "AMD_LONDON":{"activate":(16,18),"acc":"ASIA"},
    "AMD_NY":{"activate":(23,24),"acc":"LONDON"},
}

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
    y["atr"]=atr(y,14);y["vma"]=y.volume.shift(1).rolling(20).mean()
    y["jst"]=y.datetime+pd.Timedelta(hours=9)
    y["jst_date"]=y.jst.dt.normalize()
    y["jst_min"]=y.jst.dt.hour*60+y.jst.dt.minute
    return y

def precompute_session_ranges(m):
    out={}
    days=sorted(m.jst_date.unique())
    for day in days:
        day=pd.Timestamp(day)
        q=m[m.jst_date==day]
        def rg(a,b):
            w=q[(q.jst_min>=a)&(q.jst_min<b)]
            if w.empty:return None
            return float(w.high.max()),float(w.low.min())
        out[(day,"ASIA")]=rg(9*60,16*60)
        out[(day,"LONDON")]=rg(16*60,23*60)
        # overnight NY accumulation ending before Asia activation
        prev=day-pd.Timedelta(days=1)
        q1=m[(m.jst_date==prev)&(m.jst_min>=23*60)]
        q2=m[(m.jst_date==day)&(m.jst_min<6*60)]
        w=pd.concat([q1,q2]).sort_values("jst")
        out[(day,"PREV_NY")]=None if w.empty else (float(w.high.max()),float(w.low.min()))
    return out

def activation(route,row):
    a,b=ROUTES[route]["activate"];h=int(row.jst.hour)
    return a<=h<b

class Stat:
    def __init__(self):
        self.eq=100.;self.pk=100.;self.dd=0.;self.w=0;self.l=0;self.gw=0.;self.gl=0.;self.r=0.;self.trades=[];self.active=None
    def close(self,r,risk,exit_time):
        q=self.active;self.trades.append({**q,"exit_time":str(exit_time),"R":r,"win":int(r>0)})
        if r>0:self.w+=1;self.gw+=r
        else:self.l+=1;self.gl+=1.
        self.r+=r;self.eq*=max(.0001,1+risk*r/100);self.pk=max(self.pk,self.eq);self.dd=max(self.dd,100*(self.pk-self.eq)/self.pk);self.active=None
    def pf(self):return self.gw/self.gl if self.gl>0 else (np.inf if self.gw>0 else 0.)

class AMDState:
    def __init__(self,route):self.route=route;self.reset()
    def reset(self):
        self.state=0;self.age=0;self.dir=0;self.acc_hi=np.nan;self.acc_lo=np.nan;self.sweep_ext=np.nan;self.disp_ext=np.nan
        self.fvg=None;self.anchor_day=None;self.start=None;self.sweep_time=None;self.confirm_time=None
    def tick(self):
        if self.state>0:self.age+=1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True);ap.add_argument("--out",required=True)
    ap.add_argument("--risk-pct",type=float,default=.35)
    ap.add_argument("--sweep-ttl",type=int,default=8)
    ap.add_argument("--confirm-ttl",type=int,default=8)
    ap.add_argument("--pullback-ttl",type=int,default=12)
    ap.add_argument("--min-rr",type=float,default=1.25)
    a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.data);d.columns=[c.lower() for c in d.columns];d["datetime"]=pd.to_datetime(d.datetime)
    for c in ["open","high","low","close","volume"]:d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna().sort_values("datetime").reset_index(drop=True)
    m=m15_from_m1(d);ranges=precompute_session_ranges(m)
    stats={r:Stat() for r in ROUTES};states={r:AMDState(r) for r in ROUTES};evidence=[]

    for i in range(40,len(m)):
        b=m.iloc[i];t=b.datetime
        if not np.isfinite(b.atr) or b.atr<=0:continue

        # Manage positions independently by AMD route.
        for route,st in stats.items():
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
            if s.state==0:
                if not activation(route,b):continue
                day=pd.Timestamp(b.jst_date);acc=ranges.get((day,ROUTES[route]["acc"]))
                if acc is None:continue
                s.acc_hi,s.acc_lo=acc;s.anchor_day=str(day.date());s.start=str(t);s.state=1;s.age=0
                evidence.append({"time":t,"route":route,"state":"ACCUMULATION_LOCK","dir":0,"price":b.close,
                                 "acc_source":ROUTES[route]["acc"],"acc_hi":s.acc_hi,"acc_lo":s.acc_lo})
                continue

            if s.state==1:
                up=b.high>s.acc_hi+b.atr*.05 and b.close<s.acc_hi
                dn=b.low<s.acc_lo-b.atr*.05 and b.close>s.acc_lo
                if up or dn:
                    s.dir=-1 if up else 1;s.sweep_ext=float(b.high if up else b.low);s.sweep_time=str(t);s.state=2;s.age=0
                    evidence.append({"time":t,"route":route,"state":"MANIPULATION_SWEEP","dir":s.dir,"price":b.close,
                                     "acc_hi":s.acc_hi,"acc_lo":s.acc_lo})
                elif s.age>a.sweep_ttl:s.reset()
                continue

            if s.state==2:
                # Confirm structure only AFTER manipulation.
                prior=m.iloc[max(0,i-4):i]
                struct=float(prior.low.min()) if s.dir<0 else float(prior.high.max())
                body=abs(float(b.close-b.open))
                disp=body>=float(b.atr)*.45
                cisd=(b.close<struct-b.atr*.01) if s.dir<0 else (b.close>struct+b.atr*.01)
                if disp and cisd:
                    s.disp_ext=float(b.low if s.dir<0 else b.high);s.confirm_time=str(t);s.fvg=None
                    if i>=2:
                        if s.dir<0 and m.low.iat[i-2]>b.high:s.fvg=(float(b.high),float(m.low.iat[i-2]))
                        elif s.dir>0 and m.high.iat[i-2]<b.low:s.fvg=(float(m.high.iat[i-2]),float(b.low))
                    s.state=3;s.age=0
                    evidence.append({"time":t,"route":route,"state":"CISD_MSS_CONFIRMED","dir":s.dir,"price":b.close,
                                     "fvg_lo":None if s.fvg is None else min(s.fvg),"fvg_hi":None if s.fvg is None else max(s.fvg)})
                elif s.age>a.confirm_ttl:s.reset()
                continue

            if s.state==3:
                # OTE is defined only after displacement; FVG is optional confluence.
                if s.dir<0:
                    leg=s.sweep_ext-s.disp_ext
                    z1=s.disp_ext+leg*.62;z2=s.disp_ext+leg*.79
                else:
                    leg=s.disp_ext-s.sweep_ext
                    z1=s.disp_ext-leg*.79;z2=s.disp_ext-leg*.62
                if leg<=0:
                    s.reset();continue
                lo,hi=sorted((z1,z2));ote_touch=(b.high>=lo and b.low<=hi)
                fvg_touch=False
                if s.fvg:
                    fl,fh=sorted(s.fvg);fvg_touch=(b.high>=fl and b.low<=fh)
                if ote_touch or fvg_touch:
                    # M15 final trigger: rejection close in distribution direction.
                    rng=max(float(b.high-b.low),1e-9)
                    if s.dir<0:
                        trigger=(b.close<b.open and (b.high-b.close)/rng>=.45)
                    else:
                        trigger=(b.close>b.open and (b.close-b.low)/rng>=.45)
                    st=stats[route]
                    if trigger and st.active is None:
                        entry=float(b.close);sl=s.sweep_ext+b.atr*.08 if s.dir<0 else s.sweep_ext-b.atr*.08;risk=abs(entry-sl)
                        target=s.acc_lo if s.dir<0 else s.acc_hi
                        reward=(entry-target) if s.dir<0 else (target-entry);rr=reward/risk if risk>0 else 0.
                        if rr<a.min_rr:
                            rr=2.0;target=entry-risk*rr if s.dir<0 else entry+risk*rr
                        st.active={"entry_time":str(t),"dir":s.dir,"entry":entry,"sl":sl,"tp":target,"rr":rr,
                                   "route":route,"acc_source":ROUTES[route]["acc"],"sweep_time":s.sweep_time,
                                   "confirm_time":s.confirm_time,"used_ote":int(ote_touch),"used_fvg":int(fvg_touch)}
                        evidence.append({"time":t,"route":route,"state":"AMD_ENTRY","dir":s.dir,"price":entry,"rr":rr,
                                         "ote":int(ote_touch),"fvg":int(fvg_touch)})
                        s.reset()
                elif s.age>a.pullback_ttl:s.reset()

    rows=[];all_trades=[]
    for route,st in stats.items():
        N=st.w+st.l
        rows.append({"route":route,"acc_source":ROUTES[route]["acc"],"N":N,"wins":st.w,"losses":st.l,
                     "WR_pct":100*st.w/N if N else 0.,"PF_R":st.pf(),"sum_R":st.r,"Return_pct":st.eq-100,"MaxDD_pct":st.dd})
        all_trades.extend(st.trades)
    pd.DataFrame(rows).to_csv(out/"amd_m15_route_kpi.csv",index=False)
    pd.DataFrame(all_trades).to_csv(out/"amd_m15_trades.csv",index=False)
    pd.DataFrame(evidence).to_csv(out/"amd_m15_sequence.csv",index=False)
    totalN=sum(r["N"] for r in rows);totalW=sum(r["wins"] for r in rows);gp=sum(s.gw for s in stats.values());gl=sum(s.gl for s in stats.values())
    summary={"route":"AMD_M15_SESSION_RANGE","N":totalN,"wins":totalW,"losses":totalN-totalW,
             "WR_pct":100*totalW/totalN if totalN else 0.,"PF_R":gp/gl if gl>0 else 0.,
             "sum_R":sum(s.r for s in stats.values()),"timeframe":"M15",
             "data_start":str(m.datetime.iloc[0]),"data_end":str(m.datetime.iloc[-1]),
             "routes":ROUTES}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__":main()
