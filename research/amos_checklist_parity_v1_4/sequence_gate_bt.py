#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd

def atr(d,n=14):
    pc=d.close.shift(1)
    tr=pd.concat([(d.high-d.low).abs(),(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def resample(d,rule):
    x=d.set_index("datetime")
    return pd.DataFrame({
        "open":x.open.resample(rule).first(),"high":x.high.resample(rule).max(),
        "low":x.low.resample(rule).min(),"close":x.close.resample(rule).last(),
        "volume":x.volume.resample(rule).sum()
    }).dropna().reset_index()

def session_ranges(d):
    out={}; day=d.datetime.dt.normalize(); mins=d.datetime.dt.hour*60+d.datetime.dt.minute
    for k,idx in d.groupby(day).groups.items():
        q=d.loc[idx]; m=mins.loc[idx]
        def rg(a,b):
            z=q[(m>=a)&(m<b)]
            return None if z.empty else (float(z.high.max()),float(z.low.min()))
        out[pd.Timestamp(k)]={"asia":rg(0,360),"london":rg(420,600)}
    return out

def ref(r,t):
    m=t.hour*60+t.minute; q=r.get(t.normalize(),{})
    if 420<=m<600:
        z=q.get("asia"); return (*z,"ASIA") if z else None
    if 810<=m<990:
        z=q.get("london") or q.get("asia"); return (*z,"LONDON") if z else None
    return None

def macro(t):
    m=t.hour*60+t.minute
    return any(a<=m<b for a,b in [(470,500),(550,580),(830,860),(890,920)])

def swing_level(d,i,want_low,look=28,wing=2):
    vals=d.low if want_low else d.high
    for k in range(i-wing-1,max(wing,i-look)-1,-1):
        v=vals.iat[k]
        l=vals.iloc[k-wing:k]; r=vals.iloc[k+1:k+1+wing]
        if want_low and v<l.min() and v<=r.min(): return float(v)
        if not want_low and v>l.max() and v>=r.max(): return float(v)
    s=vals.iloc[max(0,i-10):i]
    return float(s.min() if want_low else s.max())

def find_pre_fvg(d,i,di,look=20):
    # IFVG branch: long wants a prior bearish FVG, short wants a prior bullish FVG.
    for k in range(i-1,max(2,i-look)-1,-1):
        if di>0 and d.low.iat[k-2] > d.high.iat[k]:
            return (float(d.high.iat[k]),float(d.low.iat[k-2]))
        if di<0 and d.high.iat[k-2] < d.low.iat[k]:
            return (float(d.high.iat[k-2]),float(d.low.iat[k]))
    return None

def ifvg_flipped(zone,b,di):
    if zone is None:return False
    lo,hi=zone
    return b.close>hi if di>0 else b.close<lo

def ifvg_retest(zone,b):
    if zone is None:return False
    lo,hi=zone; mid=(lo+hi)/2
    return b.low<=mid<=b.high

def eq_zone(sweep,disp,di):
    if di>0:
        if disp<=sweep:return None
        r=disp-sweep
        return (disp-r*.62, disp-r*.50)
    if sweep<=disp:return None
    r=sweep-disp
    return (disp+r*.50, disp+r*.62)

def touches(z,b):
    return bool(z and b.high>=z[0] and b.low<=z[1])

def rejection(b,di):
    if b.high<=b.low:return False
    pos=(b.close-b.low)/(b.high-b.low)
    return (b.close>b.open and pos>=.58) if di>0 else (b.close<b.open and pos<=.42)

def htf_context(h,t,di,px):
    j=h.datetime.searchsorted(t,side="right")-2
    if j<20:return False
    w=h.iloc[j-20:j+1]; eq=(w.high.max()+w.low.min())/2
    return px<=eq if di>0 else px>=eq

def prior_targets(htfs,t,di,px):
    c=[]
    for h in htfs:
        j=h.datetime.searchsorted(t,side="right")-2
        if j<5:continue
        a=h.iloc[max(2,j-120):j+1]
        # IRL fractal pools
        for k in range(2,len(a)-2):
            if di>0 and a.high.iloc[k]>=a.high.iloc[k-2:k].max() and a.high.iloc[k]>=a.high.iloc[k+1:k+3].max():
                v=float(a.high.iloc[k])
                if v>px:c.append(v)
            if di<0 and a.low.iloc[k]<=a.low.iloc[k-2:k].min() and a.low.iloc[k]<=a.low.iloc[k+1:k+3].min():
                v=float(a.low.iloc[k])
                if v<px:c.append(v)
        # HTF imbalance targets
        for k in range(2,len(a)):
            if di>0 and a.low.iloc[k-2] > a.high.iloc[k]: # bearish FVG above can be filled upward
                lo,hi=float(a.high.iloc[k]),float(a.low.iloc[k-2])
                if hi>px:c.append(max(px+1e-9,lo))
            if di<0 and a.high.iloc[k-2] < a.low.iloc[k]: # bullish FVG below can be filled downward
                lo,hi=float(a.high.iloc[k-2]),float(a.low.iloc[k])
                if lo<px:c.append(min(px-1e-9,hi))
    return c

def choose_target(htfs,t,di,e,sl,min_rr):
    risk=abs(e-sl)
    if risk<=0:return None
    vals=prior_targets(htfs,t,di,e)
    scored=[]
    for v in vals:
        rew=(v-e) if di>0 else (e-v)
        if rew<=0:continue
        rr=rew/risk
        if rr>=min_rr:scored.append((rr,v))
    if not scored:return None
    scored.sort(key=lambda x:x[0])
    return scored[0][1],scored[0][0]

class S:
    def __init__(self):
        self.n=self.w=self.l=0;self.gw=self.gl=self.sumr=0.;self.eq=self.peak=100.;self.dd=0.;self.pos=None
    def close(self,r,risk=.35):
        self.n+=1
        if r>0:self.w+=1;self.gw+=r
        else:self.l+=1;self.gl+=1
        self.sumr+=r;self.eq*=max(.001,1+risk*r/100);self.peak=max(self.peak,self.eq);self.dd=max(self.dd,100*(self.peak-self.eq)/self.peak);self.pos=None
    def wr(self):return 100*self.w/self.n if self.n else 0.
    def pf(self):return self.gw/self.gl if self.gl else (np.inf if self.gw else 0.)

def upd(s,b):
    p=s.pos
    if not p:return
    if p["d"]>0:
        if b.low<=p["sl"]:s.close(-1)
        elif b.high>=p["tp"]:s.close(p["rr"])
    else:
        if b.high>=p["sl"]:s.close(-1)
        elif b.low<=p["tp"]:s.close(p["rr"])

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.data); d.columns=[x.lower() for x in d.columns]; d["datetime"]=pd.to_datetime(d.datetime)
    for c in ["open","high","low","close","volume"]:d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna().sort_values("datetime").reset_index(drop=True);d["atr"]=atr(d);d["vma"]=d.volume.shift(1).rolling(20).mean()
    m15=resample(d,"15min");m30=resample(d,"30min");ranges=session_ranges(d)

    configs={}
    for model in ["IFVG","EQ_REBALANCE"]:
        for minrr in [1.5,2.0,3.0,4.0]:
            for ctx in ["CORE","CORE+HTF","CORE+MACRO","CORE+VOL"]:
                configs[(model,minrr,ctx)]=S()

    state=None;evidence=[];sid=0
    for i in range(60,len(d)):
        b=d.iloc[i];t=b.datetime
        if not np.isfinite(b.atr) or b.atr<=0:continue
        for s in configs.values(): upd(s,b)

        R=ref(ranges,t)
        if state is None:
            if not R:continue
            H,L,_=R;mn=b.atr*.03;mx=b.atr*.80
            sh=b.high>H+mn and b.close<H and (b.high-H)<=mx
            sl=b.low<L-mn and b.close>L and (L-b.low)<=mx
            if not(sh or sl):continue
            di=-1 if sh else 1;sid+=1
            state={"sid":sid,"d":di,"stage":"WAIT_MSS","age":0,
                   "sweep":float(b.high if sh else b.low),
                   "struct":swing_level(d,i,want_low=(di<0)),
                   "pre_fvg":find_pre_fvg(d,i,di),
                   "ifvg_flip":False,"disp":float(b.low if di<0 else b.high),
                   "vol":False,"macro":macro(t),"htf":htf_context(m15,t,di,float(b.close))}
            continue

        state["age"]+=1
        if state["age"]>24: state=None; continue
        di=state["d"]
        if di<0: state["sweep"]=max(state["sweep"],float(b.high))
        else: state["sweep"]=min(state["sweep"],float(b.low))

        if state["stage"]=="WAIT_MSS":
            body=abs(b.close-b.open)
            brk=(b.close<state["struct"]-b.atr*.01) if di<0 else (b.close>state["struct"]+b.atr*.01)
            disp=body>=b.atr*.45
            if ifvg_flipped(state["pre_fvg"],b,di): state["ifvg_flip"]=True
            if brk and disp:
                state["stage"]="WAIT_ENTRY";state["age"]=0
                state["disp"]=float(b.low if di<0 else b.high)
                state["vol"]=bool(np.isfinite(b.vma) and b.vma>0 and b.volume>=1.15*b.vma)
            continue

        if di<0: state["disp"]=min(state["disp"],float(b.low))
        else: state["disp"]=max(state["disp"],float(b.high))
        if ifvg_flipped(state["pre_fvg"],b,di): state["ifvg_flip"]=True

        eqz=eq_zone(state["sweep"],state["disp"],di)
        model_ready={
            "IFVG": state["ifvg_flip"] and ifvg_retest(state["pre_fvg"],b),
            "EQ_REBALANCE": touches(eqz,b),
        }

        for model,ready in model_ready.items():
            if not ready or not rejection(b,di):continue
            e=float(b.close); sl=float(state["sweep"]-b.atr*.08 if di>0 else state["sweep"]+b.atr*.08)
            for minrr in [1.5,2.0,3.0,4.0]:
                tg=choose_target([m15,m30],t,di,e,sl,minrr)
                if tg is None:continue
                tp,rr=tg
                for ctx in ["CORE","CORE+HTF","CORE+MACRO","CORE+VOL"]:
                    ok=(ctx=="CORE" or
                        (ctx=="CORE+HTF" and state["htf"]) or
                        (ctx=="CORE+MACRO" and state["macro"]) or
                        (ctx=="CORE+VOL" and state["vol"]))
                    if not ok:continue
                    s=configs[(model,minrr,ctx)]
                    if s.pos is None:s.pos={"d":di,"e":e,"sl":sl,"tp":tp,"rr":rr}
                evidence.append({"time":t,"sid":state["sid"],"dir":di,"model":model,
                                 "liquidity_sweep":1,"mss_confirmed":1,"entry_model":1,"clear_target":1,
                                 "htf_pda":int(state["htf"]),"macro_window":int(state["macro"]),
                                 "volume_influx":int(state["vol"]),"entry":e,"sl":sl,"tp":tp,"rr":rr})
            state=None
            break

    rows=[]
    for (model,minrr,ctx),s in configs.items():
        pf=s.pf();score=(min(pf,8)*math.sqrt(max(s.n,1))/(1+s.dd)) if s.n>=10 else -1
        rows.append({"model":model,"min_target_R":minrr,"context":ctx,"N":s.n,"wins":s.w,"losses":s.l,
                     "WR_pct":s.wr(),"PF_R":pf,"sum_R":s.sumr,"Return_pct":s.eq-100,"MaxDD_pct":s.dd,"objective":score})
    r=pd.DataFrame(rows);r.replace([np.inf,-np.inf],np.nan).to_csv(out/"sequence_matrix.csv",index=False)
    pd.DataFrame(evidence).to_csv(out/"entry_evidence.csv",index=False)
    top=r[r.N>=10].sort_values(["objective","PF_R","N"],ascending=False);top.to_csv(out/"ranked.csv",index=False)
    meta={"verification_level":"M1_OHLC_ORDERED_GATE_SCREEN",
          "mandatory_order":["Liquidity Sweep","MSS Confirmed","Entry Model (IFVG OR EQ Rebalance)","Clear Targets","Entry on retrace/rejection"],
          "optional_context":["HTF PDA Delivery","Macro Window","Volume Influx"],
          "start":str(d.datetime.iloc[0]),"end":str(d.datetime.iloc[-1]),"rows":len(d),
          "limitations":["mid-quote OHLCV","no spread/slippage/swap","Raw BidAsk Nautilus required for promotion"]}
    (out/"manifest.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print("DATA",meta["start"],"->",meta["end"],"ROWS",len(d),"EVIDENCE",len(evidence))
    print(top.head(20).to_string(index=False))

if __name__=="__main__": main()
