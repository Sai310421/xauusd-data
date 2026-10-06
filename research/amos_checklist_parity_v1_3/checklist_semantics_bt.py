#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd

ROWS=["sweep","htf_pda","mss","macro","equilibrium","volume","ifvg","clear_target"]
BITS={r:1<<i for i,r in enumerate(ROWS)}
PROFILES=["FVG50_REJECT","EQ_REJECT","EITHER_REJECT","FVG50_TOUCH","EQ_TOUCH","EITHER_TOUCH"]
MASKS={
 "score_only":0,
 "sweep_mss":BITS["sweep"]|BITS["mss"],
 "sweep_mss_target":BITS["sweep"]|BITS["mss"]|BITS["clear_target"],
 "sweep_ifvg":BITS["sweep"]|BITS["ifvg"],
 "sweep_mss_ifvg":BITS["sweep"]|BITS["mss"]|BITS["ifvg"],
 "sweep_structure_target":BITS["sweep"]|BITS["clear_target"], # structure handled separately
}

def atr(d,n=14):
    pc=d.close.shift(1)
    tr=pd.concat([(d.high-d.low).abs(),(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def m15(d):
    x=d.set_index("datetime")
    return pd.DataFrame({"open":x.open.resample("15min").first(),"high":x.high.resample("15min").max(),
        "low":x.low.resample("15min").min(),"close":x.close.resample("15min").last(),
        "volume":x.volume.resample("15min").sum()}).dropna().reset_index()

def day_ranges(d):
    out={};day=d.datetime.dt.normalize();mins=d.datetime.dt.hour*60+d.datetime.dt.minute
    for k,idx in d.groupby(day).groups.items():
        q=d.loc[idx];m=mins.loc[idx]
        def rg(a,b):
            z=q[(m>=a)&(m<b)]
            return None if z.empty else (float(z.high.max()),float(z.low.min()))
        out[pd.Timestamp(k)]={"asia":rg(0,360),"london":rg(420,600)}
    return out

def ref(ranges,t):
    m=t.hour*60+t.minute;r=ranges.get(t.normalize(),{})
    if 420<=m<600:
        z=r.get("asia");return (*z,"ASIA") if z else None
    if 810<=m<990:
        z=r.get("london") or r.get("asia");return (*z,"LONDON") if z else None
    return None

def ict_macro(t):
    m=t.hour*60+t.minute
    # UTC screening windows; parameters should be broker-time shifted in MT5.
    return any(a<=m<b for a,b in [(470,500),(550,580),(830,860),(890,920)])

def last_swing(d,i,want_low,look=30,wing=2):
    lo=max(wing,i-look);hi=i-wing
    for k in range(hi,lo-1,-1):
        vals=d.low if want_low else d.high
        v=vals.iat[k]
        left=vals.iloc[k-wing:k];right=vals.iloc[k+1:k+1+wing]
        if want_low and v<left.min() and v<=right.min():return float(v)
        if not want_low and v>left.max() and v>=right.max():return float(v)
    vals=d.low if want_low else d.high
    return float(vals.iloc[max(0,i-10):i].min() if want_low else vals.iloc[max(0,i-10):i].max())

def fvg_on_bar(d,i,di):
    if i<2:return None
    if di>0 and d.high.iat[i-2] < d.low.iat[i]:
        return float(d.high.iat[i-2]),float(d.low.iat[i])
    if di<0 and d.low.iat[i-2] > d.high.iat[i]:
        return float(d.high.iat[i]),float(d.low.iat[i-2])
    return None

def htf_pda(h,t,di,px,look=24):
    j=h.datetime.searchsorted(t,side="right")-2
    if j<look:return False
    w=h.iloc[j-look+1:j+1]
    # approximate HTF PDA delivery with premium/discount + active 15m imbalance context
    eq=(w.high.max()+w.low.min())/2
    pd_ok=(px<=eq if di>0 else px>=eq)
    fvg=False
    for k in range(max(2,j-12),j+1):
        if di>0 and h.high.iat[k-2] < h.low.iat[k]:
            lo,hi=h.high.iat[k-2],h.low.iat[k]
            if lo<=px<=hi:fvg=True
        if di<0 and h.low.iat[k-2] > h.high.iat[k]:
            lo,hi=h.high.iat[k],h.low.iat[k-2]
            if lo<=px<=hi:fvg=True
    return bool(pd_ok or fvg)

def target_info(di,e,sl,H,L):
    R=abs(e-sl)
    if R<=0:return False,0,0
    tp=H if di>0 else L
    rew=(tp-e) if di>0 else (e-tp)
    rr=rew/R
    return bool(rew>0 and rr>=1.0),float(tp),float(rr)

def trigger(profile,b,di,ifvg,eqzone):
    fvg_touch=False
    if ifvg:
        mid=(ifvg[0]+ifvg[1])/2
        fvg_touch=(b.low<=mid<=b.high)
    eq_touch=(b.high>=eqzone[0] and b.low<=eqzone[1]) if eqzone else False
    touch = fvg_touch if profile.startswith("FVG50") else eq_touch if profile.startswith("EQ_") else (fvg_touch or eq_touch)
    if not touch:return False
    if profile.endswith("TOUCH"):return True
    # direction-consistent rejection at the touch bar
    if di>0:return bool(b.close>b.open and b.close>(b.low+(b.high-b.low)*0.55))
    return bool(b.close<b.open and b.close<(b.low+(b.high-b.low)*0.45))

class Stat:
    __slots__=("n","w","l","r","gw","gl","eq","peak","dd","pos","day","today")
    def __init__(self):
        self.n=self.w=self.l=0;self.r=self.gw=self.gl=0.;self.eq=self.peak=100.;self.dd=0.;self.pos=None;self.day=None;self.today=0
    def close(self,hit,risk=.35):
        self.n+=1
        if hit>0:self.w+=1;self.gw+=hit
        else:self.l+=1;self.gl+=1
        self.r+=hit;self.eq*=max(.001,1+risk/100*hit);self.peak=max(self.peak,self.eq);self.dd=max(self.dd,100*(self.peak-self.eq)/self.peak);self.pos=None
    def pf(self):return self.gw/self.gl if self.gl else (np.inf if self.gw else 0.)
    def wr(self):return 100*self.w/self.n if self.n else 0.

def update(st,b):
    p=st.pos
    if not p:return
    if p["d"]>0:
        if b.low<=p["sl"]:st.close(-1);return
        if b.high>=p["tp"]:st.close(p["rr"]);return
    else:
        if b.high>=p["sl"]:st.close(-1);return
        if b.low<=p["tp"]:st.close(p["rr"]);return

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.data);d.columns=[x.lower() for x in d.columns];d["datetime"]=pd.to_datetime(d.datetime)
    for c in ["open","high","low","close","volume"]:d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna().sort_values("datetime").reset_index(drop=True)
    d["atr"]=atr(d);d["vma"]=d.volume.shift(1).rolling(20).mean()
    h=m15(d);ranges=day_ranges(d)
    configs={(p,th,mn):Stat() for p in PROFILES for th in [4,5,6] for mn in MASKS}
    active=set();evidence=[]
    state=None;setup_id=0
    for i in range(60,len(d)):
        b=d.iloc[i];t=b.datetime;day=str(t.date())
        if not np.isfinite(b.atr) or b.atr<=0:continue
        for k in tuple(active):
            st=configs[k]
            if st.day!=day:st.day=day;st.today=0
            update(st,b)
            if st.pos is None:active.discard(k)

        R=ref(ranges,t)
        if state is None:
            if not R:continue
            H,L,_=R;mn=b.atr*.03;mx=b.atr*.8
            sh=b.high>H+mn and b.close<H and b.high-H<=mx
            sl=b.low<L-mn and b.close>L and L-b.low<=mx
            if not(sh or sl):continue
            di=-1 if sh else 1;setup_id+=1
            state={"id":setup_id,"d":di,"age":0,"phase":"WAIT_CONFIRM",
                   "sweep":float(b.high if sh else b.low),
                   "struct":last_swing(d,i,want_low=(di<0)),
                   "pda":htf_pda(h,t,di,float(b.close)),
                   "macro":ict_macro(t),"mss":False,"ifvg":None,"volume":False,
                   "confirm_i":None,"disp_end":None}
            continue

        state["age"]+=1
        if state["age"]>24:
            state=None;continue
        di=state["d"]
        if state["phase"]=="WAIT_CONFIRM":
            if di<0:state["sweep"]=max(state["sweep"],float(b.high))
            else:state["sweep"]=min(state["sweep"],float(b.low))
            body=abs(b.close-b.open)
            mss=(b.close<state["struct"]-b.atr*.01 if di<0 else b.close>state["struct"]+b.atr*.01) and body>=b.atr*.45
            gap=fvg_on_bar(d,i,di)
            if mss:
                state["mss"]=True;state["confirm_i"]=i;state["disp_end"]=float(b.low if di<0 else b.high)
                state["ifvg"]=gap
                state["volume"]=bool(np.isfinite(b.vma) and b.vma>0 and b.volume>=b.vma*1.15)
                state["phase"]="WAIT_RETRACE";state["age"]=0
            continue

        # WAIT_RETRACE: event states are latched from setup/confirmation.
        if di<0:state["disp_end"]=min(state["disp_end"],float(b.low))
        else:state["disp_end"]=max(state["disp_end"],float(b.high))
        if state["ifvg"] is None:
            gap=fvg_on_bar(d,i,di)
            if gap:state["ifvg"]=gap
        if np.isfinite(b.vma) and b.vma>0 and b.volume>=b.vma*1.15:state["volume"]=True

        se=state["sweep"];de=state["disp_end"]
        if di<0 and se>de:
            z=se-de;eqzone=(de+z*.50,de+z*.79)
        elif di>0 and de>se:
            z=de-se;eqzone=(de-z*.79,de-z*.50)
        else:eqzone=None
        equilibrium=bool(eqzone and b.high>=eqzone[0] and b.low<=eqzone[1])
        entry=float(b.close);sl=float(se-b.atr*.08 if di>0 else se+b.atr*.08)
        if not R:continue
        H,L,_=R;clear,tp,nrr=target_info(di,entry,sl,H,L)
        if not clear:
            rr=1.5;dist=abs(entry-sl);tp=entry+dist*rr if di>0 else entry-dist*rr
        else:rr=nrr
        vals={"sweep":True,"htf_pda":state["pda"],"mss":state["mss"],"macro":state["macro"],
              "equilibrium":equilibrium,"volume":state["volume"],"ifvg":state["ifvg"] is not None,"clear_target":clear}
        score=sum(int(x) for x in vals.values());mask=sum(BITS[k] for k,v in vals.items() if v)
        fired_any=False
        for profile in PROFILES:
            if not trigger(profile,b,di,state["ifvg"],eqzone):continue
            for th in [4,5,6]:
                if score<th:continue
                for mn,req in MASKS.items():
                    if (req & ~mask)!=0:continue
                    if mn=="sweep_structure_target" and not (vals["mss"] or vals["ifvg"]):continue
                    key=(profile,th,mn);st=configs[key]
                    if st.day!=day:st.day=day;st.today=0
                    if st.pos is not None or st.today>=3:continue
                    st.pos={"d":di,"e":entry,"sl":sl,"tp":tp,"rr":rr};st.today+=1;active.add(key);fired_any=True
        evidence.append({"time":t,"setup":state["id"],"dir":di,"profile_touch":1 if fired_any else 0,
                         **{k:int(v) for k,v in vals.items()},"score":score,"entry":entry,"sl":sl,"tp":tp,"rr":rr})
        if fired_any:
            # one setup should not repeatedly fire the same semantic event.
            state=None

    for k in active:
        configs[k].pos=None
    rows=[]
    for (p,th,mn),s in configs.items():
        pf=s.pf();obj=(min(pf,8)*math.sqrt(max(s.n,1))/(1+s.dd)) if s.n>=15 else -1
        rows.append({"profile":p,"threshold":th,"mandatory":mn,"N":s.n,"wins":s.w,"losses":s.l,
                     "WR_pct":s.wr(),"PF_R":pf,"sum_R":s.r,"Return_pct":s.eq-100,"MaxDD_pct":s.dd,"objective":obj})
    res=pd.DataFrame(rows);res.replace([np.inf,-np.inf],np.nan).to_csv(out/"semantics_matrix.csv",index=False)
    pd.DataFrame(evidence).to_csv(out/"setup_evidence.csv",index=False)
    top=res[(res.N>=15)&res.PF_R.notna()].sort_values(["objective","PF_R","N"],ascending=False)
    top.head(50).to_csv(out/"top50.csv",index=False)
    meta={"verification_level":"M1_OHLC_SEMANTICS_SCREEN","start":str(d.datetime.iloc[0]),"end":str(d.datetime.iloc[-1]),
          "rows":len(d),"configs":len(configs),"profiles":PROFILES,
          "key_change":"event conditions latch; entry waits for IFVG50/EQ retrace; optional rejection candle",
          "limitations":["mid-quote OHLCV","no spread/slippage/swap","close-fill screening","Raw BidAsk/Nautilus required before promotion"]}
    (out/"manifest.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print("DATA",meta["start"],"->",meta["end"],"rows",len(d),"CONFIGS",len(configs),"EVIDENCE",len(evidence))
    print("TOP20")
    print(top.head(20).to_string(index=False))
    print("BY_PROFILE")
    q=res[res.N>=15].groupby("profile",as_index=False).apply(lambda x:x.sort_values("objective",ascending=False).head(1),include_groups=False)
    print(q.to_string(index=False))

if __name__=="__main__":main()
