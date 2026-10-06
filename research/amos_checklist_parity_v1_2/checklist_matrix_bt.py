#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd

ROWS=["sweep","htf_pda","mss","macro","equilibrium","volume","ifvg","clear_target"]
BIT={r:1<<i for i,r in enumerate(ROWS)}

def win(t,a,b):
    m=t.hour*60+t.minute; x,y=map(lambda s:int(s[:2])*60+int(s[3:]),(a,b))
    return x<=m<y if x<=y else (m>=x or m<y)

def atr(d,n=14):
    p=d.close.shift(1)
    tr=pd.concat([(d.high-d.low).abs(),(d.high-p).abs(),(d.low-p).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def m15(d):
    x=d.set_index("datetime")
    return pd.DataFrame({"open":x.open.resample("15min").first(),"high":x.high.resample("15min").max(),
      "low":x.low.resample("15min").min(),"close":x.close.resample("15min").last(),
      "volume":x.volume.resample("15min").sum()}).dropna().reset_index()

def ifvg(d,i,di,lb=35):
    for k in range(i-2,max(1,i-lb)-1,-1):
        if di<0 and d.low.iat[k]>d.high.iat[k-2]:
            lo,hi=d.high.iat[k-2],d.low.iat[k]
            if (d.close.iloc[k+1:i+1]<lo).any(): return True,float(lo),float(hi)
        if di>0 and d.high.iat[k]<d.low.iat[k-2]:
            lo,hi=d.high.iat[k],d.low.iat[k-2]
            if (d.close.iloc[k+1:i+1]>hi).any(): return True,float(lo),float(hi)
    return False,0.,0.

def htf_pda(h,t,di,px,look=40):
    j=h.datetime.searchsorted(t,side="right")-2
    if j<look:return False
    w=h.iloc[j-look+1:j+1]; eq=(w.high.max()+w.low.min())/2
    return px>=eq if di<0 else px<=eq

def ref(d,i):
    t=d.datetime.iat[i]; m=t.hour*60+t.minute; day=t.normalize()
    def rg(a,b):
        w=d[(d.datetime>=day+pd.Timedelta(minutes=a))&(d.datetime<day+pd.Timedelta(minutes=b))]
        return None if w.empty else (float(w.high.max()),float(w.low.min()))
    if 420<=m<600:
        r=rg(0,360); return (*r,"ASIA") if r else None
    if 810<=m<990:
        r=rg(420,600) or rg(0,360); return (*r,"LONDON") if r else None
    return None

def eqok(di,b,se,de):
    if di<0:
        z=se-de
        if z<=0:return False
        a,b2=de+z*.50,de+z*.79
        return b.high>=a and b.low<=b2 and b.close<b2
    z=de-se
    if z<=0:return False
    a,b2=de-z*.79,de-z*.50
    return b.low<=b2 and b.high>=a and b.close>a

def maskname(m): return "+".join(ROWS[i] for i in range(8) if m&(1<<i)) or "NONE"

class G:
    def __init__(self,th,mask):
        self.th=th;self.mask=mask;self.eq=100.;self.peak=100.;self.dd=0.;self.w=0;self.l=0;self.be=0
        self.gw=0.;self.gl=0.;self.r=0.;self.active=None;self.day=None;self.n_day=0
    def n(self):return self.w+self.l+self.be
    def wr(self):return 100*self.w/self.n() if self.n() else 0.
    def pf(self):return self.gw/self.gl if self.gl>0 else (np.inf if self.gw>0 else 0.)

def gpass(v,th,mask):
    return sum(int(v[r]) for r in ROWS)>=th and all(v[ROWS[i]] for i in range(8) if mask&(1<<i))

def upd(g,b,risk=.35,use_be=False):
    t=g.active
    if not t:return
    di,e,sl,tp,rr=t["d"],t["e"],t["sl"],t["tp"],t["rr"]; rd=abs(e-sl)
    hit=None
    if di>0:
        if b.low<=sl:hit=-1.
        elif b.high>=tp:hit=rr
        elif use_be and not t["be"] and b.high>=e+rd:t["be"]=True;t["sl"]=e+rd*.05
    else:
        if b.high>=sl:hit=-1.
        elif b.low<=tp:hit=rr
        elif use_be and not t["be"] and b.low<=e-rd:t["be"]=True;t["sl"]=e-rd*.05
    if hit is None:return
    if hit>0:g.w+=1;g.gw+=hit
    elif hit<0:g.l+=1;g.gl+=1.
    else:g.be+=1
    g.r+=hit;g.eq*=max(.0001,1+risk/100*hit);g.peak=max(g.peak,g.eq);g.dd=max(g.dd,100*(g.peak-g.eq)/g.peak);g.active=None

def main():
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--out",required=True)
    p.add_argument("--thresholds",default="4,5,6");p.add_argument("--risk-pct",type=float,default=.35)
    p.add_argument("--min-trades-rank",type=int,default=20);p.add_argument("--use-be",action="store_true")
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.data);d.columns=[x.lower() for x in d.columns];d["datetime"]=pd.to_datetime(d.datetime)
    for c in ["open","high","low","close","volume"]:d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna().sort_values("datetime").reset_index(drop=True);d["atr"]=atr(d);d["vma"]=d.volume.shift(1).rolling(20).mean();h=m15(d)
    gates={(th,m):G(th,m) for th in map(int,a.thresholds.split(",")) for m in range(256)}
    state="IDLE";age=0;di=0;se=st=de=0.;mss=False;il=False;setup=0;fired=set();ev=[]
    for i in range(60,len(d)):
        b=d.iloc[i];t=b.datetime;day=str(t.date())
        if not np.isfinite(b.atr) or b.atr<=0:continue
        for g in gates.values():
            if g.day!=day:g.day=day;g.n_day=0
            upd(g,b,a.risk_pct,a.use_be)
        rr=ref(d,i)
        if state=="IDLE":
            if not rr:continue
            H,L,_=rr; mn=b.atr*.03;mx=b.atr*.80
            sh=b.high>H+mn and b.close<H and b.high-H<=mx
            sl=b.low<L-mn and b.close>L and L-b.low<=mx
            if sh:
                state="SWEPT";di=-1;age=0;se=float(b.high);st=float(d.low.iloc[i-8:i].min());mss=False;il=False;setup+=1;fired=set()
            elif sl:
                state="SWEPT";di=1;age=0;se=float(b.low);st=float(d.high.iloc[i-8:i].max());mss=False;il=False;setup+=1;fired=set()
            continue
        age+=1
        if age>16:state="IDLE";continue
        if not rr:continue
        H,L,_=rr
        if state=="SWEPT":
            se=max(se,float(b.high)) if di<0 else min(se,float(b.low))
            disp=abs(b.close-b.open)>=b.atr*.55
            x=(b.close<st-b.atr*.02 and disp) if di<0 else (b.close>st+b.atr*.02 and disp)
            f,zl,zh=ifvg(d,i,di)
            mss=mss or x;il=il or f
            if x or f:state="CONFIRM";age=0;de=float(b.low if di<0 else b.high)
            continue
        de=min(de,float(b.low)) if di<0 else max(de,float(b.high))
        f,zl,zh=ifvg(d,i,di);il=il or f
        vol=bool(np.isfinite(b.vma) and b.vma>0 and b.volume>=b.vma*1.10)
        macro=win(t,"07:00","10:00") or win(t,"13:30","16:00")
        e=float(b.close);stop=se-b.atr*.08 if di>0 else se+b.atr*.08;R=abs(e-stop);target=H if di>0 else L
        reward=(target-e) if di>0 else (e-target);trade_rr=reward/R if R>0 else 0.;clear=reward>0 and trade_rr>=1.25
        v={"sweep":True,"htf_pda":htf_pda(h,t,di,e),"mss":mss,"macro":macro,"equilibrium":eqok(di,b,se,de),
           "volume":vol,"ifvg":il,"clear_target":clear}
        score=sum(map(int,v.values()));ev.append({"time":t,"setup":setup,"dir":di,**{k:int(v[k]) for k in ROWS},"score":score,"rr":trade_rr})
        if not clear:continue
        for key,g in gates.items():
            if key in fired or g.active is not None or g.n_day>=3:continue
            if gpass(v,g.th,g.mask):
                g.active={"d":di,"e":e,"sl":stop,"tp":target,"rr":trade_rr,"be":False};g.n_day+=1;fired.add(key)
    rows=[]
    for (th,m),g in gates.items():
        if g.active:g.be+=1;g.active=None
        pf=g.pf();obj=(min(pf,10)*math.sqrt(max(g.n(),1))/(1+g.dd)) if g.n()>=a.min_trades_rank else -1
        rows.append({"threshold":th,"mandatory_mask":m,"mandatory":maskname(m),"mandatory_count":m.bit_count(),"N":g.n(),
          "wins":g.w,"losses":g.l,"WR_pct":g.wr(),"PF_R":pf,"sum_R":g.r,"Return_pct_risk_compound":g.eq-100,"MaxDD_pct":g.dd,"objective":obj})
    r=pd.DataFrame(rows);r.replace([np.inf,-np.inf],np.nan).to_csv(out/"gate_matrix.csv",index=False)
    pd.DataFrame(ev).to_csv(out/"setup_evidence.csv",index=False)
    top=r[(r.N>=a.min_trades_rank)&r.PF_R.notna()].sort_values(["objective","PF_R","N"],ascending=False)
    top.head(50).to_csv(out/"top50.csv",index=False)
    core=BIT["sweep"]|BIT["clear_target"];named=[]
    for th in map(int,a.thresholds.split(",")):
        for label,m in [(f"{th}/8 Sweep+Target",core),(f"{th}/8 Sweep+MSS+Target",core|BIT["mss"]),
          (f"{th}/8 Sweep+IFVG+Target",core|BIT["ifvg"]),(f"{th}/8 Sweep+Vol+Target",core|BIT["volume"]),
          (f"{th}/8 Sweep+MSS+IFVG+Target",core|BIT["mss"]|BIT["ifvg"])]:
            x=r[(r.threshold==th)&(r.mandatory_mask==m)].iloc[0].to_dict();x["label"]=label;named.append(x)
    pd.DataFrame(named).to_csv(out/"named_gates.csv",index=False)
    meta={"verification_level":"M1_OHLC_SCREENING","rows":len(d),"start":str(d.datetime.iloc[0]),"end":str(d.datetime.iloc[-1]),
      "gate_count":len(gates),"thresholds":a.thresholds,"risk_pct":a.risk_pct,
      "limitations":["mid-quote OHLCV","no spread/slippage/swap","same-bar ambiguity resolved SL-first","Raw Bid/Ask Tick Nautilus required for promotion"]}
    (out/"manifest.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print("DATA",meta["start"],"->",meta["end"],"rows",meta["rows"],"GATES",len(gates))
    print("TOP10");print(top.head(10).to_string(index=False))
    print("NAMED");print(pd.DataFrame(named)[["label","N","WR_pct","PF_R","Return_pct_risk_compound","MaxDD_pct"]].to_string(index=False))
if __name__=="__main__":main()
