#!/usr/bin/env python3
import csv, json, math, argparse
from collections import deque, defaultdict
from dataclasses import dataclass

@dataclass
class Tick:
    ts: str
    bid: float
    ask: float
    @property
    def mid(self): return (self.bid+self.ask)/2
    @property
    def spread(self): return max(0.0,self.ask-self.bid)

class F:
    def __init__(self,n=240):
        self.m=deque(maxlen=n); self.s=deque(maxlen=n)
    def u(self,t): self.m.append(t.mid); self.s.append(t.spread)
    def ready(self,n=80): return len(self.m)>=n
    def ret(self,k): return self.m[-1]-self.m[-1-k] if len(self.m)>k else 0.0
    def mean(self,x): return sum(x)/len(x) if x else 0
    def sd(self,x):
        if len(x)<2:return 0
        a=self.mean(x); return math.sqrt(sum((v-a)**2 for v in x)/(len(x)-1))
    def vol(self,n=60):
        x=list(self.m)[-n:]; return self.sd([x[i]-x[i-1] for i in range(1,len(x))]) if len(x)>2 else 0
    def z(self,n=60):
        x=list(self.m)[-n:]; sd=self.sd(x); return (x[-1]-self.mean(x))/sd if sd else 0
    def sr(self,n=60):
        x=list(self.s)[-n:]; a=self.mean(x); return self.s[-1]/a if a else 1
    def hi(self,n=80): return max(list(self.m)[-n:])
    def lo(self,n=80): return min(list(self.m)[-n:])
    def sma(self,n=40): return self.mean(list(self.m)[-n:])

def signals(f):
    if not f.ready(120): return []
    v=max(f.vol(80),0.02); z=f.z(80); sr=f.sr(60); r3=f.ret(3); r10=f.ret(10); px=f.m[-1]
    o=[]
    if sr<1.10 and abs(r3)>.35*v:o.append(("A_CB","A1_TickFollow",1 if r3>0 else -1,1.1*v,1.2*v,180))
    if abs(z)>1.5 and sr<1.15:o.append(("A_CB","A2_MicroReversion",-1 if z>0 else 1,1.0*v,.9*v,220))
    if sr<.78 and abs(r10)>.5*v:o.append(("A_CB","A3_SpreadCompression",1 if r10>0 else -1,1.0*v,1.15*v,160))
    if sr<.85 and abs(z)<.55 and abs(r3)>.18*v:o.append(("A_CB","A4_TurnoverCycle",1 if r3>0 else -1,.8*v,.75*v,100))
    if abs(r10)>=.12 and abs(r3)>=.025 and r3*r10>0:o.append(("A_CB","A5_G75Pursuit",1 if r10>0 else -1,.20,.12,120))
    if px>=f.hi(100) and f.ret(8)>0:o.append(("B_TREND","B1_DonchianBreak",1,1.5*v,3*v,700))
    elif px<=f.lo(100) and f.ret(8)<0:o.append(("B_TREND","B1_DonchianBreak",-1,1.5*v,3*v,700))
    if abs(f.ret(12))>2.2*v:o.append(("B_TREND","B2_ATRExpansion",1 if f.ret(12)>0 else -1,1.6*v,3.2*v,800))
    if r3>0 and f.ret(25)<-v and px>f.sma(20):o.append(("B_TREND","B3_MSS_BOS",1,1.3*v,2.6*v,700))
    elif r3<0 and f.ret(25)>v and px<f.sma(20):o.append(("B_TREND","B3_MSS_BOS",-1,1.3*v,2.6*v,700))
    if abs(f.ret(40))>2.8*v and sr<1.15:o.append(("B_TREND","B4_SessionBreak",1 if f.ret(40)>0 else -1,1.5*v,3.5*v,900))
    if abs(r10)>=.12 and abs(r3)>=.025 and r3*r10>0:o.append(("B_TREND","B5_G75Pursuit",1 if r10>0 else -1,.20,.32,500))
    if abs(z)>2 and sr<1.05:o.append(("C_HIGH_WR","C1_ZScoreMeanReversion",-1 if z>0 else 1,1.35*v,1.0*v,500))
    if z>1.8 and f.ret(2)<0:o.append(("C_HIGH_WR","C2_BollingerReEntry",-1,1.2*v,1.0*v,450))
    elif z<-1.8 and f.ret(2)>0:o.append(("C_HIGH_WR","C2_BollingerReEntry",1,1.2*v,1.0*v,450))
    if f.ret(5)<-1.2*v and f.ret(20)<-2*v and f.ret(2)>0:o.append(("C_HIGH_WR","C3_RSIStochConfluence",1,1.25*v,1.05*v,420))
    elif f.ret(5)>1.2*v and f.ret(20)>2*v and f.ret(2)<0:o.append(("C_HIGH_WR","C3_RSIStochConfluence",-1,1.25*v,1.05*v,420))
    dev=px-f.sma(40)
    if abs(dev)>2*v and dev*f.ret(2)<0:o.append(("C_HIGH_WR","C4_VWAPDeviation",-1 if dev>0 else 1,1.25*v,1.0*v,450))
    if v<max(f.vol(120)*.9,.02) and abs(z)>1.6 and sr<.95:o.append(("C_HIGH_WR","C5_RegimeMicroReversion",-1 if z>0 else 1,1.1*v,.95*v,380))
    return o

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ticks",required=True); ap.add_argument("--out",default="amos_triple_engine_v2/raw_tick_kpi.json")
    a=ap.parse_args()
    f=F(); open_tr=[]; st=defaultdict(lambda:{"N":0,"W":0,"GP":0.,"GL":0.,"Net":0.,"Peak":0.,"MaxDD":0.,"Lots":0.})
    with open(a.ticks,encoding="utf-8-sig",newline="") as h:
        rd=csv.DictReader(h); req={"timestamp","bid","ask"}
        if not req.issubset(set(rd.fieldnames or [])): raise SystemExit("RAW BID/ASK TICK ONLY: required timestamp,bid,ask; OHLC rejected.")
        for row in rd:
            t=Tick(row["timestamp"],float(row["bid"]),float(row["ask"]))
            nxt=[]
            for tr in open_tr:
                tr["age"]+=1; px=t.bid if tr["side"]>0 else t.ask; reason=None
                if tr["side"]>0 and px<=tr["sl"] or tr["side"]<0 and px>=tr["sl"]:reason="SL"
                elif tr["side"]>0 and px>=tr["tp"] or tr["side"]<0 and px<=tr["tp"]:reason="TP"
                elif tr["age"]>=tr["ttl"]:reason="TTL"
                if reason:
                    pnl=tr["side"]*(px-tr["entry"])*100*.01; s=st[(tr["e"],tr["l"])]
                    s["N"]+=1; s["W"]+=pnl>0; s["GP"]+=max(0,pnl); s["GL"]+=max(0,-pnl); s["Net"]+=pnl; s["Lots"]+=.01
                    s["Peak"]=max(s["Peak"],s["Net"]); s["MaxDD"]=max(s["MaxDD"],s["Peak"]-s["Net"])
                else:nxt.append(tr)
            open_tr=nxt; f.u(t)
            for e,l,side,sl,tp,ttl in signals(f):
                entry=t.ask if side>0 else t.bid
                open_tr.append({"e":e,"l":l,"side":side,"entry":entry,"sl":entry-side*sl,"tp":entry+side*tp,"ttl":ttl,"age":0})
    out={}
    for (e,l),s in st.items():
        pf=s["GP"]/s["GL"] if s["GL"] else (999 if s["GP"] else 0)
        wr=s["W"]/s["N"] if s["N"] else 0
        ev=s["Net"]/s["N"] if s["N"] else 0
        dd=100*s["MaxDD"]/1000
        cb=s["Lots"]*6
        minN={"A_CB":35,"B_TREND":25,"C_HIGH_WR":40}[e]
        out[f"{e}/{l}"]={"N":s["N"],"WR":wr,"PF":pf,"EV_USD":ev,"TradingNetUSD":s["Net"],"VirtualGrossRTLots":s["Lots"],"VirtualCBUSD_eval_only":cb,"MaxVirtualDDPct":dd,
            "PromotionEligible":s["N"]>=minN and pf>=1.2 and ev>0 and dd<=8}
    with open(a.out,"w",encoding="utf-8") as h:json.dump(out,h,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
