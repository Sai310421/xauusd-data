from __future__ import annotations
import argparse,json,math
from collections import deque
from dataclasses import dataclass,field
from decimal import Decimal
from enum import Enum
from pathlib import Path
import numpy as np,pandas as pd,nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money,Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar,QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
SIM=Venue("SIM"); TF_MIN={"M1":1,"M5":5,"M15":15}
class Action(str,Enum):
 WAIT="WAIT"; ENTRY_LONG="ENTRY_LONG"; ENTRY_SHORT="ENTRY_SHORT"; ADD="ADD"; REDUCE="REDUCE"; EXIT="EXIT"; HEDGE="HEDGE"; UNLOCK="UNLOCK"; RECOVERY="RECOVERY"
@dataclass
class Tick:
 ts:float; bid:float; ask:float; bid_size:float=0.; ask_size:float=0.
 @property
 def mid(self): return (self.bid+self.ask)/2
 @property
 def spread(self): return self.ask-self.bid
@dataclass
class Sig:
 name:str; direction:int; score:float; expected_move:float; cost:float; meta:dict=field(default_factory=dict)
def clamp(x,a=-1.,b=1.): return max(a,min(b,x))
class HFT:
 def __init__(self): self.q=deque(maxlen=32)
 def on(self,t):
  self.q.append(t)
  if len(self.q)<8:return Sig("A",0,0,0,t.spread)
  m=[z.mid for z in self.q]; dif=[abs(b-a) for a,b in zip(m[:-1],m[1:])]; sc=sum(dif)/max(1,len(dif)) or 1e-12
  v=clamp((m[-1]-m[-4])/(3*sc)); a=clamp(((m[-1]-m[-3])-(m[-3]-m[-5]))/(3*sc))
  up=sum(b>a0 for a0,b in zip(m[:-1],m[1:])); dn=sum(b<a0 for a0,b in zip(m[:-1],m[1:])); imb=(up-dn)/max(1,up+dn)
  cont=clamp(.55*v+.25*a+.20*imb); spr=clamp(t.spread/.80,0,1)
  raw=.24*v+.18*a+.24*imb+.16*cont-.18*spr; d=1 if raw>.60 else -1 if raw<-.60 else 0
  exp=abs(cont)*(sum(dif[-7:])/max(1,len(dif[-7:])))*4
  return Sig("A",d,min(1,abs(raw)),exp,t.spread,{"v":v,"a":a,"imb":imb})
class Harm:
 def on(self,x,a,b,c,d,bias=0.,cost=0.):
  xa,ab,bc,cd=abs(a-x),abs(b-a),abs(c-b),abs(d-c)
  if min(xa,ab,bc,cd)<=1e-12:return Sig("B",0,0,0,cost)
  rb=ab/xa; dxa=abs(d-x)/xa; cdbc=cd/bc; tol=.03
  g=.5*((1-min(1,abs(rb-.618)/(tol+1e-9)))+(1-min(1,abs(dxa-.786)/(tol+1e-9))))
  bf=.5*((1-min(1,abs(rb-.786)/(tol+1e-9)))+(1-min(1,min(abs(dxa-1.272),abs(cdbc-1.618))/(tol+1e-9))))
  bok=1 if .382<=rb<=.5 else max(0,1-min(abs(rb-.382),abs(rb-.5))/.2); bat=.5*(bok+(1-min(1,abs(dxa-.886)/(tol+1e-9))))
  name,base=max((("G",g),("BF",bf),("BAT",bat)),key=lambda z:z[1]); struct=(x+a+b+c)/4; dr=1 if d<struct else -1
  score=clamp(base+.25*max(0,clamp(bias*dr)),0,1); dr=dr if score>=.60 else 0
  return Sig("B"+name,dr,score,abs(c-d)*score,cost,{"rb":rb,"dxa":dxa})
def ystar(k,it=96):
 if k<=0:return 0.
 f=lambda y:y-math.tanh(y)-k; lo=0.; hi=max(1.,k+2)
 while f(hi)<0:hi*=2
 for _ in range(it):
  mid=(lo+hi)/2
  if f(mid)>=0:hi=mid
  else:lo=mid
 return (lo+hi)/2
def dstar(sig=.8,rho=.1,lam=1.,k=.1):
 sig=max(sig,1e-12); kap=k*rho*math.sqrt(2*rho)/(2*lam*sig); return sig/math.sqrt(2*rho)*ystar(kap)
def pnr(d,mu,sig,B=10.):
 d=max(0,min(B,d))
 if abs(mu)<1e-10:return max(0,min(1,1-d/B))
 s2=max(1e-12,sig*sig); num=math.exp(-2*mu*d/s2)-math.exp(-2*mu*B/s2); den=1-math.exp(-2*mu*B/s2)
 return max(0,min(1,num/den)) if abs(den)>1e-12 else max(0,min(1,1-d/B))
class Controller:
 def decide(self,A,B,debt,mu=0.,sig=.8,floatp=0.):
  ds=dstar(sig); p=pnr(debt,mu,sig); inter=max(0,debt-ds)
  ea=(A.score*A.expected_move-A.cost) if A.direction else -abs(A.cost); eb=(B.score*B.expected_move-B.cost) if B.direction else -abs(B.cost)
  if A.direction and B.direction: comb=ea+eb+.2*min(abs(ea),abs(eb)) if A.direction==B.direction else max(ea,eb)-.35*min(abs(ea),abs(eb))
  else: comb=max(ea,eb)
  direction=A.direction if ea>=eb else B.direction
  adv={"WAIT":0,"ENTRY_LONG":comb if direction>0 else -1e9,"ENTRY_SHORT":comb if direction<0 else -1e9,"HEDGE":inter*(1-p)-.1,"RECOVERY":inter*max(0,.5-p)-.1,"REDUCE":inter*.5-.05,"EXIT":max(0,-floatp)+inter-.1,"ADD":comb*.5-(max(0,debt/ds-1) if ds>0 else 1),"UNLOCK":p*inter-.1}
  if debt>=ds and p<.35:
   adv["HEDGE"]+=inter; adv["ENTRY_LONG"]-=inter; adv["ENTRY_SHORT"]-=inter; adv["ADD"]-=inter
  n,v=max(adv.items(),key=lambda z:z[1]); return (Action(n) if v>0 else Action.WAIT),v,ds,p,inter,adv
class Cfg(StrategyConfig,frozen=True):
 instrument_id:InstrumentId; bar_type:BarType; trade_size:Decimal; tf_minutes:int; mode:str="AB100D"
class S(Strategy):
 def __init__(self,c): super().__init__(c); self.h=HFT(); self.hm=Harm(); self.ctl=Controller(); self.bs=deque(maxlen=32); self.B=Sig("B",0,0,0,0); self.direction=0; self.entries=0; self.blocked=0; self.actions={}; self.debt=0.
 @staticmethod
 def f(x): return float(x.as_double()) if hasattr(x,"as_double") else float(x)
 def on_start(self): self.subscribe_quote_ticks(self.config.instrument_id); self.subscribe_bars(self.config.bar_type)
 def on_bar(self,b):
  x={"o":self.f(b.open),"h":self.f(b.high),"l":self.f(b.low),"c":self.f(b.close)}; self.bs.append(x)
  if len(self.bs)>=10:
   z=list(self.bs); p=[z[-10]["c"],z[-8]["c"],z[-6]["c"],z[-4]["c"],z[-2]["c"]]; bias=float(np.sign(z[-2]["c"]-z[-8]["c"])); self.B=self.hm.on(*p,bias=bias,cost=0.)
 def on_quote_tick(self,t):
  bid,ask=self.f(t.bid_price),self.f(t.ask_price); A=self.h.on(Tick(float(t.ts_event),bid,ask,self.f(t.bid_size),self.f(t.ask_size))); B=self.B; act,val,ds,p,inter,adv=self.ctl.decide(A,B,self.debt,0,.8,-self.debt); self.actions[act.value]=self.actions.get(act.value,0)+1
  mode=self.config.mode.upper()
  if mode=="A": d=A.direction
  elif mode=="B": d=B.direction
  elif mode=="AB": d=(A if A.score*A.expected_move-A.cost>=B.score*B.expected_move-B.cost else B).direction
  else:d=1 if act==Action.ENTRY_LONG else -1 if act==Action.ENTRY_SHORT else 0
  if self.portfolio.is_net_flat(self.config.instrument_id):
   if not d:self.blocked+=1; return
   inst=self.cache.instrument(self.config.instrument_id); q=inst.make_qty(self.config.trade_size); side=OrderSide.BUY if d>0 else OrderSide.SELL; self.submit_order(self.order_factory.market(instrument_id=self.config.instrument_id,order_side=side,quantity=q)); self.entries+=1; self.direction=d; return
  if mode=="AB100D" and act in (Action.EXIT,Action.REDUCE,Action.HEDGE): self.close_all_positions(self.config.instrument_id)
  elif mode!="AB100D" and ((self.direction>0 and A.direction<0) or (self.direction<0 and A.direction>0)): self.close_all_positions(self.config.instrument_id)
 def on_position_closed(self,e): self.direction=0
 def on_stop(self): self.close_all_positions(self.config.instrument_id)
def parse_money(v):
 if v is None:return 0.
 if isinstance(v,(float,int,np.number)):return float(v)
 s=str(v).replace(",","").strip()
 try:return float(s.split()[0])
 except Exception:return 0.
def extract(rep,sym,tf):
 if rep is None or rep.empty:return []
 pc=next((c for c in rep.columns if str(c).lower() in ("realized_pnl","pnl","realizedpnl")),None)
 if pc is None:pc=next((c for c in rep.columns if "pnl" in str(c).lower()),None)
 tc=next((c for c in rep.columns if "closed" in str(c).lower() and ("ts" in str(c).lower() or "time" in str(c).lower())),None)
 out=[]
 for i,row in rep.iterrows():
  pnl=parse_money(row[pc]) if pc is not None else 0.; ts=row[tc] if tc is not None else i
  try:ts=pd.Timestamp(ts).value
  except Exception:
   try:ts=int(ts)
   except Exception:ts=len(out)
  out.append({"symbol":sym,"tf":tf,"pnl":pnl,"ts_closed":int(ts)})
 return out
def metr(trades,initial=1000.,days=30):
 if not trades:return {"N":0,"WR_pct":0.,"PF":0.,"NetProfit":0.,"MaxDD_pct":0.,"RF":0.,"Monthly21_pct":0.,"Expectancy":0.}
 a=np.array([t["pnl"] for t in trades],float); w=a[a>0]; l=a[a<0]; pf=float(w.sum()/abs(l.sum())) if len(l) and l.sum()!=0 else (float("inf") if len(w) else 0.); eq=initial; peak=initial; mdd=0.
 for x in a:eq+=x; peak=max(peak,eq); mdd=max(mdd,peak-eq)
 net=float(a.sum()); mddp=mdd/peak*100 if peak>0 else 0.; monthly=((max(eq,1e-9)/initial)**(21/days)-1)*100
 return {"N":int(len(a)),"WR_pct":float((a>0).mean()*100),"PF":pf,"NetProfit":net,"MaxDD_pct":float(mddp),"RF":float(net/mdd) if mdd>0 else None,"Monthly21_pct":float(monthly),"Expectancy":float(a.mean())}
def run(inst,ticks,sym,tf,mode,days):
 cfg=BacktestEngineConfig(trader_id=f"AE-{mode}-{sym}-{tf}",logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)); e=BacktestEngine(config=cfg); e.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000")); e.add_instrument(inst); e.add_data(ticks); bt=BarType.from_str(f"{inst.id.value}-{TF_MIN[tf]}-MINUTE-BID-INTERNAL"); s=S(Cfg(instrument_id=inst.id,bar_type=bt,trade_size=Decimal("1"),tf_minutes=TF_MIN[tf],mode=mode)); e.add_strategy(s); e.run(); trades=extract(e.generate_positions_report(),sym,tf); m=metr(trades,days=days); m.update({"signals":s.entries,"blocked":s.blocked,"actions":s.actions,"raw_ticks":len(ticks)}); e.dispose(); return trades,m
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--catalog",required=True); ap.add_argument("--symbols",nargs="+",required=True); ap.add_argument("--timeframes",nargs="+",required=True); ap.add_argument("--experiment-id",required=True); ap.add_argument("--raw-bidask-only",action="store_true"); a=ap.parse_args()
 if not a.raw_bidask_only:raise SystemExit("raw-bidask-only mandatory")
 cp=Path(a.catalog); manifest=json.loads((cp/"catalog_manifest.json").read_text()); days=int(manifest["days"]); cat=ParquetDataCatalog(str(cp)); insts={x.id.symbol.value.replace("/",""):x for x in cat.instruments()}; out={"verification_level":"NAUTILUS_BT_RAW_BIDASK","ohlc_resample_used":False,"nautilus_version":getattr(nautilus_trader,"__version__","unknown"),"dataset_sha256":manifest.get("catalog_sha256"),"period":{"start":manifest.get("start"),"days":days,"end_exclusive":manifest.get("end_exclusive")},"cells":{}}; all_by_mode={m:[] for m in ["A","B","AB","AB100D"]}
 for sym in a.symbols:
  inst=insts.get(sym)
  if inst is None:raise SystemExit(f"instrument missing {sym}")
  ticks=cat.query_quote_ticks(identifiers=[inst.id.value])
  if not ticks:raise SystemExit(f"no raw QuoteTicks {sym}")
  for tf in a.timeframes:
   for mode in ["A","B","AB","AB100D"]:
    tr,m=run(inst,ticks,sym,tf,mode,days); out["cells"][f"{sym}:{tf}:{mode}"]=m; all_by_mode[mode].extend(tr)
 out["aggregate"]={}
 for mode,tr in all_by_mode.items():
  tr.sort(key=lambda x:(x["ts_closed"],x["symbol"],x["tf"])); out["aggregate"][mode]=metr(tr,days=days)
 root=Path("results/ae-bt")/a.experiment_id; root.mkdir(parents=True,exist_ok=True); (root/"summary.json").write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=="__main__":main()
