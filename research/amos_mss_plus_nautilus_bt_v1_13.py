#!/usr/bin/env python3
"""AMOS MSS+ v1.13 — NautilusTrader native BacktestEngine, Raw Bid/Ask.
Signal detector is the frozen corrected v1.11 causal MSS+ pipeline.
Execution is performed by Nautilus BacktestEngine with native market orders.
One active basket at a time to avoid netting cross-contamination.
Fixed 0.01 lot equivalent = 1 XAU unit, initial equity $1000, leverage 1:2000.
Exit evaluation: structural stop, +1.5R target, 60 minutes.
"""
from __future__ import annotations
import argparse,json,subprocess,sys
from decimal import Decimal
from pathlib import Path
import numpy as np,pandas as pd,nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,OrderSide,BookType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

class Cfg(StrategyConfig,frozen=True):
 instrument_id:InstrumentId

class MSSPlusNative(Strategy):
 def __init__(self,config,candidates):
  super().__init__(config);self.c=candidates.sort_values("entry_ns").to_dict("records");self.k=0
  self.pending=None;self.active=None;self.exit_pending=False;self.submitted=0;self.skipped_busy=0;self.bad_risk=0
 def on_start(self):self.subscribe_quote_ticks(self.config.instrument_id)
 @staticmethod
 def f(x):return float(x.as_double()) if hasattr(x,"as_double") else float(x)
 def on_quote_tick(self,t:QuoteTick):
  ns=int(t.ts_event);bid=self.f(t.bid_price);ask=self.f(t.ask_price)
  flat=not self.portfolio.is_net_long(self.config.instrument_id) and not self.portfolio.is_net_short(self.config.instrument_id)
  # consume due causal signals
  while self.k<len(self.c) and int(self.c[self.k]["entry_ns"])<=ns:
   q=self.c[self.k];self.k+=1
   if self.active is not None or self.pending is not None or not flat:self.skipped_busy+=1;continue
   di=int(q["dir"]);entry=ask if di>0 else bid;stop=float(q["stop"]);risk=(entry-stop)*di
   if risk<=0:self.bad_risk+=1;continue
   instr=self.cache.instrument(self.config.instrument_id)
   order=self.order_factory.market(instrument_id=self.config.instrument_id,
      order_side=OrderSide.BUY if di>0 else OrderSide.SELL,quantity=instr.make_qty(Decimal("1")))
   self.submit_order(order)
   self.pending={"dir":di,"entry_ref":entry,"stop":stop,"target":entry+di*1.5*risk,
                 "deadline":ns+60*60*1_000_000_000,"poi_type":q["poi_type"],"signal_time":q["entry_time"]}
   self.submitted+=1;return
  if self.active is None and self.pending is not None:
   # Backtest market order is processed on the event engine; arm exits on following quote.
   self.active=self.pending;self.pending=None
  if self.active is None or self.exit_pending:return
  a=self.active;px=bid if a["dir"]>0 else ask
  stophit=px<=a["stop"] if a["dir"]>0 else px>=a["stop"]
  tphit=px>=a["target"] if a["dir"]>0 else px<=a["target"]
  if stophit or tphit or ns>=a["deadline"]:
   self.close_all_positions(self.config.instrument_id);self.exit_pending=True
 def on_position_closed(self,event):
  self.active=None;self.pending=None;self.exit_pending=False
 def on_stop(self):self.close_all_positions(self.config.instrument_id)

def money(v):
 try:return float(str(v).replace(",","").split()[0])
 except:return 0.
def trades(report):
 if report is None or report.empty:return []
 pc=next((c for c in report.columns if "pnl" in str(c).lower()),None);tc=next((c for c in report.columns if "closed" in str(c).lower()),None)
 out=[]
 for i,r in report.iterrows():
  p=money(r[pc]) if pc else 0.;ts=r[tc] if tc else i
  try:n=int(pd.Timestamp(ts).value)
  except:n=0
  out.append({"pnl":p,"ts_closed":n})
 return out
def kpi(tr,initial=1000.,days=29):
 a=np.array([x["pnl"] for x in tr],float)
 if not len(a):return {"N":0,"WR_pct":0,"PF":0,"NetProfit":0,"MaxDD_pct":0,"MaxDD_USD":0,"RF":None,"Monthly21_pct":0}
 gp=a[a>0].sum();gl=-a[a<0].sum();eq=initial;peak=initial;mdd=0.
 for p in a:eq+=p;peak=max(peak,eq);mdd=max(mdd,peak-eq)
 pf=float(gp/gl) if gl>0 else (None if gp==0 else float("inf"));net=float(a.sum())
 return {"N":len(a),"WR_pct":float(100*(a>0).mean()),"PF":pf,"NetProfit":net,
         "MaxDD_pct":float(100*mdd/peak) if peak else 0.,"MaxDD_USD":float(mdd),
         "RF":float(net/mdd) if mdd else None,"Monthly21_pct":float(((max(eq,1e-9)/initial)**(21/days)-1)*100)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);src=out/"signal_source"
 subprocess.run([sys.executable,"research/amos_raw_mss_plus_audit_v1_11.py","--catalog",a.catalog,"--out",str(src)],check=True)
 c=pd.read_csv(src/"completed_candidates.csv");c["entry_time"]=pd.to_datetime(c.entry_time)
 # Explicit nanoseconds: pandas datetime internal resolution can be us in newer versions.
 c["entry_ns"]=c["entry_time"].map(lambda x:int(pd.Timestamp(x).value))
 cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
 raw=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,
               base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"))
 eng.add_instrument(inst);eng.add_data(raw);st=MSSPlusNative(Cfg(instrument_id=inst.id),c);eng.add_strategy(st);eng.run()
 tr=trades(eng.trader.generate_positions_report());start=pd.Timestamp(int(raw[0].ts_event),unit="ns");end=pd.Timestamp(int(raw[-1].ts_event),unit="ns");days=max((end-start).total_seconds()/86400,1)
 result={"version":"v1.13","verification_level":"NAUTILUS_BACKTESTENGINE_RAW_BIDASK_NATIVE_ORDERS",
  "engine":"NautilusTrader BacktestEngine","nautilus_version":getattr(nautilus_trader,"__version__","unknown"),
  "data":{"raw_ticks":len(raw),"start":str(start),"end":str(end),"ohlc_synthetic_intrabar":False},
  "strategy":{"gate":"corrected prior MSS+ v1.11","sequence":["Sweep","CISD","MSS","Displacement","POI(FVG/IFVG/BPR)","Volume","Equilibrium","Pullback"],
              "candidate_count":len(c),"execution":"native market orders on Raw QuoteTicks","position_policy":"one active trade at a time / NETTING",
              "size":"1 XAU unit ~= 0.01 standard lot","initial_equity_USD":1000,"leverage":2000,"TP_R":1.5,"SL":"structural sweep extreme","timeout_min":60},
  "orders":{"submitted":st.submitted,"skipped_busy":st.skipped_busy,"bad_risk":st.bad_risk},
  "metrics":kpi(tr,1000.,days),
  "limitations":["Observed raw Bid/Ask spread is included natively; no extra broker commission or synthetic slippage is added.","Signals are causally precomputed from the same raw tick catalog by frozen v1.11 before BacktestEngine execution; order/portfolio accounting is native Nautilus.","One-active-trade policy intentionally prevents netting of overlapping opposite signals."]}
 pd.DataFrame(tr).to_csv(out/"trades.csv",index=False);(out/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2));eng.dispose()
if __name__=="__main__":main()
