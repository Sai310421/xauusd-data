from __future__ import annotations
import argparse,json,math
from collections import deque
from decimal import Decimal
from pathlib import Path
import numpy as np
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar,QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from goldebrave_fasttf_parity_raw import GB,Cfg,f,l1,met

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
 def _q(self,identifiers=None,start=None,end=None): return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
 ParquetDataCatalog.query_quote_ticks=_q

class BPD(GB):
 def __init__(self,c,pressure_gate:float):
  super().__init__(c)
  self.pressure_gate=pressure_gate
  self.bp_events=deque(maxlen=60)   # recent BigPlayer directional events on M1
  self.bp_rescue_pending=[]
  self.bp_active=[]
  self.bp_ledger=[]
  self.tick_count_bar=0
  self.m1_vol=deque(maxlen=200)
  self.m1_hist=deque(maxlen=64)  # tuples o,h,l,c,atr_proxy,tv
  self.rescue_signals=0

 def _pressure(self):
  if not self.bp_events:return 50.0
  b=sum(1 for x in self.bp_events if x>0);s=sum(1 for x in self.bp_events if x<0);n=b+s
  return 100.0*b/n if n else 50.0

 def on_quote_tick(self,t:QuoteTick):
  self.tick_count_bar+=1
  bid=f(t.bid_price);ask=f(t.ask_price);spread=ask-bid
  # rescue fills/exits independent from baseline GB; standard parent exit for screening
  keep=[]
  for side,p,sl,tp in self.bp_rescue_pending:
   hit=ask>=p if side>0 else bid<=p
   if hit:
    e=ask if side>0 else bid
    self.bp_active.append({'side':side,'entry':e,'stop':e-side*sl,'take':e+side*tp})
   else: keep.append((side,p,sl,tp))
  self.bp_rescue_pending=keep
  nxt=[]
  for a in self.bp_active:
   side=a['side'];mark=bid if side>0 else ask;mv=(mark-a['entry'])*side
   close=(mark<=a['stop'] if side>0 else mark>=a['stop']) or (mark>=a['take'] if side>0 else mark<=a['take'])
   if close:self.bp_ledger.append(mv)
   else:nxt.append(a)
  self.bp_active=nxt
  super().on_quote_tick(t)

 def _update_bigplayer_m1(self,o,h,l,c,tv):
  # ATR14 proxy from M1 true ranges, exact BigPlayer structure preserved for screening
  self.m1_vol.append(tv)
  self.m1_hist.append((o,h,l,c,tv))
  if len(self.m1_hist)<22 or len(self.m1_vol)<30:return
  vols=np.asarray(self.m1_vol,float);mu=float(vols.mean());sd=float(vols.std(ddof=0))
  if sd<=0:return
  z=(tv-mu)/sd
  if z<2.0:return
  hs=[x[1] for x in list(self.m1_hist)[:-1]];ls=[x[2] for x in list(self.m1_hist)[:-1]]
  prevc=[x[3] for x in list(self.m1_hist)[:-1]]
  trs=[]
  arr=list(self.m1_hist)
  for j in range(max(1,len(arr)-14),len(arr)):
   po,ph,pl,pc,_=arr[j-1];oo,hh,ll,cc,_=arr[j];trs.append(max(hh-ll,abs(hh-pc),abs(ll-pc)))
  atr=float(np.mean(trs)) if trs else 0.0
  if atr<=0:return
  rng=h-l;body=abs(c-o);br=body/rng if rng>0 else 0
  bull=c>o;bear=c<o
  imb_buy=(rng/atr>=1.5 and br>=0.60 and bull);imb_sell=(rng/atr>=1.5 and br>=0.60 and bear)
  upper=h-max(o,c);lower=min(o,c)-l
  abs_buy=(lower>=body*1.2 and lower>upper);abs_sell=(upper>=body*1.2 and upper>lower)
  sh=max(hs[-20:]);sl=min(ls[-20:]);sweep_buy=(l<sl and c>sl);sweep_sell=(h>sh and c<sh)
  if imb_buy:self.bp_events.append(1)
  if imb_sell:self.bp_events.append(-1)
  if abs_buy:self.bp_events.append(1)
  if abs_sell:self.bp_events.append(-1)
  if sweep_buy:self.bp_events.append(1)
  if sweep_sell:self.bp_events.append(-1)

 def _rescue_scan(self):
  dr=self.dayrange();vol,reg,a=self.vol_reg()
  if dr is None or a is None:return
  dhi,dlo=dr;press=self._pressure();inner=.20*vol;outer=.40*vol;off=.30*vol
  sl=min(max(a*1.2,4.0),12.0);tp=min(max(a*2.4,9.0),30.0)*(1.25 if reg==1 else .8 if reg==-1 else 1.0)
  hi,lo=self.extrema(15)
  # Rescue only candidates that are outside 0.20*vol but fail original 0.40*vol boundary.
  if press>=self.pressure_gate:
   for z in reversed(hi[-20:]):
    d=z-dhi
    if inner<d<=outer:
     p=z-off
     if all(abs(p-q[1])>=.20*vol for q in self.bp_rescue_pending):
      self.bp_rescue_pending.append((1,p,sl,tp));self.rescue_signals+=1;break
  if press<=100.0-self.pressure_gate:
   for z in reversed(lo[-20:]):
    d=dlo-z
    if inner<d<=outer:
     p=z+off
     if all(abs(p-q[1])>=.20*vol for q in self.bp_rescue_pending):
      self.bp_rescue_pending.append((-1,p,sl,tp));self.rescue_signals+=1;break
  self.bp_rescue_pending=self.bp_rescue_pending[-20:]

 def on_bar(self,bar:Bar):
  s=str(bar.bar_type);tf=60 if '-1-HOUR-' in s else 15 if '-15-MINUTE-' in s else 5 if '-5-MINUTE-' in s else 1 if '-1-MINUTE-' in s else None
  if tf==1:
   o,h,l,c=map(f,[bar.open,bar.high,bar.low,bar.close]);tv=self.tick_count_bar;self.tick_count_bar=0
   self._update_bigplayer_m1(o,h,l,c,tv)
  super().on_bar(bar)
  if tf==15 and getattr(self,'cur_hour',-1) in (11,15,16,17,18): self._rescue_scan()

 def on_stop(self):
  super().on_stop()
  for a in self.bp_active:
   mark=self.last_bid if a['side']>0 else self.last_ask
   self.bp_ledger.append((mark-a['entry'])*a['side'])

 def summary_bpd(self):
  base=self.summary()
  return {'pressure_gate_pct':self.pressure_gate,'pressure_rule':'BigPlayer v4 PressureMeter-style recent event ratio; buy >= gate, sell <= 100-gate','rescue_zone':'M15 swing distance from day range in (0.20*vol, 0.40*vol], i.e. candidates rejected by original 0.40*vol boundary','rescue_signals':self.rescue_signals,'rescue':met(self.bp_ledger),'baseline':base}

def run(catalog,gate,experiment_id):
 cat=ParquetDataCatalog(catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value])
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(l1(raw));s=inst.id.value
 st=BPD(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{s}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{s}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{s}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{s}-1-HOUR-BID-INTERNAL'),mode='m15'),gate)
 eng.add_strategy(st);eng.run();eng.end();res={'verification':'GOLDEBRAVE_BPD_PRESSURE_RESCUE_RAW','raw_ticks':len(raw),'ohlc_resample_used':False,'limitation':'Screening model. Rescue exit uses GoldeBrave parent ATR SL/TP but sizing/broker semantics remain virtual.',**st.summary_bpd()};p=Path('results/goldebrave-bpd-pressure-rescue')/experiment_id/f'g{int(gate)}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--gate',type=float,required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();run(a.catalog,a.gate,a.experiment_id)
if __name__=='__main__':main()
