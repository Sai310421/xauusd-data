from __future__ import annotations
import argparse,json,math
from collections import deque,defaultdict
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

class GBSweepBoost(GB):
 def __init__(self,c,boost_mult):
  super().__init__(c);self.boost_mult=boost_mult
  self.m1o=deque(maxlen=600);self.m1h=deque(maxlen=600);self.m1l=deque(maxlen=600);self.m1c=deque(maxlen=600);self.m1v=deque(maxlen=600)
  self.buy_age=999;self.sell_age=999;self.boosted_entries=0;self.normal_entries=0
 def _detector(self):
  self.buy_age+=1;self.sell_age+=1
  if len(self.m1c)<222:return
  h,l,c,v=self.m1h[-1],self.m1l[-1],self.m1c[-1],self.m1v[-1]
  hist=np.asarray(list(self.m1v)[-201:-1],float);mean=float(hist.mean());std=float(hist.std(ddof=0))
  if mean<=0 or std<=0:return
  z=(v-mean)/std
  ph=max(list(self.m1h)[-21:-1]);pl=min(list(self.m1l)[-21:-1])
  if z>=2.0 and l<pl and c>pl:self.buy_age=0
  if z>=2.0 and h>ph and c<ph:self.sell_age=0
 def _mult(self,side):
  return self.boost_mult if (self.buy_age<=5 if side>0 else self.sell_age<=5) else 1.0
 def boost(self):
  if self.cur_hour<9 or self.entries_day>=3:return
  dr=self.dayrange();vol,reg,a=self.vol_reg()
  if dr is None or a is None:return
  dhi,dlo=dr;off=.2*vol*(2 if reg==-1 else 1);sl=min(max(a*1.2,4),12);tp=min(max(a*2.4,9),30)*(1.25 if reg==1 else .8 if reg==-1 else 1)
  for side,p in ((1,dhi+off),(-1,dlo-off)):
   self.pending.append(('C',side,p,sl,tp,vol,self._mult(side)))
  self.pending=self.pending[-30:]
 def on_bar(self,bar:Bar):
  s=str(bar.bar_type);tf=60 if '-1-HOUR-' in s else 15 if '-15-MINUTE-' in s else 5 if '-5-MINUTE-' in s else 1 if '-1-MINUTE-' in s else None
  if tf is None:return
  o,h,l,c=map(f,[bar.open,bar.high,bar.low,bar.close]);[self.b[tf][i].append(v) for i,v in enumerate((o,h,l,c))]
  if tf==1:
   self.m1o.append(o);self.m1h.append(h);self.m1l.append(l);self.m1c.append(c);self.m1v.append(f(bar.volume));self._detector()
  self.cur_hour=self.hour(int(bar.ts_event));day=int((int(bar.ts_event)+3*3600*10**9)//(86400*10**9))
  if self.day!=day:self.day=day;self.entries_day=0;self.placed=[]
  allowed=self.cur_hour in (11,15,16,17,18)
  if not allowed:self.pending=[]
  if not allowed:return
  if tf==15:self.boost()
 def on_quote_tick(self,t:QuoteTick):
  bid=f(t.bid_price);ask=f(t.ask_price);self.last_bid=bid;self.last_ask=ask;spread=ask-bid
  if spread>2.5:return
  keep=[]
  for q in self.pending:
   layer,side,p,sl,tp,vol,mult=q
   hit=ask>=p if side>0 else bid<=p
   if hit:
    e=ask if side>0 else bid
    self.active.append({'layer':layer,'side':side,'entry':e,'sl':sl,'tp':tp,'vol':vol,'mult':mult,'stop':e-side*sl,'take':e+side*tp});self.entries_day+=1
    if mult>1:self.boosted_entries+=1
    else:self.normal_entries+=1
   else:keep.append(q)
  self.pending=keep
  if not self.active:return
  m1h=self.b[1][1][-1] if self.b[1][1] else None;m1l=self.b[1][2][-1] if self.b[1][2] else None;h1h=self.b[60][1][-1] if self.b[60][1] else None;h1l=self.b[60][2][-1] if self.b[60][2] else None;nxt=[]
  for a in self.active:
   side=a['side'];mark=bid if side>0 else ask;move=(mark-a['entry'])*side;vol=a['vol']
   if move>=1.2*vol:a['stop']=max(a['stop'],a['entry']+.2*vol) if side>0 else min(a['stop'],a['entry']-.2*vol)
   if move>=2.0*vol and None not in (m1h,m1l,h1h,h1l):
    gate=(m1h>=h1h-.3*vol) or (m1l<=h1l+.3*vol)
    if gate:
     ns=(m1h-3*vol) if side>0 else (m1l+3*vol);a['stop']=max(a['stop'],ns) if side>0 else min(a['stop'],ns)
   close=(mark<=a['stop'] if side>0 else mark>=a['stop']) or (mark>=a['take'] if side>0 else mark<=a['take'])
   if close:
    weighted=move*a['mult'];self.pnl[a['layer']].append(weighted);self.ledger.append(weighted)
   else:nxt.append(a)
  self.active=nxt
 def on_stop(self):
  for a in self.active:
   mark=self.last_bid if a['side']>0 else self.last_ask;mv=(mark-a['entry'])*a['side']*a['mult'];self.pnl[a['layer']].append(mv);self.ledger.append(mv)
 def summary_ext(self):
  return {**self.summary(),'boost_mult':self.boost_mult,'boost_window_m1_bars':5,'boosted_entries':self.boosted_entries,'normal_entries':self.normal_entries}

def run(catalog,mult,experiment_id):
 cat=ParquetDataCatalog(catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value])
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(l1(raw));s=inst.id.value
 st=GBSweepBoost(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{s}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{s}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{s}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{s}-1-HOUR-BID-INTERNAL'),mode='m15'),float(mult))
 eng.add_strategy(st);eng.run();eng.end();res={'verification':'GOLDEBRAVE_C_SWEEP_BOOST_RAW','raw_ticks':len(raw),'ohlc_resample_used':False,'logic':'C remains always enabled at 1.0x; same-direction BigPlayer-style M1 Sweep (tick-volume z>=2, 20-bar rejection) within 5 completed M1 bars boosts only PnL exposure by selected multiplier.','limitation':'Virtual lot scaling on prior GoldeBrave C approximation; Nautilus internal M1 tick-count volume proxies MT5 tick_volume.',**st.summary_ext()};p=Path('results/goldebrave-sweep-boost')/experiment_id/f'x{str(mult).replace(".","_")}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--mult',type=float,choices=[1.25,1.5,2.0],required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();run(a.catalog,a.mult,a.experiment_id)
if __name__=='__main__':main()
