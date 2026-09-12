from __future__ import annotations
import argparse,json
from decimal import Decimal
from pathlib import Path
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from goldebrave_fasttf_parity_raw import Cfg,f,l1
from goldebrave_c_g75_chase_raw import GBG75Chase

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
 def _q(self,identifiers=None,start=None,end=None): return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
 ParquetDataCatalog.query_quote_ticks=_q

class GBG75MathControl(GBG75Chase):
 def __init__(self,c,mode:str):
  super().__init__(c,10)
  self.control_mode=mode
  self.peak_equity=1000.0
  self.control_cut3=0
  self.control_cut5=0
  self.control_cut0=0
  self.control_allow10=0
  self.ledger_equity=1000.0
 def _current_equity_proxy(self):
  return 1000.0 + sum(self.ledger)
 def _dd_pct(self):
  eq=self._current_equity_proxy();self.peak_equity=max(self.peak_equity,eq)
  return 100.0*(self.peak_equity-eq)/self.peak_equity if self.peak_equity>0 else 0.0
 def _controlled_limit(self,a,move,vol):
  if self.control_mode=='off': return 10
  dd=self._dd_pct()
  # Reflected/singular-control inspired layer boundary:
  # preserve expansion in healthy states, progressively reflect exposure as DD nears boundary.
  # Parent favorable excursion acts as a state-quality term.
  trend_good = move >= max(0.60, 0.80*vol)
  strong = move >= max(1.20, 1.20*vol)
  if dd >= 5.0:
   self.control_cut0+=1;return 0
  if dd >= 4.0:
   self.control_cut3+=1;return 3
  if dd >= 3.0:
   self.control_cut5+=1;return 5
  if strong:
   self.control_allow10+=1;return 10
  if trend_good:
   return 5
  return 3
 def on_quote_tick(self,t:QuoteTick):
  bid=f(t.bid_price);ask=f(t.ask_price);self.last_bid=bid;self.last_ask=ask;spread=ask-bid
  if spread>2.5:return
  keep=[]
  for q in self.pending:
   layer,side,p,sl,tp,vol,mult=q
   hit=ask>=p if side>0 else bid<=p
   if hit:
    e=ask if side>0 else bid
    self.active.append({'layer':layer,'side':side,'entry':e,'sl':sl,'tp':tp,'vol':vol,'mult':mult,'stop':e-side*sl,'take':e+side*tp,'g75':[],'g75_next':self.g75_trigger})
    self.entries_day+=1
    if mult>1:self.boosted_entries+=1
    else:self.normal_entries+=1
   else:keep.append(q)
  self.pending=keep
  if not self.active:return
  m1h=self.b[1][1][-1] if self.b[1][1] else None;m1l=self.b[1][2][-1] if self.b[1][2] else None;h1h=self.b[60][1][-1] if self.b[60][1] else None;h1l=self.b[60][2][-1] if self.b[60][2] else None;nxt=[]
  for a in self.active:
   side=a['side'];mark=bid if side>0 else ask;move=(mark-a['entry'])*side;vol=a['vol']
   lim=self._controlled_limit(a,move,vol)
   if lim>0:
    while len(a['g75'])<lim and move>=a['g75_next']:
     a['g75'].append(mark);self.g75_total_children+=1;a['g75_next']+=self.g75_add
     self.g75_max_live=max(self.g75_max_live,len(a['g75']))
   if move>=1.2*vol:a['stop']=max(a['stop'],a['entry']+.2*vol) if side>0 else min(a['stop'],a['entry']-.2*vol)
   if move>=2.0*vol and None not in (m1h,m1l,h1h,h1l):
    gate=(m1h>=h1h-.3*vol) or (m1l<=h1l+.3*vol)
    if gate:
     ns=(m1h-3*vol) if side>0 else (m1l+3*vol);a['stop']=max(a['stop'],ns) if side>0 else min(a['stop'],ns)
   close=(mark<=a['stop'] if side>0 else mark>=a['stop']) or (mark>=a['take'] if side>0 else mark<=a['take'])
   if close:
    parent=move*a['mult'];chase=sum((mark-e)*side*self.g75_child_weight for e in a['g75']);total=parent+chase
    self.pnl[a['layer']].append(total);self.ledger.append(total)
   else:nxt.append(a)
  self.active=nxt
 def summary_math(self):
  s=self.summary_ext_g75();s.update({'control_mode':self.control_mode,'control_dd_boundary_pct':5.0,'control_cut_to_0_count':self.control_cut0,'control_cut_to_3_count':self.control_cut3,'control_cut_to_5_count':self.control_cut5,'control_allow10_count':self.control_allow10})
  return s

def run(catalog,mode,experiment_id):
 cat=ParquetDataCatalog(catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value])
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(l1(raw));s=inst.id.value
 st=GBG75MathControl(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{s}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{s}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{s}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{s}-1-HOUR-BID-INTERNAL'),mode='m15'),mode)
 eng.add_strategy(st);eng.run();eng.end();res={'verification':'GOLDEBRAVE_C_SWEEP2X_G75_MATH_CONTROL_RAW','raw_ticks':len(raw),'ohlc_resample_used':False,'logic':'C+Sweep2x parent; G75 direction-locked chase; math controller reflects layer limit 10->5->3->0 as realized-equity DD approaches 3/4/5%, while strong favorable excursion permits 10 layers.','limitation':'Screening controller uses realized-equity proxy and prior virtual PnL/lot-equivalent harness; not exact intratick floating-DD HJB solution nor broker-order parity.',**st.summary_math()};p=Path('results/goldebrave-g75-math')/experiment_id/f'{mode}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--mode',choices=['off','math'],required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();run(a.catalog,a.mode,a.experiment_id)
if __name__=='__main__':main()
