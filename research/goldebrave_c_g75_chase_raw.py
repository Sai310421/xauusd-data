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
from goldebrave_fasttf_parity_raw import Cfg,f,l1,met
from goldebrave_c_sweep_boost_raw import GBSweepBoost

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
 def _q(self,identifiers=None,start=None,end=None): return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
 ParquetDataCatalog.query_quote_ticks=_q

class GBG75Chase(GBSweepBoost):
 def __init__(self,c,max_layers:int):
  super().__init__(c,2.0)
  self.max_layers=max_layers
  self.g75_trigger=0.12
  self.g75_add=0.025
  self.g75_child_weight=0.10  # parent GoldeBrave 0.10 lot vs G75 child 0.01 lot
  self.g75_total_children=0
  self.g75_max_live=0
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
   # G75 is direction-locked to the parent N/C entry. Add only while price extends favorably.
   if self.max_layers>0:
    while len(a['g75'])<self.max_layers and move>=a['g75_next']:
     a['g75'].append(mark);self.g75_total_children+=1;a['g75_next']+=self.g75_add
     self.g75_max_live=max(self.g75_max_live,len(a['g75']))
   if move>=1.2*vol:a['stop']=max(a['stop'],a['entry']+.2*vol) if side>0 else min(a['stop'],a['entry']-.2*vol)
   if move>=2.0*vol and None not in (m1h,m1l,h1h,h1l):
    gate=(m1h>=h1h-.3*vol) or (m1l<=h1l+.3*vol)
    if gate:
     ns=(m1h-3*vol) if side>0 else (m1l+3*vol);a['stop']=max(a['stop'],ns) if side>0 else min(a['stop'],ns)
   close=(mark<=a['stop'] if side>0 else mark>=a['stop']) or (mark>=a['take'] if side>0 else mark<=a['take'])
   if close:
    parent=move*a['mult']
    chase=sum((mark-e)*side*self.g75_child_weight for e in a['g75'])
    total=parent+chase
    self.pnl[a['layer']].append(total);self.ledger.append(total)
   else:nxt.append(a)
  self.active=nxt
 def on_stop(self):
  for a in self.active:
   mark=self.last_bid if a['side']>0 else self.last_ask;side=a['side'];move=(mark-a['entry'])*side
   parent=move*a['mult'];chase=sum((mark-e)*side*self.g75_child_weight for e in a.get('g75',[]));total=parent+chase
   self.pnl[a['layer']].append(total);self.ledger.append(total)
 def summary_ext_g75(self):
  s=self.summary_ext();s.update({'g75_max_layers':self.max_layers,'g75_trigger':self.g75_trigger,'g75_add':self.g75_add,'g75_child_weight_vs_parent_0_10lot':self.g75_child_weight,'g75_total_children':self.g75_total_children,'g75_max_live_layers':self.g75_max_live,'max_total_lot_equiv':0.10+0.01*self.g75_max_live})
  return s

def run(catalog,max_layers,experiment_id):
 cat=ParquetDataCatalog(catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value])
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(l1(raw));s=inst.id.value
 st=GBG75Chase(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{s}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{s}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{s}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{s}-1-HOUR-BID-INTERNAL'),mode='m15'),max_layers)
 eng.add_strategy(st);eng.run();eng.end();res={'verification':'GOLDEBRAVE_C_SWEEP2X_G75_DIRECTIONAL_CHASE_RAW','raw_ticks':len(raw),'ohlc_resample_used':False,'logic':'C remains enabled; Sweep same-direction entries use 2x parent exposure. G75 chase is locked to parent direction, starts after +0.12 price move, adds each +0.025, child size 0.01 vs GoldeBrave parent 0.10, and all chase layers close when parent exits/TPs.','limitation':'Virtual PnL/lot-equivalent scaling on prior GoldeBrave C approximation; validates chase interaction, not exact broker order semantics.',**st.summary_ext_g75()};p=Path('results/goldebrave-g75-chase')/experiment_id/f'layers_{max_layers}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--max-layers',type=int,choices=[0,3,5,10],required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();run(a.catalog,a.max_layers,a.experiment_id)
if __name__=='__main__':main()
