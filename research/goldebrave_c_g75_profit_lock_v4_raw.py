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
from goldebrave_c_sweep_boost_raw import GBSweepBoost

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
 def _q(self,identifiers=None,start=None,end=None): return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
 ParquetDataCatalog.query_quote_ticks=_q

class GBG75ProfitLockV4(GBSweepBoost):
 def __init__(self,c,mode:str):
  super().__init__(c,2.0)
  self.mode=mode
  self.max_layers=10
  self.g75_trigger=0.12
  self.g75_add=0.025
  self.w=0.10
  self.total_children=0;self.max_live=0
  self.child_realized=0.0;self.child_partial_closes=0;self.child_be_closes=0;self.child_trail_closes=0
 def _new_child(self,entry,idx,vol):
  return {'entry':entry,'idx':idx,'open_w':self.w,'realized':0.0,'peak':0.0,'be_armed':False}
 def _manage_child(self,ch,side,mark,vol):
  if self.mode=='off': return False
  pnl_move=(mark-ch['entry'])*side
  ch['peak']=max(ch['peak'],pnl_move)
  i=ch['idx']
  # Layers 1-3: full runner to parent exit.
  if i<=3:return False
  # Layers 4-6: arm BE after +0.30R equivalent, close if returns to entry.
  if i<=6:
   arm=max(0.30,0.30*vol)
   if pnl_move>=arm: ch['be_armed']=True
   if ch['be_armed'] and pnl_move<=0:
    self.child_be_closes+=1;return True
   return False
  # Layers 7-10: realize half after +0.35R, trail remainder from peak by 0.25R.
  partial=max(0.35,0.35*vol)
  trail=max(0.25,0.25*vol)
  if ch['open_w']>self.w*0.51 and pnl_move>=partial:
   realized=pnl_move*(self.w*0.5)
   ch['realized']+=realized;self.child_realized+=realized;ch['open_w']=self.w*0.5;self.child_partial_closes+=1
  if ch['peak']>=partial and pnl_move<=max(0.0,ch['peak']-trail):
   self.child_trail_closes+=1;return True
  return False
 def on_quote_tick(self,t:QuoteTick):
  bid=f(t.bid_price);ask=f(t.ask_price);self.last_bid=bid;self.last_ask=ask;spread=ask-bid
  if spread>2.5:return
  keep=[]
  for q in self.pending:
   layer,side,p,sl,tp,vol,mult=q
   hit=ask>=p if side>0 else bid<=p
   if hit:
    e=ask if side>0 else bid
    self.active.append({'layer':layer,'side':side,'entry':e,'sl':sl,'tp':tp,'vol':vol,'mult':mult,'stop':e-side*sl,'take':e+side*tp,'g75':[],'g75_next':self.g75_trigger,'child_realized':0.0})
    self.entries_day+=1
    if mult>1:self.boosted_entries+=1
    else:self.normal_entries+=1
   else:keep.append(q)
  self.pending=keep
  if not self.active:return
  m1h=self.b[1][1][-1] if self.b[1][1] else None;m1l=self.b[1][2][-1] if self.b[1][2] else None;h1h=self.b[60][1][-1] if self.b[60][1] else None;h1l=self.b[60][2][-1] if self.b[60][2] else None;nxt=[]
  for a in self.active:
   side=a['side'];mark=bid if side>0 else ask;move=(mark-a['entry'])*side;vol=a['vol']
   while len(a['g75'])<self.max_layers and move>=a['g75_next']:
    a['g75'].append(self._new_child(mark,len(a['g75'])+1,vol));self.total_children+=1;a['g75_next']+=self.g75_add;self.max_live=max(self.max_live,len(a['g75']))
   kept=[]
   for ch in a['g75']:
    close_child=self._manage_child(ch,side,mark,vol)
    if close_child:
     pnl=(mark-ch['entry'])*side*ch['open_w']+ch['realized'];a['child_realized']+=pnl
    else:kept.append(ch)
   a['g75']=kept
   if move>=1.2*vol:a['stop']=max(a['stop'],a['entry']+.2*vol) if side>0 else min(a['stop'],a['entry']-.2*vol)
   if move>=2.0*vol and None not in (m1h,m1l,h1h,h1l):
    gate=(m1h>=h1h-.3*vol) or (m1l<=h1l+.3*vol)
    if gate:
     ns=(m1h-3*vol) if side>0 else (m1l+3*vol);a['stop']=max(a['stop'],ns) if side>0 else min(a['stop'],ns)
   close=(mark<=a['stop'] if side>0 else mark>=a['stop']) or (mark>=a['take'] if side>0 else mark<=a['take'])
   if close:
    parent=move*a['mult']
    chase=a['child_realized']+sum((mark-ch['entry'])*side*ch['open_w']+ch['realized'] for ch in a['g75'])
    total=parent+chase;self.pnl[a['layer']].append(total);self.ledger.append(total)
   else:nxt.append(a)
  self.active=nxt
 def on_stop(self):
  for a in self.active:
   mark=self.last_bid if a['side']>0 else self.last_ask;side=a['side'];move=(mark-a['entry'])*side
   parent=move*a['mult'];chase=a.get('child_realized',0.0)+sum((mark-ch['entry'])*side*ch['open_w']+ch['realized'] for ch in a.get('g75',[]));total=parent+chase
   self.pnl[a['layer']].append(total);self.ledger.append(total)
 def summary_v4(self):
  s=self.summary_ext();s.update({'control_mode':self.mode,'g75_max_layers':10,'g75_total_children':self.total_children,'g75_max_live_layers':self.max_live,'max_total_lot_equiv':0.10+0.01*self.max_live,'child_partial_closes':self.child_partial_closes,'child_be_closes':self.child_be_closes,'child_trail_closes':self.child_trail_closes,'child_realized_profit':self.child_realized})
  return s

def run(catalog,mode,experiment_id):
 cat=ParquetDataCatalog(catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value])
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(l1(raw));s=inst.id.value
 st=GBG75ProfitLockV4(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{s}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{s}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{s}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{s}-1-HOUR-BID-INTERNAL'),mode='m15'),mode)
 eng.add_strategy(st);eng.run();eng.end();res={'verification':'GOLDEBRAVE_C_SWEEP2X_G75_PROFIT_LOCK_V4_RAW','raw_ticks':len(raw),'ohlc_resample_used':False,'logic':'Keep 10-layer G75 attack. Layers 1-3 run to parent exit; 4-6 arm break-even after +0.30R; 7-10 take half profit after +0.35R and trail remainder by 0.25R.','limitation':'Virtual PnL/lot-equivalent screening harness; child partial/BE/trail logic is a strategy-equity approximation, not exact broker order semantics.',**st.summary_v4()};p=Path('results/goldebrave-g75-profit-lock-v4')/experiment_id/f'{mode}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--mode',choices=['off','v4'],required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();run(a.catalog,a.mode,a.experiment_id)
if __name__=='__main__':main()
