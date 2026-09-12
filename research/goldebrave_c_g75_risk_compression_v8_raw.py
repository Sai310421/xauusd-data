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
from goldebrave_c_g75_selective_boost_v7_raw import G75SelectiveBoostV7

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
 def _q(self,identifiers=None,start=None,end=None): return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
 ParquetDataCatalog.query_quote_ticks=_q

PROFILES={
 'a':(0.90,2.75),
 'b':(0.80,3.00),
 'c':(0.70,3.25),
}

class G75RiskCompressionV8(G75SelectiveBoostV7):
 def __init__(self,c,profile:str):
  parent_scale,child_boost=PROFILES[profile]
  super().__init__(c,child_boost)
  self.profile_v8=profile
  self.parent_scale_v8=parent_scale
  self.child_boost_v8=child_boost
 def on_quote_tick(self,t:QuoteTick):
  bid=f(t.bid_price);ask=f(t.ask_price);self.last_bid=bid;self.last_ask=ask;spread=ask-bid
  if spread>2.5:return
  keep=[]
  for q in self.pending:
   layer,side,p,sl,tp,vol,mult=q;hit=ask>=p if side>0 else bid<=p
   if hit:
    e=ask if side>0 else bid
    self.active.append({'layer':layer,'side':side,'entry':e,'sl':sl,'tp':tp,'vol':vol,'mult':mult,'stop':e-side*sl,'take':e+side*tp,'g75':[],'g75_next':self.g75_trigger,'g75_next_idx':1,'g75_fired':0,'child_realized':0.0})
    self.entries_day+=1
    if mult>1:self.boosted_entries+=1
    else:self.normal_entries+=1
   else:keep.append(q)
  self.pending=keep
  if not self.active:return
  m1h=self.b[1][1][-1] if self.b[1][1] else None;m1l=self.b[1][2][-1] if self.b[1][2] else None;h1h=self.b[60][1][-1] if self.b[60][1] else None;h1l=self.b[60][2][-1] if self.b[60][2] else None;nxt=[]
  for a in self.active:
   side=a['side'];mark=bid if side>0 else ask;move=(mark-a['entry'])*side;vol=a['vol']
   while a['g75_next_idx']<=self.max_layers and move>=a['g75_next']:
    idx=a['g75_next_idx'];threshold_move=a['g75_next'];exact_entry=a['entry']+side*threshold_move
    strong=(idx>=7 and move>=max(0.10,0.15*vol) and spread<=1.00)
    a['g75'].append(self._new_child_v5(exact_entry,idx,vol,strong));self.total_children+=1;a['g75_fired']+=1
    self.max_fired_per_parent=max(self.max_fired_per_parent,a['g75_fired']);a['g75_next_idx']+=1;a['g75_next']=self.g75_trigger+(a['g75_next_idx']-1)*self.g75_add;self.max_live=max(self.max_live,len(a['g75']))
   kept=[]
   for ch in a['g75']:
    if self._manage_child(ch,side,mark,vol):
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
    parent=move*a['mult']*self.parent_scale_v8
    chase=a['child_realized']+sum((mark-ch['entry'])*side*ch['open_w']+ch['realized'] for ch in a['g75'])
    total=parent+chase;self.pnl[a['layer']].append(total);self.ledger.append(total)
   else:nxt.append(a)
  self.active=nxt
 def summary_v8(self):
  s=self.summary_v7();s.update({'v8_profile':self.profile_v8,'v8_parent_scale':self.parent_scale_v8,'v8_child_boost':self.child_boost_v8,'v8_logic':'Keep N fixed; compress parent PnL/exposure proxy and reallocate risk to strong-state G75 layers 7-10 only.'});return s

def run(catalog,profile,experiment_id):
 cat=ParquetDataCatalog(catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value])
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(l1(raw));s=inst.id.value
 st=G75RiskCompressionV8(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{s}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{s}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{s}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{s}-1-HOUR-BID-INTERNAL'),mode='m15'),profile)
 eng.add_strategy(st);eng.run();eng.end();res={'verification':'GOLDEBRAVE_C_G75_RISK_COMPRESSION_V8_RAW','raw_ticks':len(raw),'ohlc_resample_used':False,'logic':'No-Recycle + Exact Threshold + V6-C strong-state gate. Parent risk compression with selective child reallocation; N unchanged.','limitation':'Virtual PnL/lot-equivalent screening harness; parent scale and selective boost are sizing proxies, not exact broker order semantics.',**st.summary_v8()};p=Path('results/goldebrave-g75-risk-compression-v8')/experiment_id/f'{profile}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--profile',choices=['a','b','c'],required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();run(a.catalog,a.profile,a.experiment_id)
if __name__=='__main__':main()
