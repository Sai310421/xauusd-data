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
from goldebrave_c_g75_risk_compression_v8_raw import G75RiskCompressionV8

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
 def _q(self,identifiers=None,start=None,end=None): return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
 ParquetDataCatalog.query_quote_ticks=_q

SPLITS=(0.50,0.30,0.20)
GOV={
 'off': (99.0,99.0,1.0,1.0,1.0),
 'g1': (3.0,5.0,0.75,0.50,0.0),
 'g2': (3.5,5.0,0.80,0.40,0.0),
 'g3': (4.0,5.0,0.65,0.30,0.0),
}

class FloatDD5(G75RiskCompressionV8):
 def __init__(self,c,gov):
  super().__init__(c,'c');self.gov=gov;self.eq_peak=1000.0;self.max_float_dd=0.0;self.last_eq=1000.0;self.scale_hits={1.0:0,0.8:0,0.75:0,0.65:0,0.5:0,0.4:0,0.3:0,0.0:0};self.blocked_adds=0;self.branch_children=[0,0,0];self.max_live_branch=[0,0,0]
 def _floating_equity(self,bid,ask):
  eq=1000.0+sum(self.ledger)
  for a in self.active:
   side=a['side'];mark=bid if side>0 else ask;move=(mark-a['entry'])*side
   eq += move*a['mult']*self.parent_scale_v8
   for b in range(3):
    eq += a['child_realized'][b]
    eq += sum((mark-ch['entry'])*side*ch['open_w']+ch['realized'] for ch in a['g75'][b])
  return eq
 def _scale(self,dd):
  soft,hard,s1,s2,z=GOV[self.gov]
  if dd>=hard:return z
  if dd>=4.5:return s2
  if dd>=soft:return s1
  return 1.0
 def on_quote_tick(self,t:QuoteTick):
  bid=f(t.bid_price);ask=f(t.ask_price);self.last_bid=bid;self.last_ask=ask;spread=ask-bid
  if spread>2.5:return
  keep=[]
  for q in self.pending:
   layer,side,p,sl,tp,vol,mult=q;hit=ask>=p if side>0 else bid<=p
   if hit:
    e=ask if side>0 else bid
    self.active.append({'layer':layer,'side':side,'entry':e,'sl':sl,'tp':tp,'vol':vol,'mult':mult,'stop':e-side*sl,'take':e+side*tp,'g75':[[],[],[]],'g75_next':self.g75_trigger,'g75_next_idx':1,'g75_fired':0,'child_realized':[0.0,0.0,0.0]})
    self.entries_day+=1
    if mult>1:self.boosted_entries+=1
    else:self.normal_entries+=1
   else:keep.append(q)
  self.pending=keep
  eq=self._floating_equity(bid,ask);self.eq_peak=max(self.eq_peak,eq);dd=100*(self.eq_peak-eq)/self.eq_peak if self.eq_peak else 0.0;self.max_float_dd=max(self.max_float_dd,dd);self.last_eq=eq
  if not self.active:return
  m1h=self.b[1][1][-1] if self.b[1][1] else None;m1l=self.b[1][2][-1] if self.b[1][2] else None;h1h=self.b[60][1][-1] if self.b[60][1] else None;h1l=self.b[60][2][-1] if self.b[60][2] else None;nxt=[]
  for a in self.active:
   side=a['side'];mark=bid if side>0 else ask;move=(mark-a['entry'])*side;vol=a['vol']
   while a['g75_next_idx']<=self.max_layers and move>=a['g75_next']:
    eq=self._floating_equity(bid,ask);self.eq_peak=max(self.eq_peak,eq);dd=100*(self.eq_peak-eq)/self.eq_peak if self.eq_peak else 0.0;scale=self._scale(dd);self.scale_hits[scale]=self.scale_hits.get(scale,0)+1
    idx=a['g75_next_idx'];exact_entry=a['entry']+side*a['g75_next'];strong=(idx>=7 and move>=max(0.10,0.15*vol) and spread<=1.00)
    if scale<=0:
     self.blocked_adds+=3
    else:
     for b in range(3):
      ch=self._new_child_v5(exact_entry,idx,vol,strong);ch['open_w']*=scale;ch['branch']=b;a['g75'][b].append(ch);self.branch_children[b]+=1;self.total_children+=1;self.max_live_branch[b]=max(self.max_live_branch[b],len(a['g75'][b]))
    a['g75_fired']+=1;self.max_fired_per_parent=max(self.max_fired_per_parent,a['g75_fired']);a['g75_next_idx']+=1;a['g75_next']=self.g75_trigger+(a['g75_next_idx']-1)*self.g75_add
   for b in range(3):
    kept=[]
    for ch in a['g75'][b]:
     if self._manage_child(ch,side,mark,vol):a['child_realized'][b]+=(mark-ch['entry'])*side*ch['open_w']+ch['realized']
     else:kept.append(ch)
    a['g75'][b]=kept
   self.max_live=max(self.max_live,sum(len(x) for x in a['g75']))
   if move>=1.2*vol:a['stop']=max(a['stop'],a['entry']+.2*vol) if side>0 else min(a['stop'],a['entry']-.2*vol)
   if move>=2.0*vol and None not in (m1h,m1l,h1h,h1l):
    gate=(m1h>=h1h-.3*vol) or (m1l<=h1l+.3*vol)
    if gate:
     ns=(m1h-3*vol) if side>0 else (m1l+3*vol);a['stop']=max(a['stop'],ns) if side>0 else min(a['stop'],ns)
   close=(mark<=a['stop'] if side>0 else mark>=a['stop']) or (mark>=a['take'] if side>0 else mark<=a['take'])
   if close:
    parent=move*a['mult']*self.parent_scale_v8
    chase=sum(a['child_realized'])+sum((mark-ch['entry'])*side*ch['open_w']+ch['realized'] for branch in a['g75'] for ch in branch)
    total=parent+chase;self.pnl[a['layer']].append(total);self.ledger.append(total)
   else:nxt.append(a)
  self.active=nxt
  eq=self._floating_equity(bid,ask);self.eq_peak=max(self.eq_peak,eq);dd=100*(self.eq_peak-eq)/self.eq_peak if self.eq_peak else 0.0;self.max_float_dd=max(self.max_float_dd,dd);self.last_eq=eq
 def summary_float(self):
  s=self.summary_v8();s.update({'governor':self.gov,'branch_children':self.branch_children,'max_live_branch':self.max_live_branch,'max_live_total':self.max_live,'floating_equity_peak':self.eq_peak,'floating_equity_last':self.last_eq,'max_floating_dd_pct':self.max_float_dd,'blocked_adds':self.blocked_adds,'scale_hits':{str(k):v for k,v in self.scale_hits.items()},'dd5_pass':self.max_float_dd<=5.0,'note':'Raw Bid/Ask marked-to-market floating-equity governor. Parent entry logic untouched; only new G75 child size is compressed/blocked.'});return s

def run(catalog,gov,experiment_id):
 cat=ParquetDataCatalog(catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value])
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)));eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(l1(raw));s=inst.id.value
 st=FloatDD5(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{s}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{s}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{s}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{s}-1-HOUR-BID-INTERNAL'),mode='m15'),gov);eng.add_strategy(st);eng.run();eng.end();res={'verification':'GOLDEBRAVE_V8C_SPLIT_G75_FLOATDD5_RAW','raw_ticks':len(raw),'ohlc_resample_used':False,'parent_N_target':96,'variant':'SPLIT_G75_3X10_FULL','limitation':'Marked-to-market sizing harness on Raw QuoteTick; not yet native broker order/fill semantics.',**st.summary_float()};p=Path('results/goldebrave-v8c-split-g75-floatdd5')/experiment_id/f'{gov}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--governor',choices=GOV,required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();run(a.catalog,a.governor,a.experiment_id)
if __name__=='__main__':main()
