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
 'shared10': {'caps':(4,3,3),'independent':False},
 'independent30': {'caps':(10,10,10),'independent':True},
}
WEIGHTS=(0.50,0.30,0.20)
# stage 1 enters with original parent; stage 2/3 only after favorable confirmation
STAGE_TRIGGERS=(0.00,0.06,0.12)

class ThreeSplitG75(G75SelectiveBoostV7):
 def __init__(self,c,profile):
  super().__init__(c,2.50)
  self.profile=profile;self.caps=PROFILES[profile]['caps'];self.independent=PROFILES[profile]['independent']
  self.stage_activations=[0,0,0];self.stage_children=[0,0,0]
  self.eq_peak=1000.0;self.max_float_dd=0.0;self.dd35_ticks=0;self.dd45_ticks=0;self.dd50_ticks=0
  self.realized_total=0.0

 def _new_stage(self,entry,weight,cap):
  return {'active':True,'entry':entry,'weight':weight,'cap':cap,'children':[],'next_idx':1,'next_thr':self.g75_trigger,'child_realized':0.0}

 def _float_equity(self,bid,ask):
  fl=0.0
  for a in self.active:
   side=a['side'];mark=bid if side>0 else ask
   for st in a['stages']:
    if not st or not st['active']:continue
    fl += (mark-st['entry'])*side*a['mult']*st['weight']
    for ch in st['children']:
     fl += ((mark-ch['entry'])*side*ch['open_w']+ch['realized'])*st['weight']
  return 1000.0+self.realized_total+fl

 def on_quote_tick(self,t:QuoteTick):
  bid=f(t.bid_price);ask=f(t.ask_price);self.last_bid=bid;self.last_ask=ask;spread=ask-bid
  if spread>2.5:return
  keep=[]
  for q in self.pending:
   layer,side,p,sl,tp,vol,mult=q;hit=ask>=p if side>0 else bid<=p
   if hit:
    e=ask if side>0 else bid
    stages=[self._new_stage(e,WEIGHTS[0],self.caps[0]),None,None]
    self.stage_activations[0]+=1
    self.active.append({'layer':layer,'side':side,'entry':e,'sl':sl,'tp':tp,'vol':vol,'mult':mult,'stop':e-side*sl,'take':e+side*tp,'stages':stages})
    self.entries_day+=1
    if mult>1:self.boosted_entries+=1
    else:self.normal_entries+=1
   else:keep.append(q)
  self.pending=keep
  if not self.active:return
  m1h=self.b[1][1][-1] if self.b[1][1] else None;m1l=self.b[1][2][-1] if self.b[1][2] else None;h1h=self.b[60][1][-1] if self.b[60][1] else None;h1l=self.b[60][2][-1] if self.b[60][2] else None;nxt=[]
  for a in self.active:
   side=a['side'];mark=bid if side>0 else ask;move=(mark-a['entry'])*side;vol=a['vol']
   # Activate P2/P3 only after price proves direction; preserves the original signal count N.
   for j in (1,2):
    if a['stages'][j] is None and move>=STAGE_TRIGGERS[j]:
     a['stages'][j]=self._new_stage(a['entry']+side*STAGE_TRIGGERS[j],WEIGHTS[j],self.caps[j]);self.stage_activations[j]+=1
   for j,st in enumerate(a['stages']):
    if st is None:continue
    smove=(mark-st['entry'])*side
    while st['next_idx']<=st['cap'] and smove>=st['next_thr']:
     idx=st['next_idx'];exact=st['entry']+side*st['next_thr']
     # late children get the V6-C strong-state selective boost
     local_late=(idx>=max(1,st['cap']-2))
     strong=(local_late and smove>=max(0.10,0.15*vol) and spread<=1.00)
     ch=self._new_child_v5(exact,idx,vol,strong);st['children'].append(ch)
     self.total_children+=1;self.stage_children[j]+=1;self.max_live=max(self.max_live,sum(len(x['children']) for x in a['stages'] if x))
     st['next_idx']+=1;st['next_thr']=self.g75_trigger+(st['next_idx']-1)*self.g75_add
    kept=[]
    for ch in st['children']:
     if self._manage_child(ch,side,mark,vol):
      st['child_realized']+=(mark-ch['entry'])*side*ch['open_w']+ch['realized']
     else:kept.append(ch)
    st['children']=kept
   if move>=1.2*vol:a['stop']=max(a['stop'],a['entry']+.2*vol) if side>0 else min(a['stop'],a['entry']-.2*vol)
   if move>=2.0*vol and None not in (m1h,m1l,h1h,h1l):
    gate=(m1h>=h1h-.3*vol) or (m1l<=h1l+.3*vol)
    if gate:
     ns=(m1h-3*vol) if side>0 else (m1l+3*vol);a['stop']=max(a['stop'],ns) if side>0 else min(a['stop'],ns)
   close=(mark<=a['stop'] if side>0 else mark>=a['stop']) or (mark>=a['take'] if side>0 else mark<=a['take'])
   if close:
    total=0.0
    for st in a['stages']:
     if st is None:continue
     parent=(mark-st['entry'])*side*a['mult']*st['weight']
     chase=st['child_realized']+sum(((mark-ch['entry'])*side*ch['open_w']+ch['realized'])*st['weight'] for ch in st['children'])
     total += parent+chase
    self.pnl[a['layer']].append(total);self.ledger.append(total);self.realized_total+=total
   else:nxt.append(a)
  self.active=nxt
  eq=self._float_equity(bid,ask);self.eq_peak=max(self.eq_peak,eq);dd=max(0.0,(self.eq_peak-eq)/self.eq_peak*100.0);self.max_float_dd=max(self.max_float_dd,dd)
  if dd>=3.5:self.dd35_ticks+=1
  if dd>=4.5:self.dd45_ticks+=1
  if dd>=5.0:self.dd50_ticks+=1

 def summary_split(self):
  s=self.summary_v7();s.update({'split_profile':self.profile,'parent_signal_N':len(self.ledger),'weights':WEIGHTS,'stage_triggers':STAGE_TRIGGERS,'stage_caps':self.caps,'stage_activations':self.stage_activations,'stage_children':self.stage_children,'max_floating_dd_pct_exact_proxy':self.max_float_dd,'dd_ge_3_5_ticks':self.dd35_ticks,'dd_ge_4_5_ticks':self.dd45_ticks,'dd_ge_5_ticks':self.dd50_ticks,'logic':'Each original parent signal is split 50/30/20. P2/P3 activate after +0.06/+0.12 favorable move. shared10 allocates G75 caps 4/3/3; independent30 allows 10 per stage.'});return s

def run(catalog,profile,experiment_id):
 cat=ParquetDataCatalog(catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value])
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(l1(raw));s=inst.id.value
 st=ThreeSplitG75(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{s}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{s}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{s}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{s}-1-HOUR-BID-INTERNAL'),mode='m15'),profile)
 eng.add_strategy(st);eng.run();eng.end();res={'verification':'GOLDEBRAVE_C_THREE_SPLIT_G75_RAW_V1','raw_ticks':len(raw),'ohlc_resample_used':False,'limitation':'Split/G75 sizing is a causal Raw Bid/Ask screening model, not exact broker order-margin semantics. Floating DD metric is tick-level within this internal sizing model.',**st.summary_split()};p=Path('results/goldebrave-three-split-g75')/experiment_id/f'{profile}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--profile',choices=list(PROFILES),required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();run(a.catalog,a.profile,a.experiment_id)
if __name__=='__main__':main()
