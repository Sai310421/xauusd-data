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

PROFILES=('shared10','independent30')
SPLIT_WEIGHTS=(0.50,0.30,0.20)
SPLIT_ACTIVATE=(0.00,0.06,0.12)
SHARED_BUDGETS=(4,3,3)

class ThreeSplitG75V9(G75SelectiveBoostV7):
 def __init__(self,c,profile:str):
  super().__init__(c,2.50)
  self.profile_v9=profile
  self.split_activations=[0,0,0]
  self.slice_child_counts=[0,0,0]
  self.max_total_children_per_parent=0

 def _make_slice(self,root_entry,side,sl,tp,vol,mult,idx):
  return {
   'idx':idx,'weight':SPLIT_WEIGHTS[idx],'activate_move':SPLIT_ACTIVATE[idx],
   'active':idx==0,'entry':root_entry+side*SPLIT_ACTIVATE[idx] if idx else root_entry,
   'stop':root_entry-side*sl,'take':root_entry+side*tp,
   'g75':[],'next_idx':1,'next_thr':self.g75_trigger,'fired':0,'child_realized':0.0,
  }

 def on_quote_tick(self,t:QuoteTick):
  bid=f(t.bid_price);ask=f(t.ask_price);self.last_bid=bid;self.last_ask=ask;spread=ask-bid
  if spread>2.5:return
  keep=[]
  for q in self.pending:
   layer,side,p,sl,tp,vol,mult=q;hit=ask>=p if side>0 else bid<=p
   if hit:
    e=ask if side>0 else bid
    slices=[self._make_slice(e,side,sl,tp,vol,mult,i) for i in range(3)]
    self.split_activations[0]+=1
    self.active.append({'layer':layer,'side':side,'entry':e,'sl':sl,'tp':tp,'vol':vol,'mult':mult,'slices':slices})
    self.entries_day+=1
    if mult>1:self.boosted_entries+=1
    else:self.normal_entries+=1
   else:keep.append(q)
  self.pending=keep
  if not self.active:return
  m1h=self.b[1][1][-1] if self.b[1][1] else None;m1l=self.b[1][2][-1] if self.b[1][2] else None;h1h=self.b[60][1][-1] if self.b[60][1] else None;h1l=self.b[60][2][-1] if self.b[60][2] else None
  nxt=[]
  for a in self.active:
   side=a['side'];mark=bid if side>0 else ask;root_move=(mark-a['entry'])*side;vol=a['vol']
   # Activate 30/20 slices only after price proves direction.
   for i in (1,2):
    s=a['slices'][i]
    if not s['active'] and root_move>=s['activate_move']:
     s['active']=True;s['entry']=a['entry']+side*s['activate_move'];self.split_activations[i]+=1
   total_children_live=0
   for s in a['slices']:
    if not s['active']:continue
    smove=(mark-s['entry'])*side
    budget=SHARED_BUDGETS[s['idx']] if self.profile_v9=='shared10' else 10
    while s['next_idx']<=budget and smove>=s['next_thr']:
     local_idx=s['next_idx'];global_idx=(sum(SHARED_BUDGETS[:s['idx']])+local_idx) if self.profile_v9=='shared10' else local_idx
     exact_entry=s['entry']+side*s['next_thr']
     strong=(global_idx>=7 and root_move>=max(0.10,0.15*vol) and spread<=1.00)
     ch=self._new_child_v5(exact_entry,global_idx,vol,strong)
     # Scale each child's exposure to its parent slice.
     ch['open_w']*=s['weight'];ch['init_w']*=s['weight']
     s['g75'].append(ch);s['fired']+=1;s['next_idx']+=1;s['next_thr']=self.g75_trigger+(s['next_idx']-1)*self.g75_add
     self.total_children+=1;self.slice_child_counts[s['idx']]+=1
    kept=[]
    for ch in s['g75']:
     if self._manage_child(ch,side,mark,vol):
      s['child_realized']+=(mark-ch['entry'])*side*ch['open_w']+ch['realized']
     else:kept.append(ch)
    s['g75']=kept;total_children_live+=len(kept)
   self.max_live=max(self.max_live,total_children_live)
   fired_parent=sum(s['fired'] for s in a['slices']);self.max_total_children_per_parent=max(self.max_total_children_per_parent,fired_parent)
   # Root-level BE/trail applied to all activated split parents.
   if root_move>=1.2*vol:
    for s in a['slices']:
     if s['active']:s['stop']=max(s['stop'],s['entry']+.2*vol) if side>0 else min(s['stop'],s['entry']-.2*vol)
   if root_move>=2.0*vol and None not in (m1h,m1l,h1h,h1l):
    gate=(m1h>=h1h-.3*vol) or (m1l<=h1l+.3*vol)
    if gate:
     ns=(m1h-3*vol) if side>0 else (m1l+3*vol)
     for s in a['slices']:
      if s['active']:s['stop']=max(s['stop'],ns) if side>0 else min(s['stop'],ns)
   # Close root signal when original take is hit, or when every activated slice stop is hit.
   hit_take=(mark>=a['entry']+side*a['tp'] if side>0 else mark<=a['entry']+side*a['tp'])
   act=[s for s in a['slices'] if s['active']]
   hit_all_stops=bool(act) and all((mark<=s['stop'] if side>0 else mark>=s['stop']) for s in act)
   if hit_take or hit_all_stops:
    parent=0.0;chase=0.0
    for s in act:
     parent+=(mark-s['entry'])*side*a['mult']*s['weight']
     chase+=s['child_realized']+sum((mark-ch['entry'])*side*ch['open_w']+ch['realized'] for ch in s['g75'])
    total=parent+chase;self.pnl[a['layer']].append(total);self.ledger.append(total)
   else:nxt.append(a)
  self.active=nxt

 def summary_v9(self):
  s=self.summary_v7();s.update({
   'v9_profile':self.profile_v9,'split_weights':SPLIT_WEIGHTS,'split_activate_moves':SPLIT_ACTIVATE,
   'shared_budgets':SHARED_BUDGETS if self.profile_v9=='shared10' else None,
   'split_activations':self.split_activations,'slice_child_counts':self.slice_child_counts,
   'max_total_children_per_parent':self.max_total_children_per_parent,
   'v9_logic':'N unchanged. Parent signal split 50/30/20; later slices activate only after +0.06/+0.12 favorable move. Each activated slice owns G75 state. shared10=4+3+3 children; independent30=10 each.',
  });return s

def run(catalog,profile,experiment_id):
 cat=ParquetDataCatalog(catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value])
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(l1(raw));sym=inst.id.value
 st=ThreeSplitG75V9(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{sym}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{sym}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{sym}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{sym}-1-HOUR-BID-INTERNAL'),mode='m15'),profile)
 eng.add_strategy(st);eng.run();eng.end()
 res={'verification':'GOLDEBRAVE_C_THREE_SPLIT_G75_V9_RAW','raw_ticks':len(raw),'ohlc_resample_used':False,'limitation':'Virtual PnL/lot-equivalent screening harness; split activation and child exposure are internally scaled proxies, not exact broker order semantics.',**st.summary_v9()}
 p=Path('results/goldebrave-three-split-g75-v9')/experiment_id/f'{profile}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--profile',choices=PROFILES,required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();run(a.catalog,a.profile,a.experiment_id)
if __name__=='__main__':main()
