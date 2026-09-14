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
PROFILES={
 'off': {'levels':()},
 'r1': {'levels':((3.0,0.25),(4.0,0.35),(4.5,0.50))},
 'r2': {'levels':((3.5,0.20),(4.2,0.35),(4.7,0.60))},
 'r3': {'levels':((4.0,0.25),(4.5,0.50),(4.8,1.00))},
}

class LossSideReduce(G75RiskCompressionV8):
 def __init__(self,c,profile):
  super().__init__(c,'c');self.profile=profile;self.eq_peak=1000.0;self.max_float_dd=0.0;self.last_eq=1000.0;self.reduction_events=0;self.reduced_parent_weight=0.0;self.reduced_child_weight=0.0;self.realized_reduce_pnl=0.0;self.level_armed={i:True for i,_ in enumerate(PROFILES[profile]['levels'])};self.branch_children=[0,0,0];self.max_live_branch=[0,0,0]
 def _floating_equity(self,bid,ask):
  eq=1000.0+sum(self.ledger)
  for a in self.active:
   side=a['side'];mark=bid if side>0 else ask;move=(mark-a['entry'])*side
   eq += a['parent_realized'] + move*a['mult']*self.parent_scale_v8*a['parent_w']
   for b in range(3):
    eq += a['child_realized'][b]
    eq += sum((mark-ch['entry'])*side*ch['open_w']+ch['realized'] for ch in a['g75'][b])
  return eq
 def _mark_dd(self,bid,ask):
  eq=self._floating_equity(bid,ask);self.eq_peak=max(self.eq_peak,eq);dd=100*(self.eq_peak-eq)/self.eq_peak if self.eq_peak else 0.0;self.max_float_dd=max(self.max_float_dd,dd);self.last_eq=eq;return dd
 def _reduce_worst(self,bid,ask,frac):
  worst=None
  for ai,a in enumerate(self.active):
   side=a['side'];mark=bid if side>0 else ask;move=(mark-a['entry'])*side
   ppnl=move*a['mult']*self.parent_scale_v8*a['parent_w']
   if a['parent_w']>0 and ppnl<0 and (worst is None or ppnl<worst[0]):worst=(ppnl,'p',ai,-1,-1)
   for b in range(3):
    for ci,ch in enumerate(a['g75'][b]):
     cpnl=(mark-ch['entry'])*side*ch['open_w']
     if ch['open_w']>0 and cpnl<0 and (worst is None or cpnl<worst[0]):worst=(cpnl,'c',ai,b,ci)
  if worst is None:return False
  _,kind,ai,b,ci=worst;a=self.active[ai];side=a['side'];mark=bid if side>0 else ask
  if kind=='p':
   old=a['parent_w'];cut=old*frac;move=(mark-a['entry'])*side;pnl=move*a['mult']*self.parent_scale_v8*cut;a['parent_realized']+=pnl;a['parent_w']-=cut;self.reduced_parent_weight+=cut;self.realized_reduce_pnl+=pnl
  else:
   ch=a['g75'][b][ci];old=ch['open_w'];cut=old*frac;pnl=(mark-ch['entry'])*side*cut;a['child_realized'][b]+=pnl;ch['open_w']-=cut;self.reduced_child_weight+=cut;self.realized_reduce_pnl+=pnl
  self.reduction_events+=1;return True
 def _risk_control(self,bid,ask,dd):
  levels=PROFILES[self.profile]['levels']
  for i,(thr,frac) in enumerate(levels):
   # hysteresis re-arm after DD recovers 0.75 percentage point below threshold
   if dd < max(0.0,thr-0.75):self.level_armed[i]=True
   if self.level_armed[i] and dd>=thr:
    self._reduce_worst(bid,ask,frac);self.level_armed[i]=False
 def on_quote_tick(self,t:QuoteTick):
  bid=f(t.bid_price);ask=f(t.ask_price);self.last_bid=bid;self.last_ask=ask;spread=ask-bid
  if spread>2.5:return
  keep=[]
  for q in self.pending:
   layer,side,p,sl,tp,vol,mult=q;hit=ask>=p if side>0 else bid<=p
   if hit:
    e=ask if side>0 else bid
    self.active.append({'layer':layer,'side':side,'entry':e,'sl':sl,'tp':tp,'vol':vol,'mult':mult,'stop':e-side*sl,'take':e+side*tp,'parent_w':1.0,'parent_realized':0.0,'g75':[[],[],[]],'g75_next':self.g75_trigger,'g75_next_idx':1,'g75_fired':0,'child_realized':[0.0,0.0,0.0]})
    self.entries_day+=1
    if mult>1:self.boosted_entries+=1
    else:self.normal_entries+=1
   else:keep.append(q)
  self.pending=keep
  dd=self._mark_dd(bid,ask);self._risk_control(bid,ask,dd);self._mark_dd(bid,ask)
  if not self.active:return
  m1h=self.b[1][1][-1] if self.b[1][1] else None;m1l=self.b[1][2][-1] if self.b[1][2] else None;h1h=self.b[60][1][-1] if self.b[60][1] else None;h1l=self.b[60][2][-1] if self.b[60][2] else None;nxt=[]
  for a in self.active:
   side=a['side'];mark=bid if side>0 else ask;move=(mark-a['entry'])*side;vol=a['vol']
   # Keep the profitable-direction G75 engine fully enabled. DD control never blocks a new ADD.
   while a['g75_next_idx']<=self.max_layers and move>=a['g75_next']:
    idx=a['g75_next_idx'];exact_entry=a['entry']+side*a['g75_next'];strong=(idx>=7 and move>=max(0.10,0.15*vol) and spread<=1.00)
    for b in range(3):
     ch=self._new_child_v5(exact_entry,idx,vol,strong);ch['branch']=b;a['g75'][b].append(ch);self.branch_children[b]+=1;self.total_children+=1;self.max_live_branch[b]=max(self.max_live_branch[b],len(a['g75'][b]))
    a['g75_fired']+=1;self.max_fired_per_parent=max(self.max_fired_per_parent,a['g75_fired']);a['g75_next_idx']+=1;a['g75_next']=self.g75_trigger+(a['g75_next_idx']-1)*self.g75_add
   for b in range(3):
    kept=[]
    for ch in a['g75'][b]:
     if ch['open_w']<=1e-12:continue
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
    parent=a['parent_realized']+move*a['mult']*self.parent_scale_v8*a['parent_w'];chase=sum(a['child_realized'])+sum((mark-ch['entry'])*side*ch['open_w']+ch['realized'] for branch in a['g75'] for ch in branch);total=parent+chase;self.pnl[a['layer']].append(total);self.ledger.append(total)
   else:nxt.append(a)
  self.active=nxt
  dd=self._mark_dd(bid,ask);self._risk_control(bid,ask,dd);self._mark_dd(bid,ask)
 def summary_lossreduce(self):
  s=self.summary_v8();s.update({'lossreduce_profile':self.profile,'branch_children':self.branch_children,'max_live_branch':self.max_live_branch,'max_live_total':self.max_live,'floating_equity_peak':self.eq_peak,'floating_equity_last':self.last_eq,'max_floating_dd_pct':self.max_float_dd,'dd5_pass':self.max_float_dd<=5.0,'reduction_events':self.reduction_events,'reduced_parent_weight':self.reduced_parent_weight,'reduced_child_weight':self.reduced_child_weight,'realized_reduce_pnl':self.realized_reduce_pnl,'levels':PROFILES[self.profile]['levels'],'note':'Raw Bid/Ask MTM screening. Parent signal count is preserved; profitable-direction G75 ADDs remain enabled. Risk control only partially realizes the currently worst losing parent/child exposure with hysteresis.'});return s

def run(catalog,profile,experiment_id):
 cat=ParquetDataCatalog(catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value])
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)));eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(l1(raw));s=inst.id.value
 st=LossSideReduce(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{s}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{s}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{s}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{s}-1-HOUR-BID-INTERNAL'),mode='m15'),profile);eng.add_strategy(st);eng.run();eng.end();res={'verification':'GOLDEBRAVE_V8C_SPLIT_G75_LOSSREDUCE_RAW','raw_ticks':len(raw),'ohlc_resample_used':False,'parent_N_target':96,'variant':'SPLIT_G75_3X10_FULL','limitation':'Marked-to-market sizing/partial-reduction harness on Raw QuoteTick; still not native broker order/fill semantics.',**st.summary_lossreduce()};p=Path('results/goldebrave-v8c-split-g75-lossreduce')/experiment_id/f'{profile}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--profile',choices=PROFILES,required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();run(a.catalog,a.profile,a.experiment_id)
if __name__=='__main__':main()
