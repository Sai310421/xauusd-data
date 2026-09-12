from __future__ import annotations
import argparse,json,math
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

class GBG75PredictiveV3(GBG75Chase):
 def __init__(self,c,mode:str):
  super().__init__(c,10)
  self.control_mode=mode
  self.peak_equity=1000.0
  self.max_floating_dd_pct=0.0
  self.prev_mid=None;self.prev_ts=None;self.vel_ema=0.0
  self.cut0=self.cut3=self.cut5=self.allow10=self.allow7=0
  self.score_sum=0.0;self.score_n=0
 def _equity_state(self,bid,ask):
  realized=1000.0+sum(self.ledger);floating=0.0
  for a in self.active:
   side=a['side'];mark=bid if side>0 else ask;move=(mark-a['entry'])*side
   floating+=move*a['mult']
   floating+=sum((mark-e)*side*self.g75_child_weight for e in a.get('g75',[]))
  eq=realized+floating;self.peak_equity=max(self.peak_equity,eq)
  dd=100.0*(self.peak_equity-eq)/self.peak_equity if self.peak_equity>0 else 0.0
  self.max_floating_dd_pct=max(self.max_floating_dd_pct,dd)
  return dd
 def _update_velocity(self,t,bid,ask):
  mid=(bid+ask)/2.0;ts=int(t.ts_event)
  if self.prev_mid is not None and self.prev_ts is not None and ts>self.prev_ts:
   dt=(ts-self.prev_ts)/1e9
   v=(mid-self.prev_mid)/dt if dt>0 else 0.0
   self.vel_ema=0.90*self.vel_ema+0.10*v
  self.prev_mid=mid;self.prev_ts=ts
 def _score(self,a,move,vol,spread,dd):
  side=a['side'];signed_vel=self.vel_ema*side
  move_norm=move/max(vol,0.25)
  vel_norm=signed_vel/max(vol/60.0,0.005)
  spread_pen=max(0.0,(spread-0.35)/0.35)
  layer_pen=len(a['g75'])/10.0
  dd_pen=dd/5.0
  score=1.25*move_norm+0.55*vel_norm-0.75*spread_pen-0.80*layer_pen-1.10*dd_pen
  self.score_sum+=score;self.score_n+=1
  return score
 def _limit(self,a,move,vol,spread,dd):
  if self.control_mode=='off':return 10
  score=self._score(a,move,vol,spread,dd)
  # Hard reflection first, then predictive permission.
  if dd>=4.5:self.cut0+=1;return 0
  if dd>=3.8:self.cut3+=1;return 3
  if dd>=3.0:self.cut5+=1;return 5
  if score>=2.2 and move>=max(1.00,1.00*vol):self.allow10+=1;return 10
  if score>=1.4 and move>=max(0.70,0.75*vol):self.allow7+=1;return 7
  if score>=0.7:return 5
  return 3
 def on_quote_tick(self,t:QuoteTick):
  bid=f(t.bid_price);ask=f(t.ask_price);self.last_bid=bid;self.last_ask=ask;spread=ask-bid
  if spread>2.5:return
  self._update_velocity(t,bid,ask)
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
  dd=self._equity_state(bid,ask)
  m1h=self.b[1][1][-1] if self.b[1][1] else None;m1l=self.b[1][2][-1] if self.b[1][2] else None;h1h=self.b[60][1][-1] if self.b[60][1] else None;h1l=self.b[60][2][-1] if self.b[60][2] else None;nxt=[]
  for a in self.active:
   side=a['side'];mark=bid if side>0 else ask;move=(mark-a['entry'])*side;vol=a['vol']
   lim=self._limit(a,move,vol,spread,dd)
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
 def summary_v3(self):
  s=self.summary_ext_g75();s.update({'control_mode':self.control_mode,'max_floating_dd_pct_observed':self.max_floating_dd_pct,'predictive_score_mean':self.score_sum/self.score_n if self.score_n else 0.0,'control_cut0':self.cut0,'control_cut3':self.cut3,'control_cut5':self.cut5,'control_allow7':self.allow7,'control_allow10':self.allow10})
  return s

def run(catalog,mode,experiment_id):
 cat=ParquetDataCatalog(catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value])
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(l1(raw));s=inst.id.value
 st=GBG75PredictiveV3(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{s}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{s}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{s}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{s}-1-HOUR-BID-INTERNAL'),mode='m15'),mode)
 eng.add_strategy(st);eng.run();eng.end();res={'verification':'GOLDEBRAVE_C_SWEEP2X_G75_PREDICTIVE_V3_RAW','raw_ticks':len(raw),'ohlc_resample_used':False,'logic':'Predictive controller uses intratick DD + parent excursion + signed velocity + spread + layer density to permit 3/5/7/10 G75 layers.','limitation':'Virtual PnL/lot-equivalent screening harness; predictive score is heuristic control approximation, not solved HJB policy.',**st.summary_v3()};p=Path('results/goldebrave-g75-v3')/experiment_id/f'{mode}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--mode',choices=['off','v3'],required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();run(a.catalog,a.mode,a.experiment_id)
if __name__=='__main__':main()
