from __future__ import annotations
import argparse,json,math
from collections import deque
from decimal import Decimal
from pathlib import Path
from statistics import median
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from goldebrave_fasttf_parity_raw import Cfg,f,l1
from goldebrave_v8c_split_g75_lossreduce_raw import LossSideReduce

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
 def _q(self,identifiers=None,start=None,end=None): return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
 ParquetDataCatalog.query_quote_ticks=_q

# Dynamic-CVaR + reflected minimum-intervention profiles.
# boundary: desired projected DD barrier.
# alpha: empirical tail confidence.
# vol_floor: minimum stress move as fraction of median active vol.
# horizon: scales one-tick empirical CVaR to a short adverse excursion.
# reserve: extra safety multiple on projected tail loss.
PROFILES={
 'off': {'boundary':99.0,'alpha':0.95,'vol_floor':0.0,'horizon':1.0,'reserve':1.0},
 'cv1': {'boundary':5.0,'alpha':0.95,'vol_floor':0.35,'horizon':8.0,'reserve':1.05},
 'cv2': {'boundary':5.0,'alpha':0.975,'vol_floor':0.45,'horizon':12.0,'reserve':1.10},
 'cv3': {'boundary':5.0,'alpha':0.99,'vol_floor':0.60,'horizon':16.0,'reserve':1.15},
}

class CVarReflect(LossSideReduce):
 def __init__(self,c,profile):
  # Initialize parent mechanics but disable the old threshold reducer.
  super().__init__(c,'off')
  self.cvar_profile=profile
  self.abs_moves=deque(maxlen=4000)
  self.last_mid=None
  self.cvar_control_events=0
  self.cvar_reduced_parent_weight=0.0
  self.cvar_reduced_child_weight=0.0
  self.cvar_realized_reduce_pnl=0.0
  self.projected_tail_loss_max=0.0
  self.projected_dd_max=0.0
  self.boundary_touches=0
  self.minimum_intervention_excess=0.0
  self.cvar_samples_min=128

 def on_quote_tick(self,t:QuoteTick):
  bid=f(t.bid_price);ask=f(t.ask_price);mid=0.5*(bid+ask)
  if self.last_mid is not None:
   d=abs(mid-self.last_mid)
   if math.isfinite(d): self.abs_moves.append(d)
  self.last_mid=mid
  super().on_quote_tick(t)

 def _empirical_cvar_move(self):
  if not self.abs_moves:return 0.0
  xs=sorted(self.abs_moves)
  alpha=PROFILES[self.cvar_profile]['alpha']
  k=min(len(xs)-1,max(0,int(math.floor(alpha*len(xs)))))
  tail=xs[k:]
  return sum(tail)/len(tail) if tail else xs[-1]

 def _stress_move(self):
  p=PROFILES[self.cvar_profile]
  cvar=self._empirical_cvar_move()*math.sqrt(max(1.0,p['horizon']))
  vols=[a['vol'] for a in self.active if a.get('vol',0)>0]
  vf=(median(vols)*p['vol_floor']) if vols else 0.0
  return max(cvar,vf)

 def _exposures(self,bid,ask):
  # score: lower is weaker. Losing positions are ranked before profitable ones.
  out=[]
  for ai,a in enumerate(self.active):
   side=a['side'];mark=bid if side>0 else ask;move=(mark-a['entry'])*side;vol=max(a['vol'],1e-12)
   if a.get('parent_w',0)>1e-12:
    coef=a['mult']*self.parent_scale_v8*a['parent_w']
    pnl=move*coef
    score=(0 if pnl<0 else 1, pnl/max(vol*coef,1e-12), 100.0)
    out.append({'kind':'p','ai':ai,'b':-1,'ci':-1,'coef':coef,'w':a['parent_w'],'pnl':pnl,'score':score})
   for b in range(3):
    for ci,ch in enumerate(a['g75'][b]):
     if ch.get('open_w',0)<=1e-12:continue
     cmove=(mark-ch['entry'])*side;coef=ch['open_w'];pnl=cmove*coef
     layer=float(ch.get('idx',ch.get('layer',0)) or 0)
     strong=1.0 if ch.get('strong',False) else 0.0
     # Strong/high-layer G75 receives an EDGE credit, so weak/loss-making children go first.
     edge_credit=0.10*layer+0.75*strong
     score=(0 if pnl<0 else 1, pnl/max(vol*coef,1e-12)+edge_credit, layer)
     out.append({'kind':'c','ai':ai,'b':b,'ci':ci,'coef':coef,'w':ch['open_w'],'pnl':pnl,'score':score})
  out.sort(key=lambda x:x['score'])
  return out

 def _projected_tail_loss(self,bid,ask,stress=None):
  stress=self._stress_move() if stress is None else stress
  return stress*sum(x['coef'] for x in self._exposures(bid,ask))

 def _cut_exposure(self,x,bid,ask,frac):
  if frac<=0:return 0.0
  a=self.active[x['ai']];side=a['side'];mark=bid if side>0 else ask
  frac=min(1.0,max(0.0,frac))
  if x['kind']=='p':
   old=a['parent_w'];cut=old*frac
   if cut<=1e-12:return 0.0
   move=(mark-a['entry'])*side;pnl=move*a['mult']*self.parent_scale_v8*cut
   a['parent_realized']+=pnl;a['parent_w']-=cut
   self.cvar_reduced_parent_weight+=cut;self.cvar_realized_reduce_pnl+=pnl
   return x['coef']*frac
  ch=a['g75'][x['b']][x['ci']];old=ch['open_w'];cut=old*frac
  if cut<=1e-12:return 0.0
  pnl=(mark-ch['entry'])*side*cut
  a['child_realized'][x['b']]+=pnl;ch['open_w']-=cut
  self.cvar_reduced_child_weight+=cut;self.cvar_realized_reduce_pnl+=pnl
  return x['coef']*frac

 def _risk_control(self,bid,ask,dd):
  p=PROFILES[self.cvar_profile]
  if self.cvar_profile=='off' or len(self.abs_moves)<self.cvar_samples_min or not self.active:return
  eq=self._floating_equity(bid,ask)
  peak=max(self.eq_peak,eq)
  floor=peak*(1.0-p['boundary']/100.0)
  risk_budget=max(0.0,eq-floor)
  stress=self._stress_move()
  exposures=self._exposures(bid,ask)
  projected=stress*sum(x['coef'] for x in exposures)*p['reserve']
  self.projected_tail_loss_max=max(self.projected_tail_loss_max,projected)
  projected_eq=eq-projected
  projected_dd=100.0*(peak-projected_eq)/peak if peak else 0.0
  self.projected_dd_max=max(self.projected_dd_max,projected_dd)
  if projected<=risk_budget+1e-12:return
  self.boundary_touches+=1
  # Skorokhod/reflected-control analogue: remove only the exposure required
  # to bring projected tail loss back to the boundary, weakest EDGE first.
  excess=projected-risk_budget
  self.minimum_intervention_excess+=excess
  required_coef=excess/max(stress*p['reserve'],1e-12)
  removed=0.0
  for x in exposures:
   if required_coef<=1e-12:break
   frac=min(1.0,required_coef/max(x['coef'],1e-12))
   dc=self._cut_exposure(x,bid,ask,frac)
   if dc>0:
    removed+=dc;required_coef-=dc
  if removed>0:self.cvar_control_events+=1

 def summary_cvar(self):
  s=self.summary_lossreduce()
  s.update({
   'cvar_profile':self.cvar_profile,
   'cvar_params':PROFILES[self.cvar_profile],
   'cvar_samples':len(self.abs_moves),
   'cvar_move':self._empirical_cvar_move(),
   'stress_move':self._stress_move(),
   'cvar_control_events':self.cvar_control_events,
   'cvar_reduced_parent_weight':self.cvar_reduced_parent_weight,
   'cvar_reduced_child_weight':self.cvar_reduced_child_weight,
   'cvar_realized_reduce_pnl':self.cvar_realized_reduce_pnl,
   'projected_tail_loss_max':self.projected_tail_loss_max,
   'projected_dd_max':self.projected_dd_max,
   'boundary_touches':self.boundary_touches,
   'minimum_intervention_excess_sum':self.minimum_intervention_excess,
   'dd5_pass':self.max_float_dd<=5.0,
   'control_note':'Dynamic empirical-CVaR state-asymmetric sizing plus reflected minimum intervention. Parent signals and G75 trigger/add/reversal logic are unchanged; control acts on exposure only.',
  })
  return s

def run(catalog,profile,experiment_id):
 cat=ParquetDataCatalog(catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value])
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
 eng.add_instrument(inst);eng.add_data(l1(raw));s=inst.id.value
 st=CVarReflect(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{s}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{s}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{s}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{s}-1-HOUR-BID-INTERNAL'),mode='m15'),profile)
 eng.add_strategy(st);eng.run();eng.end()
 res={'verification':'GOLDEBRAVE_V8C_SPLIT_G75_CVAR_REFLECT_RAW','raw_ticks':len(raw),'ohlc_resample_used':False,'parent_N_target':96,'variant':'SPLIT_G75_3X10_FULL','limitation':'Raw Bid/Ask MTM mathematical risk-screening harness. CVaR is empirical short-horizon stress and reductions are virtual sizing semantics, not yet native broker order/fill semantics.',**st.summary_cvar()}
 p=Path('results/goldebrave-v8c-split-g75-cvar-reflect')/experiment_id/f'{profile}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--profile',choices=PROFILES,required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();run(a.catalog,a.profile,a.experiment_id)
if __name__=='__main__':main()
