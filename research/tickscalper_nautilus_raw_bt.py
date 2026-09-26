from __future__ import annotations
import argparse,json,math
from collections import deque
from decimal import Decimal
from pathlib import Path
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,OrderSide,BookType
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
 def _q(self,identifiers=None,start=None,end=None): return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
 ParquetDataCatalog.query_quote_ticks=_q
class Cfg(StrategyConfig,frozen=True):
 instrument_id: object
 base_qty: Decimal=Decimal('0.30'); contract_units_per_lot:Decimal=Decimal('100'); first_mult:float=1.4666666667; later_mult:float=1.5; max_layers:int=10
 add_distance:float=3.11; basket_offset:float=1.20; emergency_distance:float=3.80
 session_start_hour:int=7; session_end_hour:int=17; entry_move:float=1.80; cooldown_seconds:int=45; direction_sign:int=1
 entry_mode:str='proxy'; ts_ticks_per_bar:int=3; ts_fast:int=3; ts_slow:int=5; ts_conf1:int=8; ts_conf2:int=13; ts_cross_only:bool=False; ts_use_macd:bool=False; ts_include_conf:bool=True
 risk_mode:str='none'; dd_soft_pct:float=10.0; dd_barrier_k:float=1.5; min_add_scale:float=0.25; cvar_alpha:float=0.95; cvar_window:int=400; cvar_horizon:int=20; cvar_budget_pct:float=2.0; risk_start_layer:int=1; fp_window:int=400; fp_threshold:float=0.60; fp_block_threshold:float=0.78; dd_t1:float=10.0; dd_t2:float=20.0; dd_t3:float=30.0; dd_s1:float=0.85; dd_s2:float=0.65; dd_s3:float=0.40; fdd_t1:float=10.0; fdd_t2:float=20.0; fdd_t3:float=30.0; fdd_s1:float=0.85; fdd_s2:float=0.60; fdd_s3:float=0.35; hazard_distance_enabled:bool=False; hazard_start_layer:int=7; hazard_threshold:float=0.60; hazard_lambda:float=1.0; rescue_mode:str='none'; zr_lot_coeff:float=1.35; zr_max_turns:int=4; zr_zone_width:float=3.11; zr_standard_factor:float=1.00; zr_defense_factor:float=1.10; zr_defense_fdd_trigger:float=30.0; zr_target_pnl:float=0.0; zr_hard_fdd_pct:float=45.0; early_start_layer:int=8; early_fdd_trigger:float=25.0; early_hedge_fraction:float=0.25; loss_rescue_fraction:float=0.15; loss_rescue_target:float=0.0; loss_rescue_hard_fdd:float=45.0
def floor_step(x,step=.01): return math.floor((x+1e-12)/step)*step
def layer_lot(base,n,c):
 if n==0:return base
 if n==1:return floor_step(base*c.first_mult)
 return floor_step(base*(c.later_mult**n))
class TickScalperCandidate(Strategy):
 def __init__(self,c):
  super().__init__(c);self.bid=self.ask=None;self.side=0;self.entries=[];self.trades=[];self.gw=self.gl=self.net=0.;self.eq=self.peak=1000.;self.mdd=self.max_lots=0.;self.max_layer=0;self.last_close_ns=0;self.ticket_pnls=[];self.ts_tick_count=0;self.ts_bar_open=None;self.ts_bar_hi=None;self.ts_bar_lo=None;self.ts_closes=deque(maxlen=max(256,c.ts_conf2*8));self.ts_last_bar_close=None;self.ts_prev_signal=0;self.mtm_peak=1000.;self.mfdd=0.;self.recent_mid_deltas=deque(maxlen=max(50,c.cvar_window,c.fp_window));self.prev_mid=None;self.risk_scaled_adds=0;self.risk_blocked_adds=0;self.fp_last_prob=0.0;self.fp_trigger_count=0;self.hazard_delays=0;self.hazard_last_prob=0.0;self.rescue_active=False;self.rescue_legs=[];self.rescue_turns=0;self.rescue_anchor=0.0;self.rescue_roll=0.0;self.rescue_started_count=0;self.rescue_completed_count=0;self.rescue_hard_close_count=0;self.early_rescue_used=False;self.early_rescue_started=0;self.early_rescue_completed=0;self.early_rescue_hard_closed=0;self.loss_rescue_started=0;self.loss_rescue_completed=0;self.loss_rescue_hard_closed=0
 def on_start(self):self.subscribe_quote_ticks(self.config.instrument_id)
 def _submit(self,side,lot):
  inst=self.cache.instrument(self.config.instrument_id);o=self.order_factory.market(instrument_id=self.config.instrument_id,order_side=OrderSide.BUY if side>0 else OrderSide.SELL,quantity=inst.make_qty(Decimal(str(lot))*self.config.contract_units_per_lot));self.submit_order(o)
 def _be(self):
  q=sum(l for _,l in self.entries);return sum(p*l for p,l in self.entries)/q
 def _floating_pnl(self):
  if not self.entries and not self.rescue_legs:return 0.0
  pnl=0.0
  if self.entries:
   px=self.bid if self.side>0 else self.ask
   pnl+=sum((px-p)*self.side*l for p,l in self.entries)
  for rs,rp,rl in self.rescue_legs:
   rpx=self.bid if rs>0 else self.ask
   pnl+=(rpx-rp)*rs*rl
  return pnl
 def _update_mtm_dd(self):
  mtm=self.eq+self._floating_pnl()
  self.mtm_peak=max(self.mtm_peak,mtm)
  if self.mtm_peak>0:self.mfdd=max(self.mfdd,(self.mtm_peak-mtm)/self.mtm_peak*100.0)
 def _empirical_cvar_pct(self,prospective_lot=0.0):
  vals=[abs(x) for x in self.recent_mid_deltas if x!=0]
  if len(vals)<50:return 0.0
  vals.sort(reverse=True)
  tail_n=max(1,int(math.ceil((1.0-self.config.cvar_alpha)*len(vals))))
  es=sum(vals[:tail_n])/tail_n
  gross=sum(l for _,l in self.entries)+prospective_lot
  est_loss=es*gross*self.config.cvar_horizon
  return 100.0*est_loss/max(self.mtm_peak,1e-9)
 def _first_passage_adverse_prob(self,prospective_lot):
  if len(self.recent_mid_deltas)<80 or not self.entries:return 0.5
  vals=list(self.recent_mid_deltas)[-self.config.fp_window:]
  s=self.side
  x=[d*s for d in vals]
  mu=sum(x)/len(x)
  var=sum((z-mu)*(z-mu) for z in x)/max(len(x)-1,1)
  if var<=1e-12:return 1.0 if mu<0 else 0.0
  mark=self.bid if s>0 else self.ask
  px_add=self.ask if s>0 else self.bid
  q=sum(l for _,l in self.entries)+prospective_lot
  be_new=(sum(p*l for p,l in self.entries)+px_add*prospective_lot)/max(q,1e-12)
  target=be_new + s*self.config.basket_offset
  b=max((target-mark)*s,0.01)
  a=self.config.emergency_distance if len(self.entries)+1>=self.config.max_layers else self.config.add_distance
  if abs(mu)<1e-10:
   p_lower=b/(a+b)
  else:
   k=-2.0*mu/max(var,1e-12)
   try:
    num=1.0-math.exp(k*a)
    den=1.0-math.exp(k*(a+b))
    p_up=num/den if abs(den)>1e-12 else a/(a+b)
    p_lower=1.0-p_up
   except OverflowError:
    p_lower=1.0 if mu<0 else 0.0
  return max(0.0,min(1.0,p_lower))
 def _closed_dd_pct(self):
  return 100.0*(self.peak-self.eq)/max(self.peak,1e-9)
 def _floating_dd_pct(self):
  mtm=self.eq+self._floating_pnl()
  return 100.0*(self.mtm_peak-mtm)/max(self.mtm_peak,1e-9)
 def _piecewise_scale(self,x,t1,t2,t3,s1,s2,s3):
  if x<=t1:return 1.0
  if x<=t2:return s1
  if x<=t3:return s2
  return s3
 def _dd_exposure_scale(self):
  return self._piecewise_scale(self._closed_dd_pct(),self.config.dd_t1,self.config.dd_t2,self.config.dd_t3,self.config.dd_s1,self.config.dd_s2,self.config.dd_s3)
 def _floating_exposure_scale(self):
  return self._piecewise_scale(self._floating_dd_pct(),self.config.fdd_t1,self.config.fdd_t2,self.config.fdd_t3,self.config.fdd_s1,self.config.fdd_s2,self.config.fdd_s3)
 def _risk_add_scale(self,prospective_lot):
  if self.config.risk_mode=='none' or len(self.entries)<self.config.risk_start_layer:return 1.0
  if self.config.risk_mode=='dd_adaptive':return self._dd_exposure_scale()
  if self.config.risk_mode=='floating_governor':return self._floating_exposure_scale()
  if self.config.risk_mode=='dd_floating_combo':return min(self._dd_exposure_scale(),self._floating_exposure_scale())
  mtm=self.eq+self._floating_pnl()
  dd=100.0*(self.mtm_peak-mtm)/max(self.mtm_peak,1e-9)
  scale=1.0
  if self.config.risk_mode in ('barrier','hybrid'):
   x=max(0.0,dd-self.config.dd_soft_pct)/max(self.config.dd_soft_pct,1e-9)
   if x>0: scale=min(scale,max(self.config.min_add_scale,math.exp(-self.config.dd_barrier_k*x*x)))
  if self.config.risk_mode in ('cvar','hybrid'):
   cv=self._empirical_cvar_pct(prospective_lot)
   if cv>self.config.cvar_budget_pct:
    scale=min(scale,max(self.config.min_add_scale,self.config.cvar_budget_pct/max(cv,1e-9)))
  if self.config.risk_mode in ('firstpass','fp_barrier'):
   p=self._first_passage_adverse_prob(prospective_lot);self.fp_last_prob=p
   if p>=self.config.fp_block_threshold:
    self.risk_blocked_adds+=1;self.fp_trigger_count+=1;return 0.0
   if p>=self.config.fp_threshold:
    self.fp_trigger_count+=1
    frac=(self.config.fp_block_threshold-p)/max(self.config.fp_block_threshold-self.config.fp_threshold,1e-9)
    scale=min(scale,max(self.config.min_add_scale,frac))
   if self.config.risk_mode=='fp_barrier':
    x=max(0.0,dd-self.config.dd_soft_pct)/max(self.config.dd_soft_pct,1e-9)
    if x>0:scale=min(scale,max(self.config.min_add_scale,math.exp(-self.config.dd_barrier_k*x*x)))
  return scale
 def _start_early_partial_rescue(self):
  if self.config.rescue_mode!='early_partial' or self.rescue_active or self.early_rescue_used or not self.entries:return False
  if len(self.entries)<self.config.early_start_layer:return False
  if self._floating_dd_pct()<self.config.early_fdd_trigger:return False
  gross=sum(l for _,l in self.entries)
  lot=max(0.01,floor_step(gross*self.config.early_hedge_fraction))
  rs=-self.side
  px=self.ask if rs>0 else self.bid
  self._submit(rs,lot)
  self.rescue_legs=[(rs,px,lot)]
  self.rescue_active=True;self.rescue_turns=1;self.early_rescue_used=True;self.early_rescue_started+=1;self.rescue_started_count+=1
  self.max_lots=max(self.max_lots,gross+lot)
  return True
 def _handle_early_partial_rescue(self):
  if not self.rescue_active or self.config.rescue_mode!='early_partial':return False
  if self._zr_combined_pnl()>=self.config.zr_target_pnl:
   self.early_rescue_completed+=1;self.rescue_completed_count+=1;self._close('EARLY_RESCUE_COMPLETE');return True
  if self._floating_dd_pct()>=self.config.zr_hard_fdd_pct:
   self.early_rescue_hard_closed+=1;self.rescue_hard_close_count+=1;self._close('EARLY_RESCUE_HARD_DD');return True
  return True
 def _start_loss_side_rescue(self):
  if self.config.rescue_mode!='loss_side' or self.rescue_active or not self.entries:return False
  gross=sum(l for _,l in self.entries)
  lot=max(0.01,floor_step(gross*self.config.loss_rescue_fraction))
  rs=self.side
  px=self.ask if rs>0 else self.bid
  self._submit(rs,lot)
  self.rescue_legs=[(rs,px,lot)]
  self.rescue_active=True;self.rescue_turns=1;self.loss_rescue_started+=1;self.rescue_started_count+=1
  self.max_lots=max(self.max_lots,gross+lot)
  return True
 def _handle_loss_side_rescue(self):
  if not self.rescue_active or self.config.rescue_mode!='loss_side':return False
  if self._zr_combined_pnl()>=self.config.loss_rescue_target:
   self.loss_rescue_completed+=1;self.rescue_completed_count+=1;self._close('LOSS_SIDE_RESCUE_COMPLETE');return True
  if self._floating_dd_pct()>=self.config.loss_rescue_hard_fdd:
   self.loss_rescue_hard_closed+=1;self.rescue_hard_close_count+=1;self._close('LOSS_SIDE_RESCUE_HARD_DD');return True
  return True
 def _start_zr_rescue(self):
  if self.config.rescue_mode!='zr_adaptive' or self.rescue_active or not self.entries:return False
  rs=-self.side
  prev=self.entries[-1][1]
  lot=max(0.01,floor_step(prev*self.config.zr_lot_coeff))
  px=self.ask if rs>0 else self.bid
  self._submit(rs,lot)
  self.rescue_legs=[(rs,px,lot)]
  mid=(self.bid+self.ask)/2
  self.rescue_active=True;self.rescue_turns=1;self.rescue_anchor=mid;self.rescue_roll=mid;self.rescue_started_count+=1
  self.max_lots=max(self.max_lots,sum(l for _,l in self.entries)+sum(l for _,_,l in self.rescue_legs))
  return True
 def _zr_combined_pnl(self):
  return self._floating_pnl()
 def _handle_zr_rescue(self):
  if not self.rescue_active:return False
  if self._zr_combined_pnl()>=self.config.zr_target_pnl:
   self.rescue_completed_count+=1;self._close('ZR_COMPLETE');return True
  if self._floating_dd_pct()>=self.config.zr_hard_fdd_pct:
   self.rescue_hard_close_count+=1;self._close('ZR_HARD_DD');return True
  if self.rescue_turns>=self.config.zr_max_turns:return True
  mid=(self.bid+self.ask)/2
  defense=(self._floating_dd_pct()>=self.config.zr_defense_fdd_trigger or self.rescue_turns>=2)
  factor=self.config.zr_defense_factor if defense else self.config.zr_standard_factor
  width=self.config.zr_zone_width*(factor**max(0,self.rescue_turns-1))
  anchor=self.rescue_roll if defense else self.rescue_anchor
  rs=0
  if mid>=anchor+width:rs=1
  elif mid<=anchor-width:rs=-1
  if not rs:return True
  prev=self.rescue_legs[-1][2]
  lot=max(0.01,floor_step(prev*self.config.zr_lot_coeff))
  px=self.ask if rs>0 else self.bid
  self._submit(rs,lot);self.rescue_legs.append((rs,px,lot));self.rescue_turns+=1
  if defense:self.rescue_roll=mid
  self.max_lots=max(self.max_lots,sum(l for _,l in self.entries)+sum(l for _,_,l in self.rescue_legs))
  return True
 def _effective_add_distance(self,prospective_lot):
  dist=self.config.add_distance
  if not self.config.hazard_distance_enabled or len(self.entries)<self.config.hazard_start_layer:return dist
  p=self._first_passage_adverse_prob(prospective_lot);self.hazard_last_prob=p
  if p<=self.config.hazard_threshold:return dist
  z=(p-self.config.hazard_threshold)/max(1.0-self.config.hazard_threshold,1e-9)
  return dist*(1.0+self.config.hazard_lambda*z)
 def _open(self,side):
  px=self.ask if side>0 else self.bid
  lot=layer_lot(float(self.config.base_qty),0,self.config)
  if self.config.risk_mode in ('dd_adaptive','dd_floating_combo'):
   lot=floor_step(lot*self._dd_exposure_scale())
  if lot<0.01:return
  self._submit(side,lot);self.side=side;self.entries=[(px,lot)];self.early_rescue_used=False
 def _close(self,reason):
  pnl=0.0
  px=self.bid if self.side>0 else self.ask
  for p,l in self.entries:
   leg=(px-p)*self.side*l;pnl+=leg;self.ticket_pnls.append(leg);self._submit(-self.side,l)
  for rs,rp,rl in self.rescue_legs:
   rpx=self.bid if rs>0 else self.ask
   leg=(rpx-rp)*rs*rl;pnl+=leg;self.ticket_pnls.append(leg);self._submit(-rs,rl)
  self.trades.append({'pnl':pnl,'depth':len(self.entries),'reason':reason,'rescue_turns':self.rescue_turns});self.net+=pnl;self.eq+=pnl;self.peak=max(self.peak,self.eq);self.mdd=max(self.mdd,(self.peak-self.eq)/self.peak*100)
  if pnl>0:self.gw+=pnl
  elif pnl<0:self.gl+=abs(pnl)
  self.side=0;self.entries=[];self.rescue_legs=[];self.rescue_active=False;self.rescue_turns=0;self.last_close_ns=getattr(self,'now_ns',self.last_close_ns)
 def _ts_update(self,mid):
  # SOURCE-DERIVED TickSmoother v2.1 candidate:
  # Igor described grouping by a number of ticks/seconds; public code fragments expose
  # tOpen/tClose/Hi/Lo/Fast/Slow arrays. This implementation uses N-tick synthetic bars
  # and close-based fast/slow/confirm MAs. It is not claimed byte-for-byte parity.
  if self.ts_bar_open is None:
   self.ts_bar_open=self.ts_bar_hi=self.ts_bar_lo=mid
  self.ts_bar_hi=max(self.ts_bar_hi,mid); self.ts_bar_lo=min(self.ts_bar_lo,mid); self.ts_tick_count+=1
  if self.ts_tick_count < self.config.ts_ticks_per_bar: return False
  self.ts_last_bar_close=mid
  self.ts_closes.append(mid)
  self.ts_tick_count=0; self.ts_bar_open=None; self.ts_bar_hi=None; self.ts_bar_lo=None
  return True
 def _ma(self,n):
  if len(self.ts_closes)<n:return None
  a=list(self.ts_closes)[-n:];return sum(a)/n
 def _ema_series(self,vals,n):
  if len(vals)<n:return []
  alpha=2.0/(n+1.0);e=sum(vals[:n])/n;out=[e]
  for x in vals[n:]:
   e=alpha*x+(1-alpha)*e;out.append(e)
  return out
 def _macd_ok(self,side):
  vals=list(self.ts_closes)
  if len(vals)<40:return False
  ef=self._ema_series(vals,12);es=self._ema_series(vals,26)
  offset=len(ef)-len(es);macd=[ef[i+offset]-es[i] for i in range(len(es))]
  if len(macd)<9:return False
  sig=self._ema_series(macd,9)
  if not sig:return False
  m=macd[-1];s=sig[-1]
  return (m>s) if side>0 else (m<s)
 def _entry_signal(self):
  mid=(self.bid+self.ask)/2
  if self.config.entry_mode=='ticksmoother':
   if not self._ts_update(mid): return 0
   tclose=self.ts_last_bar_close; fast=self._ma(self.config.ts_fast); slow=self._ma(self.config.ts_slow); c1=self._ma(self.config.ts_conf1); c2=self._ma(self.config.ts_conf2)
   if None in (fast,slow,c1,c2): return 0
   # Public v3.44 relation: short tClose < fastMA < slowMA < confMA1 < confMA2;
   # buy is the mirrored ordering. Optional MACD confirmation mirrors documented MC modes.
   raw=0
   if self.config.ts_include_conf:
    if tclose>fast>slow>c1>c2: raw=1
    elif tclose<fast<slow<c1<c2: raw=-1
   else:
    if tclose>fast>slow: raw=1
    elif tclose<fast<slow: raw=-1
   if raw and self.config.ts_use_macd and not self._macd_ok(raw): raw=0
   if self.config.ts_cross_only:
    fire = raw if raw!=0 and raw!=self.ts_prev_signal else 0
    self.ts_prev_signal=raw
    return fire*self.config.direction_sign
   self.ts_prev_signal=raw
   return raw*self.config.direction_sign
  # legacy statement-constrained displacement proxy
  if not hasattr(self,'anchor'): self.anchor=mid; return 0
  move=mid-self.anchor
  if abs(move)>=self.config.entry_move:
   self.anchor=mid; return (1 if move>0 else -1)*self.config.direction_sign
  return 0
 def on_quote_tick(self,t):
  self.bid=float(t.bid_price.as_double());self.ask=float(t.ask_price.as_double());self.now_ns=int(t.ts_event)
  mid=(self.bid+self.ask)/2
  if self.prev_mid is not None:self.recent_mid_deltas.append(mid-self.prev_mid)
  self.prev_mid=mid
  self._update_mtm_dd()
  if not self.entries:
   if self.last_close_ns and (self.now_ns-self.last_close_ns)<self.config.cooldown_seconds*1_000_000_000:return
   import datetime
   h=datetime.datetime.fromtimestamp(t.ts_event/1e9,datetime.timezone.utc).hour
   if self.config.session_start_hour<=h<self.config.session_end_hour:
    sig=self._entry_signal()
    if sig:self._open(sig)
   return
  if self.rescue_active:
   if self.config.rescue_mode=='early_partial':self._handle_early_partial_rescue()
   elif self.config.rescue_mode=='loss_side':self._handle_loss_side_rescue()
   else:self._handle_zr_rescue()
   return
  mark=self.bid if self.side>0 else self.ask;s=self.side;be=self._be()
  if (mark-be)*s>=self.config.basket_offset:self._close('BASKET');return
  if self._start_early_partial_rescue():return
  last=self.entries[-1][0];adverse=(last-mark) if s>0 else (mark-last)
  if len(self.entries)<self.config.max_layers:
   raw_lot=layer_lot(float(self.config.base_qty),len(self.entries),self.config)
   eff_dist=self._effective_add_distance(raw_lot)
   if adverse>=self.config.add_distance and adverse<eff_dist:self.hazard_delays+=1
   if adverse>=eff_dist:
    scale=self._risk_add_scale(raw_lot)
    lot=floor_step(raw_lot*scale)
    if lot<0.01:
     self.risk_blocked_adds+=1
     return
    if scale<0.999:self.risk_scaled_adds+=1
    self._submit(s,lot);self.entries.append((self.ask if s>0 else self.bid,lot));self.max_layer=max(self.max_layer,len(self.entries));self.max_lots=max(self.max_lots,sum(l for _,l in self.entries));return
  if len(self.entries)>=self.config.max_layers:
   adverse2=(self.entries[-1][0]-mark) if s>0 else (mark-self.entries[-1][0])
   if adverse2>=self.config.emergency_distance:
    if self.config.rescue_mode=='loss_side':
     if not self._start_loss_side_rescue():self._close('EMERGENCY')
    elif not self._start_zr_rescue():self._close('EMERGENCY')
 def on_stop(self):
  if self.entries:self._close('EOD')
 def summary(self):
  n=len(self.trades);wins=sum(x['pnl']>0 for x in self.trades);return {'N':n,'WR_pct':100*wins/max(n,1),'PF':self.gw/self.gl if self.gl else None,'EV':self.net/max(n,1),'Net':self.net,'Return_pct':100*self.net/1000,'MaxDD_pct':self.mdd,'MaxFloatingDD_pct':self.mfdd,'max_layer':self.max_layer,'max_concurrent_lots':self.max_lots,'depth10':sum(x['depth']>=10 for x in self.trades),'risk_scaled_adds':self.risk_scaled_adds,'risk_blocked_adds':self.risk_blocked_adds,'fp_trigger_count':self.fp_trigger_count,'fp_last_prob':self.fp_last_prob,'hazard_delays':self.hazard_delays,'hazard_last_prob':self.hazard_last_prob,'rescue_started':self.rescue_started_count,'rescue_completed':self.rescue_completed_count,'rescue_hard_closed':self.rescue_hard_close_count,'early_rescue_started':self.early_rescue_started,'early_rescue_completed':self.early_rescue_completed,'early_rescue_hard_closed':self.early_rescue_hard_closed,'loss_rescue_started':self.loss_rescue_started,'loss_rescue_completed':self.loss_rescue_completed,'loss_rescue_hard_closed':self.loss_rescue_hard_closed,'ticket_N':len(self.ticket_pnls),'ticket_WR_pct':100*sum(p>0 for p in self.ticket_pnls)/max(len(self.ticket_pnls),1),'ticket_PF':sum(p for p in self.ticket_pnls if p>0)/max(sum(-p for p in self.ticket_pnls if p<0),1e-12)}
def fix_sizes(ticks):
 one=Quantity.from_int(1);out=[]
 for t in ticks:
  bs=t.bid_size;qs=t.ask_size;b=float(bs.as_double());q=float(qs.as_double())
  out.append(QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=one if b<=0 else bs,ask_size=one if q<=0 else qs,ts_event=t.ts_event,ts_init=t.ts_init) if b<=0 or q<=0 else t)
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);ap.add_argument('--base-lot',type=float,default=.30);ap.add_argument('--direction-sign',type=int,default=1);ap.add_argument('--session-start',type=int,default=7);ap.add_argument('--session-end',type=int,default=17);ap.add_argument('--basket-offset',type=float,default=1.20);ap.add_argument('--max-layers',type=int,default=10);ap.add_argument('--entry-mode',choices=['proxy','ticksmoother'],default='proxy');ap.add_argument('--ts-ticks-per-bar',type=int,default=3);ap.add_argument('--ts-fast',type=int,default=3);ap.add_argument('--ts-slow',type=int,default=5);ap.add_argument('--ts-conf1',type=int,default=8);ap.add_argument('--ts-conf2',type=int,default=13);ap.add_argument('--ts-cross-only',action='store_true');ap.add_argument('--ts-use-macd',action='store_true');ap.add_argument('--ts-no-conf',action='store_true');ap.add_argument('--risk-mode',choices=['none','barrier','cvar','hybrid','firstpass','fp_barrier','dd_adaptive','floating_governor','dd_floating_combo'],default='none');ap.add_argument('--dd-soft-pct',type=float,default=10.0);ap.add_argument('--dd-barrier-k',type=float,default=1.5);ap.add_argument('--min-add-scale',type=float,default=0.25);ap.add_argument('--cvar-budget-pct',type=float,default=2.0);ap.add_argument('--risk-start-layer',type=int,default=1);ap.add_argument('--dd-t1',type=float,default=10.0);ap.add_argument('--dd-t2',type=float,default=20.0);ap.add_argument('--dd-t3',type=float,default=30.0);ap.add_argument('--dd-s1',type=float,default=0.85);ap.add_argument('--dd-s2',type=float,default=0.65);ap.add_argument('--dd-s3',type=float,default=0.40);ap.add_argument('--fdd-t1',type=float,default=10.0);ap.add_argument('--fdd-t2',type=float,default=20.0);ap.add_argument('--fdd-t3',type=float,default=30.0);ap.add_argument('--fdd-s1',type=float,default=0.85);ap.add_argument('--fdd-s2',type=float,default=0.60);ap.add_argument('--fdd-s3',type=float,default=0.35);ap.add_argument('--fp-threshold',type=float,default=0.60);ap.add_argument('--fp-block-threshold',type=float,default=0.78);ap.add_argument('--hazard-distance',action='store_true');ap.add_argument('--hazard-start-layer',type=int,default=7);ap.add_argument('--hazard-threshold',type=float,default=0.60);ap.add_argument('--hazard-lambda',type=float,default=1.0);ap.add_argument('--rescue-mode',choices=['none','zr_adaptive','early_partial','loss_side'],default='none');ap.add_argument('--zr-lot-coeff',type=float,default=1.35);ap.add_argument('--zr-max-turns',type=int,default=4);ap.add_argument('--zr-zone-width',type=float,default=3.11);ap.add_argument('--zr-defense-fdd-trigger',type=float,default=30.0);ap.add_argument('--zr-target-pnl',type=float,default=0.0);ap.add_argument('--zr-hard-fdd-pct',type=float,default=45.0);ap.add_argument('--early-start-layer',type=int,default=8);ap.add_argument('--early-fdd-trigger',type=float,default=25.0);ap.add_argument('--early-hedge-fraction',type=float,default=0.25);ap.add_argument('--loss-rescue-fraction',type=float,default=0.15);ap.add_argument('--loss-rescue-target',type=float,default=0.0);ap.add_argument('--loss-rescue-hard-fdd',type=float,default=45.0);ap.add_argument('--raw-bidask-only',action='store_true');a=ap.parse_args()
 if not a.raw_bidask_only:raise SystemExit('raw-bidask-only mandatory')
 cat=ParquetDataCatalog(a.catalog);inst=next((x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'),None)
 if inst is None:raise SystemExit('XAUUSD missing')
 ticks=fix_sizes(cat.query_quote_ticks(identifiers=[inst.id.value]));eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(ticks)
 st=TickScalperCandidate(Cfg(instrument_id=inst.id,base_qty=Decimal(str(a.base_lot)),direction_sign=a.direction_sign,session_start_hour=a.session_start,session_end_hour=a.session_end,basket_offset=a.basket_offset,max_layers=a.max_layers,entry_mode=a.entry_mode,ts_ticks_per_bar=a.ts_ticks_per_bar,ts_fast=a.ts_fast,ts_slow=a.ts_slow,ts_conf1=a.ts_conf1,ts_conf2=a.ts_conf2,ts_cross_only=a.ts_cross_only,ts_use_macd=a.ts_use_macd,ts_include_conf=not a.ts_no_conf,risk_mode=a.risk_mode,dd_soft_pct=a.dd_soft_pct,dd_barrier_k=a.dd_barrier_k,min_add_scale=a.min_add_scale,cvar_budget_pct=a.cvar_budget_pct,risk_start_layer=a.risk_start_layer,dd_t1=a.dd_t1,dd_t2=a.dd_t2,dd_t3=a.dd_t3,dd_s1=a.dd_s1,dd_s2=a.dd_s2,dd_s3=a.dd_s3,fdd_t1=a.fdd_t1,fdd_t2=a.fdd_t2,fdd_t3=a.fdd_t3,fdd_s1=a.fdd_s1,fdd_s2=a.fdd_s2,fdd_s3=a.fdd_s3,fp_threshold=a.fp_threshold,fp_block_threshold=a.fp_block_threshold,hazard_distance_enabled=a.hazard_distance,hazard_start_layer=a.hazard_start_layer,hazard_threshold=a.hazard_threshold,hazard_lambda=a.hazard_lambda,rescue_mode=a.rescue_mode,zr_lot_coeff=a.zr_lot_coeff,zr_max_turns=a.zr_max_turns,zr_zone_width=a.zr_zone_width,zr_defense_fdd_trigger=a.zr_defense_fdd_trigger,zr_target_pnl=a.zr_target_pnl,zr_hard_fdd_pct=a.zr_hard_fdd_pct,early_start_layer=a.early_start_layer,early_fdd_trigger=a.early_fdd_trigger,early_hedge_fraction=a.early_hedge_fraction,loss_rescue_fraction=a.loss_rescue_fraction,loss_rescue_target=a.loss_rescue_target,loss_rescue_hard_fdd=a.loss_rescue_hard_fdd));eng.add_strategy(st);eng.run();fills=eng.trader.generate_order_fills_report()
 o={'verification_level':'NAUTILUS_RAW_BIDASK_CANDIDATE_NOT_REPLICA','raw_ticks':len(ticks),'native_fills':len(fills) if fills is not None else 0,'ohlc_resample_used':False,'entry_rule':('TICKSMOOTHER_V21_SOURCE_DERIVED_CANDIDATE' if a.entry_mode=='ticksmoother' else 'EDGE_REVALIDATION_V1'),'entry_mode':a.entry_mode,'ts_ticks_per_bar':a.ts_ticks_per_bar,'ts_fast':a.ts_fast,'ts_slow':a.ts_slow,'ts_conf1':a.ts_conf1,'ts_conf2':a.ts_conf2,'ts_cross_only':a.ts_cross_only,'ts_use_macd':a.ts_use_macd,'ts_include_conf':not a.ts_no_conf,'risk_mode':a.risk_mode,'dd_soft_pct':a.dd_soft_pct,'dd_barrier_k':a.dd_barrier_k,'min_add_scale':a.min_add_scale,'cvar_budget_pct':a.cvar_budget_pct,'risk_start_layer':a.risk_start_layer,'dd_t1':a.dd_t1,'dd_t2':a.dd_t2,'dd_t3':a.dd_t3,'dd_s1':a.dd_s1,'dd_s2':a.dd_s2,'dd_s3':a.dd_s3,'fdd_t1':a.fdd_t1,'fdd_t2':a.fdd_t2,'fdd_t3':a.fdd_t3,'fdd_s1':a.fdd_s1,'fdd_s2':a.fdd_s2,'fdd_s3':a.fdd_s3,'fp_threshold':a.fp_threshold,'fp_block_threshold':a.fp_block_threshold,'hazard_distance':a.hazard_distance,'hazard_start_layer':a.hazard_start_layer,'hazard_threshold':a.hazard_threshold,'hazard_lambda':a.hazard_lambda,'rescue_mode':a.rescue_mode,'zr_lot_coeff':a.zr_lot_coeff,'zr_max_turns':a.zr_max_turns,'zr_zone_width':a.zr_zone_width,'zr_defense_fdd_trigger':a.zr_defense_fdd_trigger,'zr_target_pnl':a.zr_target_pnl,'zr_hard_fdd_pct':a.zr_hard_fdd_pct,'early_start_layer':a.early_start_layer,'early_fdd_trigger':a.early_fdd_trigger,'early_hedge_fraction':a.early_hedge_fraction,'loss_rescue_fraction':a.loss_rescue_fraction,'loss_rescue_target':a.loss_rescue_target,'loss_rescue_hard_fdd':a.loss_rescue_hard_fdd,'direction_sign':a.direction_sign,'session_start':a.session_start,'session_end':a.session_end,'basket_offset':a.basket_offset,'max_layers_cfg':a.max_layers,'quantity_mapping':'1.00 lot = 100 XAU units; 0.30 lot = 30 native units','instrument_size_precision':getattr(inst,'size_precision',None),**st.summary()}
 out=Path('results/tickscalper-nautilus')/a.experiment_id;out.mkdir(parents=True,exist_ok=True);(out/'kpi.json').write_text(json.dumps(o,indent=2),encoding='utf-8');print(json.dumps(o,indent=2));eng.dispose()
if __name__=='__main__':main()
