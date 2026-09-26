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
def floor_step(x,step=.01): return math.floor((x+1e-12)/step)*step
def layer_lot(base,n,c):
 if n==0:return base
 if n==1:return floor_step(base*c.first_mult)
 return floor_step(base*(c.later_mult**n))
class TickScalperCandidate(Strategy):
 def __init__(self,c):
  super().__init__(c);self.bid=self.ask=None;self.side=0;self.entries=[];self.trades=[];self.gw=self.gl=self.net=0.;self.eq=self.peak=1000.;self.mdd=self.max_lots=0.;self.max_layer=0;self.last_close_ns=0;self.ticket_pnls=[];self.ts_tick_count=0;self.ts_bar_open=None;self.ts_bar_hi=None;self.ts_bar_lo=None;self.ts_closes=deque(maxlen=max(256,c.ts_conf2*8));self.ts_last_bar_close=None;self.ts_prev_signal=0
 def on_start(self):self.subscribe_quote_ticks(self.config.instrument_id)
 def _submit(self,side,lot):
  inst=self.cache.instrument(self.config.instrument_id);o=self.order_factory.market(instrument_id=self.config.instrument_id,order_side=OrderSide.BUY if side>0 else OrderSide.SELL,quantity=inst.make_qty(Decimal(str(lot))*self.config.contract_units_per_lot));self.submit_order(o)
 def _be(self):
  q=sum(l for _,l in self.entries);return sum(p*l for p,l in self.entries)/q
 def _open(self,side):
  px=self.ask if side>0 else self.bid;lot=layer_lot(float(self.config.base_qty),0,self.config);self._submit(side,lot);self.side=side;self.entries=[(px,lot)]
 def _close(self,reason):
  px=self.bid if self.side>0 else self.ask;pnl=sum((px-p)*self.side*l for p,l in self.entries)
  for _,l in self.entries:self._submit(-self.side,l)
  for p,l in self.entries:self.ticket_pnls.append((px-p)*self.side*l)
  self.trades.append({'pnl':pnl,'depth':len(self.entries),'reason':reason});self.net+=pnl;self.eq+=pnl;self.peak=max(self.peak,self.eq);self.mdd=max(self.mdd,(self.peak-self.eq)/self.peak*100)
  if pnl>0:self.gw+=pnl
  elif pnl<0:self.gl+=abs(pnl)
  self.side=0;self.entries=[];self.last_close_ns=getattr(self,'now_ns',self.last_close_ns)
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
  if not self.entries:
   if self.last_close_ns and (self.now_ns-self.last_close_ns)<self.config.cooldown_seconds*1_000_000_000:return
   import datetime
   h=datetime.datetime.fromtimestamp(t.ts_event/1e9,datetime.timezone.utc).hour
   if self.config.session_start_hour<=h<self.config.session_end_hour:
    sig=self._entry_signal()
    if sig:self._open(sig)
   return
  mark=self.bid if self.side>0 else self.ask;s=self.side;be=self._be()
  if (mark-be)*s>=self.config.basket_offset:self._close('BASKET');return
  last=self.entries[-1][0];adverse=(last-mark) if s>0 else (mark-last)
  if adverse>=self.config.add_distance and len(self.entries)<self.config.max_layers:
   lot=layer_lot(float(self.config.base_qty),len(self.entries),self.config);self._submit(s,lot);self.entries.append((self.ask if s>0 else self.bid,lot));self.max_layer=max(self.max_layer,len(self.entries));self.max_lots=max(self.max_lots,sum(l for _,l in self.entries));return
  if len(self.entries)>=self.config.max_layers:
   adverse2=(self.entries[-1][0]-mark) if s>0 else (mark-self.entries[-1][0])
   if adverse2>=self.config.emergency_distance:self._close('EMERGENCY')
 def on_stop(self):
  if self.entries:self._close('EOD')
 def summary(self):
  n=len(self.trades);wins=sum(x['pnl']>0 for x in self.trades);return {'N':n,'WR_pct':100*wins/max(n,1),'PF':self.gw/self.gl if self.gl else None,'EV':self.net/max(n,1),'Net':self.net,'Return_pct':100*self.net/1000,'MaxDD_pct':self.mdd,'max_layer':self.max_layer,'max_concurrent_lots':self.max_lots,'depth10':sum(x['depth']>=10 for x in self.trades),'ticket_N':len(self.ticket_pnls),'ticket_WR_pct':100*sum(p>0 for p in self.ticket_pnls)/max(len(self.ticket_pnls),1),'ticket_PF':sum(p for p in self.ticket_pnls if p>0)/max(sum(-p for p in self.ticket_pnls if p<0),1e-12)}
def fix_sizes(ticks):
 one=Quantity.from_int(1);out=[]
 for t in ticks:
  bs=t.bid_size;qs=t.ask_size;b=float(bs.as_double());q=float(qs.as_double())
  out.append(QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=one if b<=0 else bs,ask_size=one if q<=0 else qs,ts_event=t.ts_event,ts_init=t.ts_init) if b<=0 or q<=0 else t)
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);ap.add_argument('--base-lot',type=float,default=.30);ap.add_argument('--direction-sign',type=int,default=1);ap.add_argument('--session-start',type=int,default=7);ap.add_argument('--session-end',type=int,default=17);ap.add_argument('--basket-offset',type=float,default=1.20);ap.add_argument('--max-layers',type=int,default=10);ap.add_argument('--entry-mode',choices=['proxy','ticksmoother'],default='proxy');ap.add_argument('--ts-ticks-per-bar',type=int,default=3);ap.add_argument('--ts-fast',type=int,default=3);ap.add_argument('--ts-slow',type=int,default=5);ap.add_argument('--ts-conf1',type=int,default=8);ap.add_argument('--ts-conf2',type=int,default=13);ap.add_argument('--ts-cross-only',action='store_true');ap.add_argument('--ts-use-macd',action='store_true');ap.add_argument('--ts-no-conf',action='store_true');ap.add_argument('--raw-bidask-only',action='store_true');a=ap.parse_args()
 if not a.raw_bidask_only:raise SystemExit('raw-bidask-only mandatory')
 cat=ParquetDataCatalog(a.catalog);inst=next((x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'),None)
 if inst is None:raise SystemExit('XAUUSD missing')
 ticks=fix_sizes(cat.query_quote_ticks(identifiers=[inst.id.value]));eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(ticks)
 st=TickScalperCandidate(Cfg(instrument_id=inst.id,base_qty=Decimal(str(a.base_lot)),direction_sign=a.direction_sign,session_start_hour=a.session_start,session_end_hour=a.session_end,basket_offset=a.basket_offset,max_layers=a.max_layers,entry_mode=a.entry_mode,ts_ticks_per_bar=a.ts_ticks_per_bar,ts_fast=a.ts_fast,ts_slow=a.ts_slow,ts_conf1=a.ts_conf1,ts_conf2=a.ts_conf2,ts_cross_only=a.ts_cross_only,ts_use_macd=a.ts_use_macd,ts_include_conf=not a.ts_no_conf));eng.add_strategy(st);eng.run();fills=eng.trader.generate_order_fills_report()
 o={'verification_level':'NAUTILUS_RAW_BIDASK_CANDIDATE_NOT_REPLICA','raw_ticks':len(ticks),'native_fills':len(fills) if fills is not None else 0,'ohlc_resample_used':False,'entry_rule':('TICKSMOOTHER_V21_SOURCE_DERIVED_CANDIDATE' if a.entry_mode=='ticksmoother' else 'EDGE_REVALIDATION_V1'),'entry_mode':a.entry_mode,'ts_ticks_per_bar':a.ts_ticks_per_bar,'ts_fast':a.ts_fast,'ts_slow':a.ts_slow,'ts_conf1':a.ts_conf1,'ts_conf2':a.ts_conf2,'ts_cross_only':a.ts_cross_only,'ts_use_macd':a.ts_use_macd,'ts_include_conf':not a.ts_no_conf,'direction_sign':a.direction_sign,'session_start':a.session_start,'session_end':a.session_end,'basket_offset':a.basket_offset,'max_layers_cfg':a.max_layers,'quantity_mapping':'1.00 lot = 100 XAU units; 0.30 lot = 30 native units','instrument_size_precision':getattr(inst,'size_precision',None),**st.summary()}
 out=Path('results/tickscalper-nautilus')/a.experiment_id;out.mkdir(parents=True,exist_ok=True);(out/'kpi.json').write_text(json.dumps(o,indent=2),encoding='utf-8');print(json.dumps(o,indent=2));eng.dispose()
if __name__=='__main__':main()
