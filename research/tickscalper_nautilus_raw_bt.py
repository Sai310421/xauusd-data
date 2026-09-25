from __future__ import annotations
import argparse,json,math
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
 base_qty: Decimal=Decimal('1'); first_mult:float=1.4666666667; later_mult:float=1.5; max_layers:int=10
 add_distance:float=3.0; basket_offset:float=.8; emergency_distance:float=9.0
def floor_step(x,step=.01): return math.floor((x+1e-12)/step)*step
def layer_lot(base,n,c):
 if n==0:return base
 if n==1:return floor_step(base*c.first_mult)
 return floor_step(base*(c.later_mult**n))
class TickScalperCandidate(Strategy):
 def __init__(self,c):
  super().__init__(c);self.bid=self.ask=None;self.side=0;self.entries=[];self.trades=[];self.gw=self.gl=self.net=0.;self.eq=self.peak=1000.;self.mdd=self.max_lots=0.;self.max_layer=0
 def on_start(self):self.subscribe_quote_ticks(self.config.instrument_id)
 def _submit(self,side,lot):
  inst=self.cache.instrument(self.config.instrument_id);o=self.order_factory.market(instrument_id=self.config.instrument_id,order_side=OrderSide.BUY if side>0 else OrderSide.SELL,quantity=inst.make_qty(Decimal(str(lot))));self.submit_order(o)
 def _be(self):
  q=sum(l for _,l in self.entries);return sum(p*l for p,l in self.entries)/q
 def _open(self,side):
  px=self.ask if side>0 else self.bid;lot=layer_lot(float(self.config.base_qty),0,self.config);self._submit(side,lot);self.side=side;self.entries=[(px,lot)]
 def _close(self,reason):
  px=self.bid if self.side>0 else self.ask;pnl=sum((px-p)*self.side*l for p,l in self.entries)
  for _,l in self.entries:self._submit(-self.side,l)
  self.trades.append({'pnl':pnl,'depth':len(self.entries),'reason':reason});self.net+=pnl;self.eq+=pnl;self.peak=max(self.peak,self.eq);self.mdd=max(self.mdd,(self.peak-self.eq)/self.peak*100)
  if pnl>0:self.gw+=pnl
  elif pnl<0:self.gl+=abs(pnl)
  self.side=0;self.entries=[]
 def on_quote_tick(self,t):
  self.bid=float(t.bid_price.as_double());self.ask=float(t.ask_price.as_double())
  if not self.entries:self._open(1 if len(self.trades)%2==0 else -1);return
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
  n=len(self.trades);wins=sum(x['pnl']>0 for x in self.trades);return {'N':n,'WR_pct':100*wins/max(n,1),'PF':self.gw/self.gl if self.gl else None,'EV':self.net/max(n,1),'Net':self.net,'Return_pct':100*self.net/1000,'MaxDD_pct':self.mdd,'max_layer':self.max_layer,'max_concurrent_lots':self.max_lots,'depth10':sum(x['depth']>=10 for x in self.trades)}
def fix_sizes(ticks):
 one=Quantity.from_int(1);out=[]
 for t in ticks:
  bs=t.bid_size;qs=t.ask_size;b=float(bs.as_double());q=float(qs.as_double())
  out.append(QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=one if b<=0 else bs,ask_size=one if q<=0 else qs,ts_event=t.ts_event,ts_init=t.ts_init) if b<=0 or q<=0 else t)
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);ap.add_argument('--base-lot',type=float,default=.30);ap.add_argument('--raw-bidask-only',action='store_true');a=ap.parse_args()
 if not a.raw_bidask_only:raise SystemExit('raw-bidask-only mandatory')
 cat=ParquetDataCatalog(a.catalog);inst=next((x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'),None)
 if inst is None:raise SystemExit('XAUUSD missing')
 ticks=fix_sizes(cat.query_quote_ticks(identifiers=[inst.id.value]));eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(ticks)
 st=TickScalperCandidate(Cfg(instrument_id=inst.id,base_qty=Decimal(str(a.base_lot))));eng.add_strategy(st);eng.run();fills=eng.trader.generate_order_fills_report()
 o={'verification_level':'NAUTILUS_RAW_BIDASK_CANDIDATE_NOT_REPLICA','raw_ticks':len(ticks),'native_fills':len(fills) if fills is not None else 0,'ohlc_resample_used':False,'entry_rule':'PLACEHOLDER_ALTERNATING_SIDE',**st.summary()}
 out=Path('results/tickscalper-nautilus')/a.experiment_id;out.mkdir(parents=True,exist_ok=True);(out/'kpi.json').write_text(json.dumps(o,indent=2),encoding='utf-8');print(json.dumps(o,indent=2));eng.dispose()
if __name__=='__main__':main()
