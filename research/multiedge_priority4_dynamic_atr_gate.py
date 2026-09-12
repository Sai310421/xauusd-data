from __future__ import annotations
import argparse,json,math
from collections import deque,Counter
from decimal import Decimal
from pathlib import Path
import numpy as np, nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar,QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,OrderSide,BookType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
 def _q(self,identifiers=None,start=None,end=None): return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
 ParquetDataCatalog.query_quote_ticks=_q

EDGES=('V1_SWEEP_MSS','V2_SWEEP_REJECT_BOS','T4_ACCEL_STRUCT','R1_RANGE_FOLLOW')
class Cfg(StrategyConfig,frozen=True):
 instrument_id:InstrumentId; m1:BarType; selected:str; unit_qty:Decimal=Decimal('1'); horizon_minutes:int=180

class S(Strategy):
 def __init__(self,cfg):
  super().__init__(cfg); self.o=deque(maxlen=300);self.h=deque(maxlen=300);self.l=deque(maxlen=300);self.c=deque(maxlen=300);self.tr=deque(maxlen=300)
  self.prev=None;self.m1_i=0;self.pending=0;self.active=None;self.last_bid=None;self.last_ask=None;self.trades=[];self.gw=self.gl=0.;self.wins=self.losses=self.fire=0
  self.sweep_state=None
 @staticmethod
 def f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)
 def on_start(self): self.subscribe_quote_ticks(self.config.instrument_id);self.subscribe_bars(self.config.m1)
 def atr(self): return float(np.mean(list(self.tr)[-20:])) if len(self.tr)>=20 else 0.
 def ema(self,n,offset=0):
  a=list(self.c); end=len(a)-offset
  if end<n:return None
  a=np.asarray(a[end-n:end],float);alpha=2/(n+1);v=float(a[0])
  for x in a[1:]:v=alpha*x+(1-alpha)*v
  return v
 def submit(self,side):
  inst=self.cache.instrument(self.config.instrument_id);q=inst.make_qty(self.config.unit_qty);self.submit_order(self.order_factory.market(instrument_id=self.config.instrument_id,order_side=OrderSide.BUY if side>0 else OrderSide.SELL,quantity=q))
 def signal(self):
  if len(self.c)<70 or self.atr()<=0:return 0
  o=np.asarray(self.o,float);h=np.asarray(self.h,float);l=np.asarray(self.l,float);c=np.asarray(self.c,float);atr=self.atr(); e21=self.ema(21); ep=self.ema(21,1); e50=self.ema(50)
  sel=self.config.selected
  # shared structural references exclude current bar
  sh=float(h[-11:-1].max()); sl=float(l[-11:-1].min()); micro_hi=float(h[-5:-1].max()); micro_lo=float(l[-5:-1].min())
  if sel in ('V1_SWEEP_MSS','V2_SWEEP_REJECT_BOS'):
   # arm after a confirmed liquidity sweep close back inside prior 10-bar range
   if self.sweep_state is None:
    if l[-1]<sl and c[-1]>sl: self.sweep_state={'side':1,'i':self.m1_i,'sweep_low':l[-1],'ref_hi':micro_hi,'reject':(c[-1]-l[-1])/max(h[-1]-l[-1],1e-9)}
    elif h[-1]>sh and c[-1]<sh: self.sweep_state={'side':-1,'i':self.m1_i,'sweep_high':h[-1],'ref_lo':micro_lo,'reject':(h[-1]-c[-1])/max(h[-1]-l[-1],1e-9)}
    return 0
   st=self.sweep_state
   if self.m1_i-st['i']>6:self.sweep_state=None;return 0
   side=st['side']
   mss=(c[-1]>st['ref_hi']) if side>0 else (c[-1]<st['ref_lo'])
   if not mss:return 0
   if sel=='V2_SWEEP_REJECT_BOS':
    # stricter: strong rejection on sweep candle + displacement/BOS close now
    body=abs(c[-1]-o[-1]); rng=max(h[-1]-l[-1],1e-9); strong=st['reject']>=0.60 and body>=0.45*atr and body/rng>=0.45
    if not strong:return 0
   self.sweep_state=None;self.fire+=1;return side
  if sel=='T4_ACCEL_STRUCT':
   # trend-aligned acceleration, not naked momentum
   if e21 is None or ep is None or e50 is None:return 0
   mom3=c[-1]-c[-4]; prev3=c[-4]-c[-7]; body=abs(c[-1]-o[-1]);rng=max(h[-1]-l[-1],1e-9)
   buy=e21>e50 and e21>ep and c[-1]>e21 and mom3>max(abs(prev3)*1.25,0.45*atr) and body>=0.40*atr and (h[-1]-c[-1])/rng<=0.25
   sell=e21<e50 and e21<ep and c[-1]<e21 and -mom3>max(abs(prev3)*1.25,0.45*atr) and body>=0.40*atr and (c[-1]-l[-1])/rng<=0.25
   if buy or sell:self.fire+=1;return 1 if buy else -1
   return 0
  if sel=='R1_RANGE_FOLLOW':
   # directional continuation inside a statistically contained 20-bar range
   if e21 is None or ep is None or e50 is None:return 0
   hi=float(h[-21:-1].max());lo=float(l[-21:-1].min());width=hi-lo
   contained=width<=4.0*atr; slope=abs(e21-ep)/atr; not_trending=slope<=0.12
   pos=(c[-1]-lo)/max(width,1e-9)
   buy=contained and not_trending and e21>=e50 and c[-2]<=e21 and c[-1]>e21 and pos>=0.45 and pos<=0.80
   sell=contained and not_trending and e21<=e50 and c[-2]>=e21 and c[-1]<e21 and pos<=0.55 and pos>=0.20
   if buy or sell:self.fire+=1;return 1 if buy else -1
   return 0
  return 0
 def open(self,side,bid,ask):
  if self.active:return
  px=ask if side>0 else bid;atr=max(self.atr(),1e-9);self.submit(side);self.active={'side':side,'entry':px,'atr':atr,'i':self.m1_i,'best':px,'worst':px,'trail':None}
 def close(self,bid,ask,reason):
  a=self.active
  if not a:return
  side=a['side'];px=bid if side>0 else ask;pnl=(px-a['entry'])*side;self.submit(-side)
  mfe=(a['best']-a['entry']) if side>0 else (a['entry']-a['best']);mae=(a['entry']-a['worst']) if side>0 else (a['worst']-a['entry'])
  self.trades.append({'pnl':pnl,'mfe_atr':max(mfe,0)/a['atr'],'mae_atr':max(mae,0)/a['atr'],'reason':reason})
  if pnl>0:self.wins+=1;self.gw+=pnl
  elif pnl<0:self.losses+=1;self.gl+=abs(pnl)
  self.active=None
 def on_bar(self,b:Bar):
  o,h,l,c=map(self.f,[b.open,b.high,b.low,b.close]);self.o.append(o);self.h.append(h);self.l.append(l);self.c.append(c);self.m1_i+=1
  tr=max(h-l,abs(h-self.prev) if self.prev is not None else 0,abs(l-self.prev) if self.prev is not None else 0);self.tr.append(tr);self.prev=c
  if not self.active and not self.pending:self.pending=self.signal()
 def on_quote_tick(self,t:QuoteTick):
  bid=self.f(t.bid_price);ask=self.f(t.ask_price);self.last_bid=bid;self.last_ask=ask
  if self.pending and not self.active:s=self.pending;self.pending=0;self.open(s,bid,ask)
  a=self.active
  if not a:return
  side=a['side'];mark=bid if side>0 else ask;atr=a['atr']
  if side>0:a['best']=max(a['best'],mark);a['worst']=min(a['worst'],mark);mfe=(a['best']-a['entry'])/atr
  else:a['best']=min(a['best'],mark);a['worst']=max(a['worst'],mark);mfe=(a['entry']-a['best'])/atr
  move=(mark-a['entry'])*side
  if move<=-0.75*atr:self.close(bid,ask,'HARD_SL');return
  if mfe>=0.75:
   dist=(0.75 if mfe>=2 else 1.0 if mfe>=1.5 else 1.2)*atr;be=a['entry']+side*0.05*atr;cand=a['best']-dist if side>0 else a['best']+dist
   cand=max(cand,be) if side>0 else min(cand,be);a['trail']=cand if a['trail'] is None else (max(a['trail'],cand) if side>0 else min(a['trail'],cand))
  if a['trail'] is not None and ((side>0 and mark<=a['trail']) or (side<0 and mark>=a['trail'])):self.close(bid,ask,'DYNAMIC_ATR_TRAIL');return
  if self.m1_i-a['i']>=self.config.horizon_minutes:self.close(bid,ask,'HORIZON')
 def on_stop(self):
  if self.active and self.last_bid is not None:self.close(self.last_bid,self.last_ask,'EOD')
 def summary(self):
  n=len(self.trades);net=sum(x['pnl'] for x in self.trades);pf=self.gw/self.gl if self.gl>0 else (math.inf if self.gw>0 else 0);r=Counter(x['reason'] for x in self.trades)
  return {'selected':self.config.selected,'N':n,'fire_count':self.fire,'WR_pct':100*self.wins/max(n,1),'PF':pf,'net_virtual':net,'expectancy':net/max(n,1),'MFE_ATR_mean':float(np.mean([x['mfe_atr'] for x in self.trades])) if n else 0,'MAE_ATR_mean':float(np.mean([x['mae_atr'] for x in self.trades])) if n else 0,'exit_reasons':dict(r)}

def l1(ticks):
 one=Quantity.from_int(1);out=[];rep=0
 for t in ticks:
  bs=float(t.bid_size.as_double()) if hasattr(t.bid_size,'as_double') else float(t.bid_size);az=float(t.ask_size.as_double()) if hasattr(t.ask_size,'as_double') else float(t.ask_size)
  if bs<=0 or az<=0:out.append(QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=one if bs<=0 else t.bid_size,ask_size=one if az<=0 else t.ask_size,ts_event=t.ts_event,ts_init=t.ts_init));rep+=1
  else:out.append(t)
 return out,rep

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--selected',choices=EDGES,required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args()
 cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value]);ticks,rep=l1(raw)
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)));eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(ticks)
 st=S(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{inst.id.value}-1-MINUTE-BID-INTERNAL'),selected=a.selected));eng.add_strategy(st);eng.run();eng.end();res={'verification_level':'NAUTILUS_RAW_PRIORITY4_DYNAMIC_ATR','raw_ticks':len(raw),'ohlc_resample_used':False,'exit_model':'DYNAMIC_ATR_NO_FIXED_TP',**st.summary()};p=Path('results/multiedge')/a.experiment_id/f'{a.selected}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
if __name__=='__main__':main()
