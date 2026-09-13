from __future__ import annotations
import argparse,json,math
from collections import deque
from decimal import Decimal
from pathlib import Path
import numpy as np
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar,QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,BookType
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
 def _q(self,identifiers=None,start=None,end=None): return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
 ParquetDataCatalog.query_quote_ticks=_q

def f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)
def met(a,initial=1000.0):
 a=np.asarray(a,float);w=a[a>0];l=a[a<0];pf=float(w.sum()/abs(l.sum())) if len(l) else (math.inf if len(w) else 0.0);eq=peak=initial;mdd=0.0
 for x in a:eq+=x;peak=max(peak,eq);mdd=max(mdd,peak-eq)
 return {'N':int(len(a)),'WR_pct':float((a>0).mean()*100) if len(a) else 0.0,'PF':pf,'Net':float(a.sum()) if len(a) else 0.0,'EV':float(a.mean()) if len(a) else 0.0,'MaxDD_pct_virtual':float(mdd/peak*100) if peak else 0.0,'RF_virtual':float(a.sum()/mdd) if mdd else None}

class Cfg(StrategyConfig,frozen=True):
 instrument_id:object;bar_type:BarType;tf_min:int

class BPA(Strategy):
 def __init__(self,c):
  super().__init__(c);self.o=deque(maxlen=260);self.h=deque(maxlen=260);self.l=deque(maxlen=260);self.c=deque(maxlen=260);self.v=deque(maxlen=260);self.tick_count=0;self.pending=None;self.active=None;self.last_bid=self.last_ask=None;self.pnl=[];self.signals=0
 def on_start(self): self.subscribe_quote_ticks(self.config.instrument_id);self.subscribe_bars(self.config.bar_type)
 def atr14(self):
  if len(self.c)<15:return None
  hh=list(self.h);ll=list(self.l);cc=list(self.c);tr=[]
  for i in range(len(cc)-14,len(cc)): tr.append(max(hh[i]-ll[i],abs(hh[i]-cc[i-1]),abs(ll[i]-cc[i-1])))
  return float(np.mean(tr))
 def on_quote_tick(self,t:QuoteTick):
  self.tick_count+=1;bid=f(t.bid_price);ask=f(t.ask_price);self.last_bid=bid;self.last_ask=ask
  if ask-bid>2.5:return
  if self.pending and not self.active:
   side,atr=self.pending;entry=ask if side>0 else bid;self.active={'side':side,'entry':entry,'stop':entry-side*atr,'take':entry+side*2.0*atr,'atr':atr};self.pending=None
  if self.active:
   a=self.active;side=a['side'];mark=bid if side>0 else ask;move=(mark-a['entry'])*side
   if move>=a['atr']: a['stop']=max(a['stop'],a['entry']) if side>0 else min(a['stop'],a['entry'])
   close=(mark<=a['stop'] if side>0 else mark>=a['stop']) or (mark>=a['take'] if side>0 else mark<=a['take'])
   if close:self.pnl.append(move);self.active=None
 def on_bar(self,bar:Bar):
  o,h,l,c=map(f,[bar.open,bar.high,bar.low,bar.close]);cur_v=self.tick_count;self.tick_count=0
  # Analyze the just-completed bar using prior history only for volume/swing references.
  if len(self.c)>=200 and not self.active and self.pending is None:
   vols=np.asarray(list(self.v)[-200:],float);mu=float(vols.mean());sd=float(vols.std(ddof=0));z=(cur_v-mu)/sd if sd>0 else -999
   atr=self.atr14();rng=h-l;body=abs(c-o);body_ratio=body/rng if rng>0 else 0.0
   prev_h=max(list(self.h)[-20:]);prev_l=min(list(self.l)[-20:])
   low_sweep=(l<prev_l and c>prev_l);high_sweep=(h>prev_h and c<prev_h)
   imb_buy=(c>o and atr and rng/atr>=1.5 and body_ratio>=0.60);imb_sell=(c<o and atr and rng/atr>=1.5 and body_ratio>=0.60)
   if z>=2.0 and atr:
    side=1 if (low_sweep and imb_buy) else -1 if (high_sweep and imb_sell) else 0
    if side:self.pending=(side,atr);self.signals+=1
  self.o.append(o);self.h.append(h);self.l.append(l);self.c.append(c);self.v.append(cur_v)
 def on_stop(self):
  if self.active:
   a=self.active;mark=self.last_bid if a['side']>0 else self.last_ask;self.pnl.append((mark-a['entry'])*a['side']);self.active=None
 def summary(self):
  s=met(self.pnl);s.update({'signals':self.signals,'tf_min':self.config.tf_min,'entry':'next QuoteTick after confirmed combo bar','signal':'z>=2.0 AND StopHunt AND same-direction Imbalance; LookbackVol=200 Swing=20 Range>=1.5ATR BodyRatio>=0.60','exit':'SL=1.0ATR14 TP=2.0ATR14 BE at +1.0ATR'});return s

def l1(xs):
 one=Quantity.from_int(1);out=[]
 for t in xs:
  bs=f(t.bid_size);az=f(t.ask_size);out.append(QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=one if bs<=0 else t.bid_size,ask_size=one if az<=0 else t.ask_size,ts_event=t.ts_event,ts_init=t.ts_init) if bs<=0 or az<=0 else t)
 return out

def run(catalog,tf,experiment_id):
 cat=ParquetDataCatalog(catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value]);eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(l1(raw));s=inst.id.value;bt=BarType.from_str(f'{s}-{tf}-MINUTE-BID-INTERNAL');st=BPA(Cfg(instrument_id=inst.id,bar_type=bt,tf_min=tf));eng.add_strategy(st);eng.run();eng.end();res={'verification':'BIGPLAYER_BPA_COMBO_RAW','raw_ticks':len(raw),'ohlc_resample_used':False,'tick_volume':'Counted directly from Raw QuoteTick callbacks inside each Nautilus internal time bar','limitation':'Screening model; exit is standardized 1ATR/2ATR, not yet GoldeBrave production exit or exact broker semantics.',**st.summary()};p=Path('results/bigplayer-bpa-combo')/experiment_id/f'm{tf}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--tf',type=int,choices=[1,5,15],required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();run(a.catalog,a.tf,a.experiment_id)
if __name__=='__main__':main()
