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
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from goldebrave_fasttf_parity_raw import GB,Cfg,f,l1

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
 def _q(self,identifiers=None,start=None,end=None): return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
 ParquetDataCatalog.query_quote_ticks=_q

class GBTenzoko(GB):
 def __init__(self,c,filter_mode='base'):
  super().__init__(c);self.filter_mode=filter_mode
  self.m1o=deque(maxlen=600);self.m1h=deque(maxlen=600);self.m1l=deque(maxlen=600);self.m1c=deque(maxlen=600);self.m1v=deque(maxlen=600)
  self.sig_buy_age=999;self.sig_sell_age=999;self.last_signal={}
  self.allowed_c_buy=0;self.allowed_c_sell=0;self.blocked_c_buy=0;self.blocked_c_sell=0
 def _atr_m1(self,n=14):
  if len(self.m1c)<n+1:return None
  h=np.asarray(self.m1h,float);l=np.asarray(self.m1l,float);c=np.asarray(self.m1c,float)
  tr=[]
  for i in range(len(c)-n,len(c)):
   tr.append(max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])))
  return float(np.mean(tr))
 def _detector(self):
  # BigPlayerDetector v4 Optimized defaults, evaluated on completed M1 bars.
  # LookbackBars=200, VolumeSigmaThreshold=2.0, RangeMultiplier=1.5, SwingLookback=20.
  if len(self.m1c)<222:return
  o,h,l,c,v = self.m1o[-1],self.m1h[-1],self.m1l[-1],self.m1c[-1],self.m1v[-1]
  hist=np.asarray(list(self.m1v)[-201:-1],float)
  mean=float(hist.mean());std=float(hist.std(ddof=0))
  if mean<=0 or std<=0:return
  z=(v-mean)/std
  atr=self._atr_m1(14)
  if atr is None or atr<=0:return
  rng=h-l
  if rng<=0:return
  body=abs(c-o);body_ratio=body/rng
  imb_buy=imb_sell=False
  if z>=2.0 and rng/atr>=1.5 and body_ratio>=0.60:
   imb_buy=c>o;imb_sell=c<o
  ph=max(list(self.m1h)[-21:-1]);pl=min(list(self.m1l)[-21:-1])
  sweep_buy=(z>=2.0 and l<pl and c>pl)
  sweep_sell=(z>=2.0 and h>ph and c<ph)
  combo_buy=sweep_buy and imb_buy;combo_sell=sweep_sell and imb_sell
  self.sig_buy_age+=1;self.sig_sell_age+=1
  buy=sweep_buy if self.filter_mode=='sweep' else combo_buy if self.filter_mode=='combo' else False
  sell=sweep_sell if self.filter_mode=='sweep' else combo_sell if self.filter_mode=='combo' else False
  if buy:self.sig_buy_age=0
  if sell:self.sig_sell_age=0
  self.last_signal={'z':z,'imb_buy':imb_buy,'imb_sell':imb_sell,'sweep_buy':sweep_buy,'sweep_sell':sweep_sell,'combo_buy':combo_buy,'combo_sell':combo_sell}
 def _c_permission(self,side):
  if self.filter_mode=='base':return True
  # Direction rejection filter: signal remains valid for 5 completed M1 bars.
  return self.sig_buy_age<=5 if side>0 else self.sig_sell_age<=5
 def boost(self):
  if self.cur_hour<9 or self.entries_day>=3:return
  dr=self.dayrange();vol,reg,a=self.vol_reg()
  if dr is None or a is None:return
  dhi,dlo=dr;off=.2*vol*(2 if reg==-1 else 1);sl=min(max(a*1.2,4),12);tp=min(max(a*2.4,9),30)*(1.25 if reg==1 else .8 if reg==-1 else 1)
  for side,p in ((1,dhi+off),(-1,dlo-off)):
   if self._c_permission(side):
    self.pending.append(('C',side,p,sl,tp,vol))
    if side>0:self.allowed_c_buy+=1
    else:self.allowed_c_sell+=1
   else:
    if side>0:self.blocked_c_buy+=1
    else:self.blocked_c_sell+=1
  self.pending=self.pending[-30:]
 def on_bar(self,bar:Bar):
  s=str(bar.bar_type);tf=60 if '-1-HOUR-' in s else 15 if '-15-MINUTE-' in s else 5 if '-5-MINUTE-' in s else 1 if '-1-MINUTE-' in s else None
  if tf is None:return
  o,h,l,c=map(f,[bar.open,bar.high,bar.low,bar.close]);[self.b[tf][i].append(v) for i,v in enumerate((o,h,l,c))]
  if tf==1:
   self.m1o.append(o);self.m1h.append(h);self.m1l.append(l);self.m1c.append(c);self.m1v.append(f(bar.volume));self._detector()
  self.cur_hour=self.hour(int(bar.ts_event));day=int((int(bar.ts_event)+3*3600*10**9)//(86400*10**9))
  if self.day!=day:self.day=day;self.entries_day=0;self.placed=[]
  allowed=self.cur_hour in (11,15,16,17,18)
  if not allowed:self.pending=[]
  if not allowed:return
  if tf==15:self.boost()
 def summary_ext(self):
  x=self.summary();x['filter_mode']=self.filter_mode;x['filter_window_m1_bars']=5;x['c_filter_counts']={'allow_buy':self.allowed_c_buy,'allow_sell':self.allowed_c_sell,'block_buy':self.blocked_c_buy,'block_sell':self.blocked_c_sell};return x

def run(catalog,mode,experiment_id):
 cat=ParquetDataCatalog(catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value])
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(l1(raw));s=inst.id.value
 st=GBTenzoko(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{s}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{s}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{s}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{s}-1-HOUR-BID-INTERNAL'),mode='m15'),filter_mode=mode)
 eng.add_strategy(st);eng.run();eng.end()
 res={'verification':'GOLDEBRAVE_C_TENZOKO_BIGPLAYER_RAW','raw_ticks':len(raw),'ohlc_resample_used':False,'indicator_logic':'BigPlayerDetector defaults: tick-volume z>=2; Imbalance range>=1.5 ATR and body>=60%; Sweep 20-bar high/low rejection; combo=same-bar Sweep+Imbalance; M1 completed bars; direction permission valid 5 bars.','limitation':'Nautilus internal M1 tick-count volume is used as MT5 tick_volume proxy. Entry/exit is same GoldeBrave C approximation as prior 30d run.',**st.summary_ext()}
 p=Path('results/goldebrave-tenzoko')/experiment_id/f'{mode}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--mode',choices=['base','sweep','combo'],required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();run(a.catalog,a.mode,a.experiment_id)
if __name__=='__main__':main()
