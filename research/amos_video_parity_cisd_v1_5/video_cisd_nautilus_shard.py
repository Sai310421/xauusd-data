from __future__ import annotations
import argparse,json,math
from collections import deque
from decimal import Decimal
from pathlib import Path
import numpy as np,pandas as pd
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar,QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,OrderSide,BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

ROUTES={"ASIA":(9,10),"LONDON":(16,18),"NY":(23,1)}
def f(x):return float(x.as_double()) if hasattr(x,'as_double') else float(x)
def ts(x):return pd.Timestamp(int(x),unit='ns',tz='UTC')
def parse_money(v):
 if v is None:return 0.
 if isinstance(v,(float,int,np.number)):return float(v)
 try:return float(str(v).replace(',','').strip().split()[0])
 except:return 0.

def extract_trades(report):
 if report is None or report.empty:return []
 pc=next((c for c in report.columns if 'pnl' in str(c).lower()),None)
 tc=next((c for c in report.columns if 'closed' in str(c).lower() and ('ts' in str(c).lower() or 'time' in str(c).lower())),None)
 out=[]
 for i,row in report.iterrows():
  pnl=parse_money(row[pc]) if pc else 0.
  t=row[tc] if tc else i
  try:t=int(pd.Timestamp(t).value)
  except:
   try:t=int(t)
   except:t=len(out)
  out.append({'pnl':pnl,'ts_closed':t})
 return out

def metrics(trades,initial=300.):
 a=np.asarray([x['pnl'] for x in trades],float)
 if len(a)==0:return {'N':0,'WR_pct':0.,'PF':0.,'NetProfit':0.,'Return_pct':0.,'MaxDD_pct':0.}
 w=a[a>0];l=a[a<0];eq=peak=initial;dd=0.
 for x in a:eq+=x;peak=max(peak,eq);dd=max(dd,peak-eq)
 return {'N':int(len(a)),'WR_pct':float((a>0).mean()*100),'PF':float(w.sum()/abs(l.sum())) if len(l) and l.sum()!=0 else (math.inf if len(w) else 0.),
         'NetProfit':float(a.sum()),'Return_pct':float(a.sum()/initial*100),'MaxDD_pct':float(dd/peak*100 if peak else 0.)}

class Cfg(StrategyConfig,frozen=True):
 instrument_id:object
 bar_type:BarType
 signal_start_ns:int
 signal_end_ns:int
 qty:Decimal=Decimal('1')

class VideoAMD(Strategy):
 def __init__(self,c):
  super().__init__(c);self.bars=deque(maxlen=800);self.phase=0;self.age=0;self.route=None;self.di=0
  self.acc_hi=self.acc_lo=self.sweep=self.cisd=None;self.confirm_idx=-1;self.signals=0;self.route_signals={k:0 for k in ROUTES};self.signal_meta=[]
 def on_start(self):self.subscribe_bars(self.config.bar_type)
 def jst(self,b):return ts(b['ts']).tz_convert('Asia/Tokyo')
 def route_at(self,b):
  h=self.jst(b).hour
  for k,(a,z) in ROUTES.items():
   if (a<=h<z) if a<z else (h>=a or h<z):return k
  return None
 def build_range(self,route,b):
  now=self.jst(b);day=now.normalize();xs=[]
  for x in list(self.bars)[:-1]:
   q=self.jst(x);m=q.hour*60+q.minute
   if route=='LONDON' and q.normalize()==day and 540<=m<960:xs.append(x)
   elif route=='NY' and q.normalize()==day and 960<=m<1380:xs.append(x)
   elif route=='ASIA' and ((q.normalize()==day-pd.Timedelta(days=1) and m>=1380) or (q.normalize()==day and m<360)):xs.append(x)
  if not xs:return None
  return max(x['h'] for x in xs),min(x['l'] for x in xs)
 def atr(self):
  x=list(self.bars)
  if len(x)<16:return None
  tr=[]
  for i in range(len(x)-14,len(x)):
   b=x[i];p=x[i-1]['c'];tr.append(max(b['h']-b['l'],abs(b['h']-p),abs(b['l']-p)))
  a=float(np.mean(tr));return a if a>0 and math.isfinite(a) else None
 def delivery(self,b,di):return b['c']>b['o'] if di<0 else b['c']<b['o']
 def cisd_origin(self,di):
  x=list(self.bars);i=len(x)-1;j=i
  if not self.delivery(x[j],di):j-=1
  if j<1 or not self.delivery(x[j],di):return None
  oldest=j
  for k in range(j-1,max(-1,i-8),-1):
   if not self.delivery(x[k],di):break
   oldest=k
  return x[oldest]['o']
 def m30(self):
  x=list(self.bars);out=[]
  for i in range(1,len(x),2):
   a,b=x[i-1],x[i]
   out.append({'o':a['o'],'h':max(a['h'],b['h']),'l':min(a['l'],b['l']),'c':b['c'],'ts':b['ts']})
  return out
 def choose_target(self,entry,sl):
  risk=abs(entry-sl)
  if risk<=0:return None
  x=list(self.bars);cs=[]
  for k in range(max(1,len(x)-25),len(x)-1):
   if self.di<0 and x[k]['l']<x[k-1]['l'] and x[k]['l']<x[k+1]['l'] and x[k]['l']<entry:cs.append(x[k]['l'])
   if self.di>0 and x[k]['h']>x[k-1]['h'] and x[k]['h']>x[k+1]['h'] and x[k]['h']>entry:cs.append(x[k]['h'])
  q=self.m30()
  for k in range(2,len(q)):
   if self.di>0 and q[k]['l']>q[k-2]['h'] and q[k-2]['h']>entry:cs.extend([q[k-2]['h'],q[k]['l']])
   if self.di<0 and q[k]['h']<q[k-2]['l'] and q[k-2]['l']<entry:cs.extend([q[k-2]['l'],q[k]['h']])
  cs.append(self.acc_lo if self.di<0 else self.acc_hi);good=[]
  for p in cs:
   rew=(entry-p) if self.di<0 else (p-entry);rr=rew/risk
   if rew>0 and rr>=.8:good.append((rew,p,rr))
  return min(good,key=lambda z:z[0]) if good else None
 def flat(self):return not self.portfolio.is_net_long(self.config.instrument_id) and not self.portfolio.is_net_short(self.config.instrument_id)
 def reset(self):
  self.phase=0;self.age=0;self.route=None;self.di=0;self.acc_hi=self.acc_lo=self.sweep=self.cisd=None;self.confirm_idx=-1
 def submit_bracket(self,b,a):
  if not (self.config.signal_start_ns<=int(b['ts'])<self.config.signal_end_ns):return False
  if not self.flat():return False
  instrument=self.cache.instrument(self.config.instrument_id)
  q=self.cache.quote_tick(self.config.instrument_id)
  if q is None:return False
  bid=f(q.bid_price);ask=f(q.ask_price);entry=ask if self.di>0 else bid
  sl=self.sweep-a*.05 if self.di>0 else self.sweep+a*.05
  z=self.choose_target(entry,sl)
  if z is None:return False
  _,tp,rr=z
  side=OrderSide.BUY if self.di>0 else OrderSide.SELL
  orders=self.order_factory.bracket(instrument_id=self.config.instrument_id,order_side=side,quantity=instrument.make_qty(self.config.qty),
      tp_price=instrument.make_price(tp),sl_trigger_price=instrument.make_price(sl),tp_post_only=False)
  self.submit_order_list(orders)
  self.signals+=1;self.route_signals[self.route]+=1
  self.signal_meta.append({'signal_ts':int(b['ts']),'route':self.route,'dir':self.di,'entry_ref':entry,'sl':sl,'tp':tp,'rr':rr})
  return True
 def on_bar(self,bar:Bar):
  b={'o':f(bar.open),'h':f(bar.high),'l':f(bar.low),'c':f(bar.close),'ts':int(bar.ts_event)}
  self.bars.append(b);a=self.atr()
  if a is None:return
  if self.phase==0:
   r=self.route_at(b)
   if not r:return
   z=self.build_range(r,b)
   if not z:return
   self.route=r;self.acc_hi,self.acc_lo=z;self.phase=1;self.age=0;return
  self.age+=1
  if self.phase==1:
   up=b['h']>self.acc_hi+a*.03 and b['c']<self.acc_hi;dn=b['l']<self.acc_lo-a*.03 and b['c']>self.acc_lo
   if up or dn:
    self.di=-1 if up else 1;self.sweep=b['h'] if up else b['l'];self.cisd=self.cisd_origin(self.di)
    if self.cisd is None:self.reset();return
    self.phase=2;self.age=0
   elif self.age>8:self.reset()
   return
  if self.phase==2:
   self.sweep=max(self.sweep,b['h']) if self.di<0 else min(self.sweep,b['l'])
   ok=b['c']<self.cisd if self.di<0 else b['c']>self.cisd
   if ok:self.confirm_idx=len(self.bars)-1;self.phase=3;self.age=0
   elif self.age>8:self.reset()
   return
  if self.phase==3:
   touch=b['h']>=self.cisd if self.di<0 else b['l']<=self.cisd;hold=b['c']<=self.cisd if self.di<0 else b['c']>=self.cisd
   if touch and hold and len(self.bars)-1>self.confirm_idx:
    self.submit_bracket(b,a);self.reset();return
   if self.age>6:self.reset()
 def on_stop(self):self.close_all_positions(self.config.instrument_id)

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--catalog',required=True);ap.add_argument('--out',required=True)
 ap.add_argument('--start',required=True);ap.add_argument('--end',required=True);ap.add_argument('--label',required=True)
 a=ap.parse_args()
 primary_start=pd.Timestamp(a.start,tz='UTC');primary_end=pd.Timestamp(a.end,tz='UTC')
 query_start=primary_start-pd.Timedelta(days=1);query_end=primary_end+pd.Timedelta(days=1)
 cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
 raw=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value],start=int(query_start.value),end=int(query_end.value))
 if not raw:raise SystemExit('no raw XAUUSD QuoteTicks in shard')
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(300,USD)],default_leverage=Decimal('2000'))
 eng.add_instrument(inst);eng.add_data(raw)
 bt=BarType.from_str(f'{inst.id.value}-15-MINUTE-MID-INTERNAL')
 st=VideoAMD(Cfg(instrument_id=inst.id,bar_type=bt,signal_start_ns=int(primary_start.value),signal_end_ns=int(primary_end.value)))
 eng.add_strategy(st);eng.run();eng.end()
 report=eng.trader.generate_positions_report();tr=extract_trades(report);m=metrics(tr,300.)
 out={'verification':'NAUTILUS_BT_RAW_BIDASK_VIDEO_CISD_V1_8_SHARD','shard':a.label,
      'primary_start':str(primary_start),'primary_end_exclusive':str(primary_end),
      'query_start':str(query_start),'query_end':str(query_end),
      'engine':'NautilusTrader BacktestEngine','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),
      'raw_ticks_loaded':len(raw),'ohlc_input_used':False,
      'signal_bars':'15-MINUTE-MID-INTERNAL built by Nautilus from raw QuoteTicks',
      'execution':'Nautilus MARKET entry + STOP_MARKET SL + LIMIT TP bracket on raw Bid/Ask',
      'initial_usd':300,'leverage':2000,'quantity_xau':1.0,
      'signals_submitted':st.signals,'route_signals':st.route_signals,**m}
 p=Path(a.out);p.mkdir(parents=True,exist_ok=True)
 (p/'summary.json').write_text(json.dumps(out,indent=2))
 pd.DataFrame(tr).to_csv(p/'trades.csv',index=False);pd.DataFrame(st.signal_meta).to_csv(p/'signals.csv',index=False)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
