from __future__ import annotations
import argparse,json,math
from collections import deque
from decimal import Decimal
from pathlib import Path
import numpy as np,pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,BookType
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

def f(x):return float(x.as_double()) if hasattr(x,'as_double') else float(x)
def nsdt(x):return pd.Timestamp(int(x),unit='ns',tz='UTC')
def metrics(rs,risk=.35):
 eq=pk=100.;dd=0.;gp=gl=0.
 for r in rs:
  if r>0:gp+=r
  else:gl-=r
  eq*=max(.0001,1+risk*r/100);pk=max(pk,eq);dd=max(dd,100*(pk-eq)/pk)
 return {'N':len(rs),'wins':sum(r>0 for r in rs),'WR_pct':100*sum(r>0 for r in rs)/len(rs) if rs else 0.,
 'PF_R':gp/gl if gl else (math.inf if gp else 0.),'sum_R':sum(rs),'Return_pct':eq-100,'MaxDD_pct':dd}

class Cfg(StrategyConfig,frozen=True):instrument_id:object
class VideoCISDRaw(Strategy):
 def __init__(self,c):
  super().__init__(c);self.ticks=[];self.bars=[];self.cur=None;self.phase=0;self.age=0;self.di=0;self.acc_hi=self.acc_lo=None
  self.sweep=None;self.cisd=None;self.confirm_i=-1;self.pos=None;self.rs=[];self.trades=[];self.route='';self.ranges={}
 def on_start(self):self.subscribe_quote_ticks(self.config.instrument_id)
 def finish_bar(self):
  if self.cur is None:return
  self.bars.append(self.cur);self.process_bar(len(self.bars)-1);self.cur=None
 def route_at(self,t):
  j=t.tz_convert('Asia/Tokyo');m=j.hour*60+j.minute
  if 540<=m<600:return 'ASIA'
  if 960<=m<1020:return 'LONDON'
  if 1380<=m<1440:return 'NY'
  return None
 def build_range(self,route,t):
  j=t.tz_convert('Asia/Tokyo');day=j.normalize();xs=[]
  for b in self.bars[:-1]:
   z=b['t'].tz_convert('Asia/Tokyo');m=z.hour*60+z.minute
   if route=='LONDON' and z.normalize()==day and 540<=m<960:xs.append(b)
   elif route=='NY' and z.normalize()==day and 960<=m<1380:xs.append(b)
   elif route=='ASIA' and ((z.normalize()==day-pd.Timedelta(days=1) and m>=1380) or (z.normalize()==day and m<360)):xs.append(b)
  if not xs:return None
  return max(x['h'] for x in xs),min(x['l'] for x in xs)
 def delivery(self,b,di):return b['c']>b['o'] if di<0 else b['c']<b['o']
 def cisd_origin(self,i,di):
  j=i
  if not self.delivery(self.bars[j],di):j-=1
  if j<1 or not self.delivery(self.bars[j],di):return None
  oldest=j
  for k in range(j-1,max(-1,i-8),-1):
   if not self.delivery(self.bars[k],di):break
   oldest=k
  return self.bars[oldest]['o']
 def atr(self,n=14):
  if len(self.bars)<n+2:return None
  x=self.bars[-n:];trs=[]
  for k,b in enumerate(x):
   prev=self.bars[len(self.bars)-n+k-1]['c'];trs.append(max(b['h']-b['l'],abs(b['h']-prev),abs(b['l']-prev)))
  return sum(trs)/len(trs)
 def target(self,i,entry,sl):
  risk=abs(entry-sl);cs=[]
  for k in range(max(1,i-24),i):
   if self.di<0 and self.bars[k]['l']<self.bars[k-1]['l'] and (k+1>=len(self.bars) or self.bars[k]['l']<self.bars[k+1]['l']) and self.bars[k]['l']<entry:cs.append(self.bars[k]['l'])
   if self.di>0 and self.bars[k]['h']>self.bars[k-1]['h'] and (k+1>=len(self.bars) or self.bars[k]['h']>self.bars[k+1]['h']) and self.bars[k]['h']>entry:cs.append(self.bars[k]['h'])
  cs.append(self.acc_lo if self.di<0 else self.acc_hi);good=[]
  for x in cs:
   rew=(entry-x) if self.di<0 else (x-entry);rr=rew/risk if risk else 0
   if rew>0 and rr>=.8:good.append((rew,x,rr))
  return min(good) if good else None
 def reset(self):self.phase=0;self.age=0;self.di=0;self.acc_hi=self.acc_lo=None;self.sweep=None;self.cisd=None;self.confirm_i=-1;self.route=''
 def process_bar(self,i):
  b=self.bars[i];a=self.atr()
  if a is None:return
  if self.phase==0:
   r=self.route_at(b['t'])
   if not r:return
   z=self.build_range(r,b['t'])
   if not z:return
   self.route=r;self.acc_hi,self.acc_lo=z;self.phase=1;self.age=0;return
  self.age+=1
  if self.phase==1:
   up=b['h']>self.acc_hi+a*.03 and b['c']<self.acc_hi;dn=b['l']<self.acc_lo-a*.03 and b['c']>self.acc_lo
   if up or dn:
    self.di=-1 if up else 1;self.sweep=b['h'] if up else b['l'];self.cisd=self.cisd_origin(i,self.di)
    if self.cisd is None:self.reset();return
    self.phase=2;self.age=0
   elif self.age>8:self.reset()
   return
  if self.phase==2:
   self.sweep=max(self.sweep,b['h']) if self.di<0 else min(self.sweep,b['l'])
   ok=b['c']<self.cisd if self.di<0 else b['c']>self.cisd
   if ok:self.confirm_i=i;self.phase=3;self.age=0
   elif self.age>8:self.reset()
   return
  if self.phase==3:
   touch=b['h']>=self.cisd if self.di<0 else b['l']<=self.cisd;hold=b['c']<=self.cisd if self.di<0 else b['c']>=self.cisd
   if touch and hold and i>self.confirm_i and self.pos is None:
    # signal at closed M15 bar; actual entry is next raw quote, preserving no-lookahead
    self.pos={'pending':True,'di':self.di,'sl_ref':self.sweep,'atr':a,'route':self.route,'signal':str(b['t']),'acc_hi':self.acc_hi,'acc_lo':self.acc_lo};self.reset()
   elif self.age>6:self.reset()
 def on_quote_tick(self,t:QuoteTick):
  bid,ask=f(t.bid_price),f(t.ask_price);ts=nsdt(t.ts_event)
  if self.pos and not self.pos.get('pending'):
   p=self.pos;mark=bid if p['di']>0 else ask;hit=None
   if p['di']>0:
    if mark<=p['sl']:hit=-1.
    elif mark>=p['tp']:hit=p['rr']
   else:
    if mark>=p['sl']:hit=-1.
    elif mark<=p['tp']:hit=p['rr']
   if hit is not None:self.rs.append(hit);self.trades.append({**p,'exit':str(ts),'R':hit});self.pos=None
  if self.pos and self.pos.get('pending'):
   p=self.pos;entry=ask if p['di']>0 else bid;sl=p['sl_ref']-p['atr']*.05 if p['di']>0 else p['sl_ref']+p['atr']*.05
   spread=ask-bid
   risk=abs(entry-sl)
   # Execution validity gate: reject stops that are effectively inside transaction cost.
   # This does not alter the video signal sequence; it only blocks non-executable geometry.
   if risk <= max(spread*2.0, 1e-9):
    self.pos=None
    return
   # target uses completed bars only
   self.di=p['di'];self.acc_hi=p['acc_hi'];self.acc_lo=p['acc_lo']
   z=self.target(len(self.bars)-1,entry,sl)
   if z is None:tp=entry+p['di']*abs(entry-sl)*2;rr=2.
   else:_,tp,rr=z
   self.pos={**p,'pending':False,'entry':entry,'sl':sl,'tp':tp,'rr':rr,'entry_time':str(ts),'spread':ask-bid}
  bucket=ts.floor('15min')
  mid=(bid+ask)/2
  if self.cur is None:self.cur={'t':bucket,'o':mid,'h':mid,'l':mid,'c':mid}
  elif bucket!=self.cur['t']:
   self.finish_bar();self.cur={'t':bucket,'o':mid,'h':mid,'l':mid,'c':mid}
  else:self.cur['h']=max(self.cur['h'],mid);self.cur['l']=min(self.cur['l'],mid);self.cur['c']=mid
 def on_stop(self):self.finish_bar()

def executable(xs):
 one=Quantity.from_int(1);return [QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=one if f(t.bid_size)<=0 else t.bid_size,ask_size=one if f(t.ask_size)<=0 else t.ask_size,ts_event=t.ts_event,ts_init=t.ts_init) for t in xs]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--out',required=True);ap.add_argument('--start');ap.add_argument('--end');a=ap.parse_args()
 cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
 raw=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value],start=a.start,end=a.end);assert raw
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
 eng.add_instrument(inst);eng.add_data(raw);st=VideoCISDRaw(Cfg(instrument_id=inst.id));eng.add_strategy(st);eng.run();eng.end()
 out={'verification':'AMOS_VIDEO_PARITY_CISD_V1_7_NAUTILUS_RAW_EXACT_AMD_WINDOWS_EXECUTION_GATE','raw_ticks':len(raw),'ohlc_resample_used_for_execution':False,'signal_bars':'M15 built causally from raw midpoint quotes','execution':'next QuoteTick Bid/Ask after closed M15 signal','initial_usd':1000,'leverage':2000,'query_start':a.start,'query_end':a.end,**metrics(st.rs)}
 p=Path(a.out);p.mkdir(parents=True,exist_ok=True);(p/'summary.json').write_text(json.dumps(out,indent=2));pd.DataFrame(st.trades).to_csv(p/'trades.csv',index=False);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
