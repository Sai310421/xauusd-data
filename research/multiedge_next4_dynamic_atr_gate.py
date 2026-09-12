from __future__ import annotations
import argparse,json,math
from collections import deque,Counter
from decimal import Decimal
from pathlib import Path
import numpy as np
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
EDGES=('T2_ADAPTIVE_MTF','T5_COMP_EXP','R3_LIQUIDITY_RUN','V5_HARMONIC_PRZ_SWEEP_CHOCH')
class Cfg(StrategyConfig,frozen=True):
 instrument_id:InstrumentId;m1:BarType;m5:BarType;m15:BarType;h1:BarType;selected:str;unit_qty:Decimal=Decimal('1');horizon_minutes:int=180
class S(Strategy):
 def __init__(self,cfg):
  super().__init__(cfg);self.buf={1:[deque(maxlen=400) for _ in range(4)],5:[deque(maxlen=400) for _ in range(4)],15:[deque(maxlen=400) for _ in range(4)],60:[deque(maxlen=400) for _ in range(4)]};self.tr=deque(maxlen=400);self.prev=None;self.m1_i=0;self.pending=0;self.active=None;self.last_bid=self.last_ask=None;self.trades=[];self.gw=self.gl=0.;self.wins=self.losses=self.fire=0;self.harm=None
 @staticmethod
 def f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)
 def on_start(self):
  self.subscribe_quote_ticks(self.config.instrument_id)
  for bt in (self.config.m1,self.config.m5,self.config.m15,self.config.h1): self.subscribe_bars(bt)
 def atr(self): return float(np.mean(list(self.tr)[-20:])) if len(self.tr)>=20 else 0.
 def ema(self,tf,n,off=0):
  a=list(self.buf[tf][3]);end=len(a)-off
  if end<n:return None
  x=np.asarray(a[end-n:end],float);alpha=2/(n+1);v=float(x[0])
  for z in x[1:]:v=alpha*z+(1-alpha)*v
  return v
 def submit(self,side):
  inst=self.cache.instrument(self.config.instrument_id);q=inst.make_qty(self.config.unit_qty);self.submit_order(self.order_factory.market(instrument_id=self.config.instrument_id,order_side=OrderSide.BUY if side>0 else OrderSide.SELL,quantity=q))
 def signal(self):
  o,h,l,c=[np.asarray(self.buf[1][i],float) for i in range(4)];atr=self.atr();sel=self.config.selected
  if len(c)<100 or atr<=0:return 0
  if sel=='T2_ADAPTIVE_MTF':
   scores=[]
   for tf in (5,15,60):
    e21=self.ema(tf,21);e50=self.ema(tf,50);ep=self.ema(tf,21,1)
    if e21 is None or e50 is None or ep is None:continue
    slope=(e21-ep)/max(atr,1e-9);strength=abs(e21-e50)/max(atr,1e-9);side=1 if e21>e50 and slope>0 else -1 if e21<e50 and slope<0 else 0;scores.append((strength,side))
   if not scores:return 0
   strength,side=max(scores,key=lambda x:x[0])
   e1=self.ema(1,21);e1p=self.ema(1,21,1)
   ok=side>0 and c[-1]>e1 and e1>e1p or side<0 and c[-1]<e1 and e1<e1p
   if side and strength>=0.35 and ok:self.fire+=1;return side
  elif sel=='T5_COMP_EXP':
   rng=np.array([max(h[i]-l[i],1e-9) for i in range(len(c))]);base=float(np.mean(rng[-25:-5]));recent=float(np.mean(rng[-5:]));body=abs(c[-1]-o[-1]);hi=float(h[-21:-1].max());lo=float(l[-21:-1].min())
   compressed=base<=0.85*float(np.mean(rng[-80:-25]));expand=recent>=1.25*base and body>=0.55*atr
   buy=compressed and expand and c[-1]>hi;sell=compressed and expand and c[-1]<lo
   if buy or sell:self.fire+=1;return 1 if buy else -1
  elif sel=='R3_LIQUIDITY_RUN':
   hi=float(h[-31:-1].max());lo=float(l[-31:-1].min());width=hi-lo;e21=self.ema(1,21);e50=self.ema(1,50);ep=self.ema(1,21,1)
   if None in (e21,e50,ep) or width<=0:return 0
   contained=width<=5*atr;pos=(c[-1]-lo)/width;liq_hi=float(h[-8:-1].max());liq_lo=float(l[-8:-1].min())
   buy=contained and e21>e50 and e21>=ep and .45<=pos<=.90 and c[-1]>liq_hi and c[-1]<hi+0.25*atr
   sell=contained and e21<e50 and e21<=ep and .10<=pos<=.55 and c[-1]<liq_lo and c[-1]>lo-0.25*atr
   if buy or sell:self.fire+=1;return 1 if buy else -1
  elif sel=='V5_HARMONIC_PRZ_SWEEP_CHOCH':
   # deterministic swing-proxy XABCD: alternating local extrema over last ~40 bars, D near 1.27-1.68 AB extension and 0.786-0.886 XA retrace; then sweep+CHOCH
   if self.harm is None:
    Xh=float(h[-41:-31].max());Xl=float(l[-41:-31].min());Ah=float(h[-31:-21].max());Al=float(l[-31:-21].min());Bh=float(h[-21:-11].max());Bl=float(l[-21:-11].min());Dh=h[-1];Dl=l[-1]
    # bullish pattern proxy: X high -> A low -> B retrace high -> D extension low
    xa_bull=Xh-Al;ab_bull=Bh-Al
    bull=xa_bull>0 and ab_bull>0 and (Xh-Bh)/xa_bull>=0.114 and (Xh-Bh)/xa_bull<=0.55 and (Bh-Dl)/ab_bull>=1.27 and (Bh-Dl)/ab_bull<=1.68 and (Xh-Dl)/xa_bull>=0.786 and (Xh-Dl)/xa_bull<=0.95
    xa_bear=Ah-Xl;ab_bear=Ah-Bl
    bear=xa_bear>0 and ab_bear>0 and (Bl-Xl)/xa_bear>=0.114 and (Bl-Xl)/xa_bear<=0.55 and (Dh-Bl)/ab_bear>=1.27 and (Dh-Bl)/ab_bear<=1.68 and (Dh-Xl)/xa_bear>=0.786 and (Dh-Xl)/xa_bear<=0.95
    if bull:self.harm={'side':1,'i':self.m1_i,'sweep':Dl,'ref':float(h[-5:-1].max())};return 0
    if bear:self.harm={'side':-1,'i':self.m1_i,'sweep':Dh,'ref':float(l[-5:-1].min())};return 0
   st=self.harm
   if self.m1_i-st['i']>6:self.harm=None;return 0
   side=st['side'];choch=c[-1]>st['ref'] if side>0 else c[-1]<st['ref']
   if choch:self.harm=None;self.fire+=1;return side
  return 0
 def open(self,side,bid,ask):
  if self.active:return
  px=ask if side>0 else bid;atr=max(self.atr(),1e-9);self.submit(side);self.active={'side':side,'entry':px,'atr':atr,'i':self.m1_i,'best':px,'worst':px,'trail':None}
 def close(self,bid,ask,reason):
  a=self.active
  if not a:return
  side=a['side'];px=bid if side>0 else ask;pnl=(px-a['entry'])*side;self.submit(-side);mfe=(a['best']-a['entry']) if side>0 else (a['entry']-a['best']);mae=(a['entry']-a['worst']) if side>0 else (a['worst']-a['entry']);self.trades.append({'pnl':pnl,'mfe_atr':max(mfe,0)/a['atr'],'mae_atr':max(mae,0)/a['atr'],'reason':reason});self.gw+=max(pnl,0);self.gl+=max(-pnl,0);self.wins+=pnl>0;self.losses+=pnl<0;self.active=None
 def on_bar(self,b:Bar):
  s=str(b.bar_type);tf=60 if '-60-MINUTE-' in s else 15 if '-15-MINUTE-' in s else 5 if '-5-MINUTE-' in s else 1 if '-1-MINUTE-' in s else None
  if tf is None:return
  vals=list(map(self.f,[b.open,b.high,b.low,b.close]));[self.buf[tf][i].append(vals[i]) for i in range(4)]
  if tf==1:
   o,h,l,c=vals;self.m1_i+=1;tr=max(h-l,abs(h-self.prev) if self.prev is not None else 0,abs(l-self.prev) if self.prev is not None else 0);self.tr.append(tr);self.prev=c
   if not self.active and not self.pending:self.pending=self.signal()
 def on_quote_tick(self,t:QuoteTick):
  bid=self.f(t.bid_price);ask=self.f(t.ask_price);self.last_bid=bid;self.last_ask=ask
  if self.pending and not self.active:s=self.pending;self.pending=0;self.open(s,bid,ask)
  a=self.active
  if not a:return
  side=a['side'];mark=bid if side>0 else ask;atr=a['atr'];a['best']=max(a['best'],mark) if side>0 else min(a['best'],mark);a['worst']=min(a['worst'],mark) if side>0 else max(a['worst'],mark);mfe=((a['best']-a['entry']) if side>0 else (a['entry']-a['best']))/atr;move=(mark-a['entry'])*side
  if move<=-0.75*atr:self.close(bid,ask,'HARD_SL');return
  if mfe>=0.75:
   dist=(0.75 if mfe>=2 else 1.0 if mfe>=1.5 else 1.2)*atr;be=a['entry']+side*0.05*atr;cand=a['best']-dist if side>0 else a['best']+dist;cand=max(cand,be) if side>0 else min(cand,be);a['trail']=cand if a['trail'] is None else (max(a['trail'],cand) if side>0 else min(a['trail'],cand))
  if a['trail'] is not None and ((side>0 and mark<=a['trail']) or (side<0 and mark>=a['trail'])):self.close(bid,ask,'DYNAMIC_ATR_TRAIL');return
  if self.m1_i-a['i']>=self.config.horizon_minutes:self.close(bid,ask,'HORIZON')
 def on_stop(self):
  if self.active and self.last_bid is not None:self.close(self.last_bid,self.last_ask,'EOD')
 def summary(self):
  n=len(self.trades);net=sum(x['pnl'] for x in self.trades);pf=self.gw/self.gl if self.gl>0 else (math.inf if self.gw>0 else 0);r=Counter(x['reason'] for x in self.trades);return {'selected':self.config.selected,'N':n,'fire_count':self.fire,'WR_pct':100*self.wins/max(n,1),'PF':pf,'net_virtual':net,'expectancy':net/max(n,1),'MFE_ATR_mean':float(np.mean([x['mfe_atr'] for x in self.trades])) if n else 0,'MAE_ATR_mean':float(np.mean([x['mae_atr'] for x in self.trades])) if n else 0,'exit_reasons':dict(r)}
def l1(ticks):
 one=Quantity.from_int(1);out=[]
 for t in ticks:
  bs=float(t.bid_size.as_double()) if hasattr(t.bid_size,'as_double') else float(t.bid_size);az=float(t.ask_size.as_double()) if hasattr(t.ask_size,'as_double') else float(t.ask_size);out.append(QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=one if bs<=0 else t.bid_size,ask_size=one if az<=0 else t.ask_size,ts_event=t.ts_event,ts_init=t.ts_init) if bs<=0 or az<=0 else t)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--selected',choices=EDGES,required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value]);ticks=l1(raw);eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)));eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(ticks);sym=inst.id.value;st=S(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{sym}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{sym}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{sym}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{sym}-60-MINUTE-BID-INTERNAL'),selected=a.selected));eng.add_strategy(st);eng.run();eng.end();res={'verification_level':'NAUTILUS_RAW_NEXT4_DYNAMIC_ATR','raw_ticks':len(raw),'ohlc_resample_used':False,'exit_model':'DYNAMIC_ATR_NO_FIXED_TP',**st.summary()};p=Path('results/multiedge')/a.experiment_id/f'{a.selected}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
if __name__=='__main__':main()
