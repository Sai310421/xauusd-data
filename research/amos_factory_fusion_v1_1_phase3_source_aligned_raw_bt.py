from __future__ import annotations
import argparse,json,math
from collections import deque
from decimal import Decimal
from pathlib import Path
import numpy as np,pandas as pd,nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money,Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar,QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
 def _q(self,identifiers=None,start=None,end=None): return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
 ParquetDataCatalog.query_quote_ticks=_q

SIM=Venue('SIM'); CONTRACT=100.0
P=dict(risk=.007,max_hold=72,min_session_bars=12,max_adx=30.,edge=.08,eq_zone=.06,sigma_mult=2.,sl_atr=.7,rr=1.8,partial_eq=.0055,partial_frac=.5,day_dd=.03,hours=(1,13),weekdays=(0,1,2),commission=7.,slip=.04,tokyo_start_min=15,tokyo_end_min=1185,cycle_target=.012,cycle_dd=.03,max_legs_cycle=4)
MODES=('REFERENCE','EQ_SESSION','ER020','ER030','ER040','CYCLE_SIGNAL','CYCLE_SAME','SET_REVERSE')

class Cfg(StrategyConfig,frozen=True):
 instrument_id:InstrumentId; bar_type:BarType; mode:str

class S(Strategy):
 def __init__(self,c):
  super().__init__(c);self.b=deque(maxlen=600);self.s=[];self.sd=None;self.bc=0;self.sb=0;self.arm=None;self.entry=None;self.side=0;self.sl=None;self.tp=None;self.qty=0.;self.eb=0;self.partial=False;self.pending=False;self.real=0.;self.peak=1000.;self.ds=1000.;self.day=None;self.halt=False;self.mdd=0.;self.dday=0.;self.legs=[];self.entries=0;self.signals=0;self.blocks={k:0 for k in ('hour','weekday','warmup','adx','edge','stale','dd','cycle_limit')};self.cycles=[];self.cycle_id=0;self.cycle_active=False;self.cycle_start_eq=0.;self.cycle_pnl=0.;self.cycle_legs=0;self.cycle_side=0;self.cycle_peak=0.;self.cycle_mdd=0.
 @staticmethod
 def f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)
 def on_start(self): self.subscribe_quote_ticks(self.config.instrument_id);self.subscribe_bars(self.config.bar_type)
 def atr(self,n=14):
  if len(self.b)<n+1:return None
  xs=list(self.b);tr=[max(xs[i]['h']-xs[i]['l'],abs(xs[i]['h']-xs[i-1]['c']),abs(xs[i]['l']-xs[i-1]['c'])) for i in range(1,len(xs))]
  a=tr[0]
  for z in tr[1:]:a=z/n+(1-1/n)*a
  return a
 def er(self,n=250):
  if len(self.b)<n+1:return None
  x=np.asarray([z['c'] for z in self.b],float)
  seg=x[-n-1:]
  den=np.abs(np.diff(seg)).sum()
  return float(abs(seg[-1]-seg[0])/den) if den>0 else 1.0
 def adx(self,n=14):
  if len(self.b)<n+2:return None
  xs=list(self.b);tr=[];pl=[];mi=[]
  for i in range(1,len(xs)):
   up=xs[i]['h']-xs[i-1]['h'];dn=xs[i-1]['l']-xs[i]['l'];pl.append(up if up>dn and up>0 else 0.);mi.append(dn if dn>up and dn>0 else 0.);tr.append(max(xs[i]['h']-xs[i]['l'],abs(xs[i]['h']-xs[i-1]['c']),abs(xs[i]['l']-xs[i-1]['c'])))
  a=1/n;at=tr[0];p=pl[0];m=mi[0];dx=[]
  for i in range(1,len(tr)):
   at=a*tr[i]+(1-a)*at;p=a*pl[i]+(1-a)*p;m=a*mi[i]+(1-a)*m
   if at>0:
    pi=100*p/at;md=100*m/at;dx.append(100*abs(pi-md)/(pi+md) if pi+md>0 else 0.)
  if not dx:return None
  v=dx[0]
  for z in dx[1:]:v=a*z+(1-a)*v
  return v
 def vs(self):
  if len(self.s)<24:return None,None
  vwap=self.s[-1]['vwap'];dev=[x['c']-x['vwap'] for x in self.s[-96:]]
  if len(dev)<24:return vwap,None
  sd=float(np.std(dev,ddof=1))
  return (vwap,sd) if math.isfinite(sd) and sd>0 else (vwap,None)
 def eq(self,bid,ask):
  fl=0 if self.entry is None else ((bid if self.side>0 else ask)-self.entry)*self.side*self.qty
  return 1000+self.real+fl
 def upd(self,ts,bid,ask):
  e=self.eq(bid,ask)
  if self.day!=ts.date():self.day=ts.date();self.ds=e;self.halt=False
  self.peak=max(self.peak,e);self.mdd=max(self.mdd,self.peak-e)
  if self.ds>0:self.dday=max(self.dday,(self.ds-e)/self.ds);self.halt=self.halt or (self.ds-e)/self.ds>=P['day_dd']
  return e
 def start_cycle(self,e,side):
  if self.cycle_active:return
  self.cycle_id+=1;self.cycle_active=True;self.cycle_start_eq=float(e);self.cycle_pnl=0.;self.cycle_legs=0;self.cycle_side=int(side);self.cycle_peak=float(e);self.cycle_mdd=0.
 def cycle_mark(self,e):
  if not self.cycle_active:return
  self.cycle_peak=max(self.cycle_peak,float(e));self.cycle_mdd=max(self.cycle_mdd,self.cycle_peak-float(e))
 def finish_cycle(self,reason):
  if not self.cycle_active:return
  self.cycles.append(dict(cycle_id=self.cycle_id,pnl=float(self.cycle_pnl),legs=int(self.cycle_legs),reason=reason,start_eq=float(self.cycle_start_eq),dd_pct=float(self.cycle_mdd/max(self.cycle_peak,1e-9)*100),win=bool(self.cycle_pnl>0)))
  self.cycle_active=False;self.cycle_start_eq=0.;self.cycle_pnl=0.;self.cycle_legs=0;self.cycle_side=0;self.cycle_peak=0.;self.cycle_mdd=0.
 def cycle_done(self):
  if not self.cycle_active:return False
  if self.cycle_pnl>=P['cycle_target']*self.cycle_start_eq:self.finish_cycle('target');return True
  if self.cycle_pnl<=-P['cycle_dd']*self.cycle_start_eq:self.finish_cycle('dd');return True
  if self.cycle_legs>=P['max_legs_cycle']:self.finish_cycle('max_legs');return True
  return False
 def book(self,px,q,tag):
  q=min(q,self.qty);lots=q/CONTRACT;pnl=(px-self.entry)*self.side*q-P['slip']*q-P['commission']*lots/2;self.real+=pnl;self.qty-=q;self.legs.append(dict(pnl=float(pnl),tag=tag,qty_oz=float(q),cycle_id=int(self.cycle_id if self.cycle_active else 0)));self.cycle_pnl+=pnl if self.cycle_active else 0.
 def reduce(self,q):
  q=max(0,int(round(min(q,self.qty))))
  if q<=0:return False
  ins=self.cache.instrument(self.config.instrument_id);o=self.order_factory.market(instrument_id=self.config.instrument_id,order_side=OrderSide.SELL if self.side>0 else OrderSide.BUY,quantity=ins.make_qty(Decimal(q)));self.submit_order(o);return True
 def on_bar(self,bar:Bar):
  self.bc+=1;ts=pd.Timestamp(int(bar.ts_event),unit='ns',tz='UTC');o,h,l,c=map(self.f,[bar.open,bar.high,bar.low,bar.close])
  try:v=max(self.f(bar.volume),1e-9)
  except:v=1.
  self.b.append(dict(o=o,h=h,l=l,c=c))
  skey=ts.date()
  if (self.config.mode=='EQ_SESSION' or self.config.mode.startswith('ER') or self.config.mode.startswith('CYCLE')) and ts.hour*60+ts.minute<P['tokyo_start_min']: skey=(ts-pd.Timedelta(days=1)).date()
  if skey!=self.sd:self.sd=skey;self.s=[];self.sb=0
  self.sb+=1;tp=(h+l+c)/3;pv=sum(x['tp']*x['v'] for x in self.s)+tp*v;vv=sum(x['v'] for x in self.s)+v;vw=pv/vv;self.s.append(dict(tp=tp,v=v,c=c,vwap=vw))
  if self.entry is not None:return
  if self.halt:self.blocks['dd']+=1;return
  if ts.hour not in P['hours']:self.blocks['hour']+=1;return
  if ts.weekday() not in P['weekdays']:self.blocks['weekday']+=1;return
  if self.sb-1<P['min_session_bars']:self.blocks['warmup']+=1;return
  atr=self.atr();adx=self.adx();vw,sd=self.vs()
  if atr is None or adx is None or sd is None:self.blocks['warmup']+=1;return
  if adx>=P['max_adx']:self.blocks['adx']+=1;return
  if self.config.mode.startswith('ER'):
   erv=self.er(250)
   if erv is None:self.blocks['warmup']+=1;return
   thr={'ER020':.20,'ER030':.30,'ER040':.40}[self.config.mode]
   if erv>thr:self.blocks['adx']+=1;return
  band=P['sigma_mult']*sd;edge=P['edge']*band;z=c-vw;d=-1 if z>=band-edge else (1 if z<=-band+edge else 0)
  if d==0:self.blocks['edge']+=1;return
  if self.config.mode=='SET_REVERSE':d=-d
  if self.config.mode=='CYCLE_SAME' and self.cycle_active and self.cycle_side!=0:d=self.cycle_side
  self.signals+=1;self.arm=dict(bar=self.bc,side=d,atr=atr)
 def on_quote_tick(self,t:QuoteTick):
  bid,ask=self.f(t.bid_price),self.f(t.ask_price);ts=pd.Timestamp(int(t.ts_event),unit='ns',tz='UTC');e=self.upd(ts,bid,ask);self.cycle_mark(e);flat=not self.portfolio.is_net_long(self.config.instrument_id) and not self.portfolio.is_net_short(self.config.instrument_id)
  if self.halt and self.entry is not None and not self.pending:
   px=bid if self.side>0 else ask;self.book(px,self.qty,'daily_dd');self.close_all_positions(self.config.instrument_id);self.reset_state();self.finish_cycle('daily_dd');return
  if self.arm and self.entry is None and flat:
   if self.arm['bar']!=self.bc:self.blocks['stale']+=1;self.arm=None;return
   d=self.arm['side'];risk=P['sl_atr']*self.arm['atr'];self.start_cycle(e,d) if self.config.mode.startswith('CYCLE') else None;lots=max(.01,round(e*P['risk']/(risk*CONTRACT),2));q=max(1,int(round(lots*CONTRACT)));ins=self.cache.instrument(self.config.instrument_id);od=self.order_factory.market(instrument_id=self.config.instrument_id,order_side=OrderSide.BUY if d>0 else OrderSide.SELL,quantity=ins.make_qty(Decimal(q)));self.submit_order(od);px=(ask+P['slip']) if d>0 else (bid-P['slip']);self.entry=px;self.side=d;self.sl=px-d*risk;self.tp=px+d*P['rr']*risk;self.qty=float(q);self.eb=self.bc;self.partial=False;self.pending=False;self.entries+=1;self.real-=P['commission']*(q/CONTRACT)/2;self.cycle_pnl-=P['commission']*(q/CONTRACT)/2 if self.cycle_active else 0.;self.cycle_legs+=1 if self.cycle_active else 0;self.arm=None;return
  if self.entry is None or self.pending:return
  px=bid if self.side>0 else ask
  if self.config.mode=='EQ_SESSION' or self.config.mode.startswith('ER') or self.config.mode.startswith('CYCLE'):
   vw,sd=self.vs()
   if vw is not None and sd is not None:
    eqw=P['eq_zone']*P['sigma_mult']*sd
    if abs(px-vw)<=eqw:
     self.book(px,self.qty,'vwap_eq');self.close_all_positions(self.config.instrument_id);self.reset_state();self.cycle_done() if self.config.mode.startswith('CYCLE') else None;return
  if not self.partial and (px-self.entry)*self.side*self.qty>=P['partial_eq']*max(e,1e-9):
   q=self.qty*P['partial_frac']
   if self.reduce(q):self.book(px,q,'partial');self.partial=True;return
  sl=(px<=self.sl if self.side>0 else px>=self.sl);tp=(px>=self.tp if self.side>0 else px<=self.tp);tm=self.bc-self.eb>=P['max_hold']
  if sl or tp or tm:self.book(px,self.qty,'sl' if sl else ('tp' if tp else 'time'));self.close_all_positions(self.config.instrument_id);self.reset_state();self.cycle_done() if self.config.mode.startswith('CYCLE') else None
 def reset_state(self):self.entry=None;self.side=0;self.sl=self.tp=None;self.qty=0.;self.pending=False;self.partial=False
 def on_position_closed(self,e):self.reset_state()
 def on_stop(self):
  if self.entry is not None and self.qty>0:self.close_all_positions(self.config.instrument_id)
  if self.cycle_active:self.finish_cycle('eod')

def met(xs,days):
 a=np.array([x['pnl'] for x in xs],float)
 if not len(a):return dict(N_legs=0,WR_legs_pct=0.,PF_legs=0.,NetProfit=0.,Monthly21_pct=0.)
 w=a[a>0].sum();l=abs(a[a<0].sum());pf=w/l if l else (math.inf if w else 0.);eq=1000+a.sum();return dict(N_legs=int(len(a)),WR_legs_pct=float((a>0).mean()*100),PF_legs=float(pf),NetProfit=float(a.sum()),FinalEquity=float(eq),Monthly21_pct=float(((max(eq,1e-9)/1000)**(21/days)-1)*100))
def cycle_met(xs):
 if not xs:return dict(N_cycles=0,WR_cycle_pct=0.,PF_cycle=0.,NetCycle=0.,MaxCycleDD_pct=0.,TailLoss=0.)
 a=np.array([x['pnl'] for x in xs],float);w=a[a>0].sum();l=abs(a[a<0].sum());pf=w/l if l else (math.inf if w else 0.)
 return dict(N_cycles=int(len(a)),WR_cycle_pct=float((a>0).mean()*100),PF_cycle=float(pf),NetCycle=float(a.sum()),MaxCycleDD_pct=float(max([x['dd_pct'] for x in xs] or [0.])),TailLoss=float(a.min() if len(a) else 0.))
def run(cat,inst,ticks,man,mode,out):
 e=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)));e.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));e.add_instrument(inst);e.add_data(ticks);bt=BarType.from_str(f'{inst.id.value}-5-MINUTE-BID-INTERNAL');s=S(Cfg(instrument_id=inst.id,bar_type=bt,mode=mode));e.add_strategy(s);e.run();r=dict(entries=s.entries,signals=s.signals,metrics=met(s.legs,int(man['days'])),cycle_metrics=cycle_met(s.cycles),max_floating_dd_pct=float(s.mdd/max(s.peak,1e-9)*100),max_daily_loss_pct=float(s.dday*100),blocks=s.blocks,realized=float(s.real));pd.DataFrame(s.legs).to_csv(out/f'{mode.lower()}_legs.csv',index=False);pd.DataFrame(s.cycles).to_csv(out/f'{mode.lower()}_cycles.csv',index=False);e.dispose();return r
def main():
 a=argparse.ArgumentParser();a.add_argument('--catalog',required=True);a.add_argument('--experiment-id',required=True);a.add_argument('--raw-bidask-only',action='store_true');a.add_argument('--mode',choices=MODES,required=True);x=a.parse_args()
 if not x.raw_bidask_only:raise SystemExit('raw-bidask-only mandatory')
 cp=Path(x.catalog);man=json.loads((cp/'catalog_manifest.json').read_text());cat=ParquetDataCatalog(str(cp));inst=next(z for z in cat.instruments() if z.id.symbol.value.replace('/','')=='XAUUSD');ticks=cat.query_quote_ticks(identifiers=[inst.id.value]);out=Path('results/amos_factory_fusion_v1_1')/x.experiment_id;out.mkdir(parents=True,exist_ok=True);m={x.mode:run(cat,inst,ticks,man,x.mode,out)};s=dict(verification_level='NAUTILUS_BT_RAW_BIDASK_PHASE3_SOURCE_ALIGNED_AB',nautilus_version=getattr(nautilus_trader,'__version__','unknown'),raw_tick_count=len(ticks),period=dict(start=man['start'],days=man['days'],end_exclusive=man['end_exclusive']),source_alignment=dict(reference='Sai310421/glowing-octo-disco scripts/pci_vwap_bt.py',known_discrepancy='set reverse_signal=true vs reference direct VWAP fade; both tested'),modes=m,limitations=['Native spread replaces reference fixed spread; extra 0.04/side slippage and $7/lot commission charged manually.','Internal M5 quote-derived volume may differ from original OrkAD parquet tick volume.']);(out/'summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
if __name__=='__main__':main()
