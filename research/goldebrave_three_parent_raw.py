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
from goldebrave_fasttf_parity_raw import GB,Cfg,f,l1,met

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
 def _q(self,identifiers=None,start=None,end=None): return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
 ParquetDataCatalog.query_quote_ticks=_q

class ThreeParent(GB):
 def __init__(self,c):
  super().__init__(c)
  self.n2_pending=[];self.n2_active=[];self.n2_ledger=[]
  self.n3_pending=[];self.n3_active=[];self.n3_ledger=[]
  self.n2_signals=0;self.n3_signals=0
  self.m1=deque(maxlen=500);self.m5=deque(maxlen=300);self.m15=deque(maxlen=300)
  self.last_n2_key=None;self.last_n3_key=None

 def _atr_local(self,arr,n=14):
  if len(arr)<n+1:return None
  a=list(arr);tr=[]
  for i in range(len(a)-n,len(a)):
   o,h,l,c=a[i];pc=a[i-1][3];tr.append(max(h-l,abs(h-pc),abs(l-pc)))
  return float(np.mean(tr)) if tr else None

 def _manage(self,bid,ask,active,ledger):
  nxt=[]
  for a in active:
   side=a['side'];mark=bid if side>0 else ask;mv=(mark-a['entry'])*side
   if mv>=a['be_at']:
    a['stop']=max(a['stop'],a['entry']+0.10*a['vol']) if side>0 else min(a['stop'],a['entry']-0.10*a['vol'])
   close=(mark<=a['stop'] if side>0 else mark>=a['stop']) or (mark>=a['take'] if side>0 else mark<=a['take'])
   if close:ledger.append(mv)
   else:nxt.append(a)
  return nxt

 def on_quote_tick(self,t:QuoteTick):
  bid=f(t.bid_price);ask=f(t.ask_price)
  # fills before baseline handler, independent screening ledgers
  for pending,active in ((self.n2_pending,self.n2_active),(self.n3_pending,self.n3_active)):
   keep=[]
   for x in pending:
    side,p,sl,tp,vol=x;hit=ask>=p if side>0 else bid<=p
    if hit:
     e=ask if side>0 else bid
     active.append({'side':side,'entry':e,'stop':e-side*sl,'take':e+side*tp,'be_at':1.0*vol,'vol':vol})
    else:keep.append(x)
   pending[:] = keep
  self.n2_active=self._manage(bid,ask,self.n2_active,self.n2_ledger)
  self.n3_active=self._manage(bid,ask,self.n3_active,self.n3_ledger)
  super().on_quote_tick(t)

 def _n2_scan(self):
  # N2: liquidity sweep -> M1 CHOCH/MSS reversal.
  # Use previous 12 M15 bars as liquidity boundary, then require latest M15 sweep/reclaim.
  if len(self.m15)<14 or len(self.m1)<8:return
  vol,reg,a=self.vol_reg()
  if a is None:return
  m15=list(self.m15);prev=m15[-13:-1];cur=m15[-1]
  ph=max(x[1] for x in prev);pl=min(x[2] for x in prev);o,h,l,c=cur
  m1=list(self.m1);recent=m1[-7:]
  prior_hi=max(x[1] for x in recent[:-1]);prior_lo=min(x[2] for x in recent[:-1]);last=recent[-1]
  # sell: swept prior high, reclaimed below, then M1 close breaks short-term low
  side=0
  if h>ph and c<ph and last[3]<prior_lo: side=-1
  # buy: swept prior low, reclaimed above, then M1 close breaks short-term high
  elif l<pl and c>pl and last[3]>prior_hi: side=1
  if side==0:return
  key=(side,round(ph if side<0 else pl,2),len(self.b[15][3]))
  if key==self.last_n2_key:return
  self.last_n2_key=key
  entry=last[3]
  sl=min(max(a*1.0,3.0),10.0);tp=min(max(a*2.0,7.0),24.0)
  self.n2_pending.append((side,entry,sl,tp,vol));self.n2_pending=self.n2_pending[-12:];self.n2_signals+=1

 def _n3_scan(self):
  # N3: displacement -> retrace into 50-79% of displacement -> continuation break.
  # M5 defines displacement/retrace; H1 regime must not oppose direction.
  if len(self.m5)<6:return
  vol,reg,a=self.vol_reg()
  if a is None:return
  x=list(self.m5);d=x[-3];r=x[-2];b=x[-1]
  do,dh,dl,dc=d;ro,rh,rl,rc=r;bo,bh,bl,bc=b
  rng=dh-dl;body=abs(dc-do);atr5=self._atr_local(self.m5,14)
  if atr5 is None or rng<1.25*atr5 or body/max(rng,1e-9)<0.60:return
  side=1 if dc>do else -1 if dc<do else 0
  if side==0 or (reg==1 and side<0):return
  lo=min(do,dc);hi=max(do,dc);span=hi-lo
  z50=hi-.50*span if side>0 else lo+.50*span
  z79=hi-.79*span if side>0 else lo+.79*span
  zlo=min(z50,z79);zhi=max(z50,z79)
  retrace=(rl<=zhi and rh>=zlo)
  rebreak=(bc>dh if side>0 else bc<dl)
  if not (retrace and rebreak):return
  key=(side,round(dh if side>0 else dl,2),len(self.b[5][3]))
  if key==self.last_n3_key:return
  self.last_n3_key=key
  entry=bc;sl=min(max(a*.9,3.0),9.0);tp=min(max(a*2.1,8.0),26.0)
  self.n3_pending.append((side,entry,sl,tp,vol));self.n3_pending=self.n3_pending[-12:];self.n3_signals+=1

 def on_bar(self,bar:Bar):
  s=str(bar.bar_type);tf=60 if '-1-HOUR-' in s else 15 if '-15-MINUTE-' in s else 5 if '-5-MINUTE-' in s else 1 if '-1-MINUTE-' in s else None
  if tf is not None:
   o,h,l,c=map(f,[bar.open,bar.high,bar.low,bar.close])
   if tf==1:self.m1.append((o,h,l,c))
   elif tf==5:self.m5.append((o,h,l,c))
   elif tf==15:self.m15.append((o,h,l,c))
  super().on_bar(bar)
  if tf is None:return
  # Preserve same trading-hour envelope as N1.
  if getattr(self,'cur_hour',-1) not in (11,15,16,17,18):
   self.n2_pending=[];self.n3_pending=[];return
  if tf==15:self._n2_scan()
  if tf==5:self._n3_scan()

 def on_stop(self):
  super().on_stop()
  for active,ledger in ((self.n2_active,self.n2_ledger),(self.n3_active,self.n3_ledger)):
   for a in active:
    mark=self.last_bid if a['side']>0 else self.last_ask;ledger.append((mark-a['entry'])*a['side'])

 def summary_three(self):
  base=self.summary();n1=base['per_layer'].get('C',met([]))
  return {'N1_GoldeBrave_C':n1,'N2_Sweep_CHOCH':met(self.n2_ledger),'N3_Continuation':met(self.n3_ledger),'N2_signals':self.n2_signals,'N3_signals':self.n3_signals,'baseline_all_layers':base}

def run(catalog,experiment_id):
 cat=ParquetDataCatalog(catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value])
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(l1(raw));s=inst.id.value
 st=ThreeParent(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{s}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{s}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{s}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{s}-1-HOUR-BID-INTERNAL'),mode='m15'))
 eng.add_strategy(st);eng.run();eng.end();res={'verification':'GOLDEBRAVE_THREE_PARENT_RAW_V1','raw_ticks':len(raw),'ohlc_resample_used':False,'design':['N1 fixed GoldeBrave C','N2 M15 liquidity sweep + M1 CHOCH/MSS reversal','N3 M5 displacement + 50-79% retrace + continuation rebreak'],'limitation':'N2/N3 are parent screening models with virtual PnL/broker semantics; G75 not yet attached.',**st.summary_three()}
 p=Path('results/goldebrave-three-parent')/experiment_id/'parents.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();run(a.catalog,a.experiment_id)
if __name__=='__main__':main()
