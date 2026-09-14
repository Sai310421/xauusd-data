from __future__ import annotations
import argparse,json
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
from goldebrave_three_parent_raw import ThreeParent

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
 def _q(self,identifiers=None,start=None,end=None): return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
 ParquetDataCatalog.query_quote_ticks=_q

class ThreeParentV2(ThreeParent):
 def __init__(self,c):
  super().__init__(c)
  self.n2_setup=None
  self.n3_setup=None
  self.n2_diag={'sweeps':0,'cisd':0,'mss':0,'disp':0,'fvg_ote':0}
  self.n3_diag={'disp':0,'retrace':0,'reaccel':0}

 def _atr_arr(self,arr,n=14):
  if len(arr)<n+1:return None
  a=list(arr);trs=[]
  for i in range(len(a)-n,len(a)):
   o,h,l,c=a[i];pc=a[i-1][3];trs.append(max(h-l,abs(h-pc),abs(l-pc)))
  return float(np.mean(trs)) if trs else None

 def _detect_n2_sweep(self):
  if len(self.m15)<14:return
  m15=list(self.m15);prev=m15[-13:-1];cur=m15[-1]
  ph=max(x[1] for x in prev);pl=min(x[2] for x in prev);o,h,l,c=cur
  side=0;liq=None
  if h>ph and c<ph: side=-1;liq=ph
  elif l<pl and c>pl: side=1;liq=pl
  if side==0:return
  self.n2_diag['sweeps']+=1
  self.n2_setup={'side':side,'liq':liq,'sweep_hi':h,'sweep_lo':l,'created_m1':len(self.m1),'stage':'cisd','ttl':24,'disp_hi':None,'disp_lo':None,'fvg':None}

 def _scan_n2_m1(self):
  s=self.n2_setup
  if not s or len(self.m1)<8:return
  s['ttl']-=1
  if s['ttl']<=0:self.n2_setup=None;return
  x=list(self.m1);cur=x[-1];prev=x[-2];side=s['side'];o,h,l,c=cur
  recent=x[-7:-1]
  prior_hi=max(z[1] for z in recent);prior_lo=min(z[2] for z in recent)
  atr=self._atr_arr(self.m1,14)
  if atr is None:return
  body=abs(c-o);rng=max(h-l,1e-9)
  # CISD proxy: close crosses previous candle open in reversal direction.
  cisd=(c>prev[0] if side>0 else c<prev[0])
  if s['stage']=='cisd' and cisd:
   s['stage']='mss';self.n2_diag['cisd']+=1
  if s['stage']=='mss':
   mss=(c>prior_hi if side>0 else c<prior_lo)
   if mss:
    s['stage']='disp';self.n2_diag['mss']+=1
  if s['stage']=='disp':
   disp=(rng>=1.15*atr and body/rng>=0.60 and ((c>o) if side>0 else (c<o)))
   if disp:
    s['disp_hi']=h;s['disp_lo']=l
    # three-candle FVG around the displacement leg if present.
    a=x[-3];fvg=None
    if side>0 and l>a[1]:fvg=(a[1],l)
    elif side<0 and h<a[2]:fvg=(h,a[2])
    s['fvg']=fvg;s['stage']='retrace';s['ttl']=12;self.n2_diag['disp']+=1
    return
  if s['stage']=='retrace':
   lo=s['disp_lo'];hi=s['disp_hi'];span=max(hi-lo,1e-9)
   # OTE 62-79% of displacement, plus FVG overlap if FVG exists.
   if side>0:ote=(hi-.79*span,hi-.62*span)
   else:ote=(lo+.62*span,lo+.79*span)
   zlo,zhi=min(ote),max(ote)
   touched=(l<=zhi and h>=zlo)
   if s['fvg'] is not None:
    flo,fhi=min(s['fvg']),max(s['fvg']);touched=touched and (l<=fhi and h>=flo)
   if not touched:return
   self.n2_diag['fvg_ote']+=1
   # confirmation close back in reversal direction after touch.
   confirm=(c>o if side>0 else c<o)
   if not confirm:return
   vol,reg,a=self.vol_reg()
   if a is None:return
   entry=c;sl=min(max(a*.85,2.5),8.0);tp=min(max(a*2.0,7.0),24.0)
   self.n2_pending.append((side,entry,sl,tp,vol));self.n2_pending=self.n2_pending[-12:]
   self.n2_signals+=1;self.n2_setup=None

 def _scan_n3_m5(self):
  if len(self.m5)<16:return
  x=list(self.m5);cur=x[-1];atr=self._atr_arr(self.m5,14)
  if atr is None:return
  o,h,l,c=cur;rng=max(h-l,1e-9);body=abs(c-o)
  if self.n3_setup is None:
   side=1 if c>o else -1 if c<o else 0
   # cleaner displacement seed, but lower than v1 to improve N.
   if side and rng>=1.05*atr and body/rng>=0.55:
    self.n3_setup={'side':side,'lo':l,'hi':h,'stage':'retrace','ttl':8}
    self.n3_diag['disp']+=1
   return
  s=self.n3_setup;s['ttl']-=1
  if s['ttl']<=0:self.n3_setup=None;return
  side=s['side'];lo=s['lo'];hi=s['hi'];span=max(hi-lo,1e-9)
  if s['stage']=='retrace':
   if side>0:zone=(hi-.79*span,hi-.50*span)
   else:zone=(lo+.50*span,lo+.79*span)
   zlo,zhi=min(zone),max(zone)
   if l<=zhi and h>=zlo:
    s['stage']='reaccel';s['ttl']=5;self.n3_diag['retrace']+=1
   return
  if s['stage']=='reaccel':
   # reacceleration = directional close + break of prior M5 high/low, not necessarily full displacement high/low.
   p=x[-2]
   ok=(c>o and c>p[1]) if side>0 else (c<o and c<p[2])
   if not ok:return
   self.n3_diag['reaccel']+=1
   vol,reg,a=self.vol_reg()
   if a is None:return
   # do not trade directly against H1 trend regime.
   if (reg==1 and side<0) or (reg==-1 and side>0):self.n3_setup=None;return
   entry=c;sl=min(max(a*.8,2.5),8.0);tp=min(max(a*1.9,7.0),23.0)
   self.n3_pending.append((side,entry,sl,tp,vol));self.n3_pending=self.n3_pending[-12:]
   self.n3_signals+=1;self.n3_setup=None

 def on_bar(self,bar:Bar):
  s=str(bar.bar_type);tf=60 if '-1-HOUR-' in s else 15 if '-15-MINUTE-' in s else 5 if '-5-MINUTE-' in s else 1 if '-1-MINUTE-' in s else None
  if tf is not None:
   o,h,l,c=map(f,[bar.open,bar.high,bar.low,bar.close])
   if tf==1:self.m1.append((o,h,l,c))
   elif tf==5:self.m5.append((o,h,l,c))
   elif tf==15:self.m15.append((o,h,l,c))
  # Call N1 baseline handler directly; bypass v1 N2/N3 scanners.
  GB.on_bar(self,bar)
  if tf is None:return
  if getattr(self,'cur_hour',-1) not in (11,15,16,17,18):
   self.n2_pending=[];self.n3_pending=[];self.n2_setup=None;self.n3_setup=None;return
  if tf==15:self._detect_n2_sweep()
  if tf==1:self._scan_n2_m1()
  if tf==5:self._scan_n3_m5()

 def summary_three_v2(self):
  base=self.summary();n1=base['per_layer'].get('C',met([]))
  return {'N1_GoldeBrave_C':n1,'N2_v2_Sweep_CISD_MSS_Displacement_OTE':met(self.n2_ledger),'N3_v2_Displacement_Fib_Reaccel':met(self.n3_ledger),'N2_signals':self.n2_signals,'N3_signals':self.n3_signals,'N2_diag':self.n2_diag,'N3_diag':self.n3_diag,'baseline_all_layers':base}

def run(catalog,experiment_id):
 cat=ParquetDataCatalog(catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value])
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(l1(raw));sym=inst.id.value
 st=ThreeParentV2(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{sym}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{sym}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{sym}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{sym}-1-HOUR-BID-INTERNAL'),mode='m15'))
 eng.add_strategy(st);eng.run();eng.end();res={'verification':'GOLDEBRAVE_THREE_PARENT_RAW_V2','raw_ticks':len(raw),'ohlc_resample_used':False,'design':['N1 fixed GoldeBrave C','N2 v2 M15 Sweep -> M1 CISD -> MSS/CHOCH -> displacement -> FVG+OTE touch -> reversal confirm','N3 v2 M5 displacement -> Fib 50-79 retrace -> directional reacceleration'],'limitation':'N2/N3 are parent screening models with virtual PnL/broker semantics; G75 is intentionally not attached until parent EDGE is positive.',**st.summary_three_v2()}
 p=Path('results/goldebrave-three-parent-v2')/experiment_id/'parents.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();run(a.catalog,a.experiment_id)
if __name__=='__main__':main()
