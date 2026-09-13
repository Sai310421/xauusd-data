from __future__ import annotations
import argparse,json
from decimal import Decimal
from pathlib import Path
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType,OmsType,BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from goldebrave_bpd_pressure_rescue_raw import BPD
from goldebrave_fasttf_parity_raw import Cfg,l1,met

class BPDV2(BPD):
 def __init__(self,c,pressure_gate:float):
  super().__init__(c,pressure_gate)
  self.zone_candidates=0
  self.pressure_pass=0

 def _rescue_scan(self):
  dr=self.dayrange();vol,reg,a=self.vol_reg()
  if dr is None or a is None:return
  dhi,dlo=dr;press=self._pressure();band=.40*vol;off=.30*vol
  sl=min(max(a*1.2,4.0),12.0);tp=min(max(a*2.4,9.0),30.0)*(1.25 if reg==1 else .8 if reg==-1 else 1.0)
  hi,lo=self.extrema(15)
  buys=[z for z in reversed(hi[-40:]) if (dhi-band)<=z<=(dhi+band)]
  sells=[z for z in reversed(lo[-40:]) if (dlo-band)<=z<=(dlo+band)]
  self.zone_candidates += len(buys)+len(sells)
  if press>=self.pressure_gate and buys:
   self.pressure_pass+=1
   z=buys[0];p=max(z-off,dhi-.15*vol)
   if all(abs(p-q[1])>=.15*vol for q in self.bp_rescue_pending):
    self.bp_rescue_pending.append((1,p,sl,tp));self.rescue_signals+=1
  if press<=100.0-self.pressure_gate and sells:
   self.pressure_pass+=1
   z=sells[0];p=min(z+off,dlo+.15*vol)
   if all(abs(p-q[1])>=.15*vol for q in self.bp_rescue_pending):
    self.bp_rescue_pending.append((-1,p,sl,tp));self.rescue_signals+=1
  self.bp_rescue_pending=self.bp_rescue_pending[-20:]

 def summary_bpd(self):
  base=self.summary()
  return {'version':'BP-D-v2','pressure_gate_pct':self.pressure_gate,'pressure_rule':'BigPlayer recent event ratio; buy >= gate, sell <= 100-gate','rescue_zone':'M15 confirmed swing within +/-0.40*vol of current day high/low boundary; original >0.40*vol breakout candidates remain baseline','zone_candidates':self.zone_candidates,'pressure_pass_scans':self.pressure_pass,'rescue_signals':self.rescue_signals,'rescue':met(self.bp_ledger),'baseline':base}

def run(catalog,gate,experiment_id):
 cat=ParquetDataCatalog(catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value])
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(l1(raw));s=inst.id.value
 st=BPDV2(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{s}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{s}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{s}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{s}-1-HOUR-BID-INTERNAL'),mode='m15'),gate)
 eng.add_strategy(st);eng.run();eng.end();res={'verification':'GOLDEBRAVE_BPD_PRESSURE_RESCUE_V2_RAW','raw_ticks':len(raw),'ohlc_resample_used':False,'limitation':'Screening model; rescue sizing/broker semantics remain virtual.',**st.summary_bpd()};p=Path('results/goldebrave-bpd-pressure-rescue-v2')/experiment_id/f'g{int(gate)}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--gate',type=float,required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();run(a.catalog,a.gate,a.experiment_id)
if __name__=='__main__':main()
