from __future__ import annotations
import argparse,json
from decimal import Decimal
from pathlib import Path
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from goldebrave_fasttf_parity_raw import Cfg,f,l1
from goldebrave_c_g75_profit_lock_v4_raw import GBG75ProfitLockV4

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
 def _q(self,identifiers=None,start=None,end=None): return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
 ParquetDataCatalog.query_quote_ticks=_q

class GBG75ProfitLockMatrix(GBG75ProfitLockV4):
 def __init__(self,c,partial_frac:float,trail_r:float):
  super().__init__(c,'v4')
  self.partial_frac=partial_frac
  self.trail_r=trail_r
 def _manage_child(self,ch,side,mark,vol):
  pnl_move=(mark-ch['entry'])*side
  ch['peak']=max(ch['peak'],pnl_move)
  i=ch['idx']
  if i<=3:return False
  if i<=6:
   arm=max(0.30,0.30*vol)
   if pnl_move>=arm: ch['be_armed']=True
   if ch['be_armed'] and pnl_move<=0:
    self.child_be_closes+=1;return True
   return False
  partial=max(0.35,0.35*vol)
  trail=max(self.trail_r,self.trail_r*vol)
  if ch['open_w']>self.w*(1.0-self.partial_frac+0.01) and pnl_move>=partial:
   close_w=self.w*self.partial_frac
   realized=pnl_move*close_w
   ch['realized']+=realized;self.child_realized+=realized;ch['open_w']=self.w-close_w;self.child_partial_closes+=1
  if ch['peak']>=partial and pnl_move<=max(0.0,ch['peak']-trail):
   self.child_trail_closes+=1;return True
  return False
 def summary_matrix(self):
  s=self.summary_v4();s.update({'partial_fraction':self.partial_frac,'trail_r':self.trail_r})
  return s

def run(catalog,partial_frac,trail_r,experiment_id):
 cat=ParquetDataCatalog(catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value])
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(l1(raw));s=inst.id.value
 st=GBG75ProfitLockMatrix(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{s}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{s}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{s}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{s}-1-HOUR-BID-INTERNAL'),mode='m15'),partial_frac,trail_r)
 eng.add_strategy(st);eng.run();eng.end();res={'verification':'GOLDEBRAVE_C_SWEEP2X_G75_PROFIT_LOCK_V4_MATRIX_RAW','raw_ticks':len(raw),'ohlc_resample_used':False,'logic':'V4 fixed core. Optimize only layers 7-10 partial fraction and trailing distance. Layers 1-3 full runner; layers 4-6 BE at +0.30R; layers 7-10 partial at +0.35R then trail remainder.','limitation':'Virtual PnL/lot-equivalent screening harness; child partial/BE/trail logic is a strategy-equity approximation, not exact broker order semantics.',**st.summary_matrix()};tag=f'p{int(partial_frac*100)}_t{int(trail_r*100)}';p=Path('results/goldebrave-g75-profit-lock-v4-matrix')/experiment_id/f'{tag}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--partial-frac',type=float,choices=[0.35,0.50,0.65],required=True);ap.add_argument('--trail-r',type=float,choices=[0.15,0.20,0.25],required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();run(a.catalog,a.partial_frac,a.trail_r,a.experiment_id)
if __name__=='__main__':main()
