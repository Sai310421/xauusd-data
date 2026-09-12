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
from goldebrave_c_g75_selective_boost_v6_raw import G75SelectiveBoostV6

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
 def _q(self,identifiers=None,start=None,end=None): return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
 ParquetDataCatalog.query_quote_ticks=_q

class G75SelectiveBoostV7(G75SelectiveBoostV6):
 def __init__(self,c,boost_mult:float):
  super().__init__(c,'c')
  self.boost_mult_v5=boost_mult
  self.boosted_children_v5=0
  self.normal_children_v5=0
 def summary_v7(self):
  s=self.summary_v6()
  s.update({'v7_boost_mult':self.boost_mult_v5,'selective_rule_v7':f'V6-C fixed: layers 7-10, move >= max(0.10,0.15*vol), spread <= 1.00; boost={self.boost_mult_v5}x.'})
  return s

def run(catalog,boost_mult,experiment_id):
 cat=ParquetDataCatalog(catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');raw=cat.query_quote_ticks(identifiers=[inst.id.value])
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(l1(raw));s=inst.id.value
 st=G75SelectiveBoostV7(Cfg(instrument_id=inst.id,m1=BarType.from_str(f'{s}-1-MINUTE-BID-INTERNAL'),m5=BarType.from_str(f'{s}-5-MINUTE-BID-INTERNAL'),m15=BarType.from_str(f'{s}-15-MINUTE-BID-INTERNAL'),h1=BarType.from_str(f'{s}-1-HOUR-BID-INTERNAL'),mode='m15'),boost_mult)
 eng.add_strategy(st);eng.run();eng.end();res={'verification':'GOLDEBRAVE_C_G75_SELECTIVE_BOOST_V7_RAW','raw_ticks':len(raw),'ohlc_resample_used':False,'logic':'No-Recycle + Exact Threshold + p35/t15. V6-C coverage fixed; sweep boost multiplier only.','limitation':'Virtual PnL/lot-equivalent screening harness; selective boost is a screening approximation, not exact broker order semantics.',**st.summary_v7()};tag=f'b{str(boost_mult).replace(".","_")}';p=Path('results/goldebrave-g75-selective-boost-v7')/experiment_id/f'{tag}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--boost-mult',type=float,choices=[1.75,2.0,2.25,2.5],required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();run(a.catalog,a.boost_mult,a.experiment_id)
if __name__=='__main__':main()
