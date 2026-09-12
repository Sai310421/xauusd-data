from __future__ import annotations
import argparse,json
from decimal import Decimal
from pathlib import Path
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money,Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType,OmsType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick
from research.rangehunter_m1_trendfollow_v2_nautilus_raw_bt import Strat,Config
SIM=Venue('SIM')

def install(cat_cls):
    if hasattr(cat_cls,'query_quote_ticks'):return
    if hasattr(cat_cls,'query'):
        cat_cls.query_quote_ticks=lambda self,identifiers=None,start=None,end=None,**kw:self.query(QuoteTick,identifiers=identifiers,start=start,end=end,**kw)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();install(ParquetDataCatalog)
    cp=Path(a.catalog);cat=ParquetDataCatalog(str(cp));inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');ticks=cat.query_quote_ticks(identifiers=[inst.id.value])
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)));eng.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(ticks)
    bt=BarType.from_str(f'{inst.id.value}-1-MINUTE-BID-INTERNAL');st=Strat(Config(instrument_id=inst.id,bar_type=bt));eng.add_strategy(st);eng.run()
    out=Path('results/ae-bt')/a.experiment_id;out.mkdir(parents=True,exist_ok=True)
    reports={}
    for name,fn in [('orders',eng.trader.generate_orders_report),('fills',eng.trader.generate_order_fills_report),('positions',eng.trader.generate_positions_report)]:
        try:
            df=fn();reports[name]={'rows':0 if df is None else len(df),'columns':[] if df is None else [str(x) for x in df.columns]};
            if df is not None:df.to_csv(out/f'{name}.csv',index=False)
        except Exception as e:reports[name]={'error':repr(e)}
    summary={'baskets_submitted':st.baskets,'reports':reports,'instrument':str(inst),'instrument_id':inst.id.value}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));eng.dispose()
if __name__=='__main__':main()
