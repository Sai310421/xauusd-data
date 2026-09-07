from __future__ import annotations

import argparse, json
from decimal import Decimal
from pathlib import Path

import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from g75_tsugi_nautilus_raw_bt import G75TsugiStrategy, G75TsugiConfig, native_report_metrics, TF_MIN

SIM = Venue('SIM')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--catalog',required=True)
    ap.add_argument('--experiment-id',required=True)
    ap.add_argument('--tf',required=True,choices=['M1','M5','M15'])
    ap.add_argument('--variant',required=True,choices=['A','B','C'])
    ap.add_argument('--raw-bidask-only',action='store_true')
    args=ap.parse_args()
    if not args.raw_bidask_only: raise SystemExit('raw-bidask-only is mandatory')
    cp=Path(args.catalog); manifest=json.loads((cp/'catalog_manifest.json').read_text())
    catalog=ParquetDataCatalog(str(cp))
    instrument=next((x for x in catalog.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'),None)
    if instrument is None: raise SystemExit('XAUUSD missing')
    ticks=catalog.query(data_cls=QuoteTick, identifiers=[instrument.id.value])
    if not ticks: raise SystemExit('no raw XAUUSD QuoteTicks')
    minutes=TF_MIN[args.tf]
    engine=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    engine.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    engine.add_instrument(instrument); engine.add_data(ticks)
    bar_type=BarType.from_str(f'{instrument.id.value}-{minutes}-MINUTE-BID-INTERNAL')
    st=G75TsugiStrategy(G75TsugiConfig(instrument_id=instrument.id,bar_type=bar_type,variant=args.variant))
    engine.add_strategy(st); engine.run(); rep=engine.trader.generate_positions_report()
    row={**st.summary(),**native_report_metrics(rep),'tf':args.tf,'raw_ticks':len(ticks),
         'nautilus_version':getattr(nautilus_trader,'__version__','unknown'),
         'period_start':manifest.get('start'),'period_days':manifest.get('days'),'period_end_exclusive':manifest.get('end_exclusive')}
    out=Path('results/ae-bt')/args.experiment_id/'cells'; out.mkdir(parents=True,exist_ok=True)
    (out/f'{args.tf}_{args.variant}.json').write_text(json.dumps(row,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(row,ensure_ascii=False))
    engine.dispose()

if __name__=='__main__': main()
