"""Known-price native reconciliation, including fee overlay and DD shutdown."""
import subprocess
import sys
import json
from pathlib import Path
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.objects import Currency, Price, Quantity
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog

root = Path('results/prop4-smoke/catalog')
inst = CurrencyPair(instrument_id=InstrumentId(Symbol('XAUUSD'),Venue('SIM')),raw_symbol=Symbol('XAUUSD'),base_currency=Currency.from_str('XAU'),quote_currency=Currency.from_str('USD'),price_precision=3,size_precision=0,price_increment=Price.from_str('0.001'),size_increment=Quantity.from_int(1),ts_event=0,ts_init=0)
cat = ParquetDataCatalog(str(root))
cat.write_data([inst])
# Proxy is replaced by deterministic monotonically ordered MAs after warmup.
start = 1785153600*10**9
quotes=[]
for n in range(300):
 bid = 2000+n*.01
 quotes.append(QuoteTick(instrument_id=inst.id,bid_price=Price.from_str(f'{bid:.3f}'),ask_price=Price.from_str(f'{bid+.1:.3f}'),bid_size=Quantity.from_int(100000),ask_size=Quantity.from_int(100000),ts_event=start+n*10**9,ts_init=start+n*10**9))
cat.write_data(quotes)
runner=Path(__file__).with_name('research_prop4_native.py')
for fee in (0,6):
 name=f'SMOKE_FEE{fee}'
 subprocess.run([sys.executable,str(runner),'--catalog',str(root),'--id',name,'--lot','.01','--initial','1000','--fee',str(fee)],check=True)
 result=json.loads((Path('results/prop4')/name/'kpi.json').read_text())
 assert result['N']>0, result
 assert result['accounting_pass'], result
 assert abs(result['native_net_after_fee']-result['Net'])<.1, result
 assert result['native_peak_DD_pct']>=0, result
print('Native reconciliation and commission overlay smoke passed')
