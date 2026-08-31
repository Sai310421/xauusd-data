from pathlib import Path
from decimal import Decimal
import json, tempfile
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from g75_mtf_simultaneous import load_df, rs, TFS, Strat, Cfg, START, DAYS, stats

def make_writable_via_csv(d):
    with tempfile.NamedTemporaryFile(suffix='.csv') as f:
        d.to_csv(f.name,index_label='datetime')
        x=pd.read_csv(f.name)
    x.index=pd.to_datetime(x.pop('datetime'),utc=True)
    for c in ['open','high','low','close','volume']:
        x[c]=pd.to_numeric(x[c],errors='coerce')
    return x.dropna().copy()

def main():
    base=load_df(); sim=Venue('SIM'); inst=TestInstrumentProvider.default_fx_ccy('XAU/USD',sim); bts=[]; allbars=[]
    unit={'M1':'1-MINUTE','M5':'5-MINUTE','M15':'15-MINUTE','H1':'1-HOUR'}
    for name,rule in TFS.items():
        bt=BarType.from_str(f'{inst.id}-{unit[name]}-LAST-EXTERNAL'); bts.append(bt)
        d=rs(base,rule); d.index=d.index+pd.Timedelta(minutes=1); d=make_writable_via_csv(d)
        allbars.extend(BarDataWrangler(bt,inst).process(d))
    allbars.sort(key=lambda x:x.ts_init)
    e=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR')))
    e.add_venue(venue=sim,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,starting_balances=[Money(1000000,USD)],base_currency=USD,default_leverage=Decimal(1)); e.add_instrument(inst); e.add_data(allbars)
    s=Strat(Cfg(instrument_id=inst.id,bar_types=tuple(bts))); e.add_strategy(s); e.run()
    per={tf:{**stats(s.pnls[tf]),'entries_N':s.cores[tf].n,'adds':s.cores[tf].adds} for tf in TFS}
    total=sum((s.pnls[t] for t in TFS),[]); w=[x for x in total if x>0]; l=[x for x in total if x<0]; net=s.balance-START; ret=100*net/START; monthly=((s.balance/START)**(21/DAYS)-1)*100
    port={'entries_N':sum(s.cores[t].n for t in TFS),'adds':sum(s.cores[t].adds for t in TFS),'cycles':len(total),'wr_pct':100*len(w)/len(total),'pf':sum(w)/abs(sum(l)),'max_dd_pct':s.maxdd,'return90_pct':ret,'monthly21_pct':monthly,'rf':ret/s.maxdd,'end_balance':s.balance}
    out={'engine':'NautilusTrader BacktestEngine','mode':'M1+M5+M15+H1 simultaneous independent cycles; fixed 0.05 lot each','per_tf':per,'portfolio':port}
    p=Path('research/g75_nautilus/results'); p.mkdir(parents=True,exist_ok=True); (p/'g75_mtf_simultaneous.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); pd.DataFrame([port]).to_csv(p/'g75_mtf_simultaneous_portfolio.csv',index=False)
    print('G75_MTF_SIMULTANEOUS='+json.dumps(out)); e.dispose()
if __name__=='__main__': main()
