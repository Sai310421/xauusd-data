from __future__ import annotations
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, StrategyConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.trading.strategy import Strategy

T=0.12; A=0.025; R=0.20; LMAX=10; LOT=0.05; V=100.0; SPREAD=0.30; COMM=7.0; START=10000.0; DAYS=65
TFS={"M1":"1min","M5":"5min","M15":"15min","H1":"1h"}

def load_df():
    p=Path('csv/XAUUSD/XAUUSD_M1_2026Q1Q2.csv'); x=pd.read_csv(p); low={c.lower():c for c in x.columns}; tc=low.get('datetime') or low.get('timestamp') or low.get('time')
    d=pd.DataFrame({k:pd.to_numeric(x[low[k]],errors='coerce') for k in ['open','high','low','close']}); d['volume']=pd.to_numeric(x[low['volume']],errors='coerce').fillna(0) if 'volume' in low else 0.0
    d.index=pd.to_datetime(x[tc],errors='coerce',utc=True); return d[~d.index.isna()].dropna().sort_index()

def rs(df, rule):
    if rule=='1min': return df.copy()
    return df.resample(rule,label='right',closed='right').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()

class Core:
    def __init__(self): self.anchor=None; self.active=False; self.side=0; self.last=None; self.ext=None; self.layers=0; self.entries=[]; self.n=0; self.adds=0
    def on(self,o,h,l,c,ts):
        ev=[]
        if self.anchor is None: self.anchor=c; return ev
        if not self.active:
            up=h>=self.anchor+T; dn=l<=self.anchor-T
            if not(up or dn): self.anchor=c; return ev
            self.side=1 if c>=self.anchor else -1; e=self.anchor+self.side*T; self.active=True; self.last=e; self.ext=e; self.layers=1; self.entries=[e]; self.n+=1; ev.append(('ENTRY',e,self.side,ts))
        if self.side==1:
            while self.layers<LMAX and self.last+A<=h+1e-12: self.last+=A; self.layers+=1; self.entries.append(self.last); self.adds+=1; ev.append(('ADD',self.last,self.side,ts))
            self.ext=max(self.ext,h); rev=c<=self.ext-R
        else:
            while self.layers<LMAX and self.last-A>=l-1e-12: self.last-=A; self.layers+=1; self.entries.append(self.last); self.adds+=1; ev.append(('ADD',self.last,self.side,ts))
            self.ext=min(self.ext,l); rev=c>=self.ext+R
        if rev:
            ev.append(('EXIT',c,self.side,ts)); self.active=False; self.anchor=c
        return ev

class Cfg(StrategyConfig,frozen=True):
    instrument_id: InstrumentId
    bar_types: tuple[BarType,...]

class Strat(Strategy):
    def __init__(self,cfg):
        super().__init__(cfg); self.cores={}; self.open={}; self.pnls={}; self.balance=START; self.peak=START; self.maxdd=0.0; self.last_close={}; self.tfmap={str(bt):name for bt,name in zip(cfg.bar_types,TFS.keys())}
        for name in TFS: self.cores[name]=Core(); self.open[name]=[]; self.pnls[name]=[]
    def on_start(self):
        for bt in self.config.bar_types: self.subscribe_bars(bt)
    def cost(self,n): return n*LOT*(SPREAD*V+COMM)
    def mark(self):
        unreal=0.0
        for tf,core in self.cores.items():
            if self.open[tf] and tf in self.last_close: unreal+=sum(core.side*(self.last_close[tf]-p)*LOT*V for p in self.open[tf])-self.cost(len(self.open[tf]))
        eq=self.balance+unreal; self.peak=max(self.peak,eq); self.maxdd=max(self.maxdd,100*(self.peak-eq)/self.peak if self.peak else 0)
    def on_bar(self,bar:Bar):
        tf=self.tfmap[str(bar.bar_type)]; c=float(bar.close); self.last_close[tf]=c; core=self.cores[tf]
        for act,p,side,ts in core.on(float(bar.open),float(bar.high),float(bar.low),c,bar.ts_event):
            if act=='ENTRY': self.open[tf]=[p]
            elif act=='ADD': self.open[tf].append(p)
            elif act=='EXIT':
                gross=sum(side*(p-x)*LOT*V for x in self.open[tf]); pnl=gross-self.cost(len(self.open[tf])); self.balance+=pnl; self.pnls[tf].append(pnl); self.open[tf]=[]
        self.mark()

def stats(pnls):
    w=[x for x in pnls if x>0]; l=[x for x in pnls if x<0]; return {'cycles':len(pnls),'wr_pct':100*len(w)/len(pnls) if pnls else 0,'pf':sum(w)/abs(sum(l)) if l else float('inf'),'net':sum(pnls)}

def main():
    base=load_df(); sim=Venue('SIM'); inst=TestInstrumentProvider.default_fx_ccy('XAU/USD',sim); bts=[]; allbars=[]
    unit={'M1':'1-MINUTE','M5':'5-MINUTE','M15':'15-MINUTE','H1':'1-HOUR'}
    for name,rule in TFS.items():
        bt=BarType.from_str(f'{inst.id}-{unit[name]}-LAST-EXTERNAL'); bts.append(bt); d=rs(base,rule); d.index=d.index+pd.Timedelta(minutes=1); allbars.extend(BarDataWrangler(bt,inst).process(d))
    allbars=sorted(allbars,key=lambda x:x.ts_init)
    e=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'))); e.add_venue(venue=sim,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,starting_balances=[Money(1000000,USD)],base_currency=USD,default_leverage=Decimal(1)); e.add_instrument(inst); e.add_data(allbars)
    s=Strat(Cfg(instrument_id=inst.id,bar_types=tuple(bts))); e.add_strategy(s); e.run()
    per={tf:{**stats(s.pnls[tf]),'entries_N':s.cores[tf].n,'adds':s.cores[tf].adds} for tf in TFS}; total_pnls=sum((s.pnls[tf] for tf in TFS),[]); net=s.balance-START; ret=100*net/START; monthly=((s.balance/START)**(21/DAYS)-1)*100
    W=[x for x in total_pnls if x>0]; L=[x for x in total_pnls if x<0]
    out={'engine':'NautilusTrader BacktestEngine','mode':'M1+M5+M15+H1 simultaneous independent cycles, fixed 0.05 lot each','per_tf':per,'portfolio':{'entries_N':sum(s.cores[t].n for t in TFS),'adds':sum(s.cores[t].adds for t in TFS),'cycles':len(total_pnls),'wr_pct':100*len(W)/len(total_pnls),'pf':sum(W)/abs(sum(L)),'max_dd_pct':s.maxdd,'return90_pct':ret,'monthly21_pct':monthly,'rf':ret/s.maxdd,'end_balance':s.balance}}
    p=Path('research/g75_nautilus/results'); p.mkdir(parents=True,exist_ok=True); (p/'g75_mtf_simultaneous.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); pd.DataFrame([out['portfolio']]).to_csv(p/'g75_mtf_simultaneous_portfolio.csv',index=False); print('G75_MTF_SIMULTANEOUS='+json.dumps(out))
    e.dispose()
if __name__=='__main__': main()
