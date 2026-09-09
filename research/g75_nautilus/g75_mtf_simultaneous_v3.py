from __future__ import annotations
import json
from pathlib import Path
from decimal import Decimal
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

T=0.12; ADD=0.025; REV=0.20; LMAX=10; LOT=0.05; V=100.0; SPREAD=0.30; COMM=7.0; START=10000.0; DAYS=65
TFS={'M1':1,'M5':5,'M15':15,'H1':60}

class Core:
    def __init__(self):
        self.anchor=None; self.active=False; self.side=0; self.last=None; self.ext=None; self.layers=0; self.n=0; self.adds=0; self.open=[]; self.pnls=[]
    def on(self,o,h,l,c,ts):
        ev=[]
        if self.anchor is None: self.anchor=c; return ev
        if not self.active:
            up=h>=self.anchor+T; dn=l<=self.anchor-T
            if not(up or dn): self.anchor=c; return ev
            self.side=1 if c>=self.anchor else -1
            e=self.anchor+self.side*T; self.active=True; self.last=e; self.ext=e; self.layers=1; self.n+=1; self.open=[e]; ev.append(('ENTRY',e))
        if self.side==1:
            while self.layers<LMAX and self.last+ADD<=h+1e-12:
                self.last+=ADD; self.layers+=1; self.adds+=1; self.open.append(self.last); ev.append(('ADD',self.last))
            self.ext=max(self.ext,h); hit=c<=self.ext-REV
        else:
            while self.layers<LMAX and self.last-ADD>=l-1e-12:
                self.last-=ADD; self.layers+=1; self.adds+=1; self.open.append(self.last); ev.append(('ADD',self.last))
            self.ext=min(self.ext,l); hit=c>=self.ext+REV
        if hit: ev.append(('EXIT',c)); self.active=False; self.anchor=c
        return ev

class Agg:
    def __init__(self,minutes): self.m=minutes; self.key=None; self.o=self.h=self.l=self.c=None
    def push(self,ts,o,h,l,c):
        stamp=pd.Timestamp(ts,unit='ns',tz='UTC'); key=stamp.floor(f'{self.m}min')
        completed=None
        if self.key is None:
            self.key=key; self.o=o; self.h=h; self.l=l; self.c=c; return None
        if key!=self.key:
            completed=(self.key,self.o,self.h,self.l,self.c)
            self.key=key; self.o=o; self.h=h; self.l=l; self.c=c
        else:
            self.h=max(self.h,h); self.l=min(self.l,l); self.c=c
        return completed

class Cfg(StrategyConfig,frozen=True): instrument_id: InstrumentId; bar_type: BarType

class Strat(Strategy):
    def __init__(self,cfg):
        super().__init__(cfg); self.cores={k:Core() for k in TFS}; self.aggs={k:Agg(v) for k,v in TFS.items() if k!='M1'}; self.balance=START; self.peak=START; self.maxdd=0.0; self.current_price=None
    def on_start(self): self.subscribe_bars(self.config.bar_type)
    def cost(self,n): return n*LOT*(SPREAD*V+COMM)
    def process_tf(self,tf,o,h,l,c,ts):
        core=self.cores[tf]
        for act,p in core.on(o,h,l,c,ts):
            if act=='EXIT':
                gross=sum(core.side*(p-x)*LOT*V for x in core.open); pnl=gross-self.cost(len(core.open)); self.balance+=pnl; core.pnls.append(pnl); core.open=[]; core.layers=0
    def mark(self,price):
        unreal=0.0
        for core in self.cores.values():
            if core.open: unreal+=sum(core.side*(price-x)*LOT*V for x in core.open)-self.cost(len(core.open))
        eq=self.balance+unreal; self.peak=max(self.peak,eq); self.maxdd=max(self.maxdd,100*(self.peak-eq)/self.peak if self.peak else 0)
    def on_bar(self,bar:Bar):
        o,h,l,c=map(float,(bar.open,bar.high,bar.low,bar.close)); self.current_price=c
        self.process_tf('M1',o,h,l,c,bar.ts_event)
        for tf,agg in self.aggs.items():
            done=agg.push(bar.ts_event,o,h,l,c)
            if done:
                ts,ao,ah,al,ac=done; self.process_tf(tf,ao,ah,al,ac,int(ts.value))
        self.mark(c)

def load_df(path):
    x=pd.read_csv(path); low={c.lower():c for c in x.columns}; tc=low.get('datetime') or low.get('timestamp') or low.get('time')
    d=pd.DataFrame({k:pd.to_numeric(x[low[k]],errors='coerce') for k in ['open','high','low','close']}); d['volume']=pd.to_numeric(x[low['volume']],errors='coerce').fillna(0) if 'volume' in low else 0.0
    d.index=pd.to_datetime(x[tc],errors='coerce',utc=True); d=d[~d.index.isna()].dropna().sort_index(); d.index=d.index+pd.Timedelta(minutes=1); return d

def tfstats(c):
    w=[x for x in c.pnls if x>0]; l=[x for x in c.pnls if x<0]
    return {'entries_N':c.n,'adds':c.adds,'cycles':len(c.pnls),'wr_pct':100*len(w)/len(c.pnls) if c.pnls else 0,'pf':sum(w)/abs(sum(l)) if l else float('inf'),'net_profit_usd':sum(c.pnls)}

def main():
    df=load_df('csv/XAUUSD/XAUUSD_M1_2026Q1Q2.csv'); sim=Venue('SIM'); inst=TestInstrumentProvider.default_fx_ccy('XAU/USD',sim); bt=BarType.from_str(f'{inst.id}-1-MINUTE-LAST-EXTERNAL'); bars=BarDataWrangler(bt,inst).process(df)
    e=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'))); e.add_venue(venue=sim,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,starting_balances=[Money(1000000,USD)],base_currency=USD,default_leverage=Decimal(1)); e.add_instrument(inst); e.add_data(bars); s=Strat(Cfg(instrument_id=inst.id,bar_type=bt)); e.add_strategy(s); e.run()
    per={tf:tfstats(c) for tf,c in s.cores.items()}; allp=sum((c.pnls for c in s.cores.values()),[]); w=[x for x in allp if x>0]; l=[x for x in allp if x<0]; net=s.balance-START; ret=100*net/START; monthly=((s.balance/START)**(21/DAYS)-1)*100
    port={'entries_N':sum(c.n for c in s.cores.values()),'adds':sum(c.adds for c in s.cores.values()),'cycles':len(allp),'wr_pct':100*len(w)/len(allp),'pf':sum(w)/abs(sum(l)),'max_dd_pct':s.maxdd,'return90_pct':ret,'monthly21_pct':monthly,'rf':ret/s.maxdd,'end_balance':s.balance}
    out={'engine':'NautilusTrader BacktestEngine','mode':'single M1 market feed -> independent simultaneous M1/M5/M15/H1 G75 cycles; fixed 0.05 lot each','per_tf':per,'portfolio':port}
    p=Path('research/g75_nautilus/results'); p.mkdir(parents=True,exist_ok=True); (p/'g75_mtf_simultaneous.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); pd.DataFrame([port]).to_csv(p/'g75_mtf_simultaneous_portfolio.csv',index=False); print('G75_MTF_SIMULTANEOUS='+json.dumps(out)); e.dispose()
if __name__=='__main__': main()
