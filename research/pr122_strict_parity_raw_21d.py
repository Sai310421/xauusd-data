from __future__ import annotations
import argparse, json, math, hashlib
from collections import deque
from decimal import Decimal
from pathlib import Path
import numpy as np, pandas as pd, nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from research.g75_tsugi_tick_strict_cell import G75TickStrict
from research.g75_tsugi_nautilus_raw_bt import G75TsugiConfig
from research.vendor.pr122.ae_math_supervisor_v1 import (
    AEMathSupervisorV1, SupervisorConfig, KellyConfig, RecoveryConfig,
    WassersteinConfig, MPCConfig, AEState, MPCAction,
)

SIM=Venue("SIM")
START=pd.Timestamp("2026-07-27T00:00:00Z")
END_EXCL=pd.Timestamp("2026-08-25T00:00:00Z")
INITIAL=1000.0
LANES=("baseline","kelly","recovery","wasserstein","mpc","kelly_recovery","kelly_recovery_mpc","full")

def lane_cfg(name):
    parts=set(name.split("_"))
    enabled=lambda k: name=="full" or k in parts
    return SupervisorConfig(
        kelly=KellyConfig(enabled=enabled("kelly")),
        recovery=RecoveryConfig(enabled=enabled("recovery")),
        wasserstein=WassersteinConfig(enabled=enabled("wasserstein")),
        mpc=MPCConfig(enabled=enabled("mpc")),
        hard_min_margin_level=500.0,soft_margin_level=650.0,max_multiplier=1.0,
    )

class StrictTrace(G75TickStrict):
    def __init__(self,cfg):
        super().__init__(cfg);self.trace=[];self._trace_ts=0
    def on_quote_tick(self,t):
        self._trace_ts=int(t.ts_event);before_layers=len(self.entry_prices);before_active=self.active;before_side=self.side
        super().on_quote_tick(t)
        after_layers=len(self.entry_prices)
        if (not before_active) and self.active:
            self.trace.append(("ENTRY",self._trace_ts,int(self.side),round(float(self.entry_prices[0]),5),1))
        elif before_active and self.active and after_layers>before_layers:
            for i in range(before_layers,after_layers):
                self.trace.append(("ADD",self._trace_ts,int(self.side),round(float(self.entry_prices[i]),5),i+1))
        if before_active and not self.active:
            self.trace.append(("EXIT",self._trace_ts,int(before_side),round(float((self.last_bid+self.last_ask)/2),5),before_layers))
    def trace_hash(self):
        return hashlib.sha256(json.dumps(self.trace,separators=(",",":")).encode()).hexdigest()

class PR122StrictShadow(StrictTrace):
    """Exact G75TickStrict chronology; PR122 only sizes a parallel economic ledger.
    Core entries/adds/exits and native strategy state remain untouched.
    """
    def __init__(self,cfg,lane):
        super().__init__(cfg);self.lane=lane;self.sup=None if lane=="baseline" else AEMathSupervisorV1(lane_cfg(lane))
        self.shadow_entries=[];self.shadow_mult=[];self.shadow_realized=0.;self.shadow_peak=INITIAL;self.shadow_maxdd=0.;self.shadow_gw=0.;self.shadow_gl=0.;self.shadow_wins=0;self.shadow_cycles=0
        self.closed_returns=deque(maxlen=40);self.mult_hist=[];self._prev_active=False;self._prev_layers=0;self._prev_side=0
    def _shadow_mark(self,bid,ask):
        if not self.shadow_entries:return 0.
        px=bid if self._prev_side>0 else ask
        return sum((px-e)*self._prev_side*m for e,m in zip(self.shadow_entries,self.shadow_mult))
    def _decision(self,bid,ask):
        if self.lane=="baseline":return 1.0
        rs=list(self.closed_returns)
        if not rs:return 0.0
        eq=INITIAL+self.shadow_realized+self._shadow_mark(bid,ask);self.shadow_peak=max(self.shadow_peak,eq)
        dd=max(0.,(self.shadow_peak-eq)/max(self.shadow_peak,1e-9))
        mu=float(np.mean(rs));sig=max(float(np.std(rs)),1e-6);coord=max(1e-6,min(.999999,1-dd/0.15))
        gross=sum(self.shadow_mult);mid=(bid+ask)/2;margin=(gross*mid)/2000 if gross>0 else 0.;ml=999999. if margin<=0 else eq/margin*100
        st=AEState(equity=eq,peak_equity=self.shadow_peak,debt=max(0.,self.shadow_peak-eq),inventory=gross,margin_level=ml,recovery_coordinate=coord)
        acts=[MPCAction("STOP",0.,0.,-.002,-.01,-1.,0.),MPCAction("HALF",.5,mu*eq*.5,0.,-.005,-.5,.0001),MPCAction("NORMAL",1.,mu*eq,0.,0.,0.,.0002)]
        d=self.sup.decide(st,rs,mu,sig,[[r] for r in rs],[1.0],acts);self.mult_hist.append(d.multiplier);return float(d.multiplier)
    def on_quote_tick(self,t):
        bid=self._f(t.bid_price);ask=self._f(t.ask_price)
        before_active=self.active;before_layers=len(self.entry_prices);before_side=self.side
        super().on_quote_tick(t)
        after_active=self.active;after_layers=len(self.entry_prices)
        if (not before_active) and after_active:
            self._prev_side=self.side;self.shadow_entries=[float(self.entry_prices[0])];self.shadow_mult=[self._decision(bid,ask)]
        elif before_active and after_active and after_layers>before_layers:
            for i in range(before_layers,after_layers):
                self.shadow_entries.append(float(self.entry_prices[i]));self.shadow_mult.append(self._decision(bid,ask))
        if before_active and not after_active:
            px=bid if before_side>0 else ask
            pnl=sum((px-e)*before_side*m for e,m in zip(self.shadow_entries,self.shadow_mult));self.shadow_realized+=pnl;self.shadow_cycles+=1
            if pnl>0:self.shadow_wins+=1;self.shadow_gw+=pnl
            elif pnl<0:self.shadow_gl+=abs(pnl)
            self.closed_returns.append(pnl/INITIAL);self.shadow_entries=[];self.shadow_mult=[];self._prev_side=0
        eq=INITIAL+self.shadow_realized+self._shadow_mark(bid,ask);self.shadow_peak=max(self.shadow_peak,eq);self.shadow_maxdd=max(self.shadow_maxdd,(self.shadow_peak-eq)/max(self.shadow_peak,1e-9)*100)
    def shadow_summary(self):
        return dict(lane=self.lane,N=self.shadow_cycles,WR_pct=100*self.shadow_wins/max(1,self.shadow_cycles),PF=self.shadow_gw/self.shadow_gl if self.shadow_gl else None,NetProfit=self.shadow_realized,Return_pct=self.shadow_realized/INITIAL*100,MaxFloatingDD_pct=self.shadow_maxdd,mean_multiplier=float(np.mean(self.mult_hist)) if self.mult_hist else (1.0 if self.lane=="baseline" else 0.),zero_multiplier_rate=float(np.mean(np.array(self.mult_hist)<=1e-12)) if self.mult_hist else 0.,event_hash=self.trace_hash(),event_count=len(self.trace))

def run(catalog,lane,wrapper):
    cat=ParquetDataCatalog(catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value]);ticks=[x for x in ticks if START.value<=int(x.ts_event)<END_EXCL.value]
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(INITIAL,USD)],default_leverage=Decimal("2000"));eng.add_instrument(inst);eng.add_data(ticks)
    bt=BarType.from_str(f"{inst.id.value}-1-MINUTE-BID-INTERNAL")
    st=PR122StrictShadow(G75TsugiConfig(instrument_id=inst.id,bar_type=bt,variant="A"),lane) if wrapper else StrictTrace(G75TsugiConfig(instrument_id=inst.id,bar_type=bt,variant="A"))
    eng.add_strategy(st);eng.run()
    core=st.summary();base=dict(N=core["cycles"],WR_pct=core["WR_pct"],PF=core["PF_virtual_raw"],NetProfit=core["realized_virtual"],MaxFloatingDD_pct=core["max_floating_DD_pct"],event_hash=st.trace_hash(),event_count=len(st.trace))
    out=st.shadow_summary() if wrapper else base;eng.dispose();return base,out,len(ticks)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--lane",choices=LANES,required=True)
    ap.add_argument("--mode",choices=["trusted","wrapper"],required=True)
    a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.mode=="trusted":
        trusted,_,n=run(a.catalog,"baseline",False)
        result={"verification_level":"PR122_STRICT_TRUSTED_21D_V1","lane":a.lane,"raw_ticks":n,"trusted_strict_baseline":trusted,"wr5_status":"INVALID"}
        p=out/f"{a.lane}.trusted.json"
    else:
        core,shadow,n=run(a.catalog,a.lane,True)
        result={"verification_level":"PR122_STRICT_WRAPPER_21D_V1","lane":a.lane,"raw_ticks":n,"wrapper_core_baseline":core,"shadow_economics":shadow,"wr5_status":"INVALID"}
        p=out/f"{a.lane}.wrapper.json"
    p.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
