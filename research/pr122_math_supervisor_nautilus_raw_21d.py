from __future__ import annotations
import argparse, hashlib, json, math
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import numpy as np, pandas as pd, nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from research.vendor.pr122.ae_math_supervisor_v1 import (
    AEMathSupervisorV1, SupervisorConfig, KellyConfig, RecoveryConfig,
    WassersteinConfig, MPCConfig, AEState, MPCAction,
)

if not hasattr(ParquetDataCatalog,"query_quote_ticks"):
    def _query_quote_ticks(self,identifiers=None,start=None,end=None):
        return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
    ParquetDataCatalog.query_quote_ticks=_query_quote_ticks

INITIAL=1000.0
START=pd.Timestamp("2026-07-27T00:00:00Z")
END_EXCL=pd.Timestamp("2026-08-25T00:00:00Z")
TRIGGER=.12; ADD=.025; REVERSAL=.20; MAX_LAYERS=10
LANES=("baseline","kelly","recovery","wasserstein","mpc","kelly_recovery","kelly_recovery_mpc","full")

class Cfg(StrategyConfig, frozen=True):
    instrument_id: object
    bar_type: BarType
    lane: str

def lane_cfg(name:str)->SupervisorConfig:
    on=lambda k: k in name.split("_") or name=="full"
    return SupervisorConfig(
        kelly=KellyConfig(enabled=on("kelly")),
        recovery=RecoveryConfig(enabled=on("recovery")),
        wasserstein=WassersteinConfig(enabled=on("wasserstein")),
        mpc=MPCConfig(enabled=on("mpc")),
        hard_min_margin_level=500.0,soft_margin_level=650.0,max_multiplier=1.0,
    )

class PR122Lane(Strategy):
    def __init__(self,cfg):
        super().__init__(cfg);self.lane=cfg.lane;self.sup=None if cfg.lane=="baseline" else AEMathSupervisorV1(lane_cfg(cfg.lane))
        self.anchor=None;self.core_active=False;self.side=0;self.core_entries=[];self.last_add=None;self.extreme=None
        self.entry_mult=[];self.realized=0.;self.peak=INITIAL;self.maxdd=0.;self.gw=0.;self.gl=0.;self.cycles=0;self.wins=0
        self.core_events=[];self.mults=[];self.hist=deque(maxlen=40);self.last_bid=None;self.last_ask=None;self.min_margin=999999.;self.max_layers=0
    @staticmethod
    def f(x): return float(x.as_double()) if hasattr(x,"as_double") else float(x)
    def on_start(self): self.subscribe_quote_ticks(self.config.instrument_id);self.subscribe_bars(self.config.bar_type)
    def _event(self,kind,ts,price,layer):
        self.core_events.append((kind,int(ts),round(float(price),5),int(self.side),int(layer)))
    def _mark(self,bid,ask):
        if not self.core_active:return 0.
        px=bid if self.side>0 else ask
        return sum((px-e)*self.side*m for e,m in zip(self.core_entries,self.entry_mult))
    def _equity(self,bid,ask): return INITIAL+self.realized+self._mark(bid,ask)
    def _margin_level(self,bid,ask):
        eq=self._equity(bid,ask);notional=sum(self.entry_mult)*((bid+ask)/2);margin=notional/2000 if notional>0 else 0.
        ml=999999. if margin<=0 else eq/margin*100
        self.min_margin=min(self.min_margin,ml);return ml
    def _mult(self,bid,ask):
        if self.lane=="baseline":return 1.0
        rs=list(self.hist)
        if not rs:return 0.0
        mu=float(np.mean(rs));sig=float(np.std(rs,ddof=0))
        eq=self._equity(bid,ask);dd=max(0.,(self.peak-eq)/max(self.peak,1e-9))
        x=max(1e-6,min(.999999,1.0-dd/0.15))
        state=AEState(equity=eq,peak_equity=self.peak,debt=max(0.,self.peak-eq),inventory=sum(self.entry_mult),margin_level=self._margin_level(bid,ask),recovery_coordinate=x)
        actions=[
          MPCAction("STOP",0.,0.,-.002,-.01,-1.,0.),
          MPCAction("HALF",.5,mu*eq*.5,0.,-.005,-.5,.0001),
          MPCAction("NORMAL",1.,mu*eq,0.,0.,0.,.0002),
        ]
        scenarios=[[r] for r in rs]
        d=self.sup.decide(state,rs,mu,max(sig,1e-6),scenarios,[1.0],actions)
        self.mults.append((d.multiplier,d.kelly_multiplier,d.recovery_multiplier,d.wasserstein_multiplier,d.mpc_multiplier,d.margin_multiplier))
        return float(d.multiplier)
    def on_bar(self,bar:Bar):
        c=self.f(bar.close);h=self.f(bar.high);l=self.f(bar.low);ts=int(bar.ts_event)
        if self.anchor is None:self.anchor=c;return
        if not self.core_active:
            up=h>=self.anchor+TRIGGER;dn=l<=self.anchor-TRIGGER
            if not(up or dn):self.anchor=c;return
            self.side=1 if c>=self.anchor else -1
            px=self.last_ask if self.side>0 else self.last_bid
            if px is None:px=c
            m=self._mult(self.last_bid or c,self.last_ask or c)
            self.core_active=True;self.core_entries=[px];self.entry_mult=[m];self.last_add=px;self.extreme=px;self.max_layers=max(self.max_layers,1);self._event("ENTRY",ts,px,1)
    def on_quote_tick(self,t:QuoteTick):
        bid=self.f(t.bid_price);ask=self.f(t.ask_price);ts=int(t.ts_event);self.last_bid=bid;self.last_ask=ask
        eq=self._equity(bid,ask);self.peak=max(self.peak,eq);self.maxdd=max(self.maxdd,(self.peak-eq)/max(self.peak,1e-9)*100);self._margin_level(bid,ask)
        if not self.core_active:return
        px=bid if self.side>0 else ask;self.extreme=max(self.extreme,px) if self.side>0 else min(self.extreme,px)
        while len(self.core_entries)<MAX_LAYERS:
            target=self.last_add+self.side*ADD
            if not(px>=target if self.side>0 else px<=target):break
            fill=ask if self.side>0 else bid;m=self._mult(bid,ask)
            self.core_entries.append(fill);self.entry_mult.append(m);self.last_add=target;self.max_layers=max(self.max_layers,len(self.core_entries));self._event("ADD",ts,fill,len(self.core_entries))
        rev=(px<=self.extreme-REVERSAL) if self.side>0 else (px>=self.extreme+REVERSAL)
        if rev:self._close(bid,ask,ts)
    def _close(self,bid,ask,ts):
        px=bid if self.side>0 else ask
        self._event("REVERSAL",ts,px,len(self.core_entries));self._event("BASKET_EXIT",ts,px,len(self.core_entries))
        pnl=sum((px-e)*self.side*m for e,m in zip(self.core_entries,self.entry_mult));self.realized+=pnl;self.cycles+=1
        if pnl>0:self.wins+=1;self.gw+=pnl
        elif pnl<0:self.gl+=abs(pnl)
        self.hist.append(pnl/max(INITIAL,1e-9))
        self.core_active=False;self.anchor=(bid+ask)/2;self.core_entries=[];self.entry_mult=[];self.last_add=None;self.extreme=None
    def on_stop(self):
        if self.core_active and self.last_bid is not None:self._close(self.last_bid,self.last_ask,int(self.clock.timestamp_ns()))
    def summary(self):
        h=hashlib.sha256(json.dumps(self.core_events,separators=(",",":")).encode()).hexdigest()
        arr=np.array(self.mults,float) if self.mults else np.empty((0,6))
        return dict(lane=self.lane,N=self.cycles,WR_pct=100*self.wins/max(1,self.cycles),PF=self.gw/self.gl if self.gl else None,
            NetProfit=self.realized,Return_pct=self.realized/INITIAL*100,MaxFloatingDD_pct=self.maxdd,min_margin_level=self.min_margin,
            max_layers=self.max_layers,event_count=len(self.core_events),event_hash=h,
            mean_multiplier=float(arr[:,0].mean()) if len(arr) else (1.0 if self.lane=="baseline" else 0.0),
            zero_multiplier_rate=float((arr[:,0]<=1e-12).mean()) if len(arr) else 0.0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
    raw=cat.query_quote_ticks(identifiers=[inst.id.value]);ticks=[x for x in raw if START.value<=int(x.ts_event)<END_EXCL.value]
    if not ticks:raise SystemExit("INVALID:no ticks")
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[]
    for lane in LANES:
        eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)))
        eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(INITIAL,USD)],default_leverage=Decimal("2000"))
        eng.add_instrument(inst);eng.add_data(ticks)
        bt=BarType.from_str(f"{inst.id.value}-1-MINUTE-BID-INTERNAL");st=PR122Lane(Cfg(instrument_id=inst.id,bar_type=bt,lane=lane));eng.add_strategy(st);eng.run();rows.append(st.summary());eng.dispose()
    hashes={r["event_hash"] for r in rows};parity=len(hashes)==1
    result=dict(verification_level="PR122_RAW_BIDASK_NAUTILUS_21D_V1",nautilus_version=getattr(nautilus_trader,"__version__","unknown"),period={"start":str(START),"end_exclusive":str(END_EXCL),"trading_days":21},initial_usd=INITIAL,leverage=2000,raw_ticks=len(ticks),frozen_core={"trigger":TRIGGER,"add":ADD,"reversal":REVERSAL,"max_layers":MAX_LAYERS},event_sequence_parity=parity,rows=rows,wr5={"status":"INVALID","present":["raw_bidask_spread","mark_to_market_dd","event_sequence_hash"],"missing":["broker_commission","slippage","execution_delay","swap_if_relevant","cashback","verified_native_margin_path","wfo","monte_carlo"]},limitations=["Supervisor ledger supports fractional size multipliers virtually; native market-order execution parity is a later gate.","Supervisor inputs use only previously closed cycle returns; no future information.","No profitability promotion permitted while WR5 INVALID."])
    (out/"summary.json").write_text(json.dumps(result,indent=2));pd.DataFrame(rows).to_csv(out/"kpi.csv",index=False);print(json.dumps(result,indent=2))
    if not parity:raise SystemExit("INVALID:event sequence parity failed")
if __name__=="__main__":main()
