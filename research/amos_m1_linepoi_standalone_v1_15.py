#!/usr/bin/env python3
"""AMOS M1 standalone v1.15.
M1 only. No M5 gate, no M15 AMD, no G75.
Reuses the frozen v19 image-interpretation for six Close-line POI patterns:
Classic V/A, QM Buy/Sell, OCL Buy/Sell.
Raw XAUUSD Bid/Ask QuoteTicks from Nautilus Parquet catalog. No OHLC input/fallback.
"""
from __future__ import annotations
import argparse, json, math, importlib.util, sys
from decimal import Decimal
from pathlib import Path
import numpy as np, pandas as pd
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

spec=importlib.util.spec_from_file_location("v19",Path(__file__).with_name("m1_linepoi_m5_g75_overlap_v19.py"))
v19=importlib.util.module_from_spec(spec);sys.modules["v19"]=v19;spec.loader.exec_module(v19)

INITIAL=1000.0

class M1Only(v19.M1LinePOIPlusM5G75OverlapV19):
    def __init__(self,c):
        super().__init__(c)
        self.m1_realized=0.0;self.m1_peak=INITIAL;self.m1_maxdd=0.0
        self.m1_max_gross=0.0

    def _m1_close(self,x,bid,ask,ts,reason):
        n=len(self.m1_closed)
        super()._m1_close(x,bid,ask,ts,reason)
        if len(self.m1_closed)>n:self.m1_realized+=float(self.m1_closed[-1]["pnl"])

    def _m1_risk(self,bid,ask):
        floating=0.0;gross=0.0
        for x in self.m1_legs:
            if not x.active:continue
            mark=bid if x.direction>0 else ask
            floating+=(mark-x.entry)*x.direction*v19.BASE_QTY
            gross+=v19.BASE_QTY
        eq=INITIAL+self.m1_realized+floating
        self.m1_peak=max(self.m1_peak,eq)
        self.m1_maxdd=max(self.m1_maxdd,self.m1_peak-eq)
        self.m1_max_gross=max(self.m1_max_gross,gross)

    def on_quote_tick(self,t):
        bid,ask=v19.ff(t.bid_price),v19.ff(t.ask_price);ts=int(t.ts_event)
        self.last_bid,self.last_ask,self.last_ns=bid,ask,ts
        if self.first_ns is None:self.first_ns=ts
        self.last_seen_ns=ts
        closed=self._m1_bar_update(ts,bid)
        self._m1_setup_tick(bid,ask,ts)
        self._m1_manage(bid,ask,ts)
        if closed:
            self._m1_after_close(ts)
            self._m1_setup_tick(bid,ask,ts)
        self._m1_risk(bid,ask)

    def on_stop(self):
        if self.last_bid is None:return
        for x in list(self.m1_legs):
            if x.active:self._m1_close(x,self.last_bid,self.last_ask,self.last_ns,"TIMEOUT")
        self._m1_risk(self.last_bid,self.last_ask)

    def metrics(self):
        rows=self.m1_closed;p=np.array([r["pnl"] for r in rows],float)
        gp=float(p[p>0].sum()) if len(p) else 0.;gl=float(-p[p<0].sum()) if len(p) else 0.
        net=float(p.sum()) if len(p) else 0.
        start=pd.Timestamp(self.first_ns,unit="ns",tz="UTC").date();end=pd.Timestamp(self.last_seen_ns,unit="ns",tz="UTC").date()
        bd=max(1,len(pd.bdate_range(start,end)));scale=21.0/bd
        by={}
        for pat in self.m1_stats:
            q=np.array([r["pnl"] for r in rows if r["pattern"]==pat],float)
            qgp=float(q[q>0].sum()) if len(q) else 0.;qgl=float(-q[q<0].sum()) if len(q) else 0.
            by[pat]={"N":int(len(q)),"WR_pct":float((q>0).mean()*100) if len(q) else 0.,
                     "PF":qgp/qgl if qgl>0 else (math.inf if qgp>0 else 0.),"Net_USD":float(q.sum()) if len(q) else 0.}
        return {"N":int(len(p)),"WR_pct":float((p>0).mean()*100) if len(p) else 0.,
                "PF":gp/gl if gl>0 else (math.inf if gp>0 else 0.),"Net_USD":net,
                "Return_pct":net/INITIAL*100,"MaxFloatingDD_USD":self.m1_maxdd,
                "MaxFloatingDD_pct_initial":self.m1_maxdd/INITIAL*100,
                "RF":net/self.m1_maxdd if self.m1_maxdd>0 else None,
                "BusinessDays":bd,"N21_linearized":len(p)*scale,
                "Monthly21_pct_linearized":net*scale/INITIAL*100,
                "MaxGrossLots_approx":self.m1_max_gross/100.0,
                "pattern_metrics":by,"pattern_stats":self.m1_stats}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
    raw=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value]);raw.sort(key=lambda t:int(t.ts_event))
    ticks,repl=v19.executable(raw)
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,
                  base_currency=USD,starting_balances=[Money(INITIAL,USD)],default_leverage=Decimal("2000"))
    eng.add_instrument(inst);eng.add_data(ticks)
    st=M1Only(v19.Cfg(instrument_id=inst.id,mode="base"));eng.add_strategy(st);eng.run();eng.end()
    pd.DataFrame(st.m1_closed).to_csv(out/"trades.csv",index=False)
    res={"version":"v1.15","verification_level":"M1_STANDALONE_NAUTILUS_RAW_BIDASK",
         "architecture":"M1 Close-line V/A/QM/OCL -> POI raw revisit -> favorable M1 Close reaction -> raw Bid/Ask entry",
         "m5_used":False,"m15_used":False,"g75_used":False,"ohlc_input_used":False,
         "raw_ticks":len(raw),"execution_ticks":len(ticks),"zero_size_replaced":repl,
         "cost_assumption":{"commission_rt_per_lot":v19.COMMISSION_RT_PER_LOT,"cashback_rt_per_lot":v19.CASHBACK_RT_PER_LOT},
         "rule_frozen_from":"m1_linepoi_m5_g75_overlap_v19.py",**st.metrics()}
    (out/"result.json").write_text(json.dumps(res,indent=2,default=str),encoding="utf-8")
    print(json.dumps(res,indent=2,default=str));eng.dispose()

if __name__=="__main__":main()
