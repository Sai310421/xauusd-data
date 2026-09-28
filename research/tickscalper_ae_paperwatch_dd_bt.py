from __future__ import annotations
import argparse, json, math
from collections import deque
from decimal import Decimal
from pathlib import Path

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType, BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from research.tickscalper_frozen4_bt import FrozenFourPartStrategy
from research.tickscalper_nautilus_raw_bt import Cfg, fix_sizes

class PaperWatchDDStrategy(FrozenFourPartStrategy):
    """
    DD research adapter from AE Math Paper Watch:
    - Drawdown reserve Phi_D = (1 - D/Dmax)^gamma
    - DD velocity/acceleration guard
    - first-passage/chance style rescue veto
    The underlying TickScalper entry/exit core is unchanged.
    """

    def __init__(self, c, dd_mode="full", dd_max=45.0, gamma=2.0, lam1=10.0, lam2=5.0, p_tail_max=0.85):
        super().__init__(c)
        self.dd_mode=dd_mode
        self.dd_max=dd_max
        self.gamma=gamma
        self.lam1=lam1
        self.lam2=lam2
        self.p_tail_max=p_tail_max
        self.dd_hist=deque(maxlen=8)
        self.dd_guard_stats={"scaled":0,"blocked":0,"reserve_min":1.0,"velocity_guard_min":1.0}
        self.dd_last={}

    def _dd_guard(self):
        d=self._floating_dd_pct()
        self.dd_hist.append((getattr(self,"now_ns",0),d))
        reserve=max(0.0,1.0-d/max(self.dd_max,1e-9))**self.gamma
        vel=acc=0.0
        if len(self.dd_hist)>=2:
            t1,d1=self.dd_hist[-2]; t2,d2=self.dd_hist[-1]
            dt=max((t2-t1)/1e9,1e-3)
            vel=(d2-d1)/dt
        if len(self.dd_hist)>=3:
            t0,d0=self.dd_hist[-3]; t1,d1=self.dd_hist[-2]; t2,d2=self.dd_hist[-1]
            dt1=max((t1-t0)/1e9,1e-3); dt2=max((t2-t1)/1e9,1e-3)
            v1=(d1-d0)/dt1; v2=(d2-d1)/dt2
            acc=(v2-v1)/max((dt1+dt2)/2.0,1e-3)
        vg=math.exp(-self.lam1*max(vel,0.0)-self.lam2*max(acc,0.0))
        self.dd_guard_stats["reserve_min"]=min(self.dd_guard_stats["reserve_min"],reserve)
        self.dd_guard_stats["velocity_guard_min"]=min(self.dd_guard_stats["velocity_guard_min"],vg)
        return reserve,vg,vel,acc

    def _four_part_fraction(self):
        frac=super()._four_part_fraction()
        if frac<=0:
            return 0.0

        reserve,vg,vel,acc=self._dd_guard()
        D,Dstar,mu,sigma,gross=self._ae_v1_boundary()
        pnr=self._ae_pnr(D,mu,sigma,gross)
        tail_prob=1.0-pnr

        scale=1.0
        if self.dd_mode in ("reserve","full"):
            scale=min(scale,reserve)
        if self.dd_mode in ("velocity","full"):
            scale=min(scale,vg)
        if self.dd_mode in ("chance","full") and tail_prob>self.p_tail_max:
            self.dd_guard_stats["blocked"]+=1
            self.dd_last={"dd":self._floating_dd_pct(),"reserve":reserve,"velocity_guard":vg,"vel":vel,"acc":acc,"p_NR":pnr,"tail_prob":tail_prob,"decision":"BLOCK"}
            return 0.0

        out=frac*scale
        if out<0.01 and frac>0:
            # Below 1% rescue proposal: prefer WAIT instead of forced minimum lot.
            self.dd_guard_stats["blocked"]+=1
            out=0.0
        elif out<frac:
            self.dd_guard_stats["scaled"]+=1

        self.dd_last={"dd":self._floating_dd_pct(),"reserve":reserve,"velocity_guard":vg,"vel":vel,"acc":acc,"p_NR":pnr,"tail_prob":tail_prob,"base_fraction":frac,"final_fraction":out,"decision":"ALLOW" if out>0 else "WAIT"}
        return out

    def summary(self):
        out=super().summary()
        out.update({"dd_mode":self.dd_mode,"dd_guard_stats":self.dd_guard_stats,"dd_last":self.dd_last})
        return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",required=True)
    ap.add_argument("--experiment-id",required=True)
    ap.add_argument("--dd-mode",choices=["none","reserve","velocity","chance","full"],default="full")
    ap.add_argument("--dd-max",type=float,default=45.0)
    ap.add_argument("--gamma",type=float,default=2.0)
    ap.add_argument("--lam1",type=float,default=10.0)
    ap.add_argument("--lam2",type=float,default=5.0)
    ap.add_argument("--p-tail-max",type=float,default=0.85)
    ap.add_argument("--fast-frac",type=float,default=0.06)
    ap.add_argument("--raw-bidask-only",action="store_true")
    a=ap.parse_args()
    if not a.raw_bidask_only: raise SystemExit("raw-bidask-only mandatory")

    cat=ParquetDataCatalog(a.catalog)
    inst=next((x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD"),None)
    if inst is None: raise SystemExit("XAUUSD missing")
    ticks=fix_sizes(cat.query_quote_ticks(identifiers=[inst.id.value]))

    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"))
    eng.add_instrument(inst);eng.add_data(ticks)

    cfg=Cfg(
        instrument_id=inst.id,base_qty=Decimal("0.30"),direction_sign=1,
        session_start_hour=7,session_end_hour=17,basket_offset=0.8575,max_layers=10,
        entry_mode="ticksmoother",ts_ticks_per_bar=5,ts_fast=3,ts_slow=5,ts_conf1=8,ts_conf2=13,ts_cross_only=True,
        risk_mode="none",rescue_mode="none",math_rescue_mode="none",
        eligibility_mode="conservative",eligibility_fast_frac=a.fast_frac,eligibility_deep_frac=0.0,eligibility_hard_fdd=a.dd_max,
        ae_stack_mode="none",ae_k=0.20,ae_rho=0.08,ae_lambda=1.0,ae_tail_distance=8.0,ae_rescue_cap=a.fast_frac,ae_hard_fdd=a.dd_max,
        fp_block_threshold=a.p_tail_max,
    )
    st=PaperWatchDDStrategy(cfg,dd_mode=a.dd_mode,dd_max=a.dd_max,gamma=a.gamma,lam1=a.lam1,lam2=a.lam2,p_tail_max=a.p_tail_max)
    eng.add_strategy(st);eng.run()
    fills=eng.trader.generate_order_fills_report()
    out={"verification_level":"NAUTILUS_RAW_BIDASK_AE_PAPERWATCH_DD_CANDIDATE","raw_ticks":len(ticks),"native_fills":len(fills) if fills is not None else 0,"ohlc_resample_used":False,"dd_mode":a.dd_mode,"dd_max":a.dd_max,"gamma":a.gamma,"lam1":a.lam1,"lam2":a.lam2,"p_tail_max":a.p_tail_max,"fast_frac":a.fast_frac,**st.summary()}
    outdir=Path("results/tickscalper-nautilus")/a.experiment_id
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/"kpi.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))
    eng.dispose()

if __name__=="__main__":
    main()
