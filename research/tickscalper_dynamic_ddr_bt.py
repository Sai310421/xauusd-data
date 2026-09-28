from __future__ import annotations
import argparse, json, math
from decimal import Decimal
from pathlib import Path
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType, BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from research.tickscalper_nautilus_raw_bt import Cfg, TickScalperCandidate, fix_sizes

class DynamicDDRStrategy(TickScalperCandidate):
    """AE Math Paper Watch Dynamic DDR applied only to ADD exposure.
    FAST Rescue remains the validated 6% eligibility rescue.
    """
    def __init__(self,c,ddr_cap=25.0,ddr_lambda=2.0,ddr_start_layer=7,ddr_min_scale=0.5):
        super().__init__(c)
        self.ddr_cap=float(ddr_cap); self.ddr_lambda=float(ddr_lambda)
        self.ddr_start_layer=int(ddr_start_layer); self.ddr_min_scale=float(ddr_min_scale)
        self.ddr_scaled=0; self.ddr_min_seen=1.0; self.ddr_last={}

    def _risk_add_scale(self,prospective_lot):
        if len(self.entries)<self.ddr_start_layer:
            return 1.0
        dd=self._floating_dd_pct()
        distance=max(0.0,self.ddr_cap-dd)
        x=min(1.0,distance/max(self.ddr_cap,1e-12))
        den=1.0-math.exp(-self.ddr_lambda)
        raw=(1.0-math.exp(-self.ddr_lambda*x))/den if den>1e-12 else x
        scale=max(self.ddr_min_scale,min(1.0,raw))
        if scale<0.999:self.ddr_scaled+=1
        self.ddr_min_seen=min(self.ddr_min_seen,scale)
        self.ddr_last={"fdd":dd,"cap":self.ddr_cap,"distance":distance,"lambda":self.ddr_lambda,"raw_scale":raw,"scale":scale,"layer":len(self.entries)}
        return scale

    def summary(self):
        out=super().summary()
        out.update({"ddr_cap":self.ddr_cap,"ddr_lambda":self.ddr_lambda,"ddr_start_layer":self.ddr_start_layer,"ddr_min_scale":self.ddr_min_scale,"ddr_scaled":self.ddr_scaled,"ddr_min_seen":self.ddr_min_seen,"ddr_last":self.ddr_last})
        return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",required=True);ap.add_argument("--experiment-id",required=True)
    ap.add_argument("--ddr-cap",type=float,required=True);ap.add_argument("--ddr-lambda",type=float,required=True)
    ap.add_argument("--ddr-start-layer",type=int,required=True);ap.add_argument("--ddr-min-scale",type=float,default=0.5)
    ap.add_argument("--fast-frac",type=float,default=0.06);ap.add_argument("--raw-bidask-only",action="store_true")
    a=ap.parse_args()
    if not a.raw_bidask_only:raise SystemExit("raw-bidask-only mandatory")
    cat=ParquetDataCatalog(a.catalog);inst=next((x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD"),None)
    if inst is None:raise SystemExit("XAUUSD missing")
    ticks=fix_sizes(cat.query_quote_ticks(identifiers=[inst.id.value]))
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"))
    eng.add_instrument(inst);eng.add_data(ticks)
    cfg=Cfg(instrument_id=inst.id,base_qty=Decimal("0.30"),direction_sign=1,session_start_hour=7,session_end_hour=17,basket_offset=0.8575,max_layers=10,
        entry_mode="ticksmoother",ts_ticks_per_bar=5,ts_fast=3,ts_slow=5,ts_conf1=8,ts_conf2=13,ts_cross_only=True,
        risk_mode="none",rescue_mode="none",math_rescue_mode="none",eligibility_mode="conservative",eligibility_fast_frac=a.fast_frac,eligibility_deep_frac=0.0,eligibility_hard_fdd=45.0,ae_stack_mode="none")
    st=DynamicDDRStrategy(cfg,a.ddr_cap,a.ddr_lambda,a.ddr_start_layer,a.ddr_min_scale)
    eng.add_strategy(st);eng.run()
    fills=eng.trader.generate_order_fills_report()
    out={"verification_level":"NAUTILUS_RAW_BIDASK_DYNAMIC_DDR_FAST6","raw_ticks":len(ticks),"native_fills":len(fills) if fills is not None else 0,"ohlc_resample_used":False,"paperwatch_formula":"scale=(1-exp(-lambda*distance/cap))/(1-exp(-lambda)); floor=min_scale","fast_rescue_fraction":a.fast_frac,**st.summary()}
    p=Path("results/tickscalper-nautilus")/a.experiment_id;p.mkdir(parents=True,exist_ok=True);(p/"kpi.json").write_text(json.dumps(out,indent=2),encoding="utf-8");print(json.dumps(out,indent=2));eng.dispose()
if __name__=="__main__":main()
