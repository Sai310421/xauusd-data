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

from research.tickscalper_frozen4_bt import FrozenFourPartStrategy
from research.tickscalper_nautilus_raw_bt import Cfg, fix_sizes

class UnifiedMathController(FrozenFourPartStrategy):
    """
    Unified numerical controller:
      Core entry/exit frozen
      -> Dynamic DDR controls ADD exposure
      -> First-Passage controls ADD allow/reduce/wait
      -> AE v1/v2/v3 + Frozen Charter controls FAST Rescue
      -> hard FDD => SAFE
    """
    def __init__(self,c,ddr_cap=28.0,ddr_lambda=3.0,ddr_start_layer=8,ddr_min_scale=0.5,
                 fp_reduce=0.60,fp_block=0.78):
        super().__init__(c)
        self.ddr_cap=float(ddr_cap); self.ddr_lambda=float(ddr_lambda)
        self.ddr_start_layer=int(ddr_start_layer); self.ddr_min_scale=float(ddr_min_scale)
        self.fp_reduce=float(fp_reduce); self.fp_block=float(fp_block)
        self.ctrl_counts={"ALLOW":0,"REDUCE":0,"WAIT":0,"SAFE":0}
        self.ddr_scaled=0; self.fp_scaled=0; self.fp_blocked=0
        self.ddr_min_seen=1.0; self.fp_last=0.0; self.ctrl_last={}

    def _ddr_scale(self):
        dd=self._floating_dd_pct()
        distance=max(0.0,self.ddr_cap-dd)
        x=min(1.0,distance/max(self.ddr_cap,1e-12))
        den=1.0-math.exp(-self.ddr_lambda)
        raw=(1.0-math.exp(-self.ddr_lambda*x))/den if den>1e-12 else x
        scale=max(self.ddr_min_scale,min(1.0,raw))
        return scale,dd,raw

    def _fp_scale(self,prospective_lot):
        p=self._first_passage_adverse_prob(prospective_lot)
        self.fp_last=p
        if p>=self.fp_block:
            self.fp_blocked+=1
            return 0.0,p,"WAIT"
        if p>=self.fp_reduce:
            frac=(self.fp_block-p)/max(self.fp_block-self.fp_reduce,1e-9)
            scale=max(self.ddr_min_scale,min(1.0,frac))
            self.fp_scaled+=1
            return scale,p,"REDUCE"
        return 1.0,p,"ALLOW"

    def _risk_add_scale(self,prospective_lot):
        if len(self.entries)<self.ddr_start_layer:
            self.ctrl_counts["ALLOW"]+=1
            return 1.0
        if self._floating_dd_pct()>=self.config.ae_hard_fdd:
            self.ctrl_counts["SAFE"]+=1
            self.ctrl_last={"action":"SAFE","reason":"HARD_FDD","fdd":self._floating_dd_pct(),"layer":len(self.entries)}
            return 0.0
        ds,dd,raw=self._ddr_scale()
        fs,p,fp_action=self._fp_scale(prospective_lot)
        final=min(ds,fs)
        if ds<0.999:self.ddr_scaled+=1
        self.ddr_min_seen=min(self.ddr_min_seen,ds)
        if final<=0.0:
            action="WAIT"
        elif final<0.999:
            action="REDUCE"
        else:
            action="ALLOW"
        self.ctrl_counts[action]+=1
        self.ctrl_last={"action":action,"layer":len(self.entries),"fdd":dd,
                        "ddr_raw":raw,"ddr_scale":ds,"fp_prob":p,"fp_action":fp_action,
                        "fp_scale":fs,"final_add_scale":final}
        return final

    def _four_part_fraction(self):
        # AE four-part rescue supervisor remains authoritative for rescue.
        frac=super()._four_part_fraction()
        if self._floating_dd_pct()>=self.config.ae_hard_fdd:
            self.ctrl_counts["SAFE"]+=1
            return 0.0
        return frac

    def summary(self):
        out=super().summary()
        out.update({
            "controller_architecture":"DDR_ADD -> FIRST_PASSAGE_ADD -> AE_V1_V2_V3_FROZEN4_RESCUE -> SAFE",
            "ddr_cap":self.ddr_cap,"ddr_lambda":self.ddr_lambda,
            "ddr_start_layer":self.ddr_start_layer,"ddr_min_scale":self.ddr_min_scale,
            "fp_reduce":self.fp_reduce,"fp_block":self.fp_block,
            "controller_counts":self.ctrl_counts,"ddr_scaled":self.ddr_scaled,
            "fp_scaled":self.fp_scaled,"fp_blocked":self.fp_blocked,
            "ddr_min_seen":self.ddr_min_seen,"fp_last":self.fp_last,"controller_last":self.ctrl_last,
        })
        return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",required=True); ap.add_argument("--experiment-id",required=True)
    ap.add_argument("--ddr-cap",type=float,default=28.0); ap.add_argument("--ddr-lambda",type=float,default=3.0)
    ap.add_argument("--ddr-start-layer",type=int,default=8); ap.add_argument("--ddr-min-scale",type=float,default=0.5)
    ap.add_argument("--fp-reduce",type=float,default=0.60); ap.add_argument("--fp-block",type=float,default=0.78)
    ap.add_argument("--fast-frac",type=float,default=0.06); ap.add_argument("--hard-fdd",type=float,default=45.0)
    ap.add_argument("--raw-bidask-only",action="store_true")
    a=ap.parse_args()
    if not a.raw_bidask_only: raise SystemExit("raw-bidask-only mandatory")
    if not (0.0<a.fp_reduce<a.fp_block<1.0): raise SystemExit("require 0 < fp-reduce < fp-block < 1")

    cat=ParquetDataCatalog(a.catalog)
    inst=next((x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD"),None)
    if inst is None: raise SystemExit("XAUUSD missing")
    ticks=fix_sizes(cat.query_quote_ticks(identifiers=[inst.id.value]))
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,
        book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"))
    eng.add_instrument(inst); eng.add_data(ticks)

    cfg=Cfg(instrument_id=inst.id,base_qty=Decimal("0.30"),direction_sign=1,
        session_start_hour=7,session_end_hour=17,basket_offset=0.8575,max_layers=10,
        entry_mode="ticksmoother",ts_ticks_per_bar=5,ts_fast=3,ts_slow=5,ts_conf1=8,ts_conf2=13,ts_cross_only=True,
        risk_mode="none",rescue_mode="none",math_rescue_mode="none",
        eligibility_mode="conservative",eligibility_fast_frac=a.fast_frac,eligibility_deep_frac=0.0,
        eligibility_hard_fdd=a.hard_fdd,ae_stack_mode="none",ae_k=0.20,ae_rho=0.08,ae_lambda=1.0,
        ae_tail_distance=8.0,ae_rescue_cap=a.fast_frac,ae_hard_fdd=a.hard_fdd,fp_block_threshold=a.fp_block)
    st=UnifiedMathController(cfg,a.ddr_cap,a.ddr_lambda,a.ddr_start_layer,a.ddr_min_scale,a.fp_reduce,a.fp_block)
    eng.add_strategy(st); eng.run()
    fills=eng.trader.generate_order_fills_report()
    out={"verification_level":"NAUTILUS_RAW_BIDASK_UNIFIED_MATH_CONTROLLER_CANDIDATE",
         "raw_ticks":len(ticks),"native_fills":len(fills) if fills is not None else 0,
         "ohlc_resample_used":False,"fast_frac":a.fast_frac,"hard_fdd":a.hard_fdd,**st.summary()}
    p=Path("results/tickscalper-nautilus")/a.experiment_id; p.mkdir(parents=True,exist_ok=True)
    (p/"kpi.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2)); eng.dispose()

if __name__=="__main__": main()
