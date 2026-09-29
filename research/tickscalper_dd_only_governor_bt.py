from __future__ import annotations
import argparse,json
from decimal import Decimal
from pathlib import Path
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType,OmsType,BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from research.tickscalper_nautilus_raw_bt import Cfg,fix_sizes
from research.tickscalper_dynamic_ddr_bt import DynamicDDRStrategy

class DDOnlyGovernor(DynamicDDRStrategy):
    """Freeze core/exit/FAST6. DD-only external governor on ADD sizing.
    Dynamic DDR + continuous First-Passage + state-asymmetric empirical CVaR.
    No hard FP veto: preserve recovery path and avoid prior over-control.
    """
    def __init__(self,c,cap,lam,start,minscale,fp_lo,fp_hi,fp_floor,cvar_budget,cvar_floor):
        super().__init__(c,cap,lam,start,minscale)
        self.fp_lo=fp_lo;self.fp_hi=fp_hi;self.fp_floor=fp_floor
        self.cvar_budget=cvar_budget;self.cvar_floor=cvar_floor
        self.fp_scaled=0;self.cvar_scaled=0;self.gov_min=1.0

    def _risk_add_scale(self,prospective_lot):
        ddr=super()._risk_add_scale(prospective_lot)
        if len(self.entries)<self.ddr_start_layer:return 1.0
        p=self._first_passage_adverse_prob(prospective_lot);self.fp_last_prob=p
        if p<=self.fp_lo: fps=1.0
        elif p>=self.fp_hi: fps=self.fp_floor
        else: fps=1.0-(1.0-self.fp_floor)*(p-self.fp_lo)/max(self.fp_hi-self.fp_lo,1e-9)
        if fps<.999:self.fp_scaled+=1

        # State-asymmetric CVaR: only binds when empirical tail budget is exceeded.
        cv=self._empirical_cvar_pct(prospective_lot)
        cvs=1.0
        if cv>self.cvar_budget:
            cvs=max(self.cvar_floor,min(1.0,self.cvar_budget/max(cv,1e-9)))
            self.cvar_scaled+=1
        scale=min(ddr,fps,cvs)
        self.gov_min=min(self.gov_min,scale)
        return scale

    def summary(self):
        o=super().summary();o.update({"governor":"DDR+continuous_FP+state_asymmetric_CVaR",
          "fp_lo":self.fp_lo,"fp_hi":self.fp_hi,"fp_floor":self.fp_floor,
          "cvar_budget":self.cvar_budget,"cvar_floor":self.cvar_floor,
          "fp_scaled":self.fp_scaled,"cvar_scaled":self.cvar_scaled,"governor_min":self.gov_min})
        return o

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--catalog",required=True);ap.add_argument("--experiment-id",required=True)
 ap.add_argument("--cap",type=float,default=28);ap.add_argument("--lam",type=float,default=3)
 ap.add_argument("--fp-lo",type=float,required=True);ap.add_argument("--fp-hi",type=float,required=True);ap.add_argument("--fp-floor",type=float,required=True)
 ap.add_argument("--cvar-budget",type=float,required=True);ap.add_argument("--cvar-floor",type=float,required=True)
 ap.add_argument("--raw-bidask-only",action="store_true");a=ap.parse_args()
 if not a.raw_bidask_only:raise SystemExit("raw-bidask-only mandatory")
 cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
 ticks=fix_sizes(cat.query_quote_ticks(identifiers=[inst.id.value]))
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"))
 eng.add_instrument(inst);eng.add_data(ticks)
 c=Cfg(instrument_id=inst.id,base_qty=Decimal(".30"),direction_sign=1,session_start_hour=7,session_end_hour=17,basket_offset=.8575,max_layers=10,
 entry_mode="ticksmoother",ts_ticks_per_bar=5,ts_fast=3,ts_slow=5,ts_conf1=8,ts_conf2=13,ts_cross_only=True,
 risk_mode="none",rescue_mode="none",math_rescue_mode="none",eligibility_mode="conservative",eligibility_fast_frac=.06,eligibility_deep_frac=0.0,eligibility_hard_fdd=45.0,ae_stack_mode="none")
 st=DDOnlyGovernor(c,a.cap,a.lam,8,.50,a.fp_lo,a.fp_hi,a.fp_floor,a.cvar_budget,a.cvar_floor)
 eng.add_strategy(st);eng.run()
 out={"verification_level":"RAW_BIDASK_DD_ONLY_GOVERNOR","raw_ticks":len(ticks),"core_frozen":True,"entry_exit_changed":False,**st.summary()}
 p=Path("results/tickscalper-nautilus")/a.experiment_id;p.mkdir(parents=True,exist_ok=True);(p/"kpi.json").write_text(json.dumps(out,indent=2),encoding="utf-8");print(json.dumps(out,indent=2));eng.dispose()
if __name__=="__main__":main()

# trigger DD-only governor sweep

# retrigger 2

# trigger fixed workflow

# trigger after yaml newline repair
