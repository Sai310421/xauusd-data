from __future__ import annotations
import argparse, json
from decimal import Decimal
from pathlib import Path
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType, BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from research.tickscalper_nautilus_raw_bt import Cfg, fix_sizes
from research.tickscalper_dynamic_ddr_bt import DynamicDDRStrategy

class CashbackDDRStrategy(DynamicDDRStrategy):
    def __init__(self,c,ddr_cap,ddr_lambda,ddr_start_layer,ddr_min_scale,cashback_per_lot=6.0):
        super().__init__(c,ddr_cap,ddr_lambda,ddr_start_layer,ddr_min_scale)
        self.cashback_per_lot=float(cashback_per_lot)
        self.cashback_closed_lots=0.0
        self.cashback_usd=0.0
        self.cb_trade_pnls=[]

    def _close(self,reason):
        closed_lots=sum(l for _,l in self.entries)+sum(l for _,_,l in self.rescue_legs)
        n0=len(self.trades)
        super()._close(reason)
        if len(self.trades)>n0:
            cb=closed_lots*self.cashback_per_lot
            self.cashback_closed_lots+=closed_lots
            self.cashback_usd+=cb
            self.cb_trade_pnls.append(self.trades[-1]["pnl"]+cb)

    def summary(self):
        out=super().summary()
        gp=sum(x for x in self.cb_trade_pnls if x>0)
        gl=sum(-x for x in self.cb_trade_pnls if x<0)
        n=len(self.cb_trade_pnls)
        wins=sum(x>0 for x in self.cb_trade_pnls)
        net_cb=self.net+self.cashback_usd
        out.update({
            "cashback_usd_per_closed_lot":self.cashback_per_lot,
            "cashback_closed_lots":self.cashback_closed_lots,
            "cashback_usd":self.cashback_usd,
            "Net_with_cashback":net_cb,
            "Return_with_cashback_pct":100.0*net_cb/1000.0,
            "WR_with_cashback_pct":100.0*wins/max(n,1),
            "PF_with_cashback":gp/gl if gl else None,
            "cashback_return_contribution_pct":100.0*self.cashback_usd/1000.0,
        })
        return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",required=True);ap.add_argument("--experiment-id",required=True)
    ap.add_argument("--cashback-per-lot",type=float,default=6.0)
    ap.add_argument("--raw-bidask-only",action="store_true")
    a=ap.parse_args()
    if not a.raw_bidask_only:raise SystemExit("raw-bidask-only mandatory")
    cat=ParquetDataCatalog(a.catalog)
    inst=next((x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD"),None)
    if inst is None:raise SystemExit("XAUUSD missing")
    ticks=fix_sizes(cat.query_quote_ticks(identifiers=[inst.id.value]))
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"))
    eng.add_instrument(inst);eng.add_data(ticks)
    cfg=Cfg(instrument_id=inst.id,base_qty=Decimal("0.30"),direction_sign=1,session_start_hour=7,session_end_hour=17,
        basket_offset=0.8575,max_layers=10,entry_mode="ticksmoother",ts_ticks_per_bar=5,ts_fast=3,ts_slow=5,ts_conf1=8,ts_conf2=13,ts_cross_only=True,
        risk_mode="none",rescue_mode="none",math_rescue_mode="none",eligibility_mode="conservative",eligibility_fast_frac=.06,
        eligibility_deep_frac=0.0,eligibility_hard_fdd=45.0,ae_stack_mode="none")
    st=CashbackDDRStrategy(cfg,28.0,3.0,8,0.50,a.cashback_per_lot)
    eng.add_strategy(st);eng.run()
    fills=eng.trader.generate_order_fills_report()
    out={"verification_level":"NAUTILUS_RAW_BIDASK_DYNAMIC_DDR_FAST6_CASHBACK_LEDGER",
         "raw_ticks":len(ticks),"native_fills":len(fills) if fills is not None else 0,
         "ohlc_resample_used":False,"cashback_assumption":"USD 6 per closed lot (round-turn ledger)",**st.summary()}
    p=Path("results/tickscalper-nautilus")/a.experiment_id;p.mkdir(parents=True,exist_ok=True)
    (p/"kpi.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2));eng.dispose()
if __name__=="__main__":main()

# workflow trigger: cashback KPI verification
