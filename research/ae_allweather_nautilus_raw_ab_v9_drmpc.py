from __future__ import annotations
import argparse, json
from pathlib import Path
from research.ae_allweather_nautilus_raw_ab_v8 import V8Cfg, ABStrategyV8, run_cell as run_v8
from research.ae_dr_mpc_recovery import RecoveryState, Disturbance, disturbance_affine_control


class ABStrategyV9DRMPC(ABStrategyV8):
    """V8 + AE-derived disturbance-affine DR-MPC proxy.

    Uses only contemporaneous V8 recovery state. It does not claim full Wasserstein
    min-max optimality; this is the Raw Bid/Ask A/B bridge for promotion testing.
    """
    def __init__(self, config):
        super().__init__(config)
        self.drmpc_calls=0; self.drmpc_add_scale_sum=0.0; self.drmpc_robust_peak=0.0

    def _drmpc_control(self,bid,ask):
        debt=self._tail_debt(bid,ask); cap=self._recovery_capacity(bid,ask)
        ratio=self._recovery_ratio(bid,ask)
        tails=self._tail_rows(bid,ask)
        max_age=max((r[3] for r in tails),default=0)
        debt_ratio=debt/max(debt+cap,1e-9)
        drift_ratio=min(1.5,self.tail_velocity_ema/max(self.config.tail_velocity_stop,1e-9))
        mae_ratio=min(1.5,debt/max(self.config.early_tail_loss,1e-9))
        age_ratio=min(1.5,max_age/max(self.config.early_tail_age,1))
        rec_err=0.0 if ratio==float('inf') else max(0.0,1.0-min(1.0,ratio/max(self.config.recovery_ratio_soft,1e-9)))
        dd=getattr(self,'max_dd_pct',0.0)
        state=RecoveryState(debt_ratio,drift_ratio,mae_ratio,age_ratio,debt_ratio,0.0,drift_ratio,min(1.5,dd/10.0))
        dist=Disturbance(drift_ratio,0.0,drift_ratio,rec_err)
        c=disturbance_affine_control(state,dist)
        self.drmpc_calls+=1; self.drmpc_add_scale_sum+=c.add_scale; self.drmpc_robust_peak=max(self.drmpc_robust_peak,c.robust_score)
        return c

    def _stress_multiplier(self,bid,ask):
        base=super()._stress_multiplier(bid,ask)
        c=self._drmpc_control(bid,ask)
        return min(base,c.add_scale)

    def summary(self):
        out=super().summary()
        out.update({'verification_level_v9':'V8_PLUS_DR_MPC_PROXY_RAW_BIDASK','drmpc_calls':self.drmpc_calls,'drmpc_mean_add_scale':self.drmpc_add_scale_sum/max(self.drmpc_calls,1),'drmpc_robust_peak':self.drmpc_robust_peak})
        return out


def run_v9(catalog_path,mode,tf,experiment_id):
    # Reuse V8 engine contract, substituting only strategy class.
    import nautilus_trader
    from decimal import Decimal
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.backtest.config import BacktestEngineConfig
    from nautilus_trader.config import LoggingConfig, RiskEngineConfig
    from nautilus_trader.model import BarType, Money
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.persistence.catalog import ParquetDataCatalog
    catalog=ParquetDataCatalog(str(catalog_path)); inst=next(x for x in catalog.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    ticks=catalog.query_quote_ticks(identifiers=[inst.id.value])
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    eng.add_instrument(inst); eng.add_data(ticks)
    bt=BarType.from_str(f'{inst.id.value}-{tf}-MINUTE-BID-INTERNAL')
    st=ABStrategyV9DRMPC(V8Cfg(instrument_id=inst.id,bar_type=bt,mode=mode)); eng.add_strategy(st); eng.run()
    obj={'verification_level':'NAUTILUS_RAW_BIDASK_AE_V8_PLUS_DR_MPC_PROXY','raw_ticks':len(ticks),'ohlc_resample_used':False,'tf_minutes':tf,**st.summary()}; eng.dispose(); return obj


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--tf',type=int,default=1); a=ap.parse_args()
    p=Path('results/math-edge-v2')/a.experiment_id; p.mkdir(parents=True,exist_ok=True)
    base=run_v8(Path(a.catalog),'CLUSTER',a.tf,a.experiment_id+'-base')
    dr=run_v9(Path(a.catalog),'CLUSTER',a.tf,a.experiment_id+'-drmpc')
    out={'base_v8':base,'drmpc_v9':dr}
    (p/'summary.json').write_text(json.dumps(out,indent=2,default=str),encoding='utf-8'); print(json.dumps(out,indent=2,default=str))

if __name__=='__main__': main()
