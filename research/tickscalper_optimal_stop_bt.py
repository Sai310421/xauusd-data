from __future__ import annotations
import argparse,json
from collections import deque
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

class OptimalStopStrategy(DynamicDDRStrategy):
    """Frozen Dynamic DDR + FAST6 core. Adds basket stop only; no entry/ADD/exit target changes."""
    def __init__(self,c,mode,loss_stop=0.0,fp_thr=0.0,fdd_thr=0.0,min_layer=8):
        super().__init__(c,28.0,3.0,8,.50)
        self.stop_mode=mode;self.loss_stop=float(loss_stop);self.stop_fp=float(fp_thr);self.stop_fdd=float(fdd_thr);self.stop_min_layer=int(min_layer)
        self.stop_count=0;self.stop_loss=0.0;self.stop_depths=[]

    def _stop_now(self):
        if not self.entries or len(self.entries)<self.stop_min_layer:return False
        pnl=self._floating_pnl()
        if pnl>=0:return False
        if self.stop_mode=="fixed":
            return pnl<=-self.loss_stop
        if self.stop_mode=="fp":
            p=self._first_passage_adverse_prob(0.0);self.fp_last_prob=p
            return p>=self.stop_fp
        if self.stop_mode=="fdd":
            return self._floating_dd_pct()>=self.stop_fdd
        if self.stop_mode=="fdd_fp":
            p=self._first_passage_adverse_prob(0.0);self.fp_last_prob=p
            return self._floating_dd_pct()>=self.stop_fdd and p>=self.stop_fp
        return False

    def _close(self,reason):
        if reason=="OPTIMAL_STOP":
            pnl=self._floating_pnl();self.stop_count+=1;self.stop_loss+=min(0.0,pnl);self.stop_depths.append(len(self.entries))
        return super()._close(reason)

    def on_quote_tick(self,t):
        # Update prices and state exactly as base does, but intercept after its standard state update is unsafe.
        # Therefore reproduce the preamble and then delegate unless stop condition fires.
        self.bid=float(t.bid_price.as_double());self.ask=float(t.ask_price.as_double());self.now_ns=int(t.ts_event)
        self._shadow_update()
        mid=(self.bid+self.ask)/2
        if self.prev_mid is not None:self.recent_mid_deltas.append(mid-self.prev_mid)
        self.prev_mid=mid
        self._update_mtm_dd()
        if self._stop_now():
            self._close("OPTIMAL_STOP");return
        # Delegate remaining logic without double state update by using local copy of core decision path.
        if not self.entries:
            if self.last_close_ns and (self.now_ns-self.last_close_ns)<self.config.cooldown_seconds*1_000_000_000:return
            import datetime
            h=datetime.datetime.fromtimestamp(t.ts_event/1e9,datetime.timezone.utc).hour
            if self.config.session_start_hour<=h<self.config.session_end_hour:
                sig=self._entry_signal()
                if sig:self._open(sig)
            return
        if self.rescue_active:
            if self.config.eligibility_mode!='none':self._handle_eligibility_rescue()
            else:self._handle_zr_rescue()
            return
        mark=self.bid if self.side>0 else self.ask;s=self.side;be=self._be()
        if (mark-be)*s>=self.config.basket_offset:self._close('BASKET');return
        last=self.entries[-1][0];adverse=(last-mark) if s>0 else (mark-last)
        if len(self.entries)<self.config.max_layers:
            from research.tickscalper_nautilus_raw_bt import layer_lot,floor_step
            raw_lot=layer_lot(float(self.config.base_qty),len(self.entries),self.config)
            eff_dist=self._effective_add_distance(raw_lot)
            if adverse>=eff_dist:
                scale=self._risk_add_scale(raw_lot);lot=floor_step(raw_lot*scale)
                if lot<0.01:self.risk_blocked_adds+=1;return
                if scale<.999:self.risk_scaled_adds+=1
                self._submit(s,lot);self.entries.append((self.ask if s>0 else self.bid,lot));self.max_layer=max(self.max_layer,len(self.entries));self.max_lots=max(self.max_lots,sum(l for _,l in self.entries));return
        if len(self.entries)>=self.config.max_layers:
            adverse2=(self.entries[-1][0]-mark) if s>0 else (mark-self.entries[-1][0])
            if adverse2>=self.config.emergency_distance:
                self._shadow_spawn_rescues()
                if self.config.eligibility_mode!='none':
                    if not self._start_eligibility_rescue():self._close('EMERGENCY_ELIGIBILITY_DENIED')
                elif not self._start_zr_rescue():self._close('EMERGENCY')

    def summary(self):
        o=super().summary();o.update({"stop_mode":self.stop_mode,"optimal_stop_count":self.stop_count,"optimal_stop_loss":self.stop_loss,"optimal_stop_depths":self.stop_depths})
        return o

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--experiment-id",required=True);ap.add_argument("--mode",choices=["none","fixed","fp","fdd_fp","fdd"],required=True)
 ap.add_argument("--loss-stop",type=float,default=0);ap.add_argument("--fp-thr",type=float,default=0);ap.add_argument("--fdd-thr",type=float,default=0);ap.add_argument("--min-layer",type=int,default=8);ap.add_argument("--raw-bidask-only",action="store_true");a=ap.parse_args()
 if not a.raw_bidask_only:raise SystemExit("raw-bidask-only mandatory")
 cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD");ticks=fix_sizes(cat.query_quote_ticks(identifiers=[inst.id.value]))
 eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)))
 eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"));eng.add_instrument(inst);eng.add_data(ticks)
 c=Cfg(instrument_id=inst.id,base_qty=Decimal(".30"),direction_sign=1,session_start_hour=7,session_end_hour=17,basket_offset=.8575,max_layers=10,entry_mode="ticksmoother",ts_ticks_per_bar=5,ts_fast=3,ts_slow=5,ts_conf1=8,ts_conf2=13,ts_cross_only=True,risk_mode="none",rescue_mode="none",math_rescue_mode="none",eligibility_mode="conservative",eligibility_fast_frac=.06,eligibility_deep_frac=0.0,eligibility_hard_fdd=45.0,ae_stack_mode="none")
 st=OptimalStopStrategy(c,a.mode,a.loss_stop,a.fp_thr,a.fdd_thr,a.min_layer);eng.add_strategy(st);eng.run()
 out={"verification_level":"RAW_BIDASK_OPTIMAL_STOP_ABLATION","core_frozen":True,"raw_ticks":len(ticks),**st.summary()}
 p=Path("results/tickscalper-nautilus")/a.experiment_id;p.mkdir(parents=True,exist_ok=True);(p/"kpi.json").write_text(json.dumps(out,indent=2),encoding="utf-8");print(json.dumps(out,indent=2));eng.dispose()
if __name__=="__main__":main()

# trigger optimal stop workflow

# trigger depth10 tail-stop refinement
