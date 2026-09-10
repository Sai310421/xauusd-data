from __future__ import annotations

import argparse, json, math
from decimal import Decimal
from pathlib import Path

import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from research.ae_allweather_nautilus_raw_ab_v1 import Regime, Side
from research.ae_allweather_nautilus_raw_ab_v7 import V7Cfg, ABStrategyV7

if not hasattr(ParquetDataCatalog, 'query_quote_ticks'):
    def _qqt(self, identifiers=None, start=None, end=None):
        return self.query(data_cls=QuoteTick, identifiers=identifiers, start=start, end=end)
    ParquetDataCatalog.query_quote_ticks = _qqt


class V8Cfg(V7Cfg, frozen=True):
    # Tail-growth control.
    tail_velocity_alpha: float = 0.02
    tail_velocity_soft: float = 0.02
    tail_velocity_hard: float = 0.08
    tail_velocity_stop: float = 0.20
    # Recovery-capacity forecast (heuristic, not a calibrated probability).
    recovery_horizon_bars: int = 240
    winner_rate_alpha: float = 0.05
    recovery_ratio_soft: float = 0.75
    recovery_ratio_hard: float = 0.40
    recovery_ratio_stop: float = 0.20
    # Toxic inventory / hedge-like entry gate.
    toxic_tail_min_loss: float = 2.0
    toxic_age_min: int = 24
    rescue_entry_only_when_stressed: bool = True
    # Early tail routing before runaway loss.
    early_tail_loss: float = 8.0
    early_tail_age: int = 60
    early_tail_economic_floor: float = 0.05


class ABStrategyV8(ABStrategyV7):
    """Tail-growth constrained cluster control.

    ORIGINAL motifs mapped:
      - viability / constrained control: keep state inside a recoverable set
      - first-passage control: intervene when recovery-capacity boundary is crossed
      - inventory control: constrain new inventory when toxic stock grows
      - model-predictive-control motif: compare forecast recovery capacity with current tail debt

    AE-DERIVED composition:
      - realized reserve + current winners + forecast winner generation define recovery capacity
      - when tail growth outruns capacity, suppress ordinary entries
      - under stress, permit only entries that offset the dominant toxic inventory direction
    """
    def __init__(self, config: V8Cfg):
        super().__init__(config)
        self.prev_tail_debt = 0.0
        self.tail_velocity_ema = 0.0
        self.tail_velocity_peak = 0.0
        self.winner_realized_ema = 0.0
        self.recovery_ratio_min = float('inf')
        self.capacity_stop_ticks = 0
        self.tail_velocity_stop_ticks = 0
        self.rescue_only_entry_bars = 0
        self.ordinary_entry_blocked_bars = 0
        self.early_tail_routes = 0
        self.early_tail_absorbed = 0.0

    def _tail_rows(self,bid,ask):
        return [r for r in self._rows(bid,ask) if r[2] < 0]

    def _tail_debt(self,bid,ask):
        return sum(-r[2] for r in self._tail_rows(bid,ask))

    def _update_tail_velocity(self,bid,ask):
        debt=self._tail_debt(bid,ask)
        dv=max(0.0,debt-self.prev_tail_debt)
        a=self.config.tail_velocity_alpha
        self.tail_velocity_ema=(1-a)*self.tail_velocity_ema+a*dv
        self.tail_velocity_peak=max(self.tail_velocity_peak,self.tail_velocity_ema)
        self.prev_tail_debt=debt
        return debt

    def _close_ids(self, ids, bid, ask, reason):
        pnl,n=super()._close_ids(ids,bid,ask,reason)
        if pnl>0 and reason in ('RECURSIVE_META_ABSORB','SYSTEM_LOSS_ABSORB','PURE_PROFIT_FALLBACK'):
            a=self.config.winner_rate_alpha
            self.winner_realized_ema=(1-a)*self.winner_realized_ema+a*pnl
        return pnl,n

    def _recovery_capacity(self,bid,ask):
        current_winners=sum(max(0.0,r[2]) for r in self._rows(bid,ask))
        future=max(0.0,self.winner_realized_ema)*self.config.recovery_horizon_bars
        reserve=max(0.0,self.harvest_reserve-self.config.reserve_min_keep)
        return reserve+current_winners+future

    def _recovery_ratio(self,bid,ask):
        debt=self._tail_debt(bid,ask)
        if debt<=1e-9:return float('inf')
        ratio=self._recovery_capacity(bid,ask)/debt
        self.recovery_ratio_min=min(self.recovery_ratio_min,ratio)
        return ratio

    def _dominant_toxic_side(self,bid,ask):
        long_loss=0.0; short_loss=0.0
        for _,p,pnl,age in self._tail_rows(bid,ask):
            if -pnl < self.config.toxic_tail_min_loss or age < self.config.toxic_age_min: continue
            if p.side>0: long_loss += -pnl
            else: short_loss += -pnl
        if long_loss==0 and short_loss==0:return 0
        return 1 if long_loss>=short_loss else -1

    def _stress_multiplier(self,bid,ask):
        ratio=self._recovery_ratio(bid,ask)
        vel=self.tail_velocity_ema
        if vel>=self.config.tail_velocity_stop:
            self.tail_velocity_stop_ticks+=1; return 0.0
        if ratio<=self.config.recovery_ratio_stop:
            self.capacity_stop_ticks+=1; return 0.0
        if vel>=self.config.tail_velocity_hard or ratio<=self.config.recovery_ratio_hard:return 0.25
        if vel>=self.config.tail_velocity_soft or ratio<=self.config.recovery_ratio_soft:return 0.50
        return 1.0

    def _try_early_tail_route(self,bid,ask):
        # Before a tail becomes huge, use reserve + current winners if economic BE is already feasible.
        tails=[r for r in self._tail_rows(bid,ask) if (-r[2]>=self.config.early_tail_loss or r[3]>=self.config.early_tail_age)]
        if not tails:return False
        tails.sort(key=self._meta_toxicity,reverse=True)
        target=tails[0]
        winners=sorted([r for r in self._rows(bid,ask) if r[2]>0],key=lambda r:r[2],reverse=True)[:self.config.meta_winner_max]
        ids=[target[0]]
        cur=target[2]
        for w in winners:
            ids.append(w[0]); cur+=w[2]
            if cur>=self.config.early_tail_economic_floor:break
        required=max(0.0,self.config.early_tail_economic_floor-cur)
        avail=max(0.0,self.harvest_reserve-self.config.reserve_min_keep)*self.config.reserve_spend_cap_fraction
        if required>avail:return False
        spend=self._spend_reserve(required)
        if cur+spend < self.config.early_tail_economic_floor:
            self.harvest_reserve+=spend; self.harvest_reserve_spent-=spend; return False
        pnl,n=super(ABStrategyV7,self)._close_ids(ids,bid,ask,'EARLY_TAIL_ROUTE')
        self.early_tail_routes+=1
        self.early_tail_absorbed += -target[2]
        self.cluster_cycles.append({'ids':ids,'pnl':pnl,'n':n,'reason':'EARLY_TAIL_ROUTE','reserve_spent':spend,'tail_absorbed':-target[2]})
        return True

    def _evaluate_cluster(self,bid,ask):
        self._update_tail_velocity(bid,ask)
        if self._recursive_cluster_control(bid,ask):return
        if self._try_early_tail_route(bid,ask):return
        if self._try_global_meta_cluster(bid,ask):
            self._recursive_cluster_control(bid,ask); return
        losers=self._tail_rows(bid,ask)
        if not losers:self._try_pure_profit_fallback(bid,ask)
        else:self.no_trade_ticks+=1

    def on_quote_tick(self,tick:QuoteTick):
        self.tick_i+=1
        bid=self.f(tick.bid_price); ask=self.f(tick.ask_price)
        self.last_bid=bid; self.last_ask=ask
        self._mark_equity(bid,ask)
        self._update_tail_velocity(bid,ask)
        if not hasattr(self,'last_entry_bar'):self.last_entry_bar=-1
        if self.bar_i>0 and self.last_entry_bar!=self.bar_i:
            desired=self.logic.desired_entry_side()
            if desired is not None:
                if self.logic.regime==Regime.RANGE:requested=1
                elif self.logic.regime==Regime.EXPANSION:requested=5 if self.logic.last_crt_signals.score_100()>=90 else 3
                else:requested=0
                mult=self._stress_multiplier(bid,ask)
                dominant=self._dominant_toxic_side(bid,ask)
                desired_i=1 if desired==Side.LONG else -1
                stressed=(mult<1.0 or dominant!=0)
                # If toxic long inventory dominates, only a short new entry can offset state exposure; vice versa.
                if self.config.rescue_entry_only_when_stressed and stressed and dominant!=0:
                    rescue_side=-dominant
                    if desired_i!=rescue_side:
                        requested=0; self.ordinary_entry_blocked_bars+=1
                    else:
                        self.rescue_only_entry_bars+=1
                room=max(0,self.config.inventory_hard_cap_v5-len(self.pos))
                n=min(room,int(requested*mult+1e-9))
                if n>0:self._enter(desired_i,ask if desired_i>0 else bid,n)
                elif requested>0:self.entry_suppressed_bars_v5+=1
            self.last_entry_bar=self.bar_i
        if self.config.mode=='BASE':self._evaluate_base(bid,ask)
        else:self._evaluate_cluster(bid,ask)

    def summary(self):
        out=super().summary()
        ratio=self._recovery_ratio(self.last_bid,self.last_ask) if self.last_bid is not None else float('inf')
        out.update({
            'verification_level_v8':'TAIL_GROWTH_RECOVERY_CAPACITY_CONTROL_V8',
            'tail_velocity_ema_end':self.tail_velocity_ema,
            'tail_velocity_peak':self.tail_velocity_peak,
            'winner_realized_ema_end':self.winner_realized_ema,
            'recovery_capacity_end':self._recovery_capacity(self.last_bid,self.last_ask) if self.last_bid is not None else 0.0,
            'recovery_ratio_end':ratio,
            'recovery_ratio_min':0.0 if self.recovery_ratio_min==float('inf') else self.recovery_ratio_min,
            'capacity_stop_ticks':self.capacity_stop_ticks,
            'tail_velocity_stop_ticks':self.tail_velocity_stop_ticks,
            'rescue_only_entry_bars':self.rescue_only_entry_bars,
            'ordinary_entry_blocked_bars':self.ordinary_entry_blocked_bars,
            'early_tail_routes':self.early_tail_routes,
            'early_tail_absorbed':self.early_tail_absorbed,
        })
        return out


def run_cell(catalog_path,mode,tf,experiment_id):
    catalog=ParquetDataCatalog(str(catalog_path)); inst=next((x for x in catalog.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'),None)
    if inst is None:raise SystemExit('XAUUSD missing')
    ticks=catalog.query_quote_ticks(identifiers=[inst.id.value]);
    if not ticks:raise SystemExit('no Raw QuoteTicks')
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    eng.add_instrument(inst); eng.add_data(ticks)
    bt=BarType.from_str(f'{inst.id.value}-{tf}-MINUTE-BID-INTERNAL')
    st=ABStrategyV8(V8Cfg(instrument_id=inst.id,bar_type=bt,mode=mode)); eng.add_strategy(st); eng.run()
    obj={'verification_level':'NAUTILUS_RAW_BIDASK_AE_TAIL_GROWTH_CONTROL_V8','engine':'NautilusTrader BacktestEngine','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'raw_ticks':len(ticks),'ohlc_resample_used':False,'signal_bars':'Nautilus INTERNAL BID bars from Raw QuoteTicks','execution':'virtual micro-position ledger marked on raw Bid/Ask; raw spread included; persistent reserve ledger; terminal inventory MTM only','tf_minutes':tf,**st.summary()}
    eng.dispose(); return obj


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--mode',choices=['BASE','CLUSTER'],required=True); ap.add_argument('--tf',type=int,default=1); ap.add_argument('--raw-bidask-only',action='store_true'); a=ap.parse_args()
    if not a.raw_bidask_only:raise SystemExit('raw-bidask-only mandatory')
    out=run_cell(Path(a.catalog),a.mode,a.tf,a.experiment_id); p=Path('results/ae-allweather-ab')/a.experiment_id; p.mkdir(parents=True,exist_ok=True)
    (p/f'{a.mode}_M{a.tf}.json').write_text(json.dumps(out,indent=2,default=str),encoding='utf-8'); print(json.dumps(out,indent=2,default=str))

if __name__=='__main__':main()
