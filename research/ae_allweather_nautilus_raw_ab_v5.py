from __future__ import annotations

import argparse
import itertools
import json
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

from research.ae_allweather_nautilus_raw_ab_v1 import Cfg, ABStrategy, Regime, Side

if not hasattr(ParquetDataCatalog, "query_quote_ticks"):
    def _qqt(self, identifiers=None, start=None, end=None):
        return self.query(data_cls=QuoteTick, identifiers=identifiers, start=start, end=end)
    ParquetDataCatalog.query_quote_ticks = _qqt


class V5Cfg(Cfg, frozen=True):
    # First-passage positive basket constraint.
    profit_floor: float = 0.05
    cluster_candidate_pool_v5: int = 16
    cluster_max_size_v5: int = 8
    inventory_soft_cap_v5: int = 18
    inventory_hard_cap_v5: int = 30

    # System-state value weights. Loss absorption is deliberately dominant.
    w_loss_absorb: float = 4.00
    w_net_profit_v5: float = 1.00
    w_dd_release_v5: float = 1.50
    w_margin_release_v5: float = 0.35
    w_inventory_release_v5: float = 0.25
    w_tail_release_v5: float = 1.25
    w_future_edge_v5: float = 1.00

    # State-dependent entry suppression.
    dd_soft_pct_v5: float = 0.08
    dd_hard_pct_v5: float = 0.18
    dd_stop_pct_v5: float = 0.30


class ABStrategyV5(ABStrategy):
    """System-state optimal, loss-absorbing cluster control.

    ORIGINAL mathematical ideas mapped here:
      * state-value / stochastic control: choose action maximizing Delta V(X)
      * impulse control: basket close is an impulse action
      * first passage: act when a feasible positive-net basket first exists
      * inventory control: penalize residual/tail inventory

    AE-derived composition:
      * floating winners absorb floating losers inside the same positive basket
      * never harvest winners first when a loss-absorbing feasible basket exists
    """
    def __init__(self, config: V5Cfg):
        super().__init__(config)
        self.loss_absorbed_total = 0.0
        self.loss_absorbing_cycles = 0
        self.system_value_gain_total = 0.0
        self.terminal_mtm = 0.0
        self.terminal_open = 0
        self.entry_suppressed_bars_v5 = 0
        self.max_state_dd = 0.0
        self.first_passage_cycles = 0

    def _rows(self, bid, ask):
        rows=[]
        for pid,p in self.pos.items():
            pnl=self._pnl_pos(p,bid,ask)
            age=self.bar_i-p.opened_bar
            rows.append((pid,p,pnl,age))
        return rows

    def _candidate_pool_v5(self,bid,ask):
        rows=self._rows(bid,ask)
        winners=sorted([r for r in rows if r[2]>0],key=lambda r:r[2],reverse=True)
        # Include the worst losers first: these are the inventory we most want winners to absorb.
        losers=sorted([r for r in rows if r[2]<0],key=lambda r:r[2])
        n=self.config.cluster_candidate_pool_v5
        nw=max(1,n//2)
        nl=n-nw
        pool=winners[:nw]+losers[:nl]
        used={x[0] for x in pool}
        remain=sorted([r for r in rows if r[0] not in used],key=lambda r:abs(r[2]),reverse=True)
        return (pool+remain)[:n]

    def _state_dd(self,bid,ask):
        floating=sum(self._pnl_pos(p,bid,ask) for p in self.pos.values())
        equity=self.config.initial_balance+self.realized+floating
        peak=max(self.peak_equity,self.config.initial_balance)
        dd=max(0.0,peak-equity)
        self.max_state_dd=max(self.max_state_dd,dd)
        return dd

    def _future_edge_cost_v5(self,sub):
        cost=0.0
        for _,p,pnl,_ in sub:
            # Preserve high-score profitable expansion inventory as potential runner.
            if pnl>0 and p.score>=90 and p.regime==Regime.EXPANSION.name:
                cost += pnl
        return cost

    def _delta_value(self,sub,bid,ask):
        net=sum(x[2] for x in sub)
        absorbed=sum(-x[2] for x in sub if x[2]<0)
        n=len(sub)
        # Closing negative positions directly releases this amount of floating DD/tail inventory.
        dd_release=absorbed
        tail_release=sum((-x[2])*(1.0+min(x[3],480)/480.0) for x in sub if x[2]<0)
        margin_release=float(n)
        future_edge=self._future_edge_cost_v5(sub)
        return (
            self.config.w_loss_absorb*absorbed
            + self.config.w_net_profit_v5*net
            + self.config.w_dd_release_v5*dd_release
            + self.config.w_margin_release_v5*margin_release
            + self.config.w_inventory_release_v5*n
            + self.config.w_tail_release_v5*tail_release
            - self.config.w_future_edge_v5*future_edge
        )

    def _best_system_cluster(self,bid,ask):
        pool=self._candidate_pool_v5(bid,ask)
        if len(pool)<2:return None
        best=None
        max_k=min(self.config.cluster_max_size_v5,len(pool))
        # A valid cluster must contain both profit and loss and remain net-positive after absorption.
        for k in range(2,max_k+1):
            for sub in itertools.combinations(pool,k):
                if not any(x[2]>0 for x in sub) or not any(x[2]<0 for x in sub):
                    continue
                net=sum(x[2] for x in sub)
                if net < self.config.profit_floor:
                    continue
                absorbed=sum(-x[2] for x in sub if x[2]<0)
                dv=self._delta_value(sub,bid,ask)
                # Lexicographic priority implements the intended control objective:
                # 1) absorb maximum loss, 2) improve system value, 3) retain positive profit.
                key=(absorbed,dv,net,len(sub))
                if best is None or key>best[0]:
                    best=(key,sub,net,absorbed,dv)
        return best

    def _best_pure_profit_cluster(self,bid,ask):
        # Fallback only when no loss-absorbing positive basket exists.
        # Keep it conservative so winners remain available to absorb future losers.
        winners=sorted([r for r in self._rows(bid,ask) if r[2]>0],key=lambda r:r[2],reverse=True)
        if len(winners)<2:return None
        max_k=min(3,len(winners))
        best=None
        for k in range(2,max_k+1):
            for sub in itertools.combinations(winners[:8],k):
                net=sum(x[2] for x in sub)
                if net < max(self.config.profit_floor, self.config.profit_buffer):continue
                future=self._future_edge_cost_v5(sub)
                score=net-future
                if best is None or score>best[0]:best=(score,sub,net)
        return best

    def _try_system_impulse(self,bid,ask):
        cand=self._best_system_cluster(bid,ask)
        if cand is None:return False
        _,sub,expected,absorbed,dv=cand
        ids=[x[0] for x in sub]
        pnl,n=self._close_ids(ids,bid,ask,'SYSTEM_LOSS_ABSORB')
        if pnl >= self.config.profit_floor:
            self.loss_absorbed_total += absorbed
            self.loss_absorbing_cycles += 1
            self.first_passage_cycles += 1
            self.system_value_gain_total += dv
        self.cluster_cycles.append({'ids':ids,'pnl':pnl,'n':n,'reason':'SYSTEM_LOSS_ABSORB','loss_absorbed':absorbed,'delta_v':dv})
        return True

    def _try_pure_profit_fallback(self,bid,ask):
        cand=self._best_pure_profit_cluster(bid,ask)
        if cand is None:return False
        _,sub,_=cand
        ids=[x[0] for x in sub]
        pnl,n=self._close_ids(ids,bid,ask,'PURE_PROFIT_FALLBACK')
        self.cluster_cycles.append({'ids':ids,'pnl':pnl,'n':n,'reason':'PURE_PROFIT_FALLBACK','loss_absorbed':0.0,'delta_v':pnl})
        return True

    def _evaluate_cluster(self,bid,ask):
        # System-state impulse has absolute priority over winner-only harvest.
        for _ in range(6):
            if len(self.pos)<2:break
            if self._try_system_impulse(bid,ask):continue
            # Do not strip winners while there is meaningful losing inventory unless capacity pressure is low.
            losers=sum(1 for r in self._rows(bid,ask) if r[2]<0)
            if losers==0 or len(self.pos)>=self.config.inventory_soft_cap_v5:
                if self._try_pure_profit_fallback(bid,ask):continue
            break

    def _entry_multiplier_v5(self,bid,ask):
        dd=self._state_dd(bid,ask)/max(self.config.initial_balance,1e-9)
        if dd>=self.config.dd_stop_pct_v5:return 0.0
        if dd>=self.config.dd_hard_pct_v5:return 0.25
        if dd>=self.config.dd_soft_pct_v5:return 0.50
        return 1.0

    def on_quote_tick(self,tick:QuoteTick):
        self.tick_i+=1
        bid=self.f(tick.bid_price); ask=self.f(tick.ask_price)
        self.last_bid=bid; self.last_ask=ask
        self._mark_equity(bid,ask)
        if not hasattr(self,'last_entry_bar'):self.last_entry_bar=-1
        if self.bar_i>0 and self.last_entry_bar!=self.bar_i:
            side=self.logic.desired_entry_side()
            if side is not None:
                if self.logic.regime==Regime.RANGE:requested=1
                elif self.logic.regime==Regime.EXPANSION:requested=5 if self.logic.last_crt_signals.score_100()>=90 else 3
                else:requested=0
                mult=self._entry_multiplier_v5(bid,ask)
                room=max(0,self.config.inventory_hard_cap_v5-len(self.pos))
                n=min(room,int(requested*mult+1e-9))
                if requested>0 and n==0:self.entry_suppressed_bars_v5+=1
                if n>0:self._enter(1 if side==Side.LONG else -1,ask if side==Side.LONG else bid,n)
            self.last_entry_bar=self.bar_i
        if self.config.mode=='BASE':self._evaluate_base(bid,ask)
        else:self._evaluate_cluster(bid,ask)

    def on_stop(self):
        # Finite-horizon accounting only: do not convert residual inventory into an artificial strategy exit.
        if self.last_bid is not None:
            self.terminal_open=len(self.pos)
            self.terminal_mtm=sum(self._pnl_pos(p,self.last_bid,self.last_ask) for p in self.pos.values())

    def summary(self):
        out=super().summary()
        realized=out['net']
        net_mtm=realized+self.terminal_mtm
        residual_loss=max(0.0,-self.terminal_mtm)
        out.update({
            'verification_level_v5':'SYSTEM_STATE_OPTIMAL_LOSS_ABSORBING_CLUSTER_V5',
            'realized_net_pre_mtm':realized,
            'terminal_mtm':self.terminal_mtm,
            'terminal_open_positions':self.terminal_open,
            'net_mtm':net_mtm,
            'return_mtm_pct':100.0*net_mtm/self.config.initial_balance,
            'loss_absorbed_total':self.loss_absorbed_total,
            'loss_absorbing_cycles':self.loss_absorbing_cycles,
            'first_passage_cycles':self.first_passage_cycles,
            'system_value_gain_total':self.system_value_gain_total,
            'residual_loss_mtm':residual_loss,
            'entry_suppressed_bars_v5':self.entry_suppressed_bars_v5,
            'max_state_DD_pct':100.0*self.max_state_dd/max(self.config.initial_balance,1e-9),
            'loss_absorption_to_residual_ratio':self.loss_absorbed_total/max(residual_loss,1e-9),
        })
        return out


def run_cell(catalog_path,mode,tf,experiment_id):
    catalog=ParquetDataCatalog(str(catalog_path))
    inst=next((x for x in catalog.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'),None)
    if inst is None:raise SystemExit('XAUUSD missing')
    ticks=catalog.query_quote_ticks(identifiers=[inst.id.value])
    if not ticks:raise SystemExit('no Raw QuoteTicks')
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    eng.add_instrument(inst); eng.add_data(ticks)
    bt=BarType.from_str(f'{inst.id.value}-{tf}-MINUTE-BID-INTERNAL')
    st=ABStrategyV5(V5Cfg(instrument_id=inst.id,bar_type=bt,mode=mode))
    eng.add_strategy(st); eng.run()
    obj={'verification_level':'NAUTILUS_RAW_BIDASK_AE_SYSTEM_STATE_CLUSTER_V5','engine':'NautilusTrader BacktestEngine','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'raw_ticks':len(ticks),'ohlc_resample_used':False,'signal_bars':'Nautilus INTERNAL BID bars from Raw QuoteTicks','execution':'virtual micro-position ledger marked on raw Bid/Ask; raw spread included; terminal inventory MTM only','tf_minutes':tf,**st.summary()}
    eng.dispose();return obj


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);ap.add_argument('--mode',choices=['BASE','CLUSTER'],required=True);ap.add_argument('--tf',type=int,default=1);ap.add_argument('--raw-bidask-only',action='store_true');a=ap.parse_args()
    if not a.raw_bidask_only:raise SystemExit('raw-bidask-only mandatory')
    out=run_cell(Path(a.catalog),a.mode,a.tf,a.experiment_id)
    p=Path('results/ae-allweather-ab')/a.experiment_id;p.mkdir(parents=True,exist_ok=True)
    (p/f'{a.mode}_M{a.tf}.json').write_text(json.dumps(out,indent=2,default=str),encoding='utf-8')
    print(json.dumps(out,indent=2,default=str))

if __name__=='__main__':main()
