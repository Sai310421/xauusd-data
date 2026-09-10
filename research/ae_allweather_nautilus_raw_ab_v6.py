from __future__ import annotations

import argparse
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

from research.ae_allweather_nautilus_raw_ab_v1 import Regime, Side
from research.ae_allweather_nautilus_raw_ab_v5 import V5Cfg, ABStrategyV5

if not hasattr(ParquetDataCatalog, 'query_quote_ticks'):
    def _qqt(self, identifiers=None, start=None, end=None):
        return self.query(data_cls=QuoteTick, identifiers=identifiers, start=start, end=end)
    ParquetDataCatalog.query_quote_ticks = _qqt


class V6Cfg(V5Cfg, frozen=True):
    # Harrison/Taksar-style state band: keep residual toxic inventory inside a controlled region.
    recursive_max_depth: int = 6
    meta_profit_floor: float = 0.05
    winner_reserve_frac: float = 0.20
    tail_age_scale: float = 240.0
    tail_depth_weight: float = 0.35
    tail_loss_weight: float = 1.00
    tail_age_weight: float = 0.55
    control_band_soft: float = 0.08
    control_band_hard: float = 0.18
    control_band_stop: float = 0.30


class ABStrategyV6(ABStrategyV5):
    """Recursive hierarchical stochastic cluster control.

    ORIGINAL source mathematics mapped:
      - Harrison-Taksar singular control: controlled state band / reflecting-barrier concept.
      - Harrison-Sellke-Taylor impulse control: state-triggered jump/impulse to a better region.
      - Davis-Norman transaction-cost control: do not intervene unless state leaves the useful region.

    AE-DERIVED composition:
      position -> cluster -> recluster -> meta-cluster -> global cluster.
      Residual inventory is never declared a failed cluster merely because one impulse could not absorb it.
      It becomes the next controlled state and is re-ranked by toxicity on every tick.
    """
    def __init__(self, config: V6Cfg):
        super().__init__(config)
        self.recursive_impulses = 0
        self.meta_cluster_cycles = 0
        self.max_recursive_depth_seen = 0
        self.tail_absorbed_total = 0.0
        self.global_absorbed_total = 0.0
        self.control_band_hits = 0
        self.no_trade_ticks = 0
        self.recluster_ticks = 0

    def _toxicity(self, row):
        _, _, pnl, age = row
        if pnl >= 0:
            return 0.0
        loss = -pnl
        age_term = min(max(age, 0) / max(self.config.tail_age_scale, 1e-9), 4.0)
        return self.config.tail_loss_weight * loss * (1.0 + self.config.tail_age_weight * age_term)

    def _candidate_pool_v5(self, bid, ask):
        rows = self._rows(bid, ask)
        winners = sorted([r for r in rows if r[2] > 0], key=lambda r: r[2], reverse=True)
        losers = sorted([r for r in rows if r[2] < 0], key=self._toxicity, reverse=True)
        n = self.config.cluster_candidate_pool_v5
        # Reserve some winners as future absorption capacity; use enough now to absorb toxic tails.
        usable_w = max(1, int(len(winners) * (1.0 - self.config.winner_reserve_frac))) if winners else 0
        pool = winners[:min(usable_w, max(1, n // 2))] + losers[:max(0, n - min(usable_w, max(1, n // 2)))]
        used = {x[0] for x in pool}
        remain = sorted([r for r in rows if r[0] not in used], key=lambda r: (self._toxicity(r), abs(r[2])), reverse=True)
        return (pool + remain)[:n]

    def _delta_value(self, sub, bid, ask):
        base = super()._delta_value(sub, bid, ask)
        tail = sum(self._toxicity(x) for x in sub if x[2] < 0)
        return base + self.config.tail_depth_weight * tail

    def _state_band(self, bid, ask):
        dd = self._state_dd(bid, ask) / max(self.config.initial_balance, 1e-9)
        if dd >= self.config.control_band_stop:
            return 'STOP'
        if dd >= self.config.control_band_hard:
            return 'HARD'
        if dd >= self.config.control_band_soft:
            return 'SOFT'
        return 'NORMAL'

    def _recursive_cluster_control(self, bid, ask):
        acted = False
        depth = 0
        while depth < self.config.recursive_max_depth and len(self.pos) >= 2:
            depth += 1
            cand = self._best_system_cluster(bid, ask)
            if cand is None:
                break
            _, sub, _, absorbed, dv = cand
            ids = [x[0] for x in sub]
            toxic_abs = sum(self._toxicity(x) for x in sub if x[2] < 0)
            pnl, n = self._close_ids(ids, bid, ask, 'RECURSIVE_META_ABSORB')
            if pnl < self.config.meta_profit_floor:
                break
            self.loss_absorbed_total += absorbed
            self.tail_absorbed_total += toxic_abs
            self.global_absorbed_total += absorbed
            self.loss_absorbing_cycles += 1
            self.first_passage_cycles += 1
            self.system_value_gain_total += dv
            self.recursive_impulses += 1
            if depth > 1:
                self.meta_cluster_cycles += 1
            self.cluster_cycles.append({
                'ids': ids, 'pnl': pnl, 'n': n, 'reason': 'RECURSIVE_META_ABSORB',
                'loss_absorbed': absorbed, 'tail_absorbed': toxic_abs,
                'delta_v': dv, 'recursive_depth': depth,
            })
            acted = True
        if depth > self.max_recursive_depth_seen:
            self.max_recursive_depth_seen = depth
        if acted:
            self.recluster_ticks += 1
        return acted

    def _evaluate_cluster(self, bid, ask):
        band = self._state_band(bid, ask)
        if band != 'NORMAL':
            self.control_band_hits += 1
        # First: recursively consume every currently feasible positive loss-absorbing meta-cluster.
        if self._recursive_cluster_control(bid, ask):
            return
        # No feasible impulse: this is a legitimate no-action region, not a failed recovery.
        self.no_trade_ticks += 1
        losers = [r for r in self._rows(bid, ask) if r[2] < 0]
        # Only harvest winner-only baskets when no toxic inventory exists. Otherwise preserve winner capacity.
        if not losers:
            self._try_pure_profit_fallback(bid, ask)

    def _entry_multiplier_v5(self, bid, ask):
        band = self._state_band(bid, ask)
        if band == 'STOP': return 0.0
        if band == 'HARD': return 0.25
        if band == 'SOFT': return 0.50
        return 1.0

    def summary(self):
        out = super().summary()
        out.update({
            'verification_level_v6': 'RECURSIVE_HIERARCHICAL_STOCHASTIC_CLUSTER_V6',
            'recursive_impulses': self.recursive_impulses,
            'meta_cluster_cycles': self.meta_cluster_cycles,
            'max_recursive_depth_seen': self.max_recursive_depth_seen,
            'tail_absorbed_total': self.tail_absorbed_total,
            'global_absorbed_total': self.global_absorbed_total,
            'control_band_hits': self.control_band_hits,
            'no_trade_ticks': self.no_trade_ticks,
            'recluster_ticks': self.recluster_ticks,
            'recursive_absorption_to_residual_ratio': self.global_absorbed_total / max(out.get('residual_loss_mtm', 0.0), 1e-9),
        })
        return out


def run_cell(catalog_path, mode, tf, experiment_id):
    catalog = ParquetDataCatalog(str(catalog_path))
    inst = next((x for x in catalog.instruments() if x.id.symbol.value.replace('/', '') == 'XAUUSD'), None)
    if inst is None: raise SystemExit('XAUUSD missing')
    ticks = catalog.query_quote_ticks(identifiers=[inst.id.value])
    if not ticks: raise SystemExit('no Raw QuoteTicks')
    eng = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'), risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue, oms_type=OmsType.NETTING, account_type=AccountType.MARGIN, base_currency=USD, starting_balances=[Money(1000, USD)], default_leverage=Decimal('2000'))
    eng.add_instrument(inst); eng.add_data(ticks)
    bt = BarType.from_str(f'{inst.id.value}-{tf}-MINUTE-BID-INTERNAL')
    st = ABStrategyV6(V6Cfg(instrument_id=inst.id, bar_type=bt, mode=mode))
    eng.add_strategy(st); eng.run()
    obj = {
        'verification_level': 'NAUTILUS_RAW_BIDASK_AE_RECURSIVE_CLUSTER_V6',
        'engine': 'NautilusTrader BacktestEngine',
        'nautilus_version': getattr(nautilus_trader, '__version__', 'unknown'),
        'raw_ticks': len(ticks), 'ohlc_resample_used': False,
        'signal_bars': 'Nautilus INTERNAL BID bars from Raw QuoteTicks',
        'execution': 'virtual micro-position ledger marked on raw Bid/Ask; raw spread included; terminal inventory MTM only',
        'tf_minutes': tf, **st.summary(),
    }
    eng.dispose(); return obj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--catalog', required=True); ap.add_argument('--experiment-id', required=True); ap.add_argument('--mode', choices=['BASE','CLUSTER'], required=True); ap.add_argument('--tf', type=int, default=1); ap.add_argument('--raw-bidask-only', action='store_true'); a = ap.parse_args()
    if not a.raw_bidask_only: raise SystemExit('raw-bidask-only mandatory')
    out = run_cell(Path(a.catalog), a.mode, a.tf, a.experiment_id)
    p = Path('results/ae-allweather-ab') / a.experiment_id; p.mkdir(parents=True, exist_ok=True)
    (p / f'{a.mode}_M{a.tf}.json').write_text(json.dumps(out, indent=2, default=str), encoding='utf-8')
    print(json.dumps(out, indent=2, default=str))

if __name__ == '__main__': main()
