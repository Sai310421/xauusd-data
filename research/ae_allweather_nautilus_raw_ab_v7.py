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

from research.ae_allweather_nautilus_raw_ab_v1 import Regime, Side
from research.ae_allweather_nautilus_raw_ab_v6 import V6Cfg, ABStrategyV6

if not hasattr(ParquetDataCatalog, 'query_quote_ticks'):
    def _qqt(self, identifiers=None, start=None, end=None):
        return self.query(data_cls=QuoteTick, identifiers=identifiers, start=start, end=end)
    ParquetDataCatalog.query_quote_ticks = _qqt


class V7Cfg(V6Cfg, frozen=True):
    # Persistent capital state: fraction of positive realized cluster PnL earmarked for future tail absorption.
    reserve_fraction: float = 0.80
    reserve_min_add: float = 0.05
    reserve_min_keep: float = 0.10
    global_economic_floor: float = 0.05
    meta_tail_max: int = 3
    meta_winner_max: int = 6
    meta_candidate_winners: int = 8
    reserve_spend_cap_fraction: float = 0.95
    # Tail priority increases with age and absolute loss.
    meta_tail_age_scale: float = 240.0
    meta_tail_age_weight: float = 0.70


class ABStrategyV7(ABStrategyV6):
    """Persistent Cluster Capital / Global Economic BE control.

    ORIGINAL mathematical motifs mapped:
      * stochastic-control state carry-forward: realized control outcome becomes part of X_{t+}
      * impulse control: liquidation/partial liquidation is an impulse
      * first passage: execute when global economic cluster value crosses a feasible boundary
      * inventory / ruin control: prioritize toxic tail removal and preserve a reserve band

    AE-DERIVED composition:
      * reserve_t is an earmarked subset of already-realized positive cluster PnL
      * reserve_t + current floating winner PnL may authorize realization of a losing tail
      * actual realized PnL remains untouched by bookkeeping: a losing close is recorded as a loss
      * reserve is consumed only as an allocation ledger, preventing double counting
    """
    def __init__(self, config: V7Cfg):
        super().__init__(config)
        self.harvest_reserve = 0.0
        self.harvest_reserve_peak = 0.0
        self.harvest_reserve_added = 0.0
        self.harvest_reserve_spent = 0.0
        self.global_meta_cycles = 0
        self.global_meta_tail_absorbed = 0.0
        self.global_meta_realized_pnl = 0.0
        self.global_meta_economic_pnl = 0.0
        self.reserve_first_passages = 0
        self.reserve_blocked_cycles = 0
        self.reserve_generation_cycles = 0

    def _add_reserve(self, pnl):
        if pnl <= 0:
            return 0.0
        add = pnl * self.config.reserve_fraction
        if add < self.config.reserve_min_add:
            return 0.0
        self.harvest_reserve += add
        self.harvest_reserve_added += add
        self.harvest_reserve_peak = max(self.harvest_reserve_peak, self.harvest_reserve)
        self.reserve_generation_cycles += 1
        return add

    def _spend_reserve(self, amount):
        if amount <= 0:
            return 0.0
        max_spend = max(0.0, self.harvest_reserve - self.config.reserve_min_keep)
        max_spend *= self.config.reserve_spend_cap_fraction
        spend = min(amount, max_spend)
        self.harvest_reserve -= spend
        self.harvest_reserve_spent += spend
        return spend

    def _close_ids(self, ids, bid, ask, reason):
        pnl, n = super()._close_ids(ids, bid, ask, reason)
        # Only positive ordinary/recursive cluster profits replenish persistent reserve.
        if reason in ('RECURSIVE_META_ABSORB', 'SYSTEM_LOSS_ABSORB', 'PURE_PROFIT_FALLBACK') and pnl > 0:
            self._add_reserve(pnl)
        return pnl, n

    def _meta_toxicity(self, row):
        _, _, pnl, age = row
        if pnl >= 0:
            return 0.0
        loss = -pnl
        age_term = min(max(age, 0) / max(self.config.meta_tail_age_scale, 1e-9), 6.0)
        return loss * (1.0 + self.config.meta_tail_age_weight * age_term)

    def _best_global_meta_cluster(self, bid, ask):
        rows = self._rows(bid, ask)
        tails = sorted([r for r in rows if r[2] < 0], key=self._meta_toxicity, reverse=True)
        winners = sorted([r for r in rows if r[2] > 0], key=lambda r: r[2], reverse=True)
        if not tails:
            return None

        tails = tails[:self.config.meta_tail_max]
        winners = winners[:self.config.meta_candidate_winners]
        reserve_available = max(0.0, self.harvest_reserve - self.config.reserve_min_keep)
        reserve_available *= self.config.reserve_spend_cap_fraction
        if reserve_available <= 0 and not winners:
            return None

        best = None
        # Allow 1..meta_tail_max toxic tails, plus 0..meta_winner_max current winners.
        for kt in range(1, min(self.config.meta_tail_max, len(tails)) + 1):
            for tsub in itertools.combinations(tails, kt):
                tail_loss = sum(-x[2] for x in tsub)
                toxic = sum(self._meta_toxicity(x) for x in tsub)
                max_kw = min(self.config.meta_winner_max, len(winners))
                for kw in range(0, max_kw + 1):
                    for wsub in itertools.combinations(winners, kw):
                        current_pnl = sum(x[2] for x in tsub) + sum(x[2] for x in wsub)
                        required = max(0.0, self.config.global_economic_floor - current_pnl)
                        if required > reserve_available:
                            continue
                        economic_pnl = current_pnl + required
                        # Lexicographic: remove most toxic/large tail first, then spend less reserve, then retain current profit.
                        key = (toxic, tail_loss, -required, current_pnl, len(tsub) + len(wsub))
                        ids = [x[0] for x in tsub] + [x[0] for x in wsub]
                        if best is None or key > best[0]:
                            best = (key, ids, current_pnl, required, economic_pnl, tail_loss, toxic)
        return best

    def _try_global_meta_cluster(self, bid, ask):
        cand = self._best_global_meta_cluster(bid, ask)
        if cand is None:
            self.reserve_blocked_cycles += 1
            return False
        _, ids, expected_current, required, economic_pnl, tail_loss, toxic = cand
        spend = self._spend_reserve(required)
        # Recheck feasibility after reserve-band/cap mechanics.
        if expected_current + spend < self.config.global_economic_floor:
            # Restore spent reserve because no impulse occurred.
            self.harvest_reserve += spend
            self.harvest_reserve_spent -= spend
            self.reserve_blocked_cycles += 1
            return False
        pnl, n = super()._close_ids(ids, bid, ask, 'GLOBAL_META_ECONOMIC_BE')
        # Do not add reserve from this close even if current basket happens to be positive; economic reserve was already allocated.
        actual_economic = pnl + spend
        self.global_meta_cycles += 1
        self.reserve_first_passages += 1
        self.global_meta_tail_absorbed += tail_loss
        self.global_meta_realized_pnl += pnl
        self.global_meta_economic_pnl += actual_economic
        self.global_absorbed_total += tail_loss
        self.tail_absorbed_total += toxic
        self.cluster_cycles.append({
            'ids': ids, 'pnl': pnl, 'n': n, 'reason': 'GLOBAL_META_ECONOMIC_BE',
            'reserve_spent': spend, 'economic_pnl': actual_economic,
            'tail_absorbed': tail_loss, 'tail_toxicity': toxic,
        })
        return True

    def _evaluate_cluster(self, bid, ask):
        # 1) Prefer currently self-financing positive loss-absorbing impulses (no historical reserve needed).
        if self._recursive_cluster_control(bid, ask):
            return
        # 2) If not currently self-financing, use persistent realized cluster capital to absorb toxic tail.
        if self._try_global_meta_cluster(bid, ask):
            # After one reserve-backed impulse, immediately re-solve state once more.
            self._recursive_cluster_control(bid, ask)
            return
        # 3) When no losses exist, harvest some winner PnL to seed/replenish reserve.
        losers = [r for r in self._rows(bid, ask) if r[2] < 0]
        if not losers:
            self._try_pure_profit_fallback(bid, ask)
        else:
            self.no_trade_ticks += 1

    def summary(self):
        out = super().summary()
        terminal_loss = max(0.0, -out.get('terminal_mtm', 0.0))
        economic_net = out.get('net_mtm', 0.0)  # reserve is already contained in realized PnL; never add it again.
        out.update({
            'verification_level_v7': 'PERSISTENT_CLUSTER_CAPITAL_GLOBAL_ECONOMIC_BE_V7',
            'harvest_reserve_end': self.harvest_reserve,
            'harvest_reserve_peak': self.harvest_reserve_peak,
            'harvest_reserve_added': self.harvest_reserve_added,
            'harvest_reserve_spent': self.harvest_reserve_spent,
            'reserve_generation_cycles': self.reserve_generation_cycles,
            'reserve_first_passages': self.reserve_first_passages,
            'reserve_blocked_cycles': self.reserve_blocked_cycles,
            'global_meta_cycles': self.global_meta_cycles,
            'global_meta_tail_absorbed': self.global_meta_tail_absorbed,
            'global_meta_realized_pnl': self.global_meta_realized_pnl,
            'global_meta_economic_pnl': self.global_meta_economic_pnl,
            'economic_net_mtm_no_double_count': economic_net,
            'terminal_loss_mtm_v7': terminal_loss,
            'tail_absorption_coverage_v7': self.global_absorbed_total / max(terminal_loss, 1e-9),
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
    st = ABStrategyV7(V7Cfg(instrument_id=inst.id, bar_type=bt, mode=mode))
    eng.add_strategy(st); eng.run()
    obj = {
        'verification_level': 'NAUTILUS_RAW_BIDASK_AE_PERSISTENT_CLUSTER_CAPITAL_V7',
        'engine': 'NautilusTrader BacktestEngine',
        'nautilus_version': getattr(nautilus_trader, '__version__', 'unknown'),
        'raw_ticks': len(ticks), 'ohlc_resample_used': False,
        'signal_bars': 'Nautilus INTERNAL BID bars from Raw QuoteTicks',
        'execution': 'virtual micro-position ledger marked on raw Bid/Ask; raw spread included; reserve is realized-profit allocation ledger; terminal inventory MTM only',
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
