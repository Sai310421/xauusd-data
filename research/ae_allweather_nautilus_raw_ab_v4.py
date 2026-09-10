from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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


@dataclass
class V4Cfg(Cfg, frozen=True):
    # Realized-profit ledger. Only this ledger may amortize historical debt.
    amortization_fraction: float = 0.35
    amortization_min_harvest: float = 0.30

    # Debt-aware inventory control.
    debt_soft_pct: float = 0.10
    debt_hard_pct: float = 0.25
    debt_stop_pct: float = 0.40
    inventory_v4_soft_cap: int = 16
    inventory_v4_hard_cap: int = 24

    # Recovery success must be economically real, not merely removed from a set.
    recovery_bundle_buffer: float = 0.05
    recovery_max_winners: int = 4


class ABStrategyV4(ABStrategy):
    def __init__(self, config: V4Cfg):
        super().__init__(config)
        self.debt_ledger = 0.0
        self.debt_ledger_peak = 0.0
        self.debt_amortized = 0.0
        self.harvest_gross = 0.0
        self.harvest_retained = 0.0
        self.recovery_success_cycles = 0
        self.recovery_failed_cycles = 0
        self.recovery_success_pnl = 0.0
        self.entry_suppressed_bars = 0
        self.eod_mark_to_market = 0.0
        self.eod_open_positions = 0

    def _current_floating_debt(self, bid, ask):
        return sum(max(0.0, -self._pnl_pos(p, bid, ask)) for p in self.pos.values())

    def _economic_debt(self, bid, ask):
        # Ledger debt + current unrealized negative inventory.
        d = self.debt_ledger + self._current_floating_debt(bid, ask)
        self.debt_ledger_peak = max(self.debt_ledger_peak, d)
        return d

    def _debt_ratio(self, bid, ask):
        return self._economic_debt(bid, ask) / max(self.config.initial_balance, 1e-9)

    def _entry_multiplier(self, bid, ask):
        r = self._debt_ratio(bid, ask)
        if r >= self.config.debt_stop_pct:
            return 0.0
        if r >= self.config.debt_hard_pct:
            return 0.25
        if r >= self.config.debt_soft_pct:
            return 0.50
        return 1.0

    def _allowed_new_entries(self, requested, bid, ask):
        mult = self._entry_multiplier(bid, ask)
        cap = self.config.inventory_v4_soft_cap if mult < 1.0 else self.config.inventory_v4_hard_cap
        room = max(0, cap - len(self.pos))
        return min(room, int(requested * mult + 1e-9))

    def _amortize_from_profit(self, realized_profit):
        if realized_profit < self.config.amortization_min_harvest or self.debt_ledger <= 0:
            self.harvest_retained += max(0.0, realized_profit)
            return 0.0
        allocation = min(self.debt_ledger, realized_profit * self.config.amortization_fraction)
        self.debt_ledger -= allocation
        self.debt_amortized += allocation
        self.harvest_retained += realized_profit - allocation
        return allocation

    def _close_ids(self, ids, bid, ask, reason):
        # Capture true recovery membership before parent removes IDs from the set.
        recovery_ids = {pid for pid in ids if pid in self.recovery_pool}
        pnl, n = super()._close_ids(ids, bid, ask, reason)

        if reason in ("PROFIT_HARVEST", "PEAK_TRAIL") and pnl > 0:
            self.harvest_gross += pnl
            self._amortize_from_profit(pnl)

        if reason == "RECOVERY_BE":
            # A recovery is successful only if the entire realized bundle is non-negative.
            if pnl >= self.config.recovery_bundle_buffer and recovery_ids:
                self.recovery_success_cycles += 1
                self.recovery_success_pnl += pnl
                # Debt represented by the recovered losing positions is retired from ledger.
                retired = min(self.debt_ledger, max(0.0, self.debt_ledger))
                if retired > 0:
                    # Do not pretend all ledger debt vanished; actual amortization is pnl-limited.
                    pay = min(retired, pnl * self.config.amortization_fraction)
                    self.debt_ledger -= pay
                    self.debt_amortized += pay
            else:
                self.recovery_failed_cycles += 1
                # Negative recovery realization becomes explicit debt.
                if pnl < 0:
                    self.debt_ledger += -pnl

        # Any forced loss is explicit debt, not silently called recovery.
        if reason in ("DEFENSE", "EOD") and pnl < 0:
            self.debt_ledger += -pnl

        self.debt_ledger_peak = max(self.debt_ledger_peak, self.debt_ledger)
        return pnl, n

    def _best_recovery_bundle(self, bid, ask):
        # Strict version: full realized bundle itself must close >= buffer.
        if not self.recovery_pool:
            return None
        rec, wins = [], []
        for pid, p in self.pos.items():
            pnl = self._pnl_pos(p, bid, ask)
            age = self.bar_i - p.opened_bar
            row = (pid, p, pnl, age)
            if pid in self.recovery_pool and pnl < 0:
                rec.append(row)
            elif pid not in self.recovery_pool and pnl > 0:
                wins.append(row)
        if not rec or not wins:
            return None

        rec = sorted(rec, key=lambda x: x[2])[:self.config.recovery_bundle_max]
        wins = sorted(wins, key=lambda x: x[2], reverse=True)[:self.config.cluster_candidate_pool]
        best = None
        import itertools
        for r in rec:
            max_k = min(self.config.recovery_max_winners, len(wins))
            for k in range(1, max_k + 1):
                for wsub in itertools.combinations(wins, k):
                    realized_net = r[2] + sum(x[2] for x in wsub)
                    if realized_net < self.config.recovery_bundle_buffer:
                        continue
                    ids = [r[0]] + [x[0] for x in wsub]
                    utility = realized_net + 0.80 * (-r[2]) + 0.10 * len(ids) - self._future_edge_penalty(wsub)
                    if best is None or utility > best[0]:
                        best = (utility, ids, realized_net)
        return best

    def _try_recovery_pool(self, bid, ask):
        cand = self._best_recovery_bundle(bid, ask)
        if cand is None:
            return False
        utility, ids, expected_net = cand
        cp, n = self._close_ids(ids, bid, ask, "RECOVERY_BE")
        self.cluster_cycles.append({"ids": ids, "pnl": cp, "n": n, "reason": "RECOVERY_BE", "utility": utility})
        self.recovery_cycles += 1
        self._recovery_debt(bid, ask)
        return cp >= self.config.recovery_bundle_buffer

    def on_quote_tick(self, tick: QuoteTick):
        self.tick_i += 1
        bid = self.f(tick.bid_price)
        ask = self.f(tick.ask_price)
        self.last_bid = bid
        self.last_ask = ask
        self._mark_equity(bid, ask)

        if not hasattr(self, "last_entry_bar"):
            self.last_entry_bar = -1
        if self.bar_i > 0 and self.last_entry_bar != self.bar_i:
            side = self.logic.desired_entry_side()
            if side is not None:
                if self.logic.regime == Regime.RANGE:
                    requested = 1
                elif self.logic.regime == Regime.EXPANSION:
                    requested = 5 if self.logic.last_crt_signals.score_100() >= 90 else 3
                else:
                    requested = 0
                n = self._allowed_new_entries(requested, bid, ask)
                if requested > 0 and n == 0:
                    self.entry_suppressed_bars += 1
                if n > 0:
                    self._enter(1 if side == Side.LONG else -1, ask if side == Side.LONG else bid, n)
            self.last_entry_bar = self.bar_i

        if self.config.mode == "BASE":
            self._evaluate_base(bid, ask)
        else:
            self._evaluate_cluster(bid, ask)

    def on_stop(self):
        # Do NOT convert finite-test endpoint into a strategy loss event.
        # Record open inventory at mark-to-market so EOD does not dominate recovery evaluation.
        if self.last_bid is not None:
            self.eod_open_positions = len(self.pos)
            self.eod_mark_to_market = sum(self._pnl_pos(p, self.last_bid, self.last_ask) for p in self.pos.values())
            self._economic_debt(self.last_bid, self.last_ask)

    def summary(self):
        out = super().summary()
        # Parent summary is realized-only because v4 intentionally leaves endpoint inventory open.
        realized_net = out["net"]
        mtm_net = realized_net + self.eod_mark_to_market
        out.update({
            "verification_level_v4": "DEBT_AMORTIZATION_INVENTORY_CONTROL_V4",
            "realized_net_pre_mtm": realized_net,
            "eod_mark_to_market": self.eod_mark_to_market,
            "eod_open_positions": self.eod_open_positions,
            "net_mtm": mtm_net,
            "return_mtm_pct": mtm_net / self.config.initial_balance * 100.0,
            "debt_ledger_end": self.debt_ledger,
            "debt_ledger_peak": self.debt_ledger_peak,
            "debt_amortized": self.debt_amortized,
            "harvest_gross": self.harvest_gross,
            "harvest_retained": self.harvest_retained,
            "true_recovery_success_cycles": self.recovery_success_cycles,
            "true_recovery_failed_cycles": self.recovery_failed_cycles,
            "true_recovery_success_pnl": self.recovery_success_pnl,
            "true_recovery_success_rate_pct": 100.0 * self.recovery_success_cycles / max(self.recovery_success_cycles + self.recovery_failed_cycles, 1),
            "entry_suppressed_bars": self.entry_suppressed_bars,
        })
        return out


def run_cell(catalog_path, mode, tf, experiment_id):
    catalog = ParquetDataCatalog(str(catalog_path))
    inst = next((x for x in catalog.instruments() if x.id.symbol.value.replace("/", "") == "XAUUSD"), None)
    if inst is None:
        raise SystemExit("XAUUSD missing")
    ticks = catalog.query_quote_ticks(identifiers=[inst.id.value])
    if not ticks:
        raise SystemExit("no Raw QuoteTicks")

    eng = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"), risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue, oms_type=OmsType.NETTING, account_type=AccountType.MARGIN, base_currency=USD,
                  starting_balances=[Money(1000, USD)], default_leverage=Decimal("2000"))
    eng.add_instrument(inst)
    eng.add_data(ticks)
    bt = BarType.from_str(f"{inst.id.value}-{tf}-MINUTE-BID-INTERNAL")
    st = ABStrategyV4(V4Cfg(instrument_id=inst.id, bar_type=bt, mode=mode))
    eng.add_strategy(st)
    eng.run()
    obj = {
        "verification_level": "NAUTILUS_RAW_BIDASK_AE_DEBT_AMORTIZATION_CLUSTER_V4",
        "engine": "NautilusTrader BacktestEngine",
        "nautilus_version": getattr(nautilus_trader, "__version__", "unknown"),
        "raw_ticks": len(ticks),
        "ohlc_resample_used": False,
        "signal_bars": "Nautilus INTERNAL BID bars from Raw QuoteTicks",
        "execution": "virtual micro-position ledger marked on raw Bid/Ask; raw spread included; endpoint inventory marked-to-market",
        "tf_minutes": tf,
        **st.summary(),
    }
    eng.dispose()
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--mode", choices=["BASE", "CLUSTER"], required=True)
    ap.add_argument("--tf", type=int, default=1)
    ap.add_argument("--raw-bidask-only", action="store_true")
    a = ap.parse_args()
    if not a.raw_bidask_only:
        raise SystemExit("raw-bidask-only mandatory")
    out = run_cell(Path(a.catalog), a.mode, a.tf, a.experiment_id)
    p = Path("results/ae-allweather-ab") / a.experiment_id
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{a.mode}_M{a.tf}.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
