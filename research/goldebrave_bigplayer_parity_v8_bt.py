from __future__ import annotations

import argparse
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

import nautilus_trader
import pandas as pd
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from research.goldebrave_bigplayer_parity_v7_bt import ParityV7Strategy
from research.goldebrave_nautilus_raw_bt import GoldeBraveConfig

SIM = Venue("SIM")


class ParityV8Strategy(ParityV7Strategy):
    """Parity v8: MT5 Examples\\ZigZag buffer/shift parity for A/B.

    Key correction versus earlier reconstructions:
    GoldeBrave scans CopyBuffer(..., 0, 0, Lookback+1) as a series and loops BAR SHIFTS
    MINSHIFT=2 .. Lookback.  Earlier code built a compact pivot list and removed the newest
    two *pivots*, which is not equivalent to ignoring shifts 0 and 1.  v8 keeps a full
    bar-indexed ZigZag buffer and scans exact shifts.

    The extremum maps reproduce the MetaQuotes Examples/ZigZag Depth/Deviation/Backstep
    pipeline and final high/low alternation. BigPlayer remains detector-only.
    """

    MINSHIFT = 2

    def __init__(self, config):
        super().__init__(config)
        self.zz_scan_calls = Counter()
        self.zz_nonzero_seen = Counter()
        self.zz_peak_seen = Counter()
        self.zz_trough_seen = Counter()
        self.zz_candidates_outside_day = Counter()
        self.zz_orders_accepted = Counter()

    def _point(self) -> float:
        # GoldeBrave XAU 3-digit setup: g_unit = K10 * _Point * scale(10) = 100*_Point.
        return self.config.unit / 100.0

    def _zigzag_buffer_exact(self, bars):
        n = len(bars)
        depth = int(self.config.zz_depth)
        backstep = int(self.config.zz_backstep)
        deviation_px = float(self.config.zz_deviation) * self._point()
        zz = [0.0] * n
        lowmap = [0.0] * n
        highmap = [0.0] * n
        if n < depth + 2:
            return zz

        last_low = None
        last_high = None

        # MetaQuotes Examples/ZigZag extremum-map pass, chronological equivalent.
        for shift in range(depth - 1, n):
            lo = min(b.l for b in bars[shift - depth + 1:shift + 1])
            val = lo
            if last_low is not None and val == last_low:
                val = 0.0
            else:
                last_low = lo
                if (bars[shift].l - lo) > deviation_px:
                    val = 0.0
                else:
                    for back in range(1, backstep + 1):
                        k = shift - back
                        if k >= 0 and lowmap[k] != 0.0 and lowmap[k] > lo:
                            lowmap[k] = 0.0
            if val != 0.0 and bars[shift].l == val:
                lowmap[shift] = val

            hi = max(b.h for b in bars[shift - depth + 1:shift + 1])
            val = hi
            if last_high is not None and val == last_high:
                val = 0.0
            else:
                last_high = hi
                if (hi - bars[shift].h) > deviation_px:
                    val = 0.0
                else:
                    for back in range(1, backstep + 1):
                        k = shift - back
                        if k >= 0 and highmap[k] != 0.0 and highmap[k] < hi:
                            highmap[k] = 0.0
            if val != 0.0 and bars[shift].h == val:
                highmap[shift] = val

        # MetaQuotes final-search pass: alternate peaks/troughs and replace same-side extrema.
        what = 0  # 0 initial, +1 seek low, -1 seek high
        last_low_val = 0.0
        last_high_val = 0.0
        last_low_pos = -1
        last_high_pos = -1

        for shift in range(depth - 1, n):
            lo = lowmap[shift]
            hi = highmap[shift]
            if what == 0:
                if last_low_val == 0.0 and last_high_val == 0.0:
                    if hi != 0.0:
                        last_high_val = hi
                        last_high_pos = shift
                        zz[shift] = hi
                        what = 1
                    if lo != 0.0:
                        last_low_val = lo
                        last_low_pos = shift
                        zz[shift] = lo
                        what = -1
                continue

            if what == 1:  # last point high; seek low
                if lo != 0.0 and lo < last_low_val if last_low_val != 0.0 else lo != 0.0:
                    if last_low_pos >= 0:
                        zz[last_low_pos] = 0.0
                    last_low_pos = shift
                    last_low_val = lo
                    zz[shift] = lo
                if hi != 0.0 and lo == 0.0:
                    last_high_val = hi
                    last_high_pos = shift
                    zz[shift] = hi
                    what = -1
                continue

            if what == -1:  # last point low; seek high
                if hi != 0.0 and hi > last_high_val if last_high_val != 0.0 else hi != 0.0:
                    if last_high_pos >= 0:
                        zz[last_high_pos] = 0.0
                    last_high_pos = shift
                    last_high_val = hi
                    zz[shift] = hi
                if lo != 0.0 and hi == 0.0:
                    last_low_val = lo
                    last_low_pos = shift
                    zz[shift] = lo
                    what = 1

        return zz

    def _scan(self, tf: str, cap: int, ns: int, layer: str):
        dr = self._day_range(ns)
        if dr is None:
            return
        dhi, dlo = dr
        hi_bar = dhi + self.min_dist()
        lo_bar = dlo - self.min_dist()

        need = int(self.config.lookback) + 1
        xs = list(self.bars[tf])[-need:]
        if len(xs) < need:
            return
        zz = self._zigzag_buffer_exact(xs)
        self.zz_scan_calls[layer] += 1
        n_buy = 0
        n_sell = 0
        eps = self._point() / 2.0

        # ArraySetAsSeries(true): shift i maps to chronological index len-1-i.
        for shift in range(self.MINSHIFT, int(self.config.lookback) + 1):
            idx = len(xs) - 1 - shift
            if idx < 0:
                break
            z = zz[idx]
            if z <= 0.0:
                continue
            self.zz_nonzero_seen[layer] += 1
            b = xs[idx]

            if abs(z - b.h) <= eps:
                self.zz_peak_seen[layer] += 1
                if z > hi_bar and n_buy < cap:
                    self.zz_candidates_outside_day[layer] += 1
                    n_buy += 1
                    p = z - self.off()
                    if not self._placed_has(p):
                        if self._place(+1, p, layer, ns, track_placed=True):
                            self.zz_orders_accepted[layer] += 1
                    hi_bar = z
                    continue

            if abs(z - b.l) <= eps:
                self.zz_trough_seen[layer] += 1
                if z < lo_bar and n_sell < cap:
                    self.zz_candidates_outside_day[layer] += 1
                    n_sell += 1
                    q = z + self.off()
                    if not self._placed_has(q):
                        if self._place(-1, q, layer, ns, track_placed=True):
                            self.zz_orders_accepted[layer] += 1
                    lo_bar = z

    def summary(self):
        x = super().summary()
        x["strategy_id"] = "GoldeBrave_v4_plus_BigPlayerDetector_parity_v8_exact_zigzag_shift"
        x["verification_level"] = "NAUTILUS_RAW_BIDASK_MT5_PARITY_V8_EXACT_ZIGZAG_SHIFT_SCAN"
        x["zigzag_parity"] = {
            "model": "MetaQuotes Examples/ZigZag maps + exact GoldeBrave series shift scan 2..Lookback",
            "depth": self.config.zz_depth,
            "deviation": self.config.zz_deviation,
            "backstep": self.config.zz_backstep,
            "lookback": self.config.lookback,
            "minshift": self.MINSHIFT,
            "scan_calls": dict(self.zz_scan_calls),
            "nonzero_buffer_points_seen": dict(self.zz_nonzero_seen),
            "peaks_seen": dict(self.zz_peak_seen),
            "troughs_seen": dict(self.zz_trough_seen),
            "outside_day_candidates": dict(self.zz_candidates_outside_day),
            "orders_accepted": dict(self.zz_orders_accepted),
        }
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--raw-bidask-only", action="store_true")
    ap.add_argument("--initial-balance", type=float, default=100000.0)
    args = ap.parse_args()
    if not args.raw_bidask_only:
        raise SystemExit("raw-bidask-only mandatory")

    catalog = ParquetDataCatalog(str(Path(args.catalog)))
    instrument = next((x for x in catalog.instruments() if x.id.symbol.value.replace("/", "") == "XAUUSD"), None)
    if instrument is None:
        raise SystemExit("XAUUSD missing")
    ticks = catalog.query(data_cls=QuoteTick, identifiers=[instrument.id.value])
    if not ticks:
        raise SystemExit("no raw XAUUSD QuoteTicks")

    engine = BacktestEngine(config=BacktestEngineConfig(
        logging=LoggingConfig(log_level="ERROR"),
        risk_engine=RiskEngineConfig(bypass=True),
    ))
    engine.add_venue(
        venue=SIM,
        oms_type=OmsType.HEDGING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(args.initial_balance, USD)],
        default_leverage=Decimal("2000"),
    )
    engine.add_instrument(instrument)
    engine.add_data(ticks)
    iid = instrument.id.value
    st = ParityV8Strategy(GoldeBraveConfig(
        instrument_id=instrument.id,
        m1=BarType.from_str(f"{iid}-1-MINUTE-BID-INTERNAL"),
        m15=BarType.from_str(f"{iid}-15-MINUTE-BID-INTERNAL"),
        h1=BarType.from_str(f"{iid}-1-HOUR-BID-INTERNAL"),
        initial_balance=args.initial_balance,
    ))
    engine.add_strategy(st)
    engine.run()
    s = st.summary()
    s.update({
        "engine": "NautilusTrader BacktestEngine",
        "nautilus_version": getattr(nautilus_trader, "__version__", "unknown"),
        "raw_ticks": len(ticks),
        "ohlc_resample_used": False,
    })
    out = Path("results/goldebrave_bigplayer") / args.experiment_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([s]).to_json(out / "arena_result.json", orient="records", indent=2)
    print(json.dumps(s, indent=2, ensure_ascii=False))
    engine.dispose()


if __name__ == "__main__":
    main()
