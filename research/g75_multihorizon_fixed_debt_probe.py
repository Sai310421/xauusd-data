from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from nautilus_trader.persistence.catalog import ParquetDataCatalog

NS = 1_000_000_000
MINUTE_NS = 60 * NS
HORIZONS_S = (300, 600, 1800, 3600, 21600, 43200, 86400)
RESCUE_SLOTS = (3, 5, 10)


@dataclass
class Q:
    ts: int
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


def _f(x) -> float:
    return float(x.as_double()) if hasattr(x, "as_double") else float(x)


def load_catalog_quotes(catalog_path: Path, plain_symbol: str) -> tuple[list[Q], dict]:
    manifest = json.loads((catalog_path / "catalog_manifest.json").read_text(encoding="utf-8"))
    catalog = ParquetDataCatalog(str(catalog_path))
    inst_by_plain = {x.id.symbol.value.replace("/", ""): x for x in catalog.instruments()}
    instrument = inst_by_plain.get(plain_symbol)
    if instrument is None:
        raise SystemExit(f"instrument missing from Nautilus catalog: {plain_symbol}")
    ticks = catalog.quote_ticks(instrument_ids=[instrument.id.value])
    if not ticks:
        raise SystemExit(f"no raw QuoteTicks: {plain_symbol}")
    qs = [Q(int(t.ts_event), _f(t.bid_price), _f(t.ask_price)) for t in ticks]
    return qs, manifest


def run_shadow(qs: list[Q], max_layers: int) -> dict:
    anchor = 0.0
    pending = 0
    active = 0
    peak = 0.0
    trough = 0.0
    entries: list[float] = []
    latest_position_fill = 0.0
    current_minute: int | None = None
    current_close = 0.0
    previous_close: float | None = None
    events: list[dict[str, object]] = []
    cycle_count = 0

    for q in qs:
        bar = (q.ts // MINUTE_NS) * MINUTE_NS
        new_bar = False
        if current_minute is None:
            current_minute = bar
            current_close = q.mid
        elif bar != current_minute:
            previous_close = current_close
            current_minute = bar
            current_close = q.mid
            new_bar = True
        else:
            current_close = q.mid

        if active:
            if active > 0:
                peak = max(peak, q.bid)
            else:
                trough = q.ask if trough <= 0 else min(trough, q.ask)

            stop = peak - 0.20 if active > 0 else trough + 0.20
            hit = q.bid <= stop if active > 0 else q.ask >= stop
            if hit:
                events.append({"ts": q.ts, "event": "REVERSAL", "side": active, "logical": stop, "layers": len(entries)})
                active = 0
                anchor = stop
                peak = trough = 0.0
                entries = []
                latest_position_fill = 0.0
            else:
                last = latest_position_fill if latest_position_fill > 0 else (sum(entries) / len(entries))
                guard = 0
                while active and len(entries) < max_layers and guard < max_layers:
                    guard += 1
                    nxt = last + active * 0.025
                    crossed = q.ask >= nxt if active > 0 else q.bid <= nxt
                    if not crossed:
                        break
                    fill = q.ask if active > 0 else q.bid
                    entries.append(fill)
                    latest_position_fill = fill
                    events.append({"ts": q.ts, "event": "ADD", "side": active, "fill": fill, "layer": len(entries)})
                    last = nxt

        if pending and not active:
            s = pending
            fill = q.ask if s > 0 else q.bid
            active = s
            pending = 0
            entries = [fill]
            latest_position_fill = fill
            peak = q.bid if s > 0 else q.ask
            trough = peak
            anchor = peak
            cycle_count += 1
            events.append({"ts": q.ts, "event": "ENTRY", "side": s, "fill": fill, "layer": 1})

        if not active and not pending and new_bar:
            c1 = previous_close
            if c1 is not None:
                if anchor <= 0:
                    anchor = c1
                elif c1 >= anchor + 0.12:
                    pending = 1
                elif c1 <= anchor - 0.12:
                    pending = -1
                else:
                    anchor = c1

    return {"metrics": {"cycle_count": cycle_count}, "events": events}


def extract_cycles(shadow: dict) -> list[dict]:
    cycles = []
    cur = None
    for ev in shadow["events"]:
        typ = ev.get("event")
        if typ == "ENTRY":
            cur = {"side": int(ev["side"]), "entries": [float(ev["fill"])], "entry_ts": int(ev["ts"])}
        elif typ == "ADD" and cur is not None:
            cur["entries"].append(float(ev["fill"]))
        elif typ == "REVERSAL" and cur is not None:
            cur["loss_state_ts"] = int(ev["ts"])
            cycles.append(cur)
            cur = None
    return cycles


def pct(xs: list[float], p: float):
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return ys[lo]
    return ys[lo] * (hi - k) + ys[hi] * (k - lo)


def dist(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "p50": pct(xs, 0.50),
        "p95": pct(xs, 0.95),
        "p99": pct(xs, 0.99),
        "max": max(xs),
    }


def analyze(qs: list[Q], max_layers: int) -> tuple[dict, list[dict]]:
    shadow = run_shadow(qs, max_layers)
    cycles = extract_cycles(shadow)
    ts = [q.ts for q in qs]
    idx_by_ts = {q.ts: i for i, q in enumerate(qs)}
    rows: list[dict] = []

    for c in cycles:
        i0 = idx_by_ts.get(c["loss_state_ts"])
        if i0 is None:
            continue
        side = int(c["side"])
        entries = [float(x) for x in c["entries"]]
        q0 = qs[i0]

        def basket_pnl(q: Q) -> float:
            px = q.bid if side > 0 else q.ask
            return sum(side * (px - e) for e in entries)

        p0 = basket_pnl(q0)
        if p0 >= 0:
            continue

        debt = -p0
        max_end = bisect.bisect_right(ts, c["loss_state_ts"] + HORIZONS_S[-1] * NS, lo=i0)
        tau_ebe = None
        worst_additional_mae = 0.0
        best_improvement = 0.0
        for q in qs[i0:max_end]:
            x = basket_pnl(q)
            worst_additional_mae = max(worst_additional_mae, max(0.0, -x - debt))
            best_improvement = max(best_improvement, x - p0)
            if x >= 0:
                tau_ebe = (q.ts - c["loss_state_ts"]) / NS
                break

        row = {
            "side": side,
            "layers": len(entries),
            "loss_state_ts": c["loss_state_ts"],
            "entry_to_lock_s": (c["loss_state_ts"] - c["entry_ts"]) / NS,
            "locked_debt_distance": debt,
            "spread_at_lock": q0.ask - q0.bid,
            "tau_economic_be_s": tau_ebe,
            "recovered_within_24h": bool(tau_ebe is not None and tau_ebe <= HORIZONS_S[-1]),
            "unlocked_path_additional_mae_before_be_or_24h": worst_additional_mae,
            "unlocked_path_best_improvement_before_be_or_24h": best_improvement,
        }
        for h in HORIZONS_S:
            row[f"be_le_{h}s"] = bool(tau_ebe is not None and tau_ebe <= h)
        for n in RESCUE_SLOTS:
            row[f"required_debt_per_{n}_rescue_slots"] = debt / n
            row[f"required_be_plus10pct_per_{n}_rescue_slots"] = debt * 1.10 / n
        rows.append(row)

    summary = {
        "classification": "G75_MULTI_HORIZON_FIXED_DEBT_RECOVERY_PROBE_V1",
        "truth_boundary": (
            "Discovery probe on Nautilus Raw Bid/Ask QuoteTicks using verified G75 event ordering. "
            "At the reversal/loss state, loss is treated as economically lockable FixedDebt. "
            "No rescue trades are injected. 3/5/10 rescue-slot values are arithmetic debt-allocation requirements, not performance. "
            "Spread is present; commission, hedge carry/swap, explicit slippage, execution delay, broker margin mechanics and actual rescue strategy costs are absent, therefore WR5=INVALID."
        ),
        "max_layers": max_layers,
        "source_cycles": shadow["metrics"]["cycle_count"],
        "loss_lock_candidates": len(rows),
        "horizons_s": list(HORIZONS_S),
        "economic_be_recovery_rate_by_horizon": {
            str(h): (sum(r[f"be_le_{h}s"] for r in rows) / len(rows) if rows else None)
            for h in HORIZONS_S
        },
        "economic_be_time_s_recovered_24h": dist([float(r["tau_economic_be_s"]) for r in rows if r["tau_economic_be_s"] is not None]),
        "locked_debt_distance": dist([float(r["locked_debt_distance"]) for r in rows]),
        "unlocked_additional_mae_avoided_by_ideal_lock": dist([float(r["unlocked_path_additional_mae_before_be_or_24h"]) for r in rows]),
        "unresolved_24h_count": sum(not r["recovered_within_24h"] for r in rows),
        "unresolved_24h_rate": (sum(not r["recovered_within_24h"] for r in rows) / len(rows) if rows else None),
        "rescue_slot_requirement_mean": {
            str(n): {
                "be": statistics.mean([r[f"required_debt_per_{n}_rescue_slots"] for r in rows]) if rows else None,
                "be_plus10pct": statistics.mean([r[f"required_be_plus10pct_per_{n}_rescue_slots"] for r in rows]) if rows else None,
            }
            for n in RESCUE_SLOTS
        },
        "wr5": "INVALID",
        "wr5_missing": [
            "round_trip_commission", "explicit_slippage", "execution_delay", "swap_or_hedge_carry",
            "cashback_assumptions", "synchronized_mark_to_market_equity", "floating_drawdown_under_actual_hedge",
            "aggregate_exposure", "broker_margin_level", "actual_rescue_entry_policy",
            "unresolved_inventory_accounting", "event_price_pitch_budget",
        ],
    }
    return summary, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--raw-bidask-only", action="store_true")
    args = ap.parse_args()
    if not args.raw_bidask_only:
        raise SystemExit("raw-bidask-only is mandatory")

    qs, manifest = load_catalog_quotes(Path(args.catalog), args.symbol)
    out = Path("results/ae-bt") / args.experiment_id
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "verification_level": "NAUTILUS_RAW_BIDASK_G75_FIXED_DEBT_PROBE",
        "symbol": args.symbol,
        "quote_rows": len(qs),
        "period": {"start": manifest.get("start"), "days": manifest.get("days"), "end_exclusive": manifest.get("end_exclusive")},
        "ohlc_resample_used": False,
    }
    for L in (10, 20):
        summary, rows = analyze(qs, L)
        result[f"L{L}"] = summary
        if rows:
            with (out / f"g75_fixed_debt_l{L}_events.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    (out / "g75_multihorizon_fixed_debt_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "catalog_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
