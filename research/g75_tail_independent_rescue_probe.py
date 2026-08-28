from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path

from research.g75_multihorizon_fixed_debt_probe import MINUTE_NS, NS, Q, load_catalog_quotes, run_shadow, extract_cycles

TAIL_START_S = 21600
END_S = 86400
RESCUE_SLOTS = (3, 5, 10)


def basket_pnl(q: Q, side: int, entries: list[float]) -> float:
    n = len(entries)
    if side > 0:
        return q.bid * n - sum(entries)
    return sum(entries) - q.ask * n


def natural_tail_cases(qs: list[Q], max_layers: int) -> list[dict]:
    shadow = run_shadow(qs, max_layers)
    cycles = extract_cycles(shadow)
    ts = [q.ts for q in qs]
    idx = {q.ts: i for i, q in enumerate(qs)}
    out: list[dict] = []

    for c in cycles:
        i0 = idx.get(c["loss_state_ts"])
        if i0 is None:
            continue
        side = int(c["side"])
        entries = [float(x) for x in c["entries"]]
        p0 = basket_pnl(qs[i0], side, entries)
        if p0 >= 0:
            continue

        i6 = bisect.bisect_left(ts, c["loss_state_ts"] + TAIL_START_S * NS, lo=i0)
        i24 = bisect.bisect_right(ts, c["loss_state_ts"] + END_S * NS, lo=i6)
        if i6 >= len(qs) or i6 >= i24:
            continue

        recovered_before_6h = False
        for q in qs[i0:i6 + 1]:
            if basket_pnl(q, side, entries) >= 0:
                recovered_before_6h = True
                break
        if recovered_before_6h:
            continue

        p6 = basket_pnl(qs[i6], side, entries)
        if p6 >= 0:
            continue

        out.append({
            "loss_state_ts": int(c["loss_state_ts"]),
            "tail_start_ts": int(qs[i6].ts),
            "tail_start_index": i6,
            "end_index": i24,
            "original_side": side,
            "original_layers": len(entries),
            "locked_debt_at_6h": -p6,
        })
    return out


def run_independent_rescue(qs: list[Q], start_i: int, end_i: int, slots: int, notional_budget: float, initial_debt: float) -> dict:
    # Fresh independent state: it does not inherit the main basket direction/anchor.
    anchor = 0.0
    pending = 0
    active = 0
    peak = 0.0
    trough = 0.0
    entries: list[float] = []
    latest_fill = 0.0
    current_minute: int | None = None
    current_close = 0.0
    previous_close: float | None = None

    qty_per_layer = notional_budget / slots
    debt = float(initial_debt)
    realized = 0.0
    wins = 0
    losses = 0
    closed_trades = 0
    max_layers_used = 0
    tau_be_s = None
    start_ts = qs[start_i].ts

    for q in qs[start_i:end_i]:
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
                exit_px = q.bid if active > 0 else q.ask
                if active > 0:
                    pnl = sum((exit_px - e) * qty_per_layer for e in entries)
                else:
                    pnl = sum((e - exit_px) * qty_per_layer for e in entries)
                realized += pnl
                debt -= pnl
                closed_trades += 1
                wins += int(pnl > 0)
                losses += int(pnl <= 0)
                active = 0
                anchor = stop
                peak = trough = 0.0
                entries = []
                latest_fill = 0.0
                if debt <= 0:
                    tau_be_s = (q.ts - start_ts) / NS
                    break
            else:
                last = latest_fill if latest_fill > 0 else sum(entries) / len(entries)
                guard = 0
                while active and len(entries) < slots and guard < slots:
                    guard += 1
                    nxt = last + active * 0.025
                    crossed = q.ask >= nxt if active > 0 else q.bid <= nxt
                    if not crossed:
                        break
                    fill = q.ask if active > 0 else q.bid
                    entries.append(fill)
                    latest_fill = fill
                    max_layers_used = max(max_layers_used, len(entries))
                    last = nxt

        if pending and not active:
            s = pending
            fill = q.ask if s > 0 else q.bid
            active = s
            pending = 0
            entries = [fill]
            latest_fill = fill
            max_layers_used = max(max_layers_used, 1)
            peak = q.bid if s > 0 else q.ask
            trough = peak
            anchor = peak

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

    # Open rescue inventory is not credited to debt because only realized rescue PnL services debt.
    return {
        "slots": slots,
        "total_notional_budget": notional_budget,
        "qty_per_layer": qty_per_layer,
        "initial_debt": initial_debt,
        "remaining_debt": max(0.0, debt),
        "realized_rescue_pnl": realized,
        "system_be_reached": debt <= 0,
        "tau_system_be_from_6h_s": tau_be_s,
        "closed_rescue_cycles": closed_trades,
        "rescue_wins": wins,
        "rescue_losses": losses,
        "max_layers_used": max_layers_used,
        "open_rescue_inventory_at_end": bool(active),
    }


def summarize(rows: list[dict], slots: int) -> dict:
    rs = [r for r in rows if r["slots"] == slots]
    n = len(rs)
    wins = [r for r in rs if r["system_be_reached"]]
    pnl_sum = sum(float(r["realized_rescue_pnl"]) for r in rs)
    pos = sum(max(0.0, float(r["realized_rescue_pnl"])) for r in rs)
    neg = sum(max(0.0, -float(r["realized_rescue_pnl"])) for r in rs)
    return {
        "tail_cases": n,
        "system_be_by_24h_rate": len(wins) / n if n else None,
        "system_be_by_24h_count": len(wins),
        "mean_remaining_debt": sum(float(r["remaining_debt"]) for r in rs) / n if n else None,
        "mean_realized_rescue_pnl": pnl_sum / n if n else None,
        "rescue_profit_factor": (pos / neg) if neg > 0 else (None if pos == 0 else float("inf")),
        "closed_rescue_cycles": sum(int(r["closed_rescue_cycles"]) for r in rs),
        "rescue_cycle_win_rate": (
            sum(int(r["rescue_wins"]) for r in rs) /
            sum(int(r["closed_rescue_cycles"]) for r in rs)
            if sum(int(r["closed_rescue_cycles"]) for r in rs) else None
        ),
        "open_rescue_inventory_at_24h_count": sum(bool(r["open_rescue_inventory_at_end"]) for r in rs),
    }


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
        "classification": "G75_6H_TAIL_INDEPENDENT_RESCUE_PROBE_V1",
        "verification_level": "NAUTILUS_RAW_BIDASK_DISCOVERY_NOT_BACKTESTENGINE",
        "symbol": args.symbol,
        "quote_rows": len(qs),
        "tail_start_s": TAIL_START_S,
        "end_s": END_S,
        "rescue_slots": list(RESCUE_SLOTS),
        "truth_boundary": (
            "6h-unresolved natural G75 baskets are frozen as debt at the executable 6h Bid/Ask mark. "
            "A fresh independent G75 Price-Follow state then trades from 6h to 24h. Main-basket state is not inherited. "
            "Rescue losses are closed normally and increase the remaining debt through the realized-PnL ledger; rescue positions are never recursively rescued. "
            "3/5/10 use the same total rescue notional budget equal to the original basket layer count, divided equally per rescue layer. "
            "Only realized rescue PnL services debt; open rescue inventory at 24h is not credited. Raw QuoteTicks are used with executable Bid/Ask and no OHLC resampling. "
            "Commission, explicit slippage, latency, swap/hedge carry, broker margin, cashback and full MTM portfolio accounting are absent, so WR5=INVALID."
        ),
        "wr5": "INVALID",
        "wr5_missing": [
            "round_trip_commission", "explicit_slippage", "execution_delay", "swap_or_hedge_carry",
            "cashback_assumptions", "synchronized_mark_to_market_equity", "floating_drawdown_under_actual_hedge",
            "aggregate_exposure", "broker_margin_level", "event_price_pitch_budget",
        ],
        "L10": {},
        "L20": {},
    }

    for max_layers in (10, 20):
        tails = natural_tail_cases(qs, max_layers)
        rows: list[dict] = []
        for t in tails:
            for slots in RESCUE_SLOTS:
                r = run_independent_rescue(
                    qs,
                    int(t["tail_start_index"]),
                    int(t["end_index"]),
                    slots,
                    float(t["original_layers"]),
                    float(t["locked_debt_at_6h"]),
                )
                rows.append({**t, **r})
        result[f"L{max_layers}"] = {
            "six_hour_tail_count": len(tails),
            "six_hour_tail_rate_vs_loss_candidates": None,
            "same_total_rescue_notional_rule": "original basket layer count; qty_per_layer=original_layers/slots",
            "variants": {str(s): summarize(rows, s) for s in RESCUE_SLOTS},
        }
        if rows:
            with (out / f"g75_tail_independent_rescue_l{max_layers}_events.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    result["period"] = {
        "start": manifest.get("start"),
        "days": manifest.get("days"),
        "end_exclusive": manifest.get("end_exclusive"),
    }
    (out / "g75_tail_independent_rescue_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
