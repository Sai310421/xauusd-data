#!/usr/bin/env python3
"""
AMOS TripleEngine v2 - NautilusTrader RAW BID/ASK TICK Virtual Entry baseline.

Formal input:
  Dukascopy daily CSV partitions containing:
  time,bid,ask,bid_size,ask_size

This script replays real bid/ask ticks through NautilusTrader BacktestEngine as
QuoteTick events. It does NOT use OHLC bars.

Stage purpose:
  Evaluate all 15 Virtual Entry candidates before promoted logics are wired to
  real order execution. Virtual PnL / virtual cashback never changes real equity.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import platform
from collections import defaultdict, deque
from decimal import Decimal
from pathlib import Path

import nautilus_trader
from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.execution import MakerTakerFeeModel
from nautilus_trader.model import AccountType
from nautilus_trader.model import AssetClass
from nautilus_trader.model import Currency
from nautilus_trader.model import InstrumentId
from nautilus_trader.model import Money
from nautilus_trader.model import OmsType
from nautilus_trader.model import Price
from nautilus_trader.model import Quantity
from nautilus_trader.model import QuoteTick
from nautilus_trader.model import Symbol
from nautilus_trader.model import Venue
from nautilus_trader.model.instruments import Cfd
from nautilus_trader.trading import Strategy


START_EQUITY = 1000.0
LOT = 0.01
CONTRACT_SIZE = 100.0
CB_PER_RT_LOT = 6.0
INSTRUMENT_ID = InstrumentId(Symbol("XAUUSD"), Venue("EXNESS"))
USD = Currency.from_str("USD")
EXNESS = Venue("EXNESS")


class Features:
    def __init__(self, n=240):
        self.m = deque(maxlen=n)
        self.s = deque(maxlen=n)

    def update(self, bid: float, ask: float):
        self.m.append((bid + ask) * 0.5)
        self.s.append(max(0.0, ask - bid))

    def ready(self, n=120):
        return len(self.m) >= n

    def mean(self, xs):
        return sum(xs) / len(xs) if xs else 0.0

    def sd(self, xs):
        if len(xs) < 2:
            return 0.0
        a = self.mean(xs)
        return math.sqrt(sum((x-a)**2 for x in xs)/(len(xs)-1))

    def ret(self, k):
        return self.m[-1] - self.m[-1-k] if len(self.m) > k else 0.0

    def vol(self, n=80):
        x = list(self.m)[-n:]
        return self.sd([x[i]-x[i-1] for i in range(1, len(x))]) if len(x) > 2 else 0.0

    def z(self, n=80):
        x = list(self.m)[-n:]
        sd = self.sd(x)
        return (x[-1]-self.mean(x))/sd if sd else 0.0

    def spread_ratio(self, n=60):
        x = list(self.s)[-n:]
        a = self.mean(x)
        return self.s[-1]/a if a else 1.0

    def high(self, n=100):
        return max(list(self.m)[-n:])

    def low(self, n=100):
        return min(list(self.m)[-n:])

    def sma(self, n=40):
        return self.mean(list(self.m)[-n:])


def candidate_signals(f: Features):
    if not f.ready(120):
        return []
    v = max(f.vol(80), 0.02)
    z = f.z(80)
    sr = f.spread_ratio(60)
    r3 = f.ret(3)
    r10 = f.ret(10)
    px = f.m[-1]
    out = []

    # A - Cashback / turnover
    if sr < 1.10 and abs(r3) > .35*v:
        out.append(("A_CB","A1_TickFollow",1 if r3 > 0 else -1,1.1*v,1.2*v,180))
    if abs(z) > 1.5 and sr < 1.15:
        out.append(("A_CB","A2_MicroReversion",-1 if z > 0 else 1,1.0*v,.9*v,220))
    if sr < .78 and abs(r10) > .5*v:
        out.append(("A_CB","A3_SpreadCompression",1 if r10 > 0 else -1,1.0*v,1.15*v,160))
    if sr < .85 and abs(z) < .55 and abs(r3) > .18*v:
        out.append(("A_CB","A4_TurnoverCycle",1 if r3 > 0 else -1,.8*v,.75*v,100))
    if abs(r10) >= .12 and abs(r3) >= .025 and r3*r10 > 0:
        out.append(("A_CB","A5_G75Pursuit",1 if r10 > 0 else -1,.20,.12,120))

    # B - Trend / breakout
    if px >= f.high(100) and f.ret(8) > 0:
        out.append(("B_TREND","B1_DonchianBreak",1,1.5*v,3.0*v,700))
    elif px <= f.low(100) and f.ret(8) < 0:
        out.append(("B_TREND","B1_DonchianBreak",-1,1.5*v,3.0*v,700))
    if abs(f.ret(12)) > 2.2*v:
        out.append(("B_TREND","B2_ATRExpansion",1 if f.ret(12) > 0 else -1,1.6*v,3.2*v,800))
    if r3 > 0 and f.ret(25) < -v and px > f.sma(20):
        out.append(("B_TREND","B3_MSS_BOS",1,1.3*v,2.6*v,700))
    elif r3 < 0 and f.ret(25) > v and px < f.sma(20):
        out.append(("B_TREND","B3_MSS_BOS",-1,1.3*v,2.6*v,700))
    if abs(f.ret(40)) > 2.8*v and sr < 1.15:
        out.append(("B_TREND","B4_SessionBreak",1 if f.ret(40) > 0 else -1,1.5*v,3.5*v,900))
    if abs(r10) >= .12 and abs(r3) >= .025 and r3*r10 > 0:
        out.append(("B_TREND","B5_G75Pursuit",1 if r10 > 0 else -1,.20,.32,500))

    # C - High win-rate / capital builder
    if abs(z) > 2.0 and sr < 1.05:
        out.append(("C_HIGH_WR","C1_ZScoreMeanReversion",-1 if z > 0 else 1,1.35*v,1.0*v,500))
    if z > 1.8 and f.ret(2) < 0:
        out.append(("C_HIGH_WR","C2_BollingerReEntry",-1,1.2*v,1.0*v,450))
    elif z < -1.8 and f.ret(2) > 0:
        out.append(("C_HIGH_WR","C2_BollingerReEntry",1,1.2*v,1.0*v,450))
    if f.ret(5) < -1.2*v and f.ret(20) < -2*v and f.ret(2) > 0:
        out.append(("C_HIGH_WR","C3_RSIStochConfluence",1,1.25*v,1.05*v,420))
    elif f.ret(5) > 1.2*v and f.ret(20) > 2*v and f.ret(2) < 0:
        out.append(("C_HIGH_WR","C3_RSIStochConfluence",-1,1.25*v,1.05*v,420))
    dev = px - f.sma(40)
    if abs(dev) > 2*v and dev*f.ret(2) < 0:
        out.append(("C_HIGH_WR","C4_VWAPDeviation",-1 if dev > 0 else 1,1.25*v,1.0*v,450))
    if v < max(f.vol(120)*.9,.02) and abs(z) > 1.6 and sr < .95:
        out.append(("C_HIGH_WR","C5_RegimeMicroReversion",-1 if z > 0 else 1,1.1*v,.95*v,380))
    return out


class VirtualTripleEngine(Strategy):
    def __new__(cls, *_args, **_kwargs):
        return super().__new__(cls)

    def __init__(self):
        super().__init__()
        self.f = Features()
        self.open_virtual = []
        self.stats = defaultdict(lambda: {
            "N":0, "W":0, "GP":0.0, "GL":0.0, "Net":0.0,
            "Peak":0.0, "MaxDD":0.0, "Lots":0.0
        })

    def on_start(self):
        self.subscribe_quotes(INSTRUMENT_ID)

    def on_quote(self, q: QuoteTick):
        bid = q.bid_price.as_double()
        ask = q.ask_price.as_double()

        survivors = []
        for tr in self.open_virtual:
            tr["age"] += 1
            px = bid if tr["side"] > 0 else ask
            reason = None
            if (tr["side"] > 0 and px <= tr["sl"]) or (tr["side"] < 0 and px >= tr["sl"]):
                reason = "SL"
            elif (tr["side"] > 0 and px >= tr["tp"]) or (tr["side"] < 0 and px <= tr["tp"]):
                reason = "TP"
            elif tr["age"] >= tr["ttl"]:
                reason = "TTL"

            if reason:
                pnl = tr["side"] * (px-tr["entry"]) * CONTRACT_SIZE * LOT
                s = self.stats[(tr["engine"], tr["logic"])]
                s["N"] += 1
                s["W"] += int(pnl > 0)
                s["GP"] += max(0.0, pnl)
                s["GL"] += max(0.0, -pnl)
                s["Net"] += pnl
                s["Lots"] += LOT
                s["Peak"] = max(s["Peak"], s["Net"])
                s["MaxDD"] = max(s["MaxDD"], s["Peak"]-s["Net"])
            else:
                survivors.append(tr)
        self.open_virtual = survivors

        self.f.update(bid, ask)
        for engine, logic, side, sl_d, tp_d, ttl in candidate_signals(self.f):
            entry = ask if side > 0 else bid
            self.open_virtual.append({
                "engine":engine, "logic":logic, "side":side, "entry":entry,
                "sl":entry-side*sl_d, "tp":entry+side*tp_d,
                "ttl":ttl, "age":0,
            })

    def export(self):
        out = {}
        min_n = {"A_CB":35, "B_TREND":25, "C_HIGH_WR":40}
        for (engine, logic), s in sorted(self.stats.items()):
            pf = s["GP"]/s["GL"] if s["GL"] > 1e-12 else (999.0 if s["GP"] > 0 else 0.0)
            wr = s["W"]/s["N"] if s["N"] else 0.0
            ev = s["Net"]/s["N"] if s["N"] else 0.0
            dd_pct = 100.0*s["MaxDD"]/START_EQUITY
            cb = s["Lots"]*CB_PER_RT_LOT
            rf = s["Net"]/s["MaxDD"] if s["MaxDD"] > 1e-12 else 0.0
            out[f"{engine}/{logic}"] = {
                "N":s["N"],
                "WR":wr,
                "PF":pf,
                "RF":rf,
                "EV_USD":ev,
                "TradingNetUSD":s["Net"],
                "ReturnPct":100.0*s["Net"]/START_EQUITY,
                "MaxVirtualDDPct":dd_pct,
                "VirtualGrossRTLots":s["Lots"],
                "VirtualCBUSD_eval_only":cb,
                "CBInclusiveReturnPct_eval_only":100.0*(s["Net"]+cb)/START_EQUITY,
                "PromotionEligible":(
                    s["N"] >= min_n[engine]
                    and pf >= 1.20
                    and ev > 0
                    and dd_pct <= 8.0
                ),
            }
        return out


def iso_to_ns(value: str) -> int:
    x = dt.datetime.fromisoformat(value.replace("Z","+00:00"))
    if x.tzinfo is None:
        x = x.replace(tzinfo=dt.timezone.utc)
    return int(x.timestamp()*1_000_000_000)


def load_quotes(raw_root: Path, days: int):
    files = sorted(raw_root.glob("*.csv"))
    if not files:
        raise SystemExit(f"No raw daily CSV files under {raw_root}")
    files = files[-days:]
    ticks = []
    rows = 0
    for p in files:
        with p.open("r", encoding="utf-8", newline="") as h:
            rd = csv.DictReader(h)
            fields = set(rd.fieldnames or [])
            if not {"time","bid","ask"}.issubset(fields):
                raise SystemExit(f"RAW BID/ASK only: bad schema {p}: {rd.fieldnames}")
            for r in rd:
                bid = float(r["bid"])
                ask = float(r["ask"])
                if bid <= 0 or ask < bid:
                    continue
                bid_size = max(float(r.get("bid_size") or 1.0), 0.01)
                ask_size = max(float(r.get("ask_size") or 1.0), 0.01)
                ts = iso_to_ns(r["time"])
                ticks.append(QuoteTick(
                    instrument_id=INSTRUMENT_ID,
                    bid_price=Price.from_str(f"{bid:.5f}"),
                    ask_price=Price.from_str(f"{ask:.5f}"),
                    bid_size=Quantity.from_str(f"{bid_size:.2f}"),
                    ask_size=Quantity.from_str(f"{ask_size:.2f}"),
                    ts_event=ts,
                    ts_init=ts,
                ))
                rows += 1
    if not ticks:
        raise SystemExit("No valid raw QuoteTicks loaded")
    return ticks, files, rows


def make_instrument():
    return Cfd(
        instrument_id=INSTRUMENT_ID,
        raw_symbol=Symbol("XAUUSD"),
        asset_class=AssetClass.COMMODITY,
        base_currency=None,
        quote_currency=USD,
        price_precision=5,
        size_precision=2,
        price_increment=Price.from_str("0.00001"),
        size_increment=Quantity.from_str("0.01"),
        lot_size=Quantity.from_str("1.00"),
        margin_init=Decimal("0.0005"),
        margin_maint=Decimal("0.0005"),
        ts_event=0,
        ts_init=0,
    )


def run(raw_root: Path, out_path: Path, days: int):
    ticks, files, rows = load_quotes(raw_root, days)
    engine = BacktestEngine(BacktestEngineConfig(bypass_logging=True, run_analysis=False))
    engine.add_venue(
        venue=EXNESS,
        oms_type=OmsType.HEDGING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money.from_str("1000 USD")],
        fee_model=MakerTakerFeeModel(
            maker_rate=Decimal("0"),
            taker_rate=Decimal("0"),
        ),
    )
    engine.add_instrument(make_instrument())
    strategy = VirtualTripleEngine()
    engine.add_strategy(strategy)
    engine.add_data(ticks)
    engine.run()

    payload = {
        "status":"NAUTILUS_RAW_VIRTUAL_BASELINE",
        "nautilus_version":getattr(nautilus_trader,"__version__","unknown"),
        "python":platform.python_version(),
        "input":{
            "raw_root":str(raw_root),
            "partitions":len(files),
            "first_file":files[0].name,
            "last_file":files[-1].name,
            "quote_ticks":rows,
            "data_type":"QuoteTick bid/ask",
            "ohlc_used":False,
        },
        "account_profile":{
            "start_usd":START_EQUITY,
            "broker_target":"Exness Pro",
            "leverage_assumption":"1:2000",
            "tari_tari_cb_usd_per_rt_lot":CB_PER_RT_LOT,
            "virtual_cb_counted_in_real_equity":False,
        },
        "stage_note":"Virtual Entry selection baseline. Real execution/margin KPI follows after logic promotion.",
        "logic_kpi":strategy.export(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    engine.dispose()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True, help="Directory containing Dukascopy YYYY-MM-DD.csv partitions")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--out", default="amos_triple_engine_v2/nautilus_raw90_kpi.json")
    a = ap.parse_args()
    run(Path(a.raw_root), Path(a.out), a.days)


if __name__ == "__main__":
    main()
