from __future__ import annotations

import argparse
import json
import math
from collections import deque
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

# Nautilus 1.230 catalog compatibility.
if not hasattr(ParquetDataCatalog, "query_quote_ticks"):
    def _query_quote_ticks(self, identifiers=None, start=None, end=None):
        return self.query(data_cls=QuoteTick, identifiers=identifiers, start=start, end=end)
    ParquetDataCatalog.query_quote_ticks = _query_quote_ticks

SIM = Venue("SIM")

P = {
    "entry_sigma": 2.0,
    "stop_sigma": 3.2,
    "allowed_utc_hours": {1, 13},
    "allowed_mql_weekdays": {1, 2, 3},
    "reverse_signal": True,
    "sl_atr": 0.70,
    "tp_rr": 1.80,
    "max_hold_bars": 72,
    "max_adx": 30.0,
    "be_rr": 1.0,
    "trail_start_rr": 1.0,
    "trail_atr": 2.0,
    "runner_trigger_rr": 1.80,
    "runner_partial": 0.50,
    "fixed_qty_oz": 2,
    "max_spread_price": 1.20,
    "max_daily_dd_pct": 3.0,
    "max_cycle_dd_pct": 3.0,
}

class Config(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType

class Fusion(Strategy):
    def __init__(self, config: Config):
        super().__init__(config)
        self.bars = deque(maxlen=500)
        self.day_key = None
        self.session = []
        self.armed = None
        self.entry = None
        self.side = 0
        self.initial_risk = None
        self.stop = None
        self.remaining_qty = 0.0
        self.entry_bar_count = 0
        self.bar_count = 0
        self.exit_pending = False
        self.runner_done = False
        self.entries = 0
        self.partial_orders = 0
        self.block = {"time":0,"weekday":0,"spread":0,"dd":0,"adx":0,"signal":0}
        self.realized = 0.0
        self.manual_trades = []
        self.equity_peak = 1000.0
        self.day_start_equity = 1000.0
        self.current_day = None
        self.max_mtm_dd = 0.0
        self.max_daily_loss = 0.0

    @staticmethod
    def f(x):
        return float(x.as_double()) if hasattr(x, "as_double") else float(x)

    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id)
        self.subscribe_bars(self.config.bar_type)

    def arr(self, k):
        return np.asarray([b[k] for b in self.bars], dtype=float)

    def ema(self, x, n):
        if len(x) < n:
            return None
        a = 2.0 / (n + 1.0)
        v = float(x[-n])
        for z in x[-n + 1:]:
            v = a * float(z) + (1.0 - a) * v
        return v

    def rsi(self, n=7):
        x = self.arr("c")
        if len(x) < n + 1:
            return None
        d = np.diff(x[-n-1:])
        up = np.maximum(d, 0.0).mean()
        dn = np.maximum(-d, 0.0).mean()
        if dn <= 0:
            return 100.0
        rs = up / dn
        return float(100.0 - 100.0 / (1.0 + rs))

    def atr(self, n=14):
        if len(self.bars) < n + 1:
            return None
        xs = list(self.bars)
        tr = []
        for i in range(-n, 0):
            cur, prev = xs[i], xs[i-1]
            tr.append(max(cur["h"]-cur["l"], abs(cur["h"]-prev["c"]), abs(cur["l"]-prev["c"])))
        v = float(np.mean(tr))
        return v if math.isfinite(v) and v > 0 else None

    def adx(self, n=14):
        xs = list(self.bars)
        if len(xs) < 2*n + 2:
            return None
        tr=[]; pdm=[]; mdm=[]
        for i in range(1, len(xs)):
            up=xs[i]["h"]-xs[i-1]["h"]; dn=xs[i-1]["l"]-xs[i]["l"]
            pdm.append(up if up>dn and up>0 else 0.0)
            mdm.append(dn if dn>up and dn>0 else 0.0)
            tr.append(max(xs[i]["h"]-xs[i]["l"], abs(xs[i]["h"]-xs[i-1]["c"]), abs(xs[i]["l"]-xs[i-1]["c"])))
        atr=sum(tr[:n]); ps=sum(pdm[:n]); ms=sum(mdm[:n]); dx=[]
        for i in range(n, len(tr)):
            if i>n:
                atr=atr-atr/n+tr[i]; ps=ps-ps/n+pdm[i]; ms=ms-ms/n+mdm[i]
            pdi=100*ps/atr if atr else 0; mdi=100*ms/atr if atr else 0
            den=pdi+mdi; dx.append(100*abs(pdi-mdi)/den if den else 0)
        if len(dx) < n:
            return None
        a=sum(dx[:n])/n
        for z in dx[n:]:
            a=(a*(n-1)+z)/n
        return float(a)

    def session_vwap_sigma(self):
        if len(self.session) < 8:
            return None, None
        p = np.asarray([x["tp"] for x in self.session], dtype=float)
        w = np.asarray([max(x["v"], 1.0) for x in self.session], dtype=float)
        sw = w.sum()
        if sw <= 0:
            return None, None
        mean = float(np.sum(p*w)/sw)
        var = float(np.sum(w*(p-mean)**2)/sw)
        sd = math.sqrt(max(var, 0.0))
        c = self.session[-1]["c"]
        sigma = (c-mean)/sd if sd > 1e-12 else 0.0
        return mean, float(sigma)

    def on_bar(self, bar: Bar):
        self.bar_count += 1
        ts = pd.Timestamp(int(bar.ts_event), unit="ns", tz="UTC")
        o,h,l,c = map(self.f, [bar.open,bar.high,bar.low,bar.close])
        try:
            v = self.f(bar.volume)
        except Exception:
            v = 1.0
        b={"o":o,"h":h,"l":l,"c":c,"v":max(v,1.0),"ts":int(bar.ts_event)}
        self.bars.append(b)
        day=ts.date()
        if day != self.day_key:
            self.day_key=day
            self.session=[]
        self.session.append({"tp":(h+l+c)/3.0,"c":c,"v":max(v,1.0)})

        if self.entry is not None:
            return
        atr=self.atr(14); adx=self.adx(14)
        if atr is None or adx is None:
            return
        if adx > P["max_adx"]:
            self.block["adx"] += 1
            return
        vwap,sigma=self.session_vwap_sigma()
        if vwap is None:
            return
        closes=self.arr("c"); e9=self.ema(closes,9); e21=self.ema(closes,21); r=self.rsi(7)
        base=0; reason=""
        if e9 is not None and e21 is not None and r is not None:
            if e9>e21 and r>=67:
                base=1; reason="EMA_RSI_BUY"
            elif e9<e21 and r<=33:
                base=-1; reason="EMA_RSI_SELL"
        if base==0 and -P["stop_sigma"] < sigma <= -P["entry_sigma"]:
            base=1; reason="VWAP_RANGE_REVERT_BUY"
        elif base==0 and P["entry_sigma"] <= sigma < P["stop_sigma"]:
            base=-1; reason="VWAP_RANGE_REVERT_SELL"
        if base==0:
            self.block["signal"] += 1
            return
        if P["reverse_signal"]:
            base=-base
        self.armed={"side":base,"atr":atr,"reason":reason,"bar":self.bar_count,"sigma":sigma,"vwap":vwap}

    def _mql_weekday(self, ts):
        return (ts.weekday()+1) % 7

    def _manual_equity(self, bid, ask):
        floating=0.0
        if self.entry is not None:
            px=bid if self.side>0 else ask
            floating=(px-self.entry)*self.side*self.remaining_qty
        return 1000.0 + self.realized + floating

    def _update_dd(self, ts, bid, ask):
        eq=self._manual_equity(bid,ask)
        if self.current_day != ts.date():
            self.current_day=ts.date()
            self.day_start_equity=eq
        self.equity_peak=max(self.equity_peak,eq)
        self.max_mtm_dd=max(self.max_mtm_dd, self.equity_peak-eq)
        if self.day_start_equity>0:
            self.max_daily_loss=max(self.max_daily_loss, (self.day_start_equity-eq)/self.day_start_equity)
        return eq

    def _dd_gate(self, eq):
        daily=(self.day_start_equity-eq)/self.day_start_equity*100 if self.day_start_equity>0 else 0
        cycle=(self.equity_peak-eq)/self.equity_peak*100 if self.equity_peak>0 else 0
        return daily < P["max_daily_dd_pct"] and cycle < P["max_cycle_dd_pct"]

    def _submit_reduce(self, qty):
        instr=self.cache.instrument(self.config.instrument_id)
        q=max(0.0, min(float(qty), self.remaining_qty))
        if q<=0:
            return
        order=self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.SELL if self.side>0 else OrderSide.BUY,
            quantity=instr.make_qty(Decimal(str(q))),
        )
        self.submit_order(order)

    def _close_manual(self, px, reason, qty=None):
        q=self.remaining_qty if qty is None else min(float(qty),self.remaining_qty)
        pnl=(px-self.entry)*self.side*q
        self.realized += pnl
        self.remaining_qty -= q
        self.manual_trades.append({"pnl":float(pnl),"reason":reason,"qty":q})
        return pnl

    def on_quote_tick(self, tick: QuoteTick):
        bid,ask=self.f(tick.bid_price),self.f(tick.ask_price)
        ts=pd.Timestamp(int(tick.ts_event),unit="ns",tz="UTC")
        eq=self._update_dd(ts,bid,ask)
        spread=ask-bid
        flat=self.portfolio.is_net_flat(self.config.instrument_id)

        if self.armed is not None and self.entry is None and flat:
            # Signal expires when a newer M5 bar appears; never carry stale entries.
            if self.armed.get("bar") != self.bar_count:
                self.armed = None
                return
            if ts.hour not in P["allowed_utc_hours"]:
                self.block["time"]+=1
                return
            if self._mql_weekday(ts) not in P["allowed_mql_weekdays"]:
                self.block["weekday"]+=1
                return
            if spread>P["max_spread_price"]:
                self.block["spread"]+=1
                return
            if not self._dd_gate(eq):
                self.block["dd"]+=1
                return
            side=self.armed["side"]; atr=self.armed["atr"]
            instr=self.cache.instrument(self.config.instrument_id)
            order=self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.BUY if side>0 else OrderSide.SELL,
                quantity=instr.make_qty(Decimal(str(P["fixed_qty_oz"]))),
            )
            self.submit_order(order)
            px=ask if side>0 else bid
            risk=P["sl_atr"]*atr
            self.entry=px; self.side=side; self.initial_risk=risk; self.stop=px-side*risk
            self.remaining_qty=float(P["fixed_qty_oz"]); self.entry_bar_count=self.bar_count
            self.runner_done=False; self.exit_pending=False; self.entries+=1; self.armed=None
            return

        if self.entry is None or self.exit_pending:
            return
        px=bid if self.side>0 else ask
        profit=(px-self.entry)*self.side
        rr=profit/self.initial_risk if self.initial_risk else 0.0
        atr=self.atr(14) or (self.initial_risk/P["sl_atr"])

        if P["be_rr"]>0 and rr>=P["be_rr"]:
            be=self.entry + self.side*0.02
            self.stop=max(self.stop,be) if self.side>0 else min(self.stop,be)
        if P["trail_atr"]>0 and rr>=P["trail_start_rr"]:
            tr=px-self.side*atr*P["trail_atr"]
            self.stop=max(self.stop,tr) if self.side>0 else min(self.stop,tr)

        if (not self.runner_done) and rr>=P["runner_trigger_rr"]:
            q=self.remaining_qty*P["runner_partial"]
            if q>0:
                self._submit_reduce(q)
                self._close_manual(px,"RUNNER_PARTIAL",q)
                self.partial_orders+=1
            self.runner_done=True
            return

        held=self.bar_count-self.entry_bar_count
        stop_hit=px<=self.stop if self.side>0 else px>=self.stop
        max_hold=held>=P["max_hold_bars"]
        if stop_hit or max_hold:
            self._close_manual(px,"STOP" if stop_hit else "MAX_HOLD")
            self.close_all_positions(self.config.instrument_id)
            self.exit_pending=True

    def on_position_closed(self, event):
        if self.remaining_qty > 1e-9:
            self.remaining_qty=0.0
        self.entry=None
        self.side=0
        self.initial_risk=None
        self.stop=None
        self.exit_pending=False
        self.runner_done=False

    def on_stop(self):
        if self.entry is not None and self.remaining_qty>0:
            self.close_all_positions(self.config.instrument_id)

def parse_money(v):
    if v is None:
        return 0.0
    try:
        return float(str(v).replace(",","").strip().split()[0])
    except Exception:
        return 0.0

def engine_trades(report):
    if report is None or report.empty:
        return []
    pc=next((c for c in report.columns if "pnl" in str(c).lower()),None)
    tc=next((c for c in report.columns if "closed" in str(c).lower()),None)
    out=[]
    for i,r in report.iterrows():
        pnl=parse_money(r[pc]) if pc is not None else 0.0
        ts=r[tc] if tc is not None else i
        try:
            ts=int(pd.Timestamp(ts).value)
        except Exception:
            ts=i
        out.append({"pnl":pnl,"ts_closed":ts})
    return out

def metrics(trades, initial=1000.0, days=90):
    a=np.asarray([x["pnl"] for x in trades],dtype=float)
    if len(a)==0:
        return {"N":0,"WR_pct":0.0,"PF":0.0,"NetProfit":0.0,"MaxDD_pct":0.0,"RF":None,"Monthly21_pct":0.0}
    wins=a[a>0]; losses=a[a<0]
    pf=float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else (float("inf") if len(wins) else 0.0)
    eq=peak=initial; mdd=0.0
    for x in a:
        eq+=x
        peak=max(peak,eq)
        mdd=max(mdd,peak-eq)
    net=float(a.sum())
    monthly=((max(eq,1e-9)/initial)**(21/max(days,1))-1)*100
    return {"N":int(len(a)),"WR_pct":float((a>0).mean()*100),"PF":pf,"NetProfit":net,
            "MaxDD_pct":float(mdd/peak*100 if peak>0 else 0.0),"RF":float(net/mdd) if mdd>0 else None,
            "Monthly21_pct":float(monthly),"FinalEquity":float(eq)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",required=True)
    ap.add_argument("--experiment-id",required=True)
    ap.add_argument("--raw-bidask-only",action="store_true")
    args=ap.parse_args()
    if not args.raw_bidask_only:
        raise SystemExit("raw-bidask-only mandatory")
    cp=Path(args.catalog)
    manifest=json.loads((cp/"catalog_manifest.json").read_text(encoding="utf-8"))
    days=int(manifest["days"])
    cat=ParquetDataCatalog(str(cp))
    inst=next((x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD"),None)
    if inst is None:
        raise SystemExit("XAUUSD missing")
    ticks=cat.query_quote_ticks(identifiers=[inst.id.value])
    if not ticks:
        raise SystemExit("no XAUUSD QuoteTicks")
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,
                  starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"))
    eng.add_instrument(inst)
    eng.add_data(ticks)
    bt=BarType.from_str(f"{inst.id.value}-5-MINUTE-BID-INTERNAL")
    st=Fusion(Config(instrument_id=inst.id,bar_type=bt))
    eng.add_strategy(st)
    eng.run()
    et=engine_trades(eng.trader.generate_positions_report())
    met=metrics(et,days=days)
    out=Path("results/amos_factory_fusion_v1")/args.experiment_id
    out.mkdir(parents=True,exist_ok=True)
    summary={
        "verification_level":"NAUTILUS_BT_RAW_BIDASK_FUSION_CLEANROOM",
        "engine":"NautilusTrader BacktestEngine",
        "nautilus_version":getattr(nautilus_trader,"__version__","unknown"),
        "strategy":"AMOS Factory Fusion v1.00",
        "symbol":"XAUUSD","signal_tf":"M5","initial_equity":1000.0,"leverage":2000,
        "data_kind":"RAW_BIDASK QuoteTick","ohlc_resample_used":False,
        "signal_bars":"Nautilus INTERNAL 5-MINUTE BID bars","execution":"market orders on raw QuoteTicks; native spread observed",
        "period":{"start":manifest["start"],"days":days,"end_exclusive":manifest["end_exclusive"]},
        "raw_tick_count":len(ticks),"entries_submitted":st.entries,"partial_runner_orders":st.partial_orders,
        "engine_metrics":met,
        "manual_mtm":{"max_floating_dd_pct":float(st.max_mtm_dd/max(st.equity_peak,1e-9)*100),"max_daily_loss_pct":float(st.max_daily_loss*100),"realized_proxy":st.realized},
        "blocks":st.block,
        "parameters":{k:(sorted(v) if isinstance(v,set) else v) for k,v in P.items()},
        "parity_status":"Phase3 parameter mapping onto Factory VWAP/EMA signal path; exact Phase3 optimizer entry source was not present in supplied ZIP.",
        "limitations":[
            "BigPlayer/StopHunt and full H1 MarketState indicator internals are simplified in the clean-room Nautilus mapping.",
            "Commission, swap, cashback and broker-specific slippage/latency are not charged in this first fusion gate; observed raw bid/ask spread is native.",
            "MT5 source must still be compiled in MetaEditor before live use; this run validates the mapped strategy path in Nautilus, not byte-for-byte MT5 parity."
        ]
    }
    pd.DataFrame(et).to_csv(out/"engine_trades.csv",index=False)
    pd.DataFrame(st.manual_trades).to_csv(out/"manual_trade_legs.csv",index=False)
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"catalog_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    eng.dispose()

if __name__ == "__main__":
    main()
