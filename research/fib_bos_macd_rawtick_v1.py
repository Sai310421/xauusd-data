#!/usr/bin/env python3
"""Causal XAUUSD Fib 0.768-0.804 retracement event study on Raw Bid/Ask."""
from __future__ import annotations
import argparse, json
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import numpy as np
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, BookType
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

NS = 1_000_000_000
def f(x): return float(x.as_double()) if hasattr(x, "as_double") else float(x)
def iso(ts): return datetime.fromtimestamp(ts // NS, timezone.utc).isoformat()

class Config(StrategyConfig, frozen=True):
    instrument_id: object
    direction_mode: str = "normal"

class FibBosMacd(Strategy):
    """All swing pivots are confirmed only after 3 subsequent completed bars."""
    def __init__(self, config):
        super().__init__(config)
        self.buckets, self.work = {}, {}
        self.bars = {1: [], 5: [], 15: []}
        self.pivots = {5: [], 15: []}
        self.arm = self.pending = self.active = None
        self.last_bid = self.last_ask = None
        self.ticks = self.signals = 0
        self.trades, self.signal_log = [], []
        self.macd_hist = []
        self.m1_closes = []
        self.ema12 = self.ema26 = self.signal9 = None
        self.macd_values = []
        self.last_m5_close = None

    def on_start(self): self.subscribe_quote_ticks(self.config.instrument_id)

    def _pivot(self, tf, ix, span=3):
        bs = self.bars[tf]
        if ix < span or ix + span >= len(bs): return
        b = bs[ix]
        left, right = bs[ix-span:ix], bs[ix+1:ix+span+1]
        if all(b["h"] > x["h"] for x in left + right):
            self.pivots[tf].append({"kind":"H", "price":b["h"], "ts":b["ts"]})
        elif all(b["l"] < x["l"] for x in left + right):
            self.pivots[tf].append({"kind":"L", "price":b["l"], "ts":b["ts"]})

    def _trend(self):
        p = self.pivots[15]
        hi = [x for x in p if x["kind"] == "H"]
        lo = [x for x in p if x["kind"] == "L"]
        if len(hi) < 2 or len(lo) < 2: return 0
        if hi[-1]["price"] > hi[-2]["price"] and lo[-1]["price"] > lo[-2]["price"]: return 1
        if hi[-1]["price"] < hi[-2]["price"] and lo[-1]["price"] < lo[-2]["price"]: return -1
        return 0

    def _atr5(self):
        bs = self.bars[5]
        if len(bs) < 15: return None
        vals = []
        for i in range(-14, 0):
            b, prev = bs[i], bs[i-1]
            vals.append(max(b["h"]-b["l"], abs(b["h"]-prev["c"]), abs(b["l"]-prev["c"])))
        return float(np.mean(vals))

    def _arm_bos(self, bar):
        trend, p, atr = self._trend(), self.pivots[5], self._atr5()
        if not trend or not atr or len(p) < 2 or self.last_m5_close is None: return
        highs, lows = [x for x in p if x["kind"]=="H"], [x for x in p if x["kind"]=="L"]
        if trend > 0 and highs and lows:
            bh, origin = highs[-1], lows[-1]
            if origin["ts"] < bh["ts"] and self.last_m5_close <= bh["price"] and bar["c"] > bh["price"] + .10*atr:
                high, low = bar["h"], origin["price"]
                if high > low:
                    self.arm = {"side":1, "high":high, "low":low, "born":bar["ts"], "expires":bar["ts"]+120*60*NS}
        elif trend < 0 and highs and lows:
            bl, origin = lows[-1], highs[-1]
            if origin["ts"] < bl["ts"] and self.last_m5_close >= bl["price"] and bar["c"] < bl["price"] - .10*atr:
                high, low = origin["price"], bar["l"]
                if high > low:
                    self.arm = {"side":-1, "high":high, "low":low, "born":bar["ts"], "expires":bar["ts"]+120*60*NS}

    def _update_macd(self, close):
        """O(1) MACD update; values exist only after each causal M1 close."""
        self.m1_closes.append(close)
        n = len(self.m1_closes)
        if n == 12: self.ema12 = float(np.mean(self.m1_closes[:12]))
        elif n > 12: self.ema12 = (2/13)*close + (11/13)*self.ema12
        if n == 26: self.ema26 = float(np.mean(self.m1_closes[:26]))
        elif n > 26: self.ema26 = (2/27)*close + (25/27)*self.ema26
        if self.ema12 is None or self.ema26 is None: return
        macd = self.ema12 - self.ema26
        self.macd_values.append(macd)
        m = len(self.macd_values)
        if m == 9: self.signal9 = float(np.mean(self.macd_values[:9]))
        elif m > 9: self.signal9 = .2*macd + .8*self.signal9
        if self.signal9 is not None: self.macd_hist.append(macd-self.signal9)


    def _on_m1_close(self, bar):
        self._update_macd(bar["c"])
        if not self.arm or self.pending or self.active or bar["ts"] > self.arm["expires"]: return
        a, side, r = self.arm, self.arm["side"], self.arm["high"]-self.arm["low"]
        zlo, zhi = (a["high"]-.804*r, a["high"]-.768*r) if side > 0 else (a["low"]+.768*r, a["low"]+.804*r)
        touch = bar["l"] <= zhi and bar["h"] >= zlo
        h = self.macd_hist[-3:] if len(self.macd_hist) >= 3 else None
        reaccelerate = h is not None and ((h[-1] > h[-2] and h[-2] <= h[-3]) if side > 0 else (h[-1] < h[-2] and h[-2] >= h[-3]))
        if touch and reaccelerate:
            self.pending = {"side":side, "arm":a, "signal_ts":bar["ts"], "zone":[zlo,zhi]}
            self.signal_log.append({"ts":iso(bar["ts"]), "side":"BUY" if side>0 else "SELL", "zone":[zlo,zhi]})

    def _on_m5_close(self, bar):
        self._pivot(5, len(self.bars[5])-4)
        self._arm_bos(bar)
        self.last_m5_close = bar["c"]

    def _on_m15_close(self):
        self._pivot(15, len(self.bars[15])-4)

    def _enter(self, bid, ask, ts):
        p, a = self.pending, self.pending["arm"]
        signal_side, width = p["side"], a["high"]-a["low"]
        side = signal_side if self.config.direction_mode == "normal" else -signal_side
        entry = ask if side > 0 else bid
        if self.config.direction_mode == "normal":
            stop = a["high"]-.933*width if side > 0 else a["low"]+.933*width
        else:
            # Contrarian stop is the BOS impulse endpoint (Fib 0.0), i.e. reversal invalidation.
            stop = a["high"] if side < 0 else a["low"]
        risk = abs(entry-stop)
        self.pending = None; self.arm = None
        if risk <= 0: return
        self.active = {"side":side, "entry":entry, "stop":stop, "risk":risk, "entry_ts":ts,
                       "end_ts":ts+120*60*NS, "mae":0.0, "mfe":0.0, "signal_ts":p["signal_ts"]}
        self.signals += 1

    def _finish(self, mark, why, ts):
        x = self.active; move = (mark-x["entry"])*x["side"]
        self.trades.append({"signal_ts":iso(x["signal_ts"]), "entry_ts":iso(x["entry_ts"]), "exit_ts":iso(ts),
          "side":"BUY" if x["side"]>0 else "SELL", "signal_side":"BUY" if x["side"]>0 else "SELL", "why":why, "entry":x["entry"], "stop":x["stop"],
          "pnl_usd":move*2.0, "r":move/x["risk"], "mae_R":-x["mae"]/x["risk"], "mfe_R":x["mfe"]/x["risk"]})
        self.active = None

    def _evaluate(self, bid, ask, ts):
        if self.pending: self._enter(bid, ask, ts)
        if self.active:
            x=self.active; mark=bid if x["side"]>0 else ask; move=(mark-x["entry"])*x["side"]
            x["mae"],x["mfe"]=min(x["mae"],move),max(x["mfe"],move)
            stop = mark <= x["stop"] if x["side"]>0 else mark >= x["stop"]
            if stop or ts >= x["end_ts"]: self._finish(mark, "SL" if stop else "120m_horizon", ts)

    def on_quote_tick(self, tick):
        self.ticks += 1; bid,ask,ts=f(tick.bid_price),f(tick.ask_price),tick.ts_event
        self.last_bid,self.last_ask=bid,ask
        for tf in (1,5,15):
            bucket=ts//(tf*60*NS); old=self.buckets.get(tf); w=self.work.get(tf)
            if old is None:
                self.buckets[tf]=bucket; self.work[tf]={"o":bid,"h":bid,"l":bid,"c":bid,"ts":ts}
            elif bucket != old:
                self.bars[tf].append(w)
                if tf==1: self._on_m1_close(w)
                elif tf==5: self._on_m5_close(w)
                else: self._on_m15_close()
                self.buckets[tf]=bucket; self.work[tf]={"o":bid,"h":bid,"l":bid,"c":bid,"ts":ts}
            else:
                w["h"],w["l"],w["c"]=max(w["h"],bid),min(w["l"],bid),bid
        self._evaluate(bid,ask,ts)

    def on_stop(self):
        if self.active: self._finish(self.last_bid if self.active["side"]>0 else self.last_ask,"dataset_end",0)

    def result(self):
        pnl=np.asarray([x["pnl_usd"] for x in self.trades],dtype=float)
        mae=np.asarray([x["mae_R"] for x in self.trades],dtype=float); mfe=np.asarray([x["mfe_R"] for x in self.trades],dtype=float)
        wins,losses=pnl[pnl>0],pnl[pnl<0]; eq=peak=1000.; dd=0.
        for z in pnl: eq+=z; peak=max(peak,eq); dd=max(dd,peak-eq)
        return {"signals":self.signals,"N":len(pnl),"WR_pct":float((pnl>0).mean()*100) if len(pnl) else 0.,
          "PF":float(wins.sum()/abs(losses.sum())) if len(losses) else None,"EV_usd":float(pnl.mean()) if len(pnl) else 0.,
          "Net_usd":float(pnl.sum()),"MaxDD_pct":float(dd/peak*100),
          "MAE_MFE_R":{"avg_MAE_R":float(mae.mean()) if len(mae) else 0.,"median_MAE_R":float(np.median(mae)) if len(mae) else 0.,
            "avg_MFE_R":float(mfe.mean()) if len(mfe) else 0.,"median_MFE_R":float(np.median(mfe)) if len(mfe) else 0.,
            "MFE_ge_1R_pct":float((mfe>=1).mean()*100) if len(mfe) else 0.,"MFE_ge_2R_pct":float((mfe>=2).mean()*100) if len(mfe) else 0.},
          "trades":self.trades,"signals_detail":self.signal_log}

def l1(ticks):
    q=Quantity.from_int(1); out=[]
    for t in ticks:
        out.append(t if f(t.bid_size)>0 and f(t.ask_size)>0 else QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=t.bid_size if f(t.bid_size)>0 else q,ask_size=t.ask_size if f(t.ask_size)>0 else q,ts_event=t.ts_event,ts_init=t.ts_init))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--catalog",action="append",required=True); ap.add_argument("--experiment-id",required=True); ap.add_argument("--direction-mode", choices=["normal","inverse"], default="normal"); a=ap.parse_args()
    raw=[]; inst=None
    for root in a.catalog:
        cat=ParquetDataCatalog(root); i=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD"); inst=inst or i
        raw+=cat.query(data_cls=QuoteTick,identifiers=[i.id.value])
    raw.sort(key=lambda x:x.ts_event)
    engine=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)))
    engine.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"))
    engine.add_instrument(inst); engine.add_data(l1(raw)); s=FibBosMacd(Config(instrument_id=inst.id, direction_mode=a.direction_mode)); engine.add_strategy(s); engine.run(); engine.end()
    out={"status":"COMPLETE","engine":"NautilusTrader BacktestEngine","raw_ticks":len(raw),"ohlc_resample_used":False,
      "execution":"Raw QuoteTick BUY Ask / SELL Bid","direction_mode":a.direction_mode,"hypothesis":"M15 confirmed HH/HL or LL/LH; M5 confirmed-close BOS; fixed BOS-leg Fib .768-.804 retracement; M1 MACD(12,26,9) histogram reacceleration.",
      "anti_lookahead":"Pivots require three subsequent completed bars. BOS uses completed M5 close. MACD trigger uses completed M1 close; entry occurs on the following QuoteTick.",
      "parameters":{"fib_entry_zone":"0.768-0.804","stop":"0.933","pivot_confirmation":"3 completed bars each side","BOS":"M5 close beyond pivot by 0.10xATR14","measurement_exit":"stop or 120 minutes after entry; no discretionary TP","inverse_definition":"Opposite trade at the same causal signal; stop at BOS impulse endpoint (Fib 0.0)" if a.direction_mode=="inverse" else None,"macd":"12,26,9 histogram reacceleration"},
      "results":{"fib_bos_macd_v1":s.result()}}
    path=Path("results/fib-bos-macd")/a.experiment_id/"summary.json"; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
