from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

SIM = Venue("SIM")
D = Decimal
MINUTE_NS = 60_000_000_000
MODES = ("BASE", "A0", "A1")
EPS = 1e-12
BIG = 1e6


def vgrsi_value(close: np.ndarray, t: int, ws: int = 35, wv: int = 35, mode: str = "A0") -> float | None:
    if ws < 2 or wv < 2 or t < ws or t >= len(close):
        return None
    Spos = Sneg = 0.0
    Npos = Nneg = 0
    j_first = t - ws + 1
    if j_first < 1:
        return None
    for j in range(j_first, t + 1):
        pj = float(close[j])
        i_min = max(0, j - wv)
        taken = 0
        run_min = BIG
        for i in range(j - 1, i_min - 1, -1):
            if taken >= ws:
                break
            den = float(i - j)
            if abs(den) < EPS:
                continue
            slope_i = (float(close[i]) - pj) / den
            visible = True if i == j - 1 else (slope_i < run_min - EPS)
            if slope_i < run_min:
                run_min = slope_i
            if not visible or i < 1:
                continue
            dp = float(close[i]) - float(close[i - 1])
            taken += 1
            if dp > EPS:
                Spos += dp
                Npos += 1
            elif dp < -EPS:
                Sneg += -dp
                Nneg += 1
    rS = Spos / Sneg if Sneg > EPS else (BIG if Spos > EPS else 1.0)
    rN = Npos / Nneg if Nneg > 0 else (BIG if Npos > 0 else 1.0)
    rA = 0.5 * (rS + rN) if mode == "A0" else (rS / rN if rN > EPS else BIG)
    rA = max(rA, 0.0)
    return 100.0 - 100.0 / (1.0 + rA)


@dataclass
class PendingOpen:
    side: int
    tag: str
    raw_level: float
    trigger_ts: int
    trigger_bid: float
    trigger_ask: float


class G75VGRSIConfig(StrategyConfig, frozen=True):
    instrument_id: object
    mode: str = "BASE"
    base_lot: Decimal = D("0.01")
    contract_size: Decimal = D("100")
    trigger_step: Decimal = D("0.12")
    add_step: Decimal = D("0.025")
    reversal_step: Decimal = D("0.20")
    max_layers: int = 10
    vgrsi_ws: int = 35
    vgrsi_wv: int = 35
    vgrsi_center: float = 50.0


class G75VGRSIStrategy(Strategy):
    def __init__(self, config: G75VGRSIConfig):
        super().__init__(config)
        self.market_bar_type = BarType.from_str(f"{config.instrument_id.value}-1-MINUTE-BID-INTERNAL")
        self.anchor = 0.0
        self.pending_side = 0
        self.active_side = 0
        self.peak = 0.0
        self.trough = 0.0
        self.last_bid = self.last_ask = self.last_mid = 0.0
        self.last_ts = 0
        self.current_minute = None
        self.current_close = 0.0
        self.previous_close = None
        self.minute_closes: list[float] = []
        self.entry_order_id = None
        self.pending_opens: dict[str, PendingOpen] = {}
        self.pending_adds = 0
        self.fill_prices: list[float] = []
        self.closing = False
        self.triggers = self.entries = self.adds = self.exits = 0
        self.order_denied = self.order_rejected = self.order_canceled = 0
        self.vgrsi_checked = self.vgrsi_aligned = self.vgrsi_rejected = self.vgrsi_warmup = 0
        self.realized = 0.0
        self.closed_peak = self.equity_peak = 1000.0
        self.max_closed_dd = self.max_floating_dd = 0.0
        self.cycle_pnls: list[float] = []
        self.fill_sequence: list[str] = []
        self.max_layers_seen = 0

    def on_start(self):
        # QuoteTicks remain the execution/PnL source. The INTERNAL BID bar subscription
        # mirrors the known-working Nautilus raw execution path and initializes the
        # simulated market without using OHLC-resampled execution.
        self.subscribe_quote_ticks(self.config.instrument_id)
        self.subscribe_bars(self.market_bar_type)

    def on_bar(self, bar):
        pass

    def _clear_failed_open(self, event):
        oid = str(event.client_order_id)
        p = self.pending_opens.pop(oid, None)
        if p is None:
            return
        if p.tag == "ENTRY": self.entry_order_id = None
        else: self.pending_adds = max(0, self.pending_adds - 1)

    def on_order_denied(self, event): self.order_denied += 1; self._clear_failed_open(event)
    def on_order_rejected(self, event): self.order_rejected += 1; self._clear_failed_open(event)
    def on_order_canceled(self, event): self.order_canceled += 1; self._clear_failed_open(event)

    def _submit_market(self, side: int, raw_level: float, tag: str, trigger_ts=None, trigger_bid=None, trigger_ask=None):
        inst = self.cache.instrument(self.config.instrument_id)
        qty = inst.make_qty(self.config.base_lot * self.config.contract_size)
        order = self.order_factory.market(self.config.instrument_id, OrderSide.BUY if side > 0 else OrderSide.SELL, qty)
        oid = str(order.client_order_id)
        self.pending_opens[oid] = PendingOpen(side, tag, raw_level, self.last_ts if trigger_ts is None else int(trigger_ts), self.last_bid if trigger_bid is None else float(trigger_bid), self.last_ask if trigger_ask is None else float(trigger_ask))
        if tag == "ENTRY": self.entry_order_id = oid
        else: self.pending_adds += 1
        self.submit_order(order)

    def _try_same_tick_add(self, side, bid, ask, last_cursor, trigger_ts):
        if self.closing or not self.active_side or self.pending_adds or len(self.fill_prices) >= self.config.max_layers: return False
        nxt = last_cursor + side * float(self.config.add_step)
        if not (ask >= nxt if side > 0 else bid <= nxt): return False
        self._submit_market(side, nxt, "ADD", trigger_ts, bid, ask); return True

    def _basket_money(self, side, bid, ask):
        if not self.fill_prices: return 0.0
        liq = bid if side > 0 else ask
        units = float(self.config.base_lot * self.config.contract_size)
        return sum(side * (liq - px) * units for px in self.fill_prices)

    def _update_equity_dd(self):
        mtm = self._basket_money(self.active_side, self.last_bid, self.last_ask) if self.active_side else 0.0
        eq = 1000.0 + self.realized + mtm
        self.equity_peak = max(self.equity_peak, eq)
        self.max_floating_dd = max(self.max_floating_dd, self.equity_peak - eq)

    def on_order_filled(self, event):
        p = self.pending_opens.pop(str(event.client_order_id), None)
        if p is None: return
        px = float(event.last_px); self.fill_sequence.append(f"{int(event.ts_event)}:{p.tag}:{p.side}:{px:.5f}")
        if p.tag == "ENTRY":
            self.entry_order_id = None; self.active_side = p.side; self.fill_prices = [px]
            ref = self.last_bid if p.side > 0 else self.last_ask
            self.peak = self.trough = self.anchor = ref; self.entries += 1
        else:
            self.pending_adds = max(0, self.pending_adds - 1); self.fill_prices.append(px); self.adds += 1
        self.max_layers_seen = max(self.max_layers_seen, len(self.fill_prices))
        if p.tag == "ADD" and self.active_side and not self.closing: self._try_same_tick_add(p.side, p.trigger_bid, p.trigger_ask, p.raw_level, p.trigger_ts)

    def on_position_closed(self, event):
        if not self.closing: return
        self.active_side = 0; self.fill_prices = []; self.peak = self.trough = 0.0; self.closing = False; self.exits += 1

    def _manage_active(self):
        if not self.active_side or self.closing or self.pending_adds: return
        side = self.active_side
        if side > 0:
            self.peak = max(self.peak, self.last_bid); stop = self.peak - float(self.config.reversal_step); hit = self.last_bid <= stop
        else:
            self.trough = self.last_ask if self.trough <= 0 else min(self.trough, self.last_ask); stop = self.trough + float(self.config.reversal_step); hit = self.last_ask >= stop
        if hit:
            pnl = self._basket_money(side, self.last_bid, self.last_ask); self.cycle_pnls.append(pnl); self.realized += pnl
            closed_eq = 1000.0 + self.realized; self.closed_peak = max(self.closed_peak, closed_eq); self.max_closed_dd = max(self.max_closed_dd, self.closed_peak - closed_eq)
            self.anchor = stop; self.closing = True; self.close_all_positions(self.config.instrument_id); return
        if self.fill_prices: self._try_same_tick_add(side, self.last_bid, self.last_ask, self.fill_prices[-1], self.last_ts)

    def _execute_pending(self):
        if self.pending_side == 0 or self.active_side or self.entry_order_id is not None or self.closing: return
        side = self.pending_side; self.pending_side = 0; self._submit_market(side, 0.0, "ENTRY")

    def _direction_allows(self, side):
        if self.config.mode == "BASE": return True
        if len(self.minute_closes) <= self.config.vgrsi_ws: self.vgrsi_warmup += 1; return False
        arr = np.asarray(self.minute_closes, dtype=float); val = vgrsi_value(arr, len(arr)-1, self.config.vgrsi_ws, self.config.vgrsi_wv, self.config.mode); self.vgrsi_checked += 1
        if val is None or not np.isfinite(val): self.vgrsi_warmup += 1; return False
        vside = 1 if val > self.config.vgrsi_center else (-1 if val < self.config.vgrsi_center else 0); ok = vside == side
        if ok: self.vgrsi_aligned += 1
        else: self.vgrsi_rejected += 1
        return ok

    def _evaluate_initial(self, new_bar):
        if not new_bar or self.active_side or self.pending_side or self.entry_order_id is not None or self.closing: return
        c1 = self.previous_close
        if c1 is None: return
        if self.anchor <= 0: self.anchor = c1; return
        trig = float(self.config.trigger_step); side = 1 if c1 >= self.anchor + trig else (-1 if c1 <= self.anchor - trig else 0)
        if side == 0: self.anchor = c1; return
        self.triggers += 1
        if self._direction_allows(side): self.pending_side = side

    def on_quote_tick(self, tick):
        self.last_ts = int(tick.ts_event); self.last_bid = float(tick.bid_price); self.last_ask = float(tick.ask_price); self.last_mid = (self.last_bid+self.last_ask)/2.0
        minute = (self.last_ts // MINUTE_NS) * MINUTE_NS; new_bar = False
        if self.current_minute is None: self.current_minute = minute; self.current_close = self.last_mid
        elif minute != self.current_minute:
            self.previous_close = self.current_close; self.minute_closes.append(self.previous_close); self.current_minute = minute; self.current_close = self.last_mid; new_bar = True
        else: self.current_close = self.last_mid
        self._manage_active(); self._execute_pending(); self._evaluate_initial(new_bar); self._update_equity_dd()

    def on_stop(self):
        if self.active_side and not self.closing:
            pnl = self._basket_money(self.active_side, self.last_bid, self.last_ask); self.cycle_pnls.append(pnl); self.realized += pnl
        self.close_all_positions(self.config.instrument_id); self.unsubscribe_quote_ticks(self.config.instrument_id); self.unsubscribe_bars(self.market_bar_type)


def calc_metrics(strat, trading_days):
    a = np.asarray(strat.cycle_pnls, dtype=float); wins=a[a>0]; losses=a[a<0]
    gw=float(wins.sum()) if len(wins) else 0.0; gl=float(-losses.sum()) if len(losses) else 0.0; pf=gw/gl if gl>0 else (float("inf") if gw>0 else 0.0)
    net=float(a.sum()) if len(a) else 0.0; final=1000.0+net
    monthly=None if final<=0 or trading_days<=0 else ((final/1000.0)**(21.0/trading_days)-1)*100.0; daily=None if final<=0 or trading_days<=0 else ((final/1000.0)**(1.0/trading_days)-1)*100.0
    return {"N":int(len(a)),"N_per_day":float(len(a)/trading_days) if trading_days else None,"WR_pct":float((a>0).mean()*100) if len(a) else 0.0,"PF":pf,"NetProfit":net,"FinalBalance":final,"MaxClosedDD_pct":strat.max_closed_dd/strat.closed_peak*100 if strat.closed_peak>0 else 0.0,"MaxFloatingDD_pct":strat.max_floating_dd/strat.equity_peak*100 if strat.equity_peak>0 else 0.0,"RF_closed":net/strat.max_closed_dd if strat.max_closed_dd>0 else None,"RF_floating":net/strat.max_floating_dd if strat.max_floating_dd>0 else None,"DailyCompound_pct":daily,"Monthly21_pct":monthly,"Triggers":strat.triggers,"EntriesSubmitted":strat.entries,"Adds":strat.adds,"Exits":strat.exits,"OrderDenied":strat.order_denied,"OrderRejected":strat.order_rejected,"OrderCanceled":strat.order_canceled,"MaxLayersSeen":strat.max_layers_seen,"VGRSI_checked":strat.vgrsi_checked,"VGRSI_aligned":strat.vgrsi_aligned,"VGRSI_rejected":strat.vgrsi_rejected,"VGRSI_warmup_rejects":strat.vgrsi_warmup,"fill_sequence_hash":hashlib.sha256("\n".join(strat.fill_sequence).encode()).hexdigest()}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--catalog",required=True); ap.add_argument("--experiment-id",required=True); ap.add_argument("--symbol",default="XAUUSD"); ap.add_argument("--modes",nargs="+",default=list(MODES),choices=MODES); ap.add_argument("--max-layers",type=int,default=10); ap.add_argument("--base-lot",type=Decimal,default=D("0.01")); ap.add_argument("--raw-bidask-only",action="store_true"); args=ap.parse_args()
    if not args.raw_bidask_only: raise SystemExit("raw-bidask-only is mandatory")
    catalog_path=Path(args.catalog).resolve(); manifest=json.loads((catalog_path/"catalog_manifest.json").read_text()); catalog=ParquetDataCatalog(str(catalog_path)); inst_by_plain={x.id.symbol.value.replace("/",""):x for x in catalog.instruments()}; instrument=inst_by_plain.get(args.symbol)
    if instrument is None: raise SystemExit(f"instrument missing: {args.symbol}")
    ticks=catalog.quote_ticks(instrument_ids=[instrument.id.value]);
    if not ticks: raise SystemExit("no raw QuoteTicks")
    trading_days=len({datetime.fromtimestamp(int(t.ts_event)/1e9,tz=timezone.utc).date() for t in ticks}); outdir=Path("results/ae-bt")/args.experiment_id; outdir.mkdir(parents=True,exist_ok=True); results={}
    for mode in args.modes:
        engine=BacktestEngine(config=BacktestEngineConfig(trader_id=TraderId(f"G75VG-{mode[:3]}"),logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)))
        engine.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"))
        engine.add_instrument(instrument); engine.add_data(ticks)
        strat=G75VGRSIStrategy(G75VGRSIConfig(instrument_id=instrument.id,mode=mode,max_layers=args.max_layers,base_lot=args.base_lot)); engine.add_strategy(strat); engine.run(); results[mode]=calc_metrics(strat,trading_days); engine.dispose()
    summary={"verification_level":"NAUTILUS_BT_RAW_BIDASK","engine":"NautilusTrader BacktestEngine","nautilus_version":getattr(nautilus_trader,"__version__","unknown"),"symbol":args.symbol,"initial_balance":1000.0,"data_kind":"RAW_BIDASK QuoteTick","ohlc_resample_used":False,"native_spread":True,"signal_market_init":"Nautilus INTERNAL 1-MINUTE BID bars built directly from raw QuoteTicks; not used for execution/PnL","execution_model":"NETTING simulated venue, MARKET orders on raw QuoteTicks","commission_model":"NOT_INCLUDED","slippage_model":"NOT_INCLUDED","latency_model":"NOT_INCLUDED","production_gate":"INVALID_UNTIL_NATIVE_FEE_SLIPPAGE_LATENCY_MODELS_ADDED","g75":{"trigger":0.12,"add":0.025,"reversal":0.20,"max_layers":args.max_layers,"base_lot":float(args.base_lot)},"vgrsi":{"ws":35,"wv":35,"center":50.0,"usage":"direction-only hard alignment; never creates entries"},"period":{"start":manifest.get("start"),"end_exclusive":manifest.get("end_exclusive"),"calendar_days":manifest.get("days"),"trading_days":trading_days},"modes":results}
    (outdir/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)); (outdir/"catalog_manifest.json").write_text(json.dumps(manifest,indent=2)); print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
