from __future__ import annotations
import argparse, json, math
from collections import deque
from decimal import Decimal
from pathlib import Path
import numpy as np, pandas as pd, nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, BookType
from nautilus_trader.model.objects import Quantity\nfrom nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

P = dict(
    jump_window_sec=2.0, jump_z=3.5, sigma_ticks=500, min_jump_abs=0.60,
    exhaustion_min_sec=0.40, exhaustion_max_sec=8.0, recovery_ratio=0.18,
    momentum_ticks=8, max_spread=1.20, stop_mult=0.65, tp_mult=1.10,
    max_hold_sec=120, cooldown_sec=8, continuation_momentum_ticks=12,
    a25_hard_stop_mult=1.80, a25_wait_cap_sec=6.0, a22_recovery_cap_sec=25.0,
    a22_recovery_ratio=0.12, g75_trigger=0.12, g75_add=0.025, g75_max_layers=10,
    base_units=1, commission_per_unit_roundturn=0.0,
)

ARMS = ("BASE", "A25", "A25_A22", "A25_A22_G75")

def f(x):
    return float(x.as_double()) if hasattr(x, "as_double") else float(x)

def money(x):
    try: return float(str(x).replace(",", "").split()[0])
    except Exception: return 0.0

def ensure_executable_l1(raw_ticks, depth_units=1000):
    depth = Quantity.from_int(depth_units); out=[]; replaced=0
    for t in raw_ticks:
        bs, az = f(t.bid_size), f(t.ask_size)
        if bs <= 0 or az <= 0:
            out.append(QuoteTick(
                instrument_id=t.instrument_id, bid_price=t.bid_price, ask_price=t.ask_price,
                bid_size=depth if bs <= 0 else t.bid_size,
                ask_size=depth if az <= 0 else t.ask_size,
                ts_event=t.ts_event, ts_init=t.ts_init))
            replaced += 1
        else:
            out.append(t)
    return out, replaced

class Config(StrategyConfig, frozen=True):
    instrument_id: object
    arm: str

class A25JumpRecovery(Strategy):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.arm=cfg.arm; self.hist=deque(maxlen=4000); self.rets=deque(maxlen=P["sigma_ticks"])
        self.jump=None; self.entry=None; self.side=0; self.risk=0.; self.tp=0.; self.entry_ts=0
        self.units=0; self.layers=0; self.next_add=None; self.cool_until=0; self.exit_pending=False
        self.stop_hit_ts=None; self.realized=0.; self.peak_equity=1000.; self.max_floating_dd=0.
        self.jump_count=0; self.exhaustion_count=0; self.entries=0; self.stop_events=0
        self.a25_waits=0; self.a25_saved=0; self.a25_worsened=0; self.a22_timeouts=0; self.g75_adds=0
        self.wait_reference_pnl=None; self.events=[]

    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id)

    def _flat(self):
        return (not self.portfolio.is_net_long(self.config.instrument_id)
                and not self.portfolio.is_net_short(self.config.instrument_id))

    def _submit(self, side, units):
        instr=self.cache.instrument(self.config.instrument_id)
        os=OrderSide.BUY if side>0 else OrderSide.SELL
        q=instr.make_qty(Decimal(int(units)))
        self.submit_order(self.order_factory.market(
            instrument_id=self.config.instrument_id, order_side=os, quantity=q))

    def _price_ago(self, ts, ns):
        target=ts-ns
        for t,p in reversed(self.hist):
            if t <= target: return p
        return None

    def _momentum(self, n):
        if len(self.hist) < n+1: return 0.
        a=list(self.hist)[-n-1:]
        return a[-1][1]-a[0][1]

    def _continuation_value_proxy(self, mid):
        # Operational A25 proxy, not the closed-form Levy value function.
        # Positive means recovery/continuation evidence dominates immediate exit.
        mom=self.side*self._momentum(P["continuation_momentum_ticks"])
        if self.jump is None: return -1.
        age=(self.hist[-1][0]-self.jump["ts"])/1e9
        shock_decay=max(0., 1.-age/max(P["exhaustion_max_sec"],1e-9))
        dist_from_extreme=self.side*(mid-self.jump["extreme"])
        return 0.55*mom/max(self.risk,1e-9)+0.30*dist_from_extreme/max(self.risk,1e-9)-0.15*shock_decay

    def _mark_equity(self, bid, ask):
        if self.entry is None or self.units<=0:
            eq=1000.+self.realized
        else:
            px=bid if self.side>0 else ask
            unrl=self.side*(px-self.entry)*self.units
            eq=1000.+self.realized+unrl
        self.peak_equity=max(self.peak_equity,eq)
        dd=(self.peak_equity-eq)/max(self.peak_equity,1e-9)*100.
        self.max_floating_dd=max(self.max_floating_dd,dd)

    def on_quote_tick(self, t):
        bid,ask=f(t.bid_price),f(t.ask_price); mid=(bid+ask)/2.; ts=int(t.ts_event)
        if self.hist:
            prev=self.hist[-1][1]
            if prev>0: self.rets.append(mid-prev)
        self.hist.append((ts,mid)); self._mark_equity(bid,ask)

        if self.entry is not None and not self.exit_pending:
            px=bid if self.side>0 else ask
            pnl_per=self.side*(px-self.entry)
            # G75 is pursuit only: add strictly while profitable.
            if self.arm=="A25_A22_G75" and pnl_per>0 and self.layers<P["g75_max_layers"]:
                if self.next_add is None: self.next_add=P["g75_trigger"]
                if pnl_per >= self.next_add:
                    old_notional=self.entry*self.units
                    self._submit(self.side,P["base_units"]); self.units+=P["base_units"]
                    self.entry=(old_notional+px*P["base_units"])/self.units
                    self.layers+=1; self.g75_adds+=1; self.next_add+=P["g75_add"]
            hit_tp=(px>=self.tp if self.side>0 else px<=self.tp)
            adverse=self.side*(px-self.entry)
            hit_stop=adverse <= -self.risk
            hard_stop=adverse <= -P["a25_hard_stop_mult"]*self.risk
            timeout=(ts-self.entry_ts)>=P["max_hold_sec"]*1_000_000_000
            if hit_tp or timeout:
                self.close_all_positions(self.config.instrument_id); self.exit_pending=True; return
            if hit_stop:
                if self.stop_hit_ts is None:
                    self.stop_hit_ts=ts; self.stop_events+=1; self.wait_reference_pnl=adverse*self.units
                if self.arm=="BASE":
                    self.close_all_positions(self.config.instrument_id); self.exit_pending=True; return
                cv=self._continuation_value_proxy(mid)
                if cv>0 and not hard_stop:
                    self.a25_waits+=1
                    grace=P["a22_recovery_cap_sec"] if "A22" in self.arm else P["a25_wait_cap_sec"]
                    if (ts-self.stop_hit_ts) < grace*1e9:
                        # A22 requires observable recovery while under water.
                        if "A22" not in self.arm or self.side*(mid-self.jump["extreme"]) >= P["a22_recovery_ratio"]*self.risk:
                            return
                if self.wait_reference_pnl is not None:
                    now=adverse*self.units
                    if now>self.wait_reference_pnl: self.a25_saved+=1
                    elif now<self.wait_reference_pnl: self.a25_worsened+=1
                if "A22" in self.arm and (ts-self.stop_hit_ts)>=P["a22_recovery_cap_sec"]*1e9:
                    self.a22_timeouts+=1
                self.close_all_positions(self.config.instrument_id); self.exit_pending=True; return
            return

        if self.exit_pending or ts<self.cool_until or not self._flat() or ask-bid>P["max_spread"]:
            return

        # Detect a standardized raw-tick jump over event time.
        ago=self._price_ago(ts,int(P["jump_window_sec"]*1e9))
        if ago is not None and len(self.rets)>=100:
            sigma=float(np.std(self.rets))
            move=mid-ago
            z=abs(move)/(sigma*math.sqrt(max(2.,len(self.rets)**0.25))) if sigma>1e-9 else 0.
            if abs(move)>=P["min_jump_abs"] and z>=P["jump_z"]:
                direction=1 if move>0 else -1
                if self.jump is None or direction!=self.jump["dir"]:
                    self.jump=dict(ts=ts,start=ago,extreme=mid,dir=direction,size=abs(move))
                    self.jump_count+=1
        if self.jump is None: return

        age=(ts-self.jump["ts"])/1e9
        if self.jump["dir"]<0: self.jump["extreme"]=min(self.jump["extreme"],mid)
        else: self.jump["extreme"]=max(self.jump["extreme"],mid)
        if age>P["exhaustion_max_sec"]:
            self.jump=None; return
        if age<P["exhaustion_min_sec"]: return
        rec=(mid-self.jump["extreme"]) if self.jump["dir"]<0 else (self.jump["extreme"]-mid)
        ratio=rec/max(self.jump["size"],1e-9)
        mom=self._momentum(P["momentum_ticks"])
        exhausted=(ratio>=P["recovery_ratio"] and ((self.jump["dir"]<0 and mom>0) or (self.jump["dir"]>0 and mom<0)))
        if not exhausted: return

        self.exhaustion_count+=1
        # Fade exhausted jump: down-jump -> long, up-jump -> short.
        self.side=1 if self.jump["dir"]<0 else -1
        px=ask if self.side>0 else bid
        self.risk=max(0.25,P["stop_mult"]*self.jump["size"])
        self.tp=px+self.side*P["tp_mult"]*self.risk
        self._submit(self.side,P["base_units"])
        self.entry=px; self.entry_ts=ts; self.units=P["base_units"]; self.layers=1
        self.next_add=P["g75_trigger"]; self.entries+=1; self.stop_hit_ts=None
        self.wait_reference_pnl=None; self.exit_pending=False

    def on_position_closed(self, e):
        rp=getattr(e,"realized_pnl",None)
        self.realized += money(rp) if rp is not None else 0.
        self.cool_until=int(e.ts_event)+P["cooldown_sec"]*1_000_000_000
        self.entry=None; self.side=0; self.risk=0.; self.tp=0.; self.entry_ts=0
        self.units=0; self.layers=0; self.next_add=None; self.stop_hit_ts=None
        self.wait_reference_pnl=None; self.exit_pending=False; self.jump=None

    def on_stop(self):
        self.close_all_positions(self.config.instrument_id)

def extract_positions(report):
    if report is None or report.empty: return []
    pc=next((c for c in report.columns if "pnl" in str(c).lower()),None)
    tc=next((c for c in report.columns if "closed" in str(c).lower()),None)
    out=[]
    for i,row in report.iterrows():
        try: ts=int(pd.Timestamp(row[tc]).value) if tc else int(i)
        except Exception: ts=0
        out.append(dict(pnl=money(row[pc]) if pc else 0.,ts_closed=ts))
    return out

def metrics(trades, floating_dd, initial=1000., days=90):
    a=np.array([x["pnl"] for x in trades],float)
    if not len(a):
        return dict(N=0,WR_pct=0.,PF=0.,EV_trade=0.,NetProfit=0.,Return_pct=0.,
                    MaxClosedDD_pct=0.,MaxFloatingDD_pct=floating_dd,RF=None,MaxConsecutiveLosses=0)
    win=a[a>0].sum(); loss=abs(a[a<0].sum()); eq=peak=initial; mdd=0.; run=mx=0
    for z in a:
        eq+=z; peak=max(peak,eq); mdd=max(mdd,peak-eq); run=run+1 if z<0 else 0; mx=max(mx,run)
    return dict(N=int(len(a)),WR_pct=float((a>0).mean()*100),PF=float(win/loss) if loss else None,
                EV_trade=float(a.mean()),NetProfit=float(a.sum()),Return_pct=float(a.sum()/initial*100),
                MaxClosedDD_pct=float(100*mdd/max(peak,1e-9)),MaxFloatingDD_pct=float(floating_dd),
                RF=float(a.sum()/mdd) if mdd else None,MaxConsecutiveLosses=int(mx))

def run_arm(cat, inst, ticks, arm, days):
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,
                  book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],
                  default_leverage=Decimal("2000"))
    eng.add_instrument(inst); eng.add_data(ticks)
    st=A25JumpRecovery(Config(instrument_id=inst.id,arm=arm)); eng.add_strategy(st); eng.run()
    pos=eng.trader.generate_positions_report(); tr=extract_positions(pos)
    met=metrics(tr,st.max_floating_dd,days=days)
    diag=dict(jumps=st.jump_count,exhaustions=st.exhaustion_count,entries=st.entries,stop_events=st.stop_events,
              a25_waits=st.a25_waits,a25_saved=st.a25_saved,a25_worsened=st.a25_worsened,
              a22_timeouts=st.a22_timeouts,g75_adds=st.g75_adds)
    eng.dispose(); return tr,met,diag

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--catalog",required=True); ap.add_argument("--experiment-id",required=True)
    ap.add_argument("--raw-bidask-only",action="store_true"); a=ap.parse_args()
    if not a.raw_bidask_only: raise SystemExit("Raw BidAsk mandatory")
    cp=Path(a.catalog); man=json.loads((cp/"catalog_manifest.json").read_text())
    cat=ParquetDataCatalog(str(cp)); inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
    raw=cat.query_quote_ticks(identifiers=[inst.id.value]); ticks,replaced=ensure_executable_l1(raw,1000)
    out=Path("results/ae-bt")/a.experiment_id; out.mkdir(parents=True,exist_ok=True)
    summaries={}
    for arm in ARMS:
        tr,met,diag=run_arm(cat,inst,ticks,arm,int(man["days"]))
        summaries[arm]=dict(metrics=met,diagnostics=diag)
        pd.DataFrame(tr).to_csv(out/f"trades_{arm}.csv",index=False)
    base=summaries["BASE"]["metrics"]
    for arm in ARMS[1:]:
        m=summaries[arm]["metrics"]
        summaries[arm]["delta_vs_BASE"]=dict(
            Return_pct=(m["Return_pct"]-base["Return_pct"]),
            PF=(None if m["PF"] is None or base["PF"] is None else m["PF"]-base["PF"]),
            EV_trade=m["EV_trade"]-base["EV_trade"],
            MaxFloatingDD_pct=m["MaxFloatingDD_pct"]-base["MaxFloatingDD_pct"])
    result=dict(
        verification_level="NAUTILUS_IS_RAW_BIDASK_4ARM",
        strategy="AE_A25_JumpRecovery_XAUUSD_v0.1",
        engine="NautilusTrader BacktestEngine",nautilus_version=getattr(nautilus_trader,"__version__","unknown"),
        data_kind="Dukascopy RAW_BIDASK QuoteTick prices/timestamps; synthetic nonzero L1 sizes only when source size is zero",
        period=dict(start=man["start"],days=man["days"],end_exclusive=man["end_exclusive"]),
        raw_tick_count=len(raw),l1_size_replacements=replaced,params=P,arms=summaries,
        interpretation=dict(
            A25="operational continuation-value proxy; NOT a claim of closed-form Levy optimal-stopping value",
            A22="time-under-water recovery grace after A25 WAIT",
            G75="profit-direction pursuit only; 0.01-lot-equivalent 1-unit layers",
            a25_saved_worsened="event diagnostic at eventual stop decision, not causal proof"),
        limitations=[
            "This first run is NAUTILUS_IS, not fresh OOS/VALIDATED.",
            "A25 continuation value is an operational proxy requiring later calibration/fresh holdout.",
            "Native Bid/Ask spread is included; explicit commission and probabilistic slippage are not yet injected.",
            "Floating DD is tick-marked from strategy state and submitted quantities; fill-level reconciliation is a next gate.",
            "No OHLC fallback is permitted."])
    (out/"summary.json").write_text(json.dumps(result,indent=2))
    (out/"catalog_manifest.json").write_text(json.dumps(man,indent=2))
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
