from __future__ import annotations

import argparse, json, math
from collections import deque
from dataclasses import dataclass
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

SIM = Venue('SIM')
TF_MIN = {'M1':1,'M5':5,'M15':15}

@dataclass
class CtrlStats:
    soft_events:int=0; lock_events:int=0; hard_events:int=0; recovery_cycles:int=0; recovery_success:int=0
    max_debt:float=0.0; debt_ticks:int=0; debt_clear_ticks:list=None
    def __post_init__(self):
        if self.debt_clear_ticks is None: self.debt_clear_ticks=[]

class G75TsugiConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    variant: str
    initial_balance: float = 1000.0
    trigger: float = 0.12
    add: float = 0.025
    reversal: float = 0.20
    max_layers: int = 10
    unit_qty: Decimal = Decimal('1')
    soft_dd_pct: float = 1.5
    hedge_dd_pct: float = 3.0
    hard_dd_pct: float = 4.5
    g75_profit_allocation: float = 0.70

class G75TsugiStrategy(Strategy):
    """Raw Bid/Ask G75 with A/B/C supervisory lanes.
    C uses a synthetic hedge-lock equivalent: native market flatten at lock point,
    debt ledger preserves the frozen loss, then reduced G75 cycles repay debt.
    This isolates the economics of lock+recovery without pretending NETTING is HEDGING.
    """
    def __init__(self, config:G75TsugiConfig):
        super().__init__(config)
        self.anchor=None; self.side=0; self.active=False; self.entry_prices=[]; self.last_add=None; self.extreme=None
        self.exit_pending=False; self.pending_start=None
        self.realized=0.0; self.equity_peak=config.initial_balance; self.max_floating_dd=0.0
        self.cycle_count=0; self.win_count=0; self.loss_count=0; self.total_gross_win=0.0; self.total_gross_loss=0.0
        self.total_entries=0; self.total_adds=0; self.max_layers_seen=0
        self.mode='NORMAL'; self.debt=0.0; self.ctrl=CtrlStats(); self.recovery_pnls=deque(maxlen=20); self.lock_tick=None
        self.tick_index=0; self.recovery_active=False; self.stopped=False
        self.last_bid=None; self.last_ask=None

    @staticmethod
    def _f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)

    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id)
        self.subscribe_bars(self.config.bar_type)

    def _mark_pnl(self,bid,ask):
        if not self.active or not self.entry_prices: return 0.0
        px = bid if self.side>0 else ask
        return sum((px-e)*self.side for e in self.entry_prices)

    def _dd_pct(self,bid,ask):
        eq=self.config.initial_balance+self.realized+self._mark_pnl(bid,ask)
        self.equity_peak=max(self.equity_peak,eq)
        dd=max(0.0,(self.equity_peak-eq)/max(self.equity_peak,1e-9)*100.0)
        self.max_floating_dd=max(self.max_floating_dd,dd)
        return dd

    def _submit_market(self, side:int, qty_units:int):
        if qty_units<=0: return
        inst=self.cache.instrument(self.config.instrument_id)
        order=self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY if side>0 else OrderSide.SELL,
            quantity=inst.make_qty(Decimal(qty_units)),
        )
        self.submit_order(order)

    def _effective_max_layers(self):
        if self.config.variant=='A': return self.config.max_layers
        if self.mode in ('REDUCED','RECOVERY'): return max(1,self.config.max_layers//2)
        return self.config.max_layers

    def _recovery_rate_ok(self):
        if self.mode!='RECOVERY': return True
        if len(self.recovery_pnls)<5: return True
        pos=sum(x for x in self.recovery_pnls if x>0); neg=abs(sum(x for x in self.recovery_pnls if x<0))
        return pos>neg

    def _start_cycle(self, side:int, px:float):
        if self.stopped or self.exit_pending: return
        if self.mode=='RECOVERY' and not self._recovery_rate_ok(): return
        self.side=side; self.active=True; self.entry_prices=[px]; self.last_add=px; self.extreme=px; self.exit_pending=False
        self.total_entries+=1; self.max_layers_seen=max(self.max_layers_seen,1)
        self._submit_market(side,1)

    def _close_cycle(self, bid:float, ask:float, reason:str):
        if not self.active or self.exit_pending: return
        px=bid if self.side>0 else ask
        pnl=sum((px-e)*self.side for e in self.entry_prices)
        qty=len(self.entry_prices)
        self._submit_market(-self.side,qty)
        self.exit_pending=True
        self.realized+=pnl; self.cycle_count+=1
        if pnl>0: self.win_count+=1; self.total_gross_win+=pnl
        elif pnl<0: self.loss_count+=1; self.total_gross_loss+=abs(pnl)
        was_recovery=self.mode=='RECOVERY'
        if was_recovery:
            self.ctrl.recovery_cycles+=1; self.recovery_pnls.append(pnl)
            if pnl>0 and self.debt>0:
                self.debt=max(0.0,self.debt-pnl*self.config.g75_profit_allocation)
                if self.debt<=1e-12:
                    self.ctrl.recovery_success+=1
                    if self.lock_tick is not None: self.ctrl.debt_clear_ticks.append(self.tick_index-self.lock_tick)
                    self.mode='NORMAL'; self.recovery_active=False; self.lock_tick=None
        self.active=False; self.side=0; self.entry_prices=[]; self.last_add=None; self.extreme=None
        self.anchor=(bid+ask)/2; self.exit_pending=False

    def _lock_to_debt(self,bid,ask):
        if not self.active: return
        locked=max(0.0,-self._mark_pnl(bid,ask))
        self.debt+=locked; self.ctrl.max_debt=max(self.ctrl.max_debt,self.debt); self.ctrl.lock_events+=1
        self.lock_tick=self.tick_index; self.recovery_active=True
        self._close_cycle(bid,ask,'HEDGE_LOCK_EQUIV')
        self.mode='RECOVERY'; self.anchor=(bid+ask)/2

    def on_bar(self, bar:Bar):
        if self.active or self.stopped or self.exit_pending: return
        c=self._f(bar.close); h=self._f(bar.high); l=self._f(bar.low)
        if self.anchor is None:
            self.anchor=c; return
        up=h>=self.anchor+self.config.trigger; dn=l<=self.anchor-self.config.trigger
        if not (up or dn):
            self.anchor=c; return
        side=1 if c>=self.anchor else -1
        self.pending_start=(side,c)

    def on_quote_tick(self,tick:QuoteTick):
        self.tick_index+=1
        bid=self._f(tick.bid_price); ask=self._f(tick.ask_price); self.last_bid=bid; self.last_ask=ask
        if self.debt>0: self.ctrl.debt_ticks+=1; self.ctrl.max_debt=max(self.ctrl.max_debt,self.debt)
        if self.pending_start is not None and not self.active and not self.stopped:
            side,_=self.pending_start; self.pending_start=None
            px=ask if side>0 else bid
            self._start_cycle(side,px)
        dd=self._dd_pct(bid,ask)
        if self.config.variant!='A':
            if dd>=self.config.hard_dd_pct:
                self.ctrl.hard_events+=1
                if self.active: self._close_cycle(bid,ask,'HARD_DD')
                self.stopped=True; self.mode='STOPPED'; return
            if self.config.variant=='C' and self.active and self.mode!='RECOVERY' and dd>=self.config.hedge_dd_pct:
                self._lock_to_debt(bid,ask); return
            if self.mode not in ('RECOVERY','STOPPED'):
                if dd>=self.config.soft_dd_pct:
                    if self.mode!='REDUCED': self.ctrl.soft_events+=1
                    self.mode='REDUCED'
                else:
                    self.mode='NORMAL'
        if not self.active: return
        px=bid if self.side>0 else ask
        if self.extreme is None: self.extreme=px
        self.extreme=max(self.extreme,px) if self.side>0 else min(self.extreme,px)
        cap=self._effective_max_layers()
        while len(self.entry_prices)<cap:
            nxt=self.last_add+self.side*self.config.add
            crossed = px>=nxt if self.side>0 else px<=nxt
            if not crossed: break
            fill=ask if self.side>0 else bid
            self.entry_prices.append(fill); self.last_add=nxt; self.total_adds+=1
            self.max_layers_seen=max(self.max_layers_seen,len(self.entry_prices)); self._submit_market(self.side,1)
            if len(self.entry_prices)>=cap: break
        reversal_hit=(px<=self.extreme-self.config.reversal) if self.side>0 else (px>=self.extreme+self.config.reversal)
        if reversal_hit: self._close_cycle(bid,ask,'REVERSAL')

    def on_stop(self):
        if self.active and self.last_bid is not None:
            self._close_cycle(self.last_bid,self.last_ask,'EOD')

    def summary(self):
        pf=self.total_gross_win/self.total_gross_loss if self.total_gross_loss>0 else (math.inf if self.total_gross_win>0 else 0.0)
        wr=self.win_count/max(self.cycle_count,1)*100.0
        dct=self.ctrl.debt_clear_ticks
        return {
            'variant':self.config.variant,'cycles':self.cycle_count,'WR_pct':wr,'PF_virtual_raw':pf,
            'realized_virtual':self.realized,'max_floating_DD_pct':self.max_floating_dd,
            'entries':self.total_entries,'adds':self.total_adds,'max_layers_seen':self.max_layers_seen,
            'mode_final':self.mode,'debt_final':self.debt,'max_debt':self.ctrl.max_debt,
            'soft_events':self.ctrl.soft_events,'lock_events':self.ctrl.lock_events,'hard_events':self.ctrl.hard_events,
            'recovery_cycles':self.ctrl.recovery_cycles,'recovery_success':self.ctrl.recovery_success,
            'debt_clear_ticks_p50':float(np.median(dct)) if dct else None,
            'debt_clear_ticks_p95':float(np.percentile(dct,95)) if dct else None,
            'stopped':self.stopped,
        }

def parse_money(v):
    if v is None:return 0.0
    if isinstance(v,(float,int,np.number)):return float(v)
    s=str(v).replace(',','').strip()
    try:return float(s.split()[0])
    except:return 0.0

def native_report_metrics(report, initial=1000.0):
    if report is None or report.empty:return {'N_native':0,'NetProfit_native':0.0}
    pnl_col=next((c for c in report.columns if 'pnl' in str(c).lower()),None)
    vals=[] if pnl_col is None else [parse_money(x) for x in report[pnl_col].tolist()]
    return {'N_native':len(vals),'NetProfit_native':float(sum(vals))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--raw-bidask-only',action='store_true'); ap.add_argument('--timeframes',nargs='+',default=['M1','M5','M15'])
    args=ap.parse_args()
    if not args.raw_bidask_only: raise SystemExit('raw-bidask-only is mandatory')
    cp=Path(args.catalog); manifest=json.loads((cp/'catalog_manifest.json').read_text()); catalog=ParquetDataCatalog(str(cp))
    instrument=next((x for x in catalog.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'),None)
    if instrument is None: raise SystemExit('XAUUSD missing')
    ticks=catalog.query_quote_ticks(identifiers=[instrument.id.value])
    if not ticks: raise SystemExit('no raw XAUUSD QuoteTicks')
    out=Path('results/ae-bt')/args.experiment_id; out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for tf in args.timeframes:
        minutes=TF_MIN[tf]
        for variant in ['A','B','C']:
            engine=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
            engine.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
            engine.add_instrument(instrument); engine.add_data(ticks)
            bar_type=BarType.from_str(f'{instrument.id.value}-{minutes}-MINUTE-BID-INTERNAL')
            st=G75TsugiStrategy(G75TsugiConfig(instrument_id=instrument.id,bar_type=bar_type,variant=variant))
            engine.add_strategy(st); engine.run(); rep=engine.trader.generate_positions_report()
            m={**st.summary(),**native_report_metrics(rep),'tf':tf,'raw_ticks':len(ticks)}; rows.append(m)
            engine.dispose()
    df=pd.DataFrame(rows); df.to_csv(out/'kpi.csv',index=False)
    rel=[]
    for tf,g in df.groupby('tf'):
        a=g[g.variant=='A'].iloc[0]
        for _,r in g.iterrows():
            rel.append({'tf':tf,'variant':r.variant,'N_retention':float(r.cycles/max(a.cycles,1)),'return_delta_vs_A':float(r.realized_virtual-a.realized_virtual),'dd_delta_vs_A':float(r.max_floating_DD_pct-a.max_floating_DD_pct)})
    summary={
      'verification_level':'NAUTILUS_BT_RAW_BIDASK_G75_TSUGI_ABC','engine':'NautilusTrader BacktestEngine','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),
      'data_kind':'RAW_BIDASK QuoteTick','ohlc_resample_used':False,'signal_bars':'Nautilus INTERNAL bars built from raw QuoteTicks','execution':'Native Nautilus market orders; raw Bid/Ask observed. Controller metrics use a synchronized virtual basket ledger.',
      'period':{'start':manifest.get('start'),'days':manifest.get('days'),'end_exclusive':manifest.get('end_exclusive')},'raw_ticks':len(ticks),'results':rows,'relative':rel,
      'controller_semantics':{'A':'Frozen G75 raw baseline','B':'soft DD layer throttle + 4.5% hard stop','C':'B + 3.0% synthetic hedge-lock equivalent + economic debt recovery; 70% positive recovery PnL allocated to debt'},
      'limitations':['C lock is economic-equivalent flatten-and-debt, not simultaneous long/short broker hedging; dedicated HEDGING-account validation remains required.','Controller DD/PF use a virtual basket ledger synchronized to raw Bid/Ask; native position report is included as an execution cross-check.','No explicit commission/slippage/reject model beyond observed raw spread in this first controller gate.']
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
