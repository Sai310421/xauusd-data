from __future__ import annotations

import argparse, json
from decimal import Decimal
from pathlib import Path

import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from g75_tsugi_nautilus_raw_bt import G75TsugiStrategy, G75TsugiConfig, native_report_metrics, TF_MIN

SIM=Venue('SIM')

class G75TickStrict(G75TsugiStrategy):
    """Strict chronology: completed bar only refreshes the inactive anchor.
    Trigger/Add/Extreme/Reversal are evaluated on ordered Raw Bid/Ask QuoteTicks.
    This removes OHLC same-bar ordering assumptions.
    """
    def on_bar(self, bar:Bar):
        if self.stopped or self.active or self.exit_pending:
            return
        self.anchor=self._f(bar.close)

    def on_quote_tick(self,tick:QuoteTick):
        self.tick_index+=1
        bid=self._f(tick.bid_price); ask=self._f(tick.ask_price); mid=(bid+ask)/2.0
        self.last_bid=bid; self.last_ask=ask
        if self.debt>0:
            self.ctrl.debt_ticks+=1; self.ctrl.max_debt=max(self.ctrl.max_debt,self.debt)
        if self.anchor is None:
            self.anchor=mid
        if not self.active and not self.stopped and not self.exit_pending:
            if mid>=self.anchor+self.config.trigger:
                self._start_cycle(1,ask)
            elif mid<=self.anchor-self.config.trigger:
                self._start_cycle(-1,bid)
            else:
                return
        dd=self._dd_pct(bid,ask)
        if self.config.variant!='A':
            if dd>=self.config.hard_dd_pct:
                self.ctrl.hard_events+=1
                if self.active:self._close_cycle(bid,ask,'HARD_DD')
                self.stopped=True; self.mode='STOPPED'; return
            if self.config.variant=='C' and self.active and self.mode!='RECOVERY' and dd>=self.config.hedge_dd_pct:
                self._lock_to_debt(bid,ask); return
            if self.mode not in ('RECOVERY','STOPPED'):
                if dd>=self.config.soft_dd_pct:
                    if self.mode!='REDUCED':self.ctrl.soft_events+=1
                    self.mode='REDUCED'
                else:self.mode='NORMAL'
        if not self.active:return
        px=bid if self.side>0 else ask
        self.extreme=max(self.extreme,px) if self.side>0 else min(self.extreme,px)
        cap=self._effective_max_layers()
        while len(self.entry_prices)<cap:
            nxt=self.last_add+self.side*self.config.add
            crossed=px>=nxt if self.side>0 else px<=nxt
            if not crossed:break
            fill=ask if self.side>0 else bid
            self.entry_prices.append(fill); self.last_add=nxt; self.total_adds+=1
            self.max_layers_seen=max(self.max_layers_seen,len(self.entry_prices)); self._submit_market(self.side,1)
        reversal_hit=(px<=self.extreme-self.config.reversal) if self.side>0 else (px>=self.extreme+self.config.reversal)
        if reversal_hit:self._close_cycle(bid,ask,'REVERSAL')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--tf',required=True,choices=['M1','M5','M15']); ap.add_argument('--variant',required=True,choices=['A','B','C']); ap.add_argument('--raw-bidask-only',action='store_true'); args=ap.parse_args()
    if not args.raw_bidask_only:raise SystemExit('raw-bidask-only mandatory')
    cp=Path(args.catalog); manifest=json.loads((cp/'catalog_manifest.json').read_text()); catalog=ParquetDataCatalog(str(cp))
    instrument=next((x for x in catalog.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'),None)
    ticks=catalog.query(data_cls=QuoteTick,identifiers=[instrument.id.value]); minutes=TF_MIN[args.tf]
    engine=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    engine.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    engine.add_instrument(instrument); engine.add_data(ticks)
    bt=BarType.from_str(f'{instrument.id.value}-{minutes}-MINUTE-BID-INTERNAL')
    st=G75TickStrict(G75TsugiConfig(instrument_id=instrument.id,bar_type=bt,variant=args.variant))
    engine.add_strategy(st); engine.run(); rep=engine.trader.generate_positions_report()
    row={**st.summary(),**native_report_metrics(rep),'tf':args.tf,'raw_ticks':len(ticks),'chronology':'RAW_TICK_STRICT','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'period_start':manifest.get('start'),'period_days':manifest.get('days'),'period_end_exclusive':manifest.get('end_exclusive')}
    out=Path('results/ae-bt')/args.experiment_id/'cells'; out.mkdir(parents=True,exist_ok=True); (out/f'{args.tf}_{args.variant}.json').write_text(json.dumps(row,indent=2,ensure_ascii=False),encoding='utf-8'); print(json.dumps(row,ensure_ascii=False)); engine.dispose()

if __name__=='__main__':main()
