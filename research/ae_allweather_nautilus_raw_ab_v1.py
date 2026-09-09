from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from collections import defaultdict
import numpy as np
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar as NBar, QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from research.ae_allweather_range_crt_cluster_v1 import AEAllWeatherEngine, EngineConfig, Bar, Side, Regime

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
    def _qqt(self,identifiers=None,start=None,end=None): return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
    ParquetDataCatalog.query_quote_ticks=_qqt

@dataclass
class VPos:
    id:int; side:int; entry:float; opened_bar:int; cluster:int

class Cfg(StrategyConfig, frozen=True):
    instrument_id: object
    bar_type: BarType
    mode: str
    initial_balance: float = 1000.0
    max_hold_bars: int = 240
    individual_tp: float = 1.50
    individual_sl: float = 1.50
    cluster_tp: float = 5.00
    cluster_trail_activation: float = 7.00
    cluster_trail_distance: float = 2.00
    cluster_max_loss: float = 15.00
    unit_qty: float = 1.0

class ABStrategy(Strategy):
    def __init__(self, config: Cfg):
        super().__init__(config)
        ecfg=EngineConfig(cluster_tp=config.cluster_tp, cluster_trail_activation=config.cluster_trail_activation, cluster_trail_distance=config.cluster_trail_distance)
        self.logic=AEAllWeatherEngine(ecfg)
        self.bar_i=0; self.tick_i=0; self.last_bid=None; self.last_ask=None
        self.next_id=1; self.next_cluster=1; self.pos={}; self.clusters=defaultdict(list); self.cluster_peak=defaultdict(float)
        self.realized=0.0; self.trades=[]; self.cluster_cycles=[]; self.equity_path=[config.initial_balance]
        self.regime_counts=defaultdict(int); self.entries_by_regime=defaultdict(int); self.poi_types=defaultdict(int)
        self.max_open=0; self.max_clusters=0; self.max_float_dd=0.0; self.peak_equity=config.initial_balance
    @staticmethod
    def f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)
    def on_start(self): self.subscribe_quote_ticks(self.config.instrument_id); self.subscribe_bars(self.config.bar_type)
    def on_bar(self, bar:NBar):
        self.bar_i+=1
        b=Bar(int(bar.ts_event),self.f(bar.open),self.f(bar.high),self.f(bar.low),self.f(bar.close))
        st=self.logic.on_bar(b); self.regime_counts[self.logic.regime.name]+=1
    def _new_cluster_for_side(self,side:int):
        open_clusters=[cid for cid,ids in self.clusters.items() if ids and self.pos.get(ids[0]) and self.pos[ids[0]].side==side and len(ids)<5]
        if open_clusters:return open_clusters[-1]
        cid=self.next_cluster;self.next_cluster+=1;return cid
    def _enter(self,side:int,px:float,n:int):
        for _ in range(n):
            cid=self._new_cluster_for_side(side) if self.config.mode=='CLUSTER' else self.next_cluster
            if self.config.mode!='CLUSTER': self.next_cluster+=1
            p=VPos(self.next_id,side,px,self.bar_i,cid);self.next_id+=1;self.pos[p.id]=p;self.clusters[cid].append(p.id)
            self.entries_by_regime[self.logic.regime.name]+=1
        self.max_open=max(self.max_open,len(self.pos)); self.max_clusters=max(self.max_clusters,sum(1 for ids in self.clusters.values() if any(pid in self.pos for pid in ids)))
    def _pnl_pos(self,p,bid,ask):
        mark=bid if p.side>0 else ask; return (mark-p.entry)*p.side*self.config.unit_qty
    def _cluster_pnl(self,cid,bid,ask): return sum(self._pnl_pos(self.pos[pid],bid,ask) for pid in self.clusters[cid] if pid in self.pos)
    def _close_ids(self,ids,bid,ask,reason):
        pnl=0.0; n=0
        for pid in list(ids):
            p=self.pos.get(pid)
            if p is None:continue
            x=self._pnl_pos(p,bid,ask); pnl+=x; n+=1; self.trades.append({'pnl':x,'reason':reason,'side':p.side,'hold_bars':self.bar_i-p.opened_bar,'cluster':p.cluster}); self.pos.pop(pid,None)
        self.realized+=pnl; self.equity_path.append(self.config.initial_balance+self.realized)
        return pnl,n
    def _evaluate_base(self,bid,ask):
        for pid,p in list(self.pos.items()):
            pnl=self._pnl_pos(p,bid,ask); hold=self.bar_i-p.opened_bar
            if pnl>=self.config.individual_tp:self._close_ids([pid],bid,ask,'IND_TP')
            elif pnl<=-self.config.individual_sl:self._close_ids([pid],bid,ask,'IND_SL')
            elif hold>=self.config.max_hold_bars:self._close_ids([pid],bid,ask,'TIME')
    def _evaluate_cluster(self,bid,ask):
        for cid,ids in list(self.clusters.items()):
            live=[pid for pid in ids if pid in self.pos]
            if not live:continue
            pnl=self._cluster_pnl(cid,bid,ask); self.cluster_peak[cid]=max(self.cluster_peak[cid],pnl)
            reason=None
            if pnl>=self.config.cluster_tp:reason='CLUSTER_TP'
            elif self.cluster_peak[cid]>=self.config.cluster_trail_activation and pnl<=self.cluster_peak[cid]-self.config.cluster_trail_distance:reason='CLUSTER_TRAIL'
            elif pnl<=-self.config.cluster_max_loss:reason='CLUSTER_DEFENSE'
            elif max(self.bar_i-self.pos[pid].opened_bar for pid in live)>=self.config.max_hold_bars:reason='CLUSTER_TIME'
            if reason:
                cp,n=self._close_ids(live,bid,ask,reason); self.cluster_cycles.append({'cluster':cid,'pnl':cp,'n':n,'reason':reason})
    def _mark_equity(self,bid,ask):
        fl=sum(self._pnl_pos(p,bid,ask) for p in self.pos.values()); eq=self.config.initial_balance+self.realized+fl; self.peak_equity=max(self.peak_equity,eq); dd=self.peak_equity-eq; self.max_float_dd=max(self.max_float_dd,dd)
    def on_quote_tick(self,tick:QuoteTick):
        self.tick_i+=1; bid=self.f(tick.bid_price); ask=self.f(tick.ask_price); self.last_bid=bid;self.last_ask=ask
        self._mark_equity(bid,ask)
        # only create once per completed bar using a simple tick latch
        if not hasattr(self,'last_entry_bar'):self.last_entry_bar=-1
        if self.bar_i>0 and self.last_entry_bar!=self.bar_i:
            side=self.logic.desired_entry_side()
            if side is not None:
                if self.logic.regime==Regime.RANGE:n=1
                elif self.logic.regime==Regime.EXPANSION:n=5 if self.logic.last_crt_signals.score_100()>=90 else 3
                else:n=0
                if n>0:self._enter(1 if side==Side.LONG else -1, ask if side==Side.LONG else bid,n)
            self.last_entry_bar=self.bar_i
        if self.config.mode=='BASE':self._evaluate_base(bid,ask)
        else:self._evaluate_cluster(bid,ask)
    def on_stop(self):
        if self.last_bid is not None and self.pos:
            self._close_ids(list(self.pos),self.last_bid,self.last_ask,'EOD')
    def summary(self):
        a=np.array([x['pnl'] for x in self.trades],float); wins=a[a>0]; losses=a[a<0]
        pf=float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else (float('inf') if len(wins) else 0.0)
        wr=float((a>0).mean()*100) if len(a) else 0.0
        eq=self.config.initial_balance;peak=eq;mdd=0
        for x in a: eq+=x;peak=max(peak,eq);mdd=max(mdd,peak-eq)
        cyc=np.array([x['pnl'] for x in self.cluster_cycles],float) if self.cluster_cycles else np.array([])
        cwin=cyc[cyc>0];closs=cyc[cyc<0]; cpf=float(cwin.sum()/abs(closs.sum())) if len(closs) and closs.sum()!=0 else (float('inf') if len(cwin) else 0.0)
        return {'mode':self.config.mode,'N':int(len(a)),'WR_pct':wr,'PF':pf,'net':float(a.sum()) if len(a) else 0.0,'return_pct':float((eq/self.config.initial_balance-1)*100),'max_DD_realized_pct':float(mdd/max(peak,1e-9)*100),'max_FloatingDD_pct':float(self.max_float_dd/max(self.peak_equity,1e-9)*100),'max_open_positions':self.max_open,'max_clusters':self.max_clusters,'cluster_cycles':len(cyc),'cluster_PF':cpf,'regime_counts':dict(self.regime_counts),'entries_by_regime':dict(self.entries_by_regime)}

def run_cell(catalog_path,mode,tf,experiment_id):
    catalog=ParquetDataCatalog(str(catalog_path)); inst=next((x for x in catalog.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'),None)
    if inst is None:raise SystemExit('XAUUSD missing')
    ticks=catalog.query_quote_ticks(identifiers=[inst.id.value]);
    if not ticks:raise SystemExit('no Raw QuoteTicks')
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    eng.add_instrument(inst);eng.add_data(ticks);bt=BarType.from_str(f'{inst.id.value}-{tf}-MINUTE-BID-INTERNAL');st=ABStrategy(Cfg(instrument_id=inst.id,bar_type=bt,mode=mode));eng.add_strategy(st);eng.run();obj={'verification_level':'NAUTILUS_RAW_BIDASK_AE_ALLWEATHER_AB_V1','engine':'NautilusTrader BacktestEngine','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'raw_ticks':len(ticks),'ohlc_resample_used':False,'signal_bars':'Nautilus INTERNAL BID bars from Raw QuoteTicks','execution':'virtual micro-position ledger marked on raw Bid/Ask; raw spread included; native venue used for event engine/time ordering','tf_minutes':tf,**st.summary()};eng.dispose();return obj

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);ap.add_argument('--mode',choices=['BASE','CLUSTER'],required=True);ap.add_argument('--tf',type=int,default=1);ap.add_argument('--raw-bidask-only',action='store_true');a=ap.parse_args()
    if not a.raw_bidask_only:raise SystemExit('raw-bidask-only mandatory')
    out=run_cell(Path(a.catalog),a.mode,a.tf,a.experiment_id);p=Path('results/ae-allweather-ab')/a.experiment_id;p.mkdir(parents=True,exist_ok=True);(p/f'{a.mode}_M{a.tf}.json').write_text(json.dumps(out,indent=2,default=str),encoding='utf-8');print(json.dumps(out,indent=2,default=str))
if __name__=='__main__':main()
