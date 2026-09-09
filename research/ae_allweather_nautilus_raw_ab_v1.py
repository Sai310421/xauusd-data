from __future__ import annotations
import argparse, json, itertools
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
    def _qqt(self,identifiers=None,start=None,end=None):
        return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
    ParquetDataCatalog.query_quote_ticks=_qqt

@dataclass
class VPos:
    id:int
    side:int
    entry:float
    opened_bar:int
    regime:str
    score:float

class Cfg(StrategyConfig, frozen=True):
    instrument_id: object
    bar_type: BarType
    mode: str
    initial_balance: float = 1000.0
    max_hold_bars: int = 240
    individual_tp: float = 1.50
    individual_sl: float = 1.50
    unit_qty: float = 1.0

    # Dynamic Profit Cluster
    cluster_min_size: int = 2
    cluster_max_size: int = 5
    cluster_candidate_pool: int = 12
    profit_buffer: float = 0.30
    economic_be_buffer: float = 0.05
    trail_activation: float = 2.50
    trail_distance: float = 0.80
    defense_loss: float = 3.00
    defense_min_age: int = 24
    inventory_soft_cap: int = 20
    inventory_hard_cap: int = 40

    # Recovery pool
    recovery_age_bars: int = 480
    recovery_pool_cap: int = 24
    recovery_bundle_max: int = 5
    recovery_min_cover: float = 0.05
    recovery_priority_bonus: float = 0.50
    recovery_reserve_fraction: float = 0.55

    # Utility weights
    w_profit: float = 1.00
    w_dd_release: float = 0.35
    w_inventory_release: float = 0.20
    w_runner_penalty: float = 0.25

class ABStrategy(Strategy):
    def __init__(self, config: Cfg):
        super().__init__(config)
        self.logic=AEAllWeatherEngine(EngineConfig())
        self.bar_i=0; self.tick_i=0; self.last_bid=None; self.last_ask=None
        self.next_id=1; self.pos={}
        self.realized=0.0; self.trades=[]; self.cluster_cycles=[]; self.equity_path=[config.initial_balance]
        self.regime_counts=defaultdict(int); self.entries_by_regime=defaultdict(int)
        self.max_open=0; self.max_float_dd=0.0; self.peak_equity=config.initial_balance
        self.cluster_peak={}; self.last_profit_signature=None; self.recluster_changes=0
        self.exit_counts=defaultdict(int); self.exit_pnl=defaultdict(float); self.inventory_release=defaultdict(int)
        self.recovery_pool=set(); self.recovery_added=0; self.recovery_released=0; self.recovery_cycles=0
        self.recovery_debt_peak=0.0; self.recovery_debt_end=0.0

    @staticmethod
    def f(x): return float(x.as_double()) if hasattr(x,'as_double') else float(x)

    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar:NBar):
        self.bar_i+=1
        b=Bar(int(bar.ts_event),self.f(bar.open),self.f(bar.high),self.f(bar.low),self.f(bar.close))
        self.logic.on_bar(b); self.regime_counts[self.logic.regime.name]+=1

    def _enter(self,side:int,px:float,n:int):
        score=max(self.logic.last_range_signals.score,self.logic.last_crt_signals.score_100())
        for _ in range(n):
            if len(self.pos)>=self.config.inventory_hard_cap: break
            p=VPos(self.next_id,side,px,self.bar_i,self.logic.regime.name,score)
            self.next_id+=1; self.pos[p.id]=p; self.entries_by_regime[self.logic.regime.name]+=1
        self.max_open=max(self.max_open,len(self.pos))

    def _pnl_pos(self,p,bid,ask):
        mark=bid if p.side>0 else ask
        return (mark-p.entry)*p.side*self.config.unit_qty

    def _close_ids(self,ids,bid,ask,reason):
        pnl=0.0; n=0
        for pid in list(ids):
            p=self.pos.get(pid)
            if p is None: continue
            x=self._pnl_pos(p,bid,ask); pnl+=x; n+=1
            self.trades.append({'pnl':x,'reason':reason,'side':p.side,'hold_bars':self.bar_i-p.opened_bar,'regime':p.regime,'score':p.score})
            self.pos.pop(pid,None)
            if pid in self.recovery_pool:
                self.recovery_pool.discard(pid); self.recovery_released+=1
        self.realized+=pnl; self.equity_path.append(self.config.initial_balance+self.realized)
        if n:
            self.exit_counts[reason]+=1; self.exit_pnl[reason]+=pnl; self.inventory_release[reason]+=n
        return pnl,n

    def _evaluate_base(self,bid,ask):
        for pid,p in list(self.pos.items()):
            pnl=self._pnl_pos(p,bid,ask); hold=self.bar_i-p.opened_bar
            if pnl>=self.config.individual_tp:self._close_ids([pid],bid,ask,'IND_TP')
            elif pnl<=-self.config.individual_sl:self._close_ids([pid],bid,ask,'IND_SL')
            elif hold>=self.config.max_hold_bars:self._close_ids([pid],bid,ask,'TIME')

    def _candidate_pool(self,bid,ask):
        rows=[]
        for pid,p in self.pos.items():
            pnl=self._pnl_pos(p,bid,ask); age=self.bar_i-p.opened_bar
            rows.append((pid,p,pnl,age))
        winners=sorted([r for r in rows if r[2]>=0],key=lambda r:r[2],reverse=True)
        losers=sorted([r for r in rows if r[2]<0],key=lambda r:r[2],reverse=True)
        half=max(1,self.config.cluster_candidate_pool//2)
        pool=winners[:half]+losers[:self.config.cluster_candidate_pool-half]
        used={r[0] for r in pool}
        remain=sorted([r for r in rows if r[0] not in used],key=lambda r:abs(r[2]),reverse=True)
        return (pool+remain)[:self.config.cluster_candidate_pool]

    def _future_edge_penalty(self, subset):
        pen=0.0
        for _,p,pnl,_ in subset:
            if pnl>0 and p.score>=90: pen += pnl*self.config.w_runner_penalty
        return pen

    def _cluster_utility(self, subset):
        net=sum(x[2] for x in subset)
        losing_released=sum(-x[2] for x in subset if x[2]<0)
        n=len(subset)
        recovery_bonus=sum(self.config.recovery_priority_bonus for x in subset if x[0] in self.recovery_pool)
        return (self.config.w_profit*net + self.config.w_dd_release*losing_released
                + self.config.w_inventory_release*n + recovery_bonus
                - self._future_edge_penalty(subset))

    def _best_profit_cluster(self,bid,ask):
        pool=self._candidate_pool(bid,ask)
        if len(pool)<self.config.cluster_min_size:return None
        best=None; max_k=min(self.config.cluster_max_size,len(pool))
        for k in range(self.config.cluster_min_size,max_k+1):
            for sub in itertools.combinations(pool,k):
                net=sum(x[2] for x in sub)
                if net<self.config.profit_buffer:continue
                utility=self._cluster_utility(sub)
                if any(x[2]<0 for x in sub):utility+=0.25
                if best is None or utility>best[0]:best=(utility,sub,net)
        return best

    def _best_economic_be_cluster(self,bid,ask):
        pool=self._candidate_pool(bid,ask)
        if len(pool)<self.config.cluster_min_size:return None
        best=None; max_k=min(self.config.cluster_max_size,len(pool))
        for k in range(self.config.cluster_min_size,max_k+1):
            for sub in itertools.combinations(pool,k):
                net=sum(x[2] for x in sub)
                if net<self.config.economic_be_buffer or not any(x[2]<0 for x in sub):continue
                losing_released=sum(-x[2] for x in sub if x[2]<0)
                recovery_bonus=sum(self.config.recovery_priority_bonus for x in sub if x[0] in self.recovery_pool)
                utility=net+0.75*losing_released+0.20*len(sub)+recovery_bonus-self._future_edge_penalty(sub)
                if best is None or utility>best[0]:best=(utility,sub,net)
        return best

    def _recovery_debt(self,bid,ask):
        debt=0.0
        for pid in list(self.recovery_pool):
            p=self.pos.get(pid)
            if p is None:
                self.recovery_pool.discard(pid); continue
            debt+=max(0.0,-self._pnl_pos(p,bid,ask))
        self.recovery_debt_peak=max(self.recovery_debt_peak,debt)
        self.recovery_debt_end=debt
        return debt

    def _refresh_recovery_pool(self,bid,ask):
        stale=[]
        for pid,p in self.pos.items():
            if pid in self.recovery_pool:continue
            age=self.bar_i-p.opened_bar
            pnl=self._pnl_pos(p,bid,ask)
            if age>=self.config.recovery_age_bars and pnl<0:
                stale.append((pnl,pid))
        stale.sort()  # worst first
        room=max(0,self.config.recovery_pool_cap-len(self.recovery_pool))
        for _,pid in stale[:room]:
            self.recovery_pool.add(pid); self.recovery_added+=1
        self._recovery_debt(bid,ask)

    def _best_recovery_bundle(self,bid,ask):
        if not self.recovery_pool:return None
        rec=[]; wins=[]
        for pid,p in self.pos.items():
            pnl=self._pnl_pos(p,bid,ask); age=self.bar_i-p.opened_bar
            row=(pid,p,pnl,age)
            if pid in self.recovery_pool:rec.append(row)
            elif pnl>0:wins.append(row)
        if not rec or not wins:return None
        rec=sorted(rec,key=lambda x:x[2])[:self.config.recovery_bundle_max]
        wins=sorted(wins,key=lambda x:x[2],reverse=True)[:self.config.cluster_candidate_pool]
        best=None
        for r in rec:
            needed=max(self.config.recovery_min_cover,-r[2])
            for k in range(1,min(self.config.recovery_bundle_max-1,len(wins))+1):
                for wsub in itertools.combinations(wins,k):
                    gross=sum(x[2] for x in wsub)
                    # Keep some winner profit as reserve; do not spend all harvest on debt.
                    usable=gross*self.config.recovery_reserve_fraction
                    net=r[2]+usable
                    if net<self.config.recovery_min_cover:continue
                    ids=[r[0]]+[x[0] for x in wsub]
                    utility=net+(-r[2])*0.9+0.15*len(ids)-self._future_edge_penalty(wsub)
                    if best is None or utility>best[0]:best=(utility,ids,net)
        return best

    def _trail_signature(self,subset):return tuple(sorted(x[0] for x in subset))

    def _try_profit_harvest(self,bid,ask):
        cand=self._best_profit_cluster(bid,ask)
        if cand is None:return False
        _,sub,net=cand; sig=self._trail_signature(sub)
        if self.last_profit_signature is not None and sig!=self.last_profit_signature:self.recluster_changes+=1
        self.last_profit_signature=sig
        peak=max(self.cluster_peak.get(sig,-1e18),net); self.cluster_peak[sig]=peak
        reason='PROFIT_HARVEST'
        if peak>=self.config.trail_activation and net<=peak-self.config.trail_distance:reason='PEAK_TRAIL'
        ids=[x[0] for x in sub]
        cp,n=self._close_ids(ids,bid,ask,reason)
        self.cluster_cycles.append({'ids':ids,'pnl':cp,'n':n,'reason':reason,'utility':cand[0]})
        return True

    def _try_economic_be(self,bid,ask):
        if len(self.pos)<self.config.inventory_soft_cap:return False
        cand=self._best_economic_be_cluster(bid,ask)
        if cand is None:return False
        _,sub,_=cand; ids=[x[0] for x in sub]
        cp,n=self._close_ids(ids,bid,ask,'ECONOMIC_BE')
        self.cluster_cycles.append({'ids':ids,'pnl':cp,'n':n,'reason':'ECONOMIC_BE','utility':cand[0]})
        return True

    def _try_recovery_pool(self,bid,ask):
        cand=self._best_recovery_bundle(bid,ask)
        if cand is None:return False
        _,ids,_=cand
        cp,n=self._close_ids(ids,bid,ask,'RECOVERY_BE')
        self.cluster_cycles.append({'ids':ids,'pnl':cp,'n':n,'reason':'RECOVERY_BE','utility':cand[0]})
        self.recovery_cycles+=1
        self._recovery_debt(bid,ask)
        return True

    def _try_defense(self,bid,ask):
        # Hard defense only for non-recovery inventory when capacity is exhausted.
        if len(self.pos)<self.config.inventory_hard_cap:return False
        desired=self.logic.desired_entry_side(); desired_side=1 if desired==Side.LONG else (-1 if desired==Side.SHORT else 0)
        candidates=[]
        for pid,p in self.pos.items():
            if pid in self.recovery_pool:continue
            pnl=self._pnl_pos(p,bid,ask); age=self.bar_i-p.opened_bar
            if pnl<=-self.config.defense_loss and age>=self.config.defense_min_age and desired_side!=0 and p.side!=desired_side:
                candidates.append((pnl,pid))
        if not candidates:return False
        candidates.sort(); ids=[pid for _,pid in candidates[:self.config.cluster_max_size]]
        cp,n=self._close_ids(ids,bid,ask,'DEFENSE')
        self.cluster_cycles.append({'ids':ids,'pnl':cp,'n':n,'reason':'DEFENSE','utility':cp})
        return True

    def _evaluate_cluster(self,bid,ask):
        self._refresh_recovery_pool(bid,ask)
        for _ in range(4):
            if len(self.pos)<self.config.cluster_min_size:break
            if self._try_profit_harvest(bid,ask):continue
            if self._try_recovery_pool(bid,ask):continue
            if self._try_economic_be(bid,ask):continue
            self._try_defense(bid,ask); break
        self._recovery_debt(bid,ask)

    def _mark_equity(self,bid,ask):
        fl=sum(self._pnl_pos(p,bid,ask) for p in self.pos.values())
        eq=self.config.initial_balance+self.realized+fl
        self.peak_equity=max(self.peak_equity,eq); dd=self.peak_equity-eq
        self.max_float_dd=max(self.max_float_dd,dd)

    def on_quote_tick(self,tick:QuoteTick):
        self.tick_i+=1; bid=self.f(tick.bid_price); ask=self.f(tick.ask_price); self.last_bid=bid; self.last_ask=ask
        self._mark_equity(bid,ask)
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
        # EOD remains an accounting close for finite backtest only.
        if self.last_bid is not None and self.pos:
            self._recovery_debt(self.last_bid,self.last_ask)
            self._close_ids(list(self.pos),self.last_bid,self.last_ask,'EOD')

    def summary(self):
        a=np.array([x['pnl'] for x in self.trades],float)
        wins=a[a>0]; losses=a[a<0]
        pf=float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else (float('inf') if len(wins) else 0.0)
        wr=float((a>0).mean()*100) if len(a) else 0.0
        eq=self.config.initial_balance;peak=eq;mdd=0
        for x in a: eq+=x;peak=max(peak,eq);mdd=max(mdd,peak-eq)
        cyc=np.array([x['pnl'] for x in self.cluster_cycles],float) if self.cluster_cycles else np.array([])
        cwin=cyc[cyc>0]; closs=cyc[cyc<0]
        cpf=float(cwin.sum()/abs(closs.sum())) if len(closs) and closs.sum()!=0 else (float('inf') if len(cwin) else 0.0)
        profitable_cycles=int((cyc>0).sum()) if len(cyc) else 0
        recovery_success=float(self.recovery_released/max(self.recovery_added,1)*100)
        return {
            'mode':self.config.mode,'N':int(len(a)),'WR_pct':wr,'PF':pf,
            'net':float(a.sum()) if len(a) else 0.0,
            'return_pct':float((eq/self.config.initial_balance-1)*100),
            'max_DD_realized_pct':float(mdd/max(peak,1e-9)*100),
            'max_FloatingDD_pct':float(self.max_float_dd/max(self.peak_equity,1e-9)*100),
            'max_open_positions':self.max_open,
            'cluster_cycles':len(cyc),'cluster_PF':cpf,
            'profitable_cluster_cycles':profitable_cycles,
            'profit_harvest_rate_pct':float(profitable_cycles/max(len(cyc),1)*100),
            'avg_pnl_per_cluster':float(cyc.mean()) if len(cyc) else 0.0,
            'recluster_changes':self.recluster_changes,
            'recovery_added':self.recovery_added,
            'recovery_released':self.recovery_released,
            'recovery_cycles':self.recovery_cycles,
            'recovery_success_rate_pct':recovery_success,
            'recovery_debt_peak':self.recovery_debt_peak,
            'recovery_debt_pre_eod':self.recovery_debt_end,
            'recovery_pool_open_pre_eod':len(self.recovery_pool),
            'exit_counts':dict(self.exit_counts),'exit_pnl':dict(self.exit_pnl),
            'inventory_release':dict(self.inventory_release),
            'regime_counts':dict(self.regime_counts),'entries_by_regime':dict(self.entries_by_regime)
        }

def run_cell(catalog_path,mode,tf,experiment_id):
    catalog=ParquetDataCatalog(str(catalog_path)); inst=next((x for x in catalog.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'),None)
    if inst is None:raise SystemExit('XAUUSD missing')
    ticks=catalog.query_quote_ticks(identifiers=[inst.id.value])
    if not ticks:raise SystemExit('no Raw QuoteTicks')
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    eng.add_instrument(inst); eng.add_data(ticks)
    bt=BarType.from_str(f'{inst.id.value}-{tf}-MINUTE-BID-INTERNAL')
    st=ABStrategy(Cfg(instrument_id=inst.id,bar_type=bt,mode=mode)); eng.add_strategy(st); eng.run()
    obj={'verification_level':'NAUTILUS_RAW_BIDASK_AE_RECOVERY_POOL_CLUSTER_V3','engine':'NautilusTrader BacktestEngine','nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'raw_ticks':len(ticks),'ohlc_resample_used':False,'signal_bars':'Nautilus INTERNAL BID bars from Raw QuoteTicks','execution':'virtual micro-position ledger marked on raw Bid/Ask; raw spread included; native venue used for event engine/time ordering','tf_minutes':tf,**st.summary()}
    eng.dispose(); return obj

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--mode',choices=['BASE','CLUSTER'],required=True); ap.add_argument('--tf',type=int,default=1); ap.add_argument('--raw-bidask-only',action='store_true'); a=ap.parse_args()
    if not a.raw_bidask_only:raise SystemExit('raw-bidask-only mandatory')
    out=run_cell(Path(a.catalog),a.mode,a.tf,a.experiment_id)
    p=Path('results/ae-allweather-ab')/a.experiment_id; p.mkdir(parents=True,exist_ok=True)
    (p/f'{a.mode}_M{a.tf}.json').write_text(json.dumps(out,indent=2,default=str),encoding='utf-8')
    print(json.dumps(out,indent=2,default=str))

if __name__=='__main__':main()
