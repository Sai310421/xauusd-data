from __future__ import annotations
import argparse, json, math
from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from g75_tsugi_causal_rawtick_v2 import G75TsugiV2, Cfg, TF_SEC

SIM=Venue('SIM')

@dataclass(frozen=True)
class EVStats:
    n:int
    wins:int
    sum_win:float
    sum_loss:float
    @property
    def p(self): return self.wins/max(1,self.n)
    @property
    def avg_win(self): return self.sum_win/max(1,self.wins)
    @property
    def avg_loss(self): return self.sum_loss/max(1,self.n-self.wins)
    @property
    def ev(self): return self.p*self.avg_win-(1-self.p)*self.avg_loss


def qbin(x, cuts):
    for i,c in enumerate(cuts):
        if x < c:return i
    return len(cuts)

class G75FeatureProbe(G75TsugiV2):
    """Frozen G75 core plus causal feature capture at entry; no gating."""
    def __init__(self,cfg):
        super().__init__(cfg); self.feature_at_open=None; self.trade_rows=[]
    def feature_key(self,side,mid):
        overshoot=max(0.0, (mid-(self.anchor+self.config.trigger))*side)
        vel=0.0
        if len(self.tick_mids)>=3: vel=(self.tick_mids[-1]-self.tick_mids[0])*side
        mom=0.0
        if len(self.prev_closes)>=2: mom=(self.prev_closes[-1]-self.prev_closes[-2])*side
        ds=self.direction_score(side,mid)
        return {
            'direction_score':int(ds),
            'overshoot':float(overshoot),
            'velocity':float(vel),
            'momentum':float(mom),
        }
    def close(self,bid,ask,reason):
        if not self.active:return 0.0
        feat=self.feature_at_open
        p=super().close(bid,ask,reason)
        if feat is not None:self.trade_rows.append({**feat,'pnl':float(p),'reason':reason})
        self.feature_at_open=None
        return p
    def on_quote_tick(self,t):
        was=self.active
        bid=self.f(t.bid_price); ask=self.f(t.ask_price); mid=(bid+ask)/2
        super().on_quote_tick(t)
        if (not was) and self.active and self.feature_at_open is None:
            self.feature_at_open=self.feature_key(self.side,mid)


def build_table(rows):
    agg=defaultdict(lambda:[0,0,0.0,0.0])
    for r in rows:
        key=(int(r['direction_score']), qbin(abs(r['velocity']),[0.005,0.015,0.03,0.06]), qbin(abs(r['momentum']),[0.02,0.05,0.10,0.20]))
        a=agg[key]; a[0]+=1
        if r['pnl']>0:a[1]+=1;a[2]+=r['pnl']
        elif r['pnl']<0:a[3]+=abs(r['pnl'])
    return {k:EVStats(*v) for k,v in agg.items()}

class G75EV(G75TsugiV2):
    """Frozen price core. EV layer only decides candidate acceptance and effective layer cap."""
    def __init__(self,cfg,table,mode='EV_GATE',ev_min=0.0,min_samples=4):
        super().__init__(cfg); self.table=table; self.ev_mode=mode; self.ev_min=ev_min; self.min_samples=min_samples
        self.ev_accepts=0; self.ev_rejects=0; self.ev_unknown=0; self.ev_sum=0.0; self.ev_obs=0; self.current_ev=0.0
    def _ev_lookup(self,side,mid):
        vel=0.0
        if len(self.tick_mids)>=3:vel=(self.tick_mids[-1]-self.tick_mids[0])*side
        mom=0.0
        if len(self.prev_closes)>=2:mom=(self.prev_closes[-1]-self.prev_closes[-2])*side
        key=(int(self.direction_score(side,mid)), qbin(abs(vel),[0.005,0.015,0.03,0.06]), qbin(abs(mom),[0.02,0.05,0.10,0.20]))
        s=self.table.get(key)
        if s is None or s.n<self.min_samples:return None
        return s.ev
    def cap(self):
        base=super().cap()
        if self.ev_mode!='EV_SIZER':return base
        # discrete expectancy sizing via layer budget; Frozen trigger/add/reversal unchanged.
        if self.current_ev<=0:return 1
        if self.current_ev<0.05:return max(1,int(round(base*0.3)))
        if self.current_ev<0.15:return max(1,int(round(base*0.6)))
        return base
    def on_quote_tick(self,t:QuoteTick):
        bid=self.f(t.bid_price);ask=self.f(t.ask_price);mid=(bid+ask)/2
        # replicate parent chronology until just before fresh-entry block, while preserving frozen core.
        self.last_bid=bid;self.last_ask=ask;self.tick_mids.append(mid)
        ns=int(t.ts_event); b=(ns//1_000_000_000)//self.config.tf_sec
        if self.bucket is None:self.bucket=b;self.anchor=mid
        elif b!=self.bucket:
            self.finish_bar();self.bucket=b;self.started_bucket=False
        self.bar_close_mid=mid;self.maybe_rearm(bid,ask)
        if self.mode=='COOLDOWN':self.dd(bid,ask);return
        rd=self.dd(bid,ask)
        if self.config.variant!='A':
            if rd>=self.config.hard_dd_pct:self.hard_reset(bid,ask);return
            if self.config.variant=='C' and self.active and self.mode not in ('RECOVERY','COOLDOWN') and rd>=self.config.lock_dd_pct:self.lock(bid,ask);return
            if self.mode not in ('RECOVERY','COOLDOWN'):
                if rd>=self.config.soft_dd_pct:
                    if self.mode!='REDUCED':self.soft+=1
                    self.mode='REDUCED'
                else:self.mode='NORMAL'
        if not self.active and not self.started_bucket and self.anchor is not None:
            side=1 if mid>=self.anchor+self.config.trigger else (-1 if mid<=self.anchor-self.config.trigger else 0)
            if side:
                self.trigger_candidates+=1
                needed=1 if self.config.variant=='A' else (3 if self.mode=='RECOVERY' else 2)
                if self.config.variant!='A' and self.direction_score(side,mid)<needed:
                    self.direction_rejects+=1;return
                ev=self._ev_lookup(side,mid)
                if ev is None:
                    self.ev_unknown+=1;return
                self.ev_sum+=ev;self.ev_obs+=1
                if ev<=self.ev_min:
                    self.ev_rejects+=1;return
                self.ev_accepts+=1;self.current_ev=ev
                entry=ask if side>0 else bid
                self.side=side;self.active=True;self.entries=[entry];self.last_add=entry
                self.extreme=(bid if side>0 else ask);self.started_bucket=True;self.maxlayers=max(self.maxlayers,1)
        if not self.active:return
        px=bid if self.side>0 else ask;self.extreme=max(self.extreme,px) if self.side>0 else min(self.extreme,px)
        cap=self.cap()
        while len(self.entries)<cap:
            target=self.last_add+self.side*self.config.add
            cross=px>=target if self.side>0 else px<=target
            if not cross:break
            # add requires positive EV inherited from entry state; no loss-triggered add.
            if self.current_ev<=self.ev_min:break
            if self.config.variant!='A' and len(self.tick_mids)>=3 and (self.tick_mids[-1]-self.tick_mids[0])*self.side<=0:break
            fill=ask if self.side>0 else bid;self.entries.append(fill);self.last_add=target;self.adds+=1;self.maxlayers=max(self.maxlayers,len(self.entries))
        self.dd(bid,ask)
    def summary(self):
        x=super().summary();x.update({'ev_mode':self.ev_mode,'ev_accepts':self.ev_accepts,'ev_rejects':self.ev_rejects,'ev_unknown':self.ev_unknown,'mean_predicted_ev':self.ev_sum/max(1,self.ev_obs)})
        return x


def run(inst,ticks,tf,kind,table=None):
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    eng.add_instrument(inst);eng.add_data(ticks)
    cfg=Cfg(instrument_id=inst.id,tf_sec=TF_SEC[tf],variant='A')
    if kind=='PURE':s=G75FeatureProbe(cfg)
    else:s=G75EV(cfg,table,mode=kind)
    eng.add_strategy(s);eng.run();res=s.summary();rows=getattr(s,'trade_rows',[]);eng.dispose();return res,rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);ap.add_argument('--tf',choices=TF_SEC,required=True);ap.add_argument('--variant',choices=['PURE','EV_GATE','EV_SIZER'],required=True);ap.add_argument('--raw-bidask-only',action='store_true');a=ap.parse_args()
    if not a.raw_bidask_only:raise SystemExit('Raw BidAsk mandatory')
    cat=ParquetDataCatalog(str(Path(a.catalog)));inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value]);
    if not ticks:raise SystemExit('no ticks')
    split=int(len(ticks)*0.40); train=ticks[:split]; test=ticks[split:]
    train_res,rows=run(inst,train,a.tf,'PURE'); table=build_table(rows)
    test_res,_=run(inst,test,a.tf,a.variant,table)
    out={**test_res,'tf':a.tf,'variant_eval':a.variant,'raw_ticks_total':len(ticks),'train_ticks':len(train),'test_ticks':len(test),'train_cycles':len(rows),'ev_cells':len(table),'split':'40% chronological train / 60% causal OOS test','frozen_core':{'trigger':0.12,'add':0.025,'reversal':0.20,'max_layers':10},'verification_level':'CAUSAL_RAW_BIDASK_G75_EV_V1','note':'EV layer changes candidate acceptance/layer budget only; price thresholds remain frozen.'}
    d=Path('results/g75-ev')/a.experiment_id;d.mkdir(parents=True,exist_ok=True);(d/f'{a.tf}_{a.variant}.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
