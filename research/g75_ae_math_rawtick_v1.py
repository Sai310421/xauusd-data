from __future__ import annotations
import argparse,json,math
from collections import deque
from decimal import Decimal
from pathlib import Path
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import Money,Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from research.g75_tsugi_causal_rawtick_v2 import G75TsugiV2,Cfg,TF_SEC
from research.ae_math_supervisor_v1_ref import (
    AEState,KellyConfig,RecoveryConfig,WassersteinConfig,MPCConfig,MPCAction,
    risk_constrained_kelly_fraction,brownian_first_passage_recovery_probability,
    probability_multiplier,wasserstein_linear_robust_edge,robust_multiplier,
    mpc_multiplier,margin_multiplier,
)

SIM=Venue('SIM')
LANES={'BASE','KELLY','RECOVERY','DRO','MPC','KELLY_RECOVERY','KRM','FULL'}

class G75AEMathV1(G75TsugiV2):
    """G75 causal Raw Bid/Ask core with supervisory size throttles only.

    Frozen event geometry/order remains inherited from G75TsugiV2.
    The supervisor never boosts above 1.0. Entry/Add prices are unchanged.
    v1 uses weighted internal layer PnL because G75TsugiV2 is itself an
    internal raw-tick state-machine benchmark rather than broker order sizing.
    """
    def __init__(self,cfg:Cfg,lane:str):
        super().__init__(cfg)
        self.lane=lane
        self.layer_weights=[]
        self.cycle_returns=deque(maxlen=128)
        self.tick_returns=deque(maxlen=256)
        self.last_mid_for_math=None
        self.pending_weight=1.0
        self.gate_counts={k:0 for k in ['kelly','recovery','dro','mpc','margin']}
        self.mult_sum=0.;self.mult_n=0;self.zero_throttles=0
        self.math_decisions=[]

    def mark(self,bid,ask):
        if not self.active:return 0.
        px=bid if self.side>0 else ask
        total=0.0
        for i,e in enumerate(self.entries):
            w=self.layer_weights[i] if i<len(self.layer_weights) else self.pending_weight
            total += (px-e)*self.side*w
        return total

    def close(self,bid,ask,reason):
        before=self.realized
        p=super().close(bid,ask,reason)
        delta=self.realized-before
        if abs(delta)>0:
            self.cycle_returns.append(delta/max(self.config.initial_balance,1e-9))
        self.layer_weights=[]
        return p

    def _math_multiplier(self,bid,ask):
        if self.lane=='BASE': return 1.0,{'binding':'base'}
        eq=self.equity(bid,ask);peak=max(self.account_peak,eq)
        debt=max(0.,self.debt);inv=float(len(self.entries))
        # Margin proxy: this benchmark has no native broker margin report.
        # Approximate only for throttling research; BROKER_REALITY_BT must replace it.
        exposure=max(1.0,sum(self.layer_weights) if self.layer_weights else 1.0)
        margin_level=max(0.,2000.0/exposure)
        # Recovery coordinate is normalized remaining distance from ruin(0) to BE(1).
        debt_scale=max(40.0,self.maxdebt,1.0)
        rc=max(0.,min(1.,1.0-debt/debt_scale))
        state=AEState(eq,peak,debt/debt_scale,inv/max(1,self.config.max_layers),margin_level,rc)

        rs=list(self.cycle_returns)
        k=1.0
        if self.lane in {'KELLY','KELLY_RECOVERY','KRM','FULL'} and len(rs)>=8:
            k=risk_constrained_kelly_fraction(rs,KellyConfig())

        rec=1.0;p=1.0
        if self.lane in {'RECOVERY','KELLY_RECOVERY','KRM','FULL'} and debt>0:
            xs=list(self.tick_returns)
            if len(xs)>=16:
                mu=sum(xs)/len(xs); var=sum((x-mu)**2 for x in xs)/max(1,len(xs)-1);sig=max(1e-6,var**.5)
                p=brownian_first_passage_recovery_probability(rc,0.,1.,mu,sig)
                rec=probability_multiplier(p,RecoveryConfig())

        dro=1.0;edge=0.0
        if self.lane in {'DRO','FULL'} and len(rs)>=8:
            scenarios=[[r] for r in rs[-64:]]
            edge=wasserstein_linear_robust_edge(scenarios,[1.0],WassersteinConfig().rho)
            dro=robust_multiplier(edge,WassersteinConfig())

        mm=1.0;act='NORMAL'
        if self.lane in {'MPC','KRM','FULL'}:
            recent=(sum(rs[-8:])/max(1,len(rs[-8:]))) if rs else 0.0
            actions=[
                MPCAction('STOP',0.0,0.0,-state.dd,-.05,-.10,0.0),
                MPCAction('HALF',0.5,recent*.5,0.0,-.02,-.02,.00005),
                MPCAction('NORMAL',1.0,recent,0.0,0.0,0.0,.0001),
            ]
            mm,act=mpc_multiplier(state,actions,MPCConfig())

        mg=margin_multiplier(margin_level)
        vals={'kelly':k,'recovery':rec,'dro':dro,'mpc':mm,'margin':mg}
        mult=max(0.,min(1.0,*vals.values()))
        binding=min(vals,key=vals.get)
        self.gate_counts[binding]+=1
        self.mult_sum+=mult;self.mult_n+=1
        if mult<=1e-12:self.zero_throttles+=1
        return mult,{'binding':binding,'k':k,'p':p,'recovery':rec,'robust_edge':edge,'dro':dro,'mpc':mm,'action':act,'margin':mg}

    def on_quote_tick(self,t:QuoteTick):
        bid=self.f(t.bid_price);ask=self.f(t.ask_price);mid=(bid+ask)/2
        if self.last_mid_for_math is not None and self.last_mid_for_math!=0:
            self.tick_returns.append((mid-self.last_mid_for_math)/self.last_mid_for_math)
        self.last_mid_for_math=mid
        before_len=len(self.entries)
        mult,detail=self._math_multiplier(bid,ask)
        self.pending_weight=mult
        super().on_quote_tick(t)
        after_len=len(self.entries)
        if after_len>before_len:
            self.layer_weights.extend([mult]*(after_len-before_len))
            self.math_decisions.append({'ts':int(t.ts_event),'lane':self.lane,'added_layers':after_len-before_len,'multiplier':mult,**detail})

    def summary(self):
        x=super().summary()
        x.update({
            'math_lane':self.lane,
            'math_avg_multiplier':self.mult_sum/max(1,self.mult_n),
            'math_zero_throttles':self.zero_throttles,
            'math_binding_counts':self.gate_counts,
            'math_decisions':len(self.math_decisions),
            'math_contract':'SUPERVISORY_SIZE_ONLY_NO_BOOST',
            'margin_metric':'PROXY_ONLY_NOT_BROKER_NATIVE',
        })
        return x

def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',required=True);p.add_argument('--experiment-id',required=True)
    p.add_argument('--tf',choices=TF_SEC,required=True);p.add_argument('--lane',choices=sorted(LANES),required=True)
    p.add_argument('--raw-bidask-only',action='store_true');a=p.parse_args()
    if not a.raw_bidask_only:raise SystemExit('Raw BidAsk mandatory')
    cp=Path(a.catalog);man=json.loads((cp/'catalog_manifest.json').read_text());cat=ParquetDataCatalog(str(cp))
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    try:ticks=cat.query_quote_ticks(identifiers=[inst.id.value])
    except AttributeError:ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'))
    eng.add_instrument(inst);eng.add_data(ticks)
    s=G75AEMathV1(Cfg(instrument_id=inst.id,tf_sec=TF_SEC[a.tf],variant='A'),a.lane);eng.add_strategy(s);eng.run()
    r={**s.summary(),'tf':a.tf,'raw_ticks':len(ticks),'period_start':man.get('start'),'period_days':man.get('days'),'period_end_exclusive':man.get('end_exclusive'),'nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'verification_level':'NAUTILUS_BT_RAW_BIDASK_MATH_AB','ohlc_resample_used':False}
    out=Path('results/ae-bt')/a.experiment_id/'cells';out.mkdir(parents=True,exist_ok=True)
    (out/f'{a.tf}_{a.lane}.json').write_text(json.dumps(r,indent=2));
    (out/f'{a.tf}_{a.lane}_decisions.json').write_text(json.dumps(s.math_decisions,indent=2));print(json.dumps(r));eng.dispose()
if __name__=='__main__':main()
