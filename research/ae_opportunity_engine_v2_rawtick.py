from __future__ import annotations
import argparse,json,math,statistics
from collections import deque,defaultdict
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
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
SIM=Venue('SIM')
ALPHAS=('TREND','MEANREV','BREAKOUT','SHOCK_FADE')
class Cfg(StrategyConfig,frozen=True):
    instrument_id:InstrumentId
    mode:str
    max_entries_per_day:int
    initial_balance:float=1000.0
    dd_limit:float=3.0
    tp:float=0.20
    sl:float=0.12
    max_hold_min:int=5
    min_ev_samples:int=5
class S(Strategy):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.cl=deque(maxlen=64);self.bucket=None;self.cur=None;self.bar_index=0
        self.active=False;self.side=0;self.entry=None;self.entry_bar=0;self.alpha=None
        self.real=0.;self.peak=cfg.initial_balance;self.maxdd=0.;self.gw=self.gl=0.;self.wins=self.cycles=0
        self.day=None;self.entries_today=0;self.total_entries=0;self.dd_stops=0;self.candidates=0;self.ev_rejects=0
        self.last_bid=self.last_ask=None
        self.pnl_hist={a:deque(maxlen=50) for a in ALPHAS};self.alpha_trades=defaultdict(int);self.alpha_wins=defaultdict(int);self.alpha_pnl=defaultdict(float);self.alpha_candidates=defaultdict(int)
    @staticmethod
    def f(x):return float(x.as_double()) if hasattr(x,'as_double') else float(x)
    def on_start(self):self.subscribe_quote_ticks(self.config.instrument_id)
    def mark(self,bid,ask):
        if not self.active:return 0.
        px=bid if self.side>0 else ask
        return (px-self.entry)*self.side
    def dd(self,bid,ask):
        eq=self.config.initial_balance+self.real+self.mark(bid,ask);self.peak=max(self.peak,eq)
        d=max(0.,(self.peak-eq)/max(self.peak,1e-9)*100);self.maxdd=max(self.maxdd,d);return d
    def close(self,bid,ask):
        if not self.active:return
        px=bid if self.side>0 else ask;p=(px-self.entry)*self.side
        self.real+=p;self.cycles+=1;self.alpha_trades[self.alpha]+=1;self.alpha_pnl[self.alpha]+=p;self.pnl_hist[self.alpha].append(p)
        if p>0:self.wins+=1;self.gw+=p;self.alpha_wins[self.alpha]+=1
        elif p<0:self.gl+=abs(p)
        self.active=False;self.side=0;self.entry=None;self.alpha=None
    def alpha_score(self,a):
        h=list(self.pnl_hist[a]);n=len(h)
        if n<self.config.min_ev_samples:return None
        m=sum(h)/n
        sd=statistics.pstdev(h) if n>1 else 0.0
        return m/(sd/math.sqrt(n)+0.02)
    def signals(self):
        x=list(self.cl)
        if len(x)<25:return []
        c=x[-1];r1=c-x[-2]
        sma20=sum(x[-20:])/20;prev_sma=sum(x[-21:-1])/20;slope=sma20-prev_sma
        sd20=statistics.pstdev(x[-20:]) or 1e-9
        hi=max(x[-21:-1]);lo=min(x[-21:-1])
        rets=[x[i]-x[i-1] for i in range(len(x)-19,len(x))]
        rsd=statistics.pstdev(rets) or 1e-9
        out=[]
        if c>sma20+0.04 and slope>0:out.append(('TREND',1,abs(c-sma20)/sd20))
        elif c<sma20-0.04 and slope<0:out.append(('TREND',-1,abs(c-sma20)/sd20))
        z=(c-sma20)/sd20
        if z>1.5:out.append(('MEANREV',-1,abs(z)))
        elif z<-1.5:out.append(('MEANREV',1,abs(z)))
        if c>hi+0.02:out.append(('BREAKOUT',1,(c-hi)/0.02))
        elif c<lo-0.02:out.append(('BREAKOUT',-1,(lo-c)/0.02))
        shock=r1/rsd
        if shock>2.5:out.append(('SHOCK_FADE',-1,abs(shock)))
        elif shock<-2.5:out.append(('SHOCK_FADE',1,abs(shock)))
        for a,_,_ in out:self.alpha_candidates[a]+=1
        self.candidates+=len(out)
        return out
    def choose(self,sigs):
        if not sigs:return None
        if self.config.mode=='ALL':return max(sigs,key=lambda q:q[2])
        warm=[q for q in sigs if len(self.pnl_hist[q[0]])<self.config.min_ev_samples]
        if warm:return min(warm,key=lambda q:len(self.pnl_hist[q[0]]))
        scored=[]
        for q in sigs:
            s=self.alpha_score(q[0])
            if s is not None and s>0:scored.append((s*q[2],q))
        if not scored:
            self.ev_rejects+=len(sigs);return None
        return max(scored,key=lambda z:z[0])[1]
    def finish_bar(self,bid,ask):
        if self.cur is None:return
        self.cl.append(self.cur);self.bar_index+=1;self.cur=None
        if self.active:
            px=bid if self.side>0 else ask;move=(px-self.entry)*self.side
            if move>=self.config.tp or move<=-self.config.sl or self.bar_index-self.entry_bar>=self.config.max_hold_min:
                self.close(bid,ask)
        if self.active:return
        if self.entries_today>=self.config.max_entries_per_day:return
        q=self.choose(self.signals())
        if q is None:return
        a,s,_=q;self.alpha=a;self.side=s;self.entry=ask if s>0 else bid;self.entry_bar=self.bar_index;self.active=True;self.entries_today+=1;self.total_entries+=1
    def on_quote_tick(self,t):
        bid=self.f(t.bid_price);ask=self.f(t.ask_price);mid=(bid+ask)/2;self.last_bid=bid;self.last_ask=ask
        sec=int(t.ts_event)//1_000_000_000;day=sec//86400
        if self.day is None or day!=self.day:self.day=day;self.entries_today=0
        b=sec//60
        if self.bucket is None:self.bucket=b
        elif b!=self.bucket:self.finish_bar(bid,ask);self.bucket=b
        self.cur=mid
        if self.dd(bid,ask)>=self.config.dd_limit:
            if self.active:self.close(bid,ask)
            self.dd_stops+=1
    def on_stop(self):
        if self.active and self.last_bid is not None:self.close(self.last_bid,self.last_ask)
    def summary(self):
        pf=self.gw/self.gl if self.gl else (math.inf if self.gw else 0.)
        alpha={a:{'trades':self.alpha_trades[a],'wins':self.alpha_wins[a],'pnl':self.alpha_pnl[a],'candidates':self.alpha_candidates[a],'score':self.alpha_score(a)} for a in ALPHAS}
        return dict(mode=self.config.mode,max_entries_per_day=self.config.max_entries_per_day,cycles=self.cycles,total_entries=self.total_entries,WR_pct=100*self.wins/max(1,self.cycles),PF=pf,realized_usd_0p01lot_equiv=self.real,return_pct_on_1000=self.real/10,max_DD_pct=self.maxdd,candidates=self.candidates,ev_rejects=self.ev_rejects,dd_stops=self.dd_stops,alpha=alpha)
def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',required=True);p.add_argument('--experiment-id',required=True);p.add_argument('--mode',choices=['ALL','AUCTION'],required=True);p.add_argument('--max-entries-per-day',type=int,required=True);p.add_argument('--raw-bidask-only',action='store_true');a=p.parse_args()
    if not a.raw_bidask_only:raise SystemExit('Raw BidAsk mandatory')
    cp=Path(a.catalog);man=json.loads((cp/'catalog_manifest.json').read_text());cat=ParquetDataCatalog(str(cp));inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD');ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)));eng.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));eng.add_instrument(inst);eng.add_data(ticks)
    s=S(Cfg(instrument_id=inst.id,mode=a.mode,max_entries_per_day=a.max_entries_per_day));eng.add_strategy(s);eng.run();r={**s.summary(),'raw_ticks':len(ticks),'nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'period_start':man.get('start'),'period_days':man.get('days'),'period_end_exclusive':man.get('end_exclusive'),'chronology':'RAW_BIDASK_MULTI_ALPHA_EDGE_AUCTION_V2'}
    out=Path('results/ae-bt')/a.experiment_id/'cells';out.mkdir(parents=True,exist_ok=True);(out/f'{a.mode}_N{a.max_entries_per_day}.json').write_text(json.dumps(r,indent=2));print(json.dumps(r));eng.dispose()
if __name__=='__main__':main()
