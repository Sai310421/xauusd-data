from __future__ import annotations
import argparse,json,math
from collections import deque
from decimal import Decimal
from pathlib import Path
import numpy as np,pandas as pd,nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar,QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,OrderSide,BookType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from research.rangehunter_m1_trendfollow_v2_1_nautilus_raw_bt import ensure_executable_l1

if not hasattr(ParquetDataCatalog,"query_quote_ticks"):
    def _query_quote_ticks(self,identifiers=None,start=None,end=None):
        return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
    ParquetDataCatalog.query_quote_ticks=_query_quote_ticks

INITIAL=1000.0
START=pd.Timestamp("2026-07-27T00:00:00Z")
END_EXCL=pd.Timestamp("2026-08-25T00:00:00Z")
TRADING_DAYS=21
PARAMS={
"A":dict(risk_pct=1.00,max_dd_pct=4.,max_daily_dd_pct=2.,max_spread_points=35.,max_lot=.10,bands_period=30,band_dev=2.2,rsi_period=14,adx_period=14,atr_period=14,max_adx=20.,min_z=2.,buy_rsi_max=32.,sell_rsi_min=68.,max_mid_slope_atr=.12,stop_atr=1.55,rr=1.15),
"B":dict(risk_pct=1.00,max_dd_pct=4.,max_daily_dd_pct=2.,max_spread_points=35.,max_lot=.10,fast_ema=20,slow_ema=80,adx_period=14,atr_period=14,breakout=40,adx_min=25.,min_atr_exp=1.05,min_breakout_atr=.10,stop_atr=1.65,rr=3.,trigger=.12,add=.025,max_adds=3),
"C":dict(risk_pct=.50,max_dd_pct=15.,max_daily_dd_pct=5.,max_spread_points=35.,max_lot=.05,atr_period=14,warmup_ticks=400,ewma_alpha=.025,entry_z=1.10,mom_alpha=.18,mom_min_points=.35,cooldown_sec=8,stop_atr=.60,rr=.85,reversal_z=2.50),
}
class Cfg(StrategyConfig,frozen=True):
    instrument_id:InstrumentId
    bar_type:BarType
    engine:str
class Base(Strategy):
    def __init__(self,cfg):
        super().__init__(cfg);self.p=PARAMS[cfg.engine];self.b=deque(maxlen=240);self.side=None;self.entry=None;self.qty=0.;self.stop_px=None;self.tp=None;self.entry_ts=None;self.realized=0.;self.peak=INITIAL;self.max_fdd=0.;self.max_exp=0.;self.max_age=0.;self.day=None;self.day_eq=INITIAL;self.n_entries=0;self.adds=0;self.last_add=None
    @staticmethod
    def f(x):return float(x.as_double()) if hasattr(x,"as_double") else float(x)
    def on_start(self):self.subscribe_quote_ticks(self.config.instrument_id);self.subscribe_bars(self.config.bar_type)
    def on_bar(self,bar:Bar):self.b.append(dict(o=self.f(bar.open),h=self.f(bar.high),l=self.f(bar.low),c=self.f(bar.close),ts=int(bar.ts_event)));self.signal_bar()
    def signal_bar(self):pass
    def arr(self,k):return np.array([x[k] for x in self.b],float)
    def ema(self,x,n):
        if len(x)<n:return None
        a=2/(n+1);v=float(np.mean(x[:n]))
        for z in x[n:]:v=a*float(z)+(1-a)*v
        return v
    def atr(self,n):
        if len(self.b)<n+1:return None
        x=list(self.b);tr=[max(x[i]["h"]-x[i]["l"],abs(x[i]["h"]-x[i-1]["c"]),abs(x[i]["l"]-x[i-1]["c"])) for i in range(1,len(x))];a=sum(tr[:n])/n
        for z in tr[n:]:a=(a*(n-1)+z)/n
        return float(a)
    def rsi(self,n):
        c=self.arr("c")
        if len(c)<n+1:return None
        d=np.diff(c);g=np.where(d>0,d,0.);l=np.where(d<0,-d,0.);ag=float(np.mean(g[:n]));al=float(np.mean(l[:n]))
        for i in range(n,len(g)):ag=(ag*(n-1)+g[i])/n;al=(al*(n-1)+l[i])/n
        return 100. if al==0 else 100-100/(1+ag/al)
    def adx(self,n):
        x=list(self.b)
        if len(x)<2*n+2:return None
        tr=[];pdv=[];mdv=[]
        for i in range(1,len(x)):
            up=x[i]["h"]-x[i-1]["h"];dn=x[i-1]["l"]-x[i]["l"];pdv.append(up if up>dn and up>0 else 0);mdv.append(dn if dn>up and dn>0 else 0);tr.append(max(x[i]["h"]-x[i]["l"],abs(x[i]["h"]-x[i-1]["c"]),abs(x[i]["l"]-x[i-1]["c"])))
        at=sum(tr[:n]);ps=sum(pdv[:n]);ms=sum(mdv[:n]);dx=[]
        for i in range(n,len(tr)):
            if i>n:at=at-at/n+tr[i];ps=ps-ps/n+pdv[i];ms=ms-ms/n+mdv[i]
            p=100*ps/at if at else 0;m=100*ms/at if at else 0;dx.append(100*abs(p-m)/(p+m) if p+m else 0)
        if len(dx)<n:return None
        a=sum(dx[:n])/n
        for z in dx[n:]:a=(a*(n-1)+z)/n
        return float(a)
    def eq(self,bid,ask):
        u=0.
        if self.side=="BUY":u=(bid-self.entry)*self.qty
        elif self.side=="SELL":u=(self.entry-ask)*self.qty
        return INITIAL+self.realized+u
    def guard(self,bid,ask,ts):
        eq=self.eq(bid,ask);self.peak=max(self.peak,eq);self.max_fdd=max(self.max_fdd,self.peak-eq);k=str(pd.Timestamp(ts,unit="ns",tz="UTC").date())
        if k!=self.day:self.day=k;self.day_eq=eq
        if 100*(self.peak-eq)/max(self.peak,1e-9)>=self.p["max_dd_pct"]:return False
        if 100*(self.day_eq-eq)/max(self.day_eq,1e-9)>=self.p["max_daily_dd_pct"]:return False
        return (ask-bid)/.01<=self.p["max_spread_points"]
    def size(self,entry,stop):
        cash=max(0.,INITIAL+self.realized)*self.p["risk_pct"]/100;dist=abs(entry-stop);q=min(self.p["max_lot"]*100,cash/dist if dist else 0)
        return max(0.,float(math.floor(q)))
    def enter(self,side,px,atr,ts):
        if self.side is not None:return False
        r=self.p["stop_atr"]*atr;stop=px-r if side=="BUY" else px+r;q=self.size(px,stop)
        if q<=0:return False
        instr=self.cache.instrument(self.config.instrument_id);os=OrderSide.BUY if side=="BUY" else OrderSide.SELL
        self.submit_order(self.order_factory.market(instrument_id=self.config.instrument_id,order_side=os,quantity=instr.make_qty(Decimal(str(q)))))
        self.side=side;self.entry=px;self.qty=q;self.stop_px=stop;self.tp=px+self.p["rr"]*r if side=="BUY" else px-self.p["rr"]*r;self.entry_ts=ts;self.n_entries+=1;self.max_exp=max(self.max_exp,q);self.last_add=px;return True
    def add(self,px):
        q=max(1.0,float(math.floor(min(self.p["max_lot"]*100,self.qty/max(1,self.adds+1)))));instr=self.cache.instrument(self.config.instrument_id);os=OrderSide.BUY if self.side=="BUY" else OrderSide.SELL
        self.submit_order(self.order_factory.market(instrument_id=self.config.instrument_id,order_side=os,quantity=instr.make_qty(Decimal(str(q)))))
        t=self.qty+q;self.entry=(self.entry*self.qty+px*q)/t;self.qty=t;self.adds+=1;self.last_add=px;self.max_exp=max(self.max_exp,self.qty)
    def exit(self,px,ts):
        pnl=(px-self.entry)*self.qty if self.side=="BUY" else (self.entry-px)*self.qty;self.realized+=pnl;self.max_age=max(self.max_age,(ts-self.entry_ts)/1e9);self.close_all_positions(self.config.instrument_id);self.side=None;self.entry=None;self.qty=0.;self.stop_px=None;self.tp=None;self.entry_ts=None;self.adds=0;self.last_add=None
    def manage(self,bid,ask,ts):
        if self.side is None:return
        px=bid if self.side=="BUY" else ask;hit=(px<=self.stop_px or px>=self.tp) if self.side=="BUY" else (px>=self.stop_px or px<=self.tp)
        if hit:self.exit(px,ts)
    def on_stop(self):self.close_all_positions(self.config.instrument_id)
class A(Base):
    def __init__(self,cfg):super().__init__(cfg);self.armed=None
    def signal_bar(self):
        p=self.p
        if self.side is not None or len(self.b)<60:return
        c=self.arr("c");w=c[-p["bands_period"]:];mid=float(np.mean(w));sd=float(np.std(w));rsi=self.rsi(p["rsi_period"]);adx=self.adx(p["adx_period"]);atr=self.atr(p["atr_period"])
        if sd<=0 or rsi is None or adx is None or atr is None or atr<=0 or adx>p["max_adx"]:return
        prev=float(np.mean(c[-p["bands_period"]-1:-1]))
        if abs(mid-prev)>atr*p["max_mid_slope_atr"]:return
        z=(c[-1]-mid)/sd
        if z<=-p["min_z"] and rsi<=p["buy_rsi_max"]:self.armed=("BUY",atr,int(self.b[-1]["ts"]))
        elif z>=p["min_z"] and rsi>=p["sell_rsi_min"]:self.armed=("SELL",atr,int(self.b[-1]["ts"]))
    def on_quote_tick(self,t):
        bid=self.f(t.bid_price);ask=self.f(t.ask_price);ts=int(t.ts_event);self.manage(bid,ask,ts)
        if self.armed and self.side is None and self.guard(bid,ask,ts):
            s,a,z=self.armed
            if ts-z<=60_000_000_000:self.enter(s,ask if s=="BUY" else bid,a,ts)
            self.armed=None
class B(Base):
    def __init__(self,cfg):super().__init__(cfg);self.armed=None
    def signal_bar(self):
        p=self.p
        if self.side is not None or len(self.b)<p["slow_ema"]+p["breakout"]+5:return
        c=self.arr("c");f1=self.ema(c,p["fast_ema"]);f2=self.ema(c[:-1],p["fast_ema"]);s1=self.ema(c,p["slow_ema"]);s2=self.ema(c[:-1],p["slow_ema"]);adx=self.adx(p["adx_period"]);a1=self.atr(p["atr_period"])
        if any(x is None for x in (f1,f2,s1,s2,adx,a1)) or a1<=0:return
        save=self.b;self.b=deque(list(self.b)[:-1],maxlen=240);a2=self.atr(p["atr_period"]);self.b=save
        if a2 is None or adx<p["adx_min"] or a1/a2<p["min_atr_exp"]:return
        prior=list(self.b)[:-1][-p["breakout"]:];hi=max(x["h"] for x in prior);lo=min(x["l"] for x in prior);last=c[-1]
        if f1>s1 and f1>f2 and s1>=s2 and last>hi+a1*p["min_breakout_atr"]:self.armed=("BUY",a1,int(self.b[-1]["ts"]))
        elif f1<s1 and f1<f2 and s1<=s2 and last<lo-a1*p["min_breakout_atr"]:self.armed=("SELL",a1,int(self.b[-1]["ts"]))
    def on_quote_tick(self,t):
        bid=self.f(t.bid_price);ask=self.f(t.ask_price);ts=int(t.ts_event);self.manage(bid,ask,ts)
        if self.armed and self.side is None and self.guard(bid,ask,ts):
            s,a,z=self.armed
            if ts-z<=60_000_000_000:self.enter(s,ask if s=="BUY" else bid,a,ts)
            self.armed=None
        if self.side is not None and self.adds<self.p["max_adds"] and self.guard(bid,ask,ts):
            px=bid if self.side=="BUY" else ask;need=self.p["trigger"] if self.adds==0 else self.p["add"];fav=px-self.last_add>=need if self.side=="BUY" else self.last_add-px>=need
            if fav:self.add(px)
class C(Base):
    def __init__(self,cfg):super().__init__(cfg);self.last=None;self.mean=0.;self.var=0.;self.mom=0.;self.n=0;self.last_entry=-10**30
    def on_quote_tick(self,t):
        bid=self.f(t.bid_price);ask=self.f(t.ask_price);ts=int(t.ts_event);self.manage(bid,ask,ts);mid=(ask+bid)/2
        if self.last is None:self.last=mid;return
        d=mid-self.last;self.last=mid;p=self.p;sd=math.sqrt(max(self.var,0.)) if self.n>20 else 0.;z=(d-self.mean)/sd if sd>.0001 else 0.;e=d-self.mean;self.mean+=p["ewma_alpha"]*e;self.var=(1-p["ewma_alpha"])*(self.var+p["ewma_alpha"]*e*e);self.mom=(1-p["mom_alpha"])*self.mom+p["mom_alpha"]*d;self.n+=1
        if self.side is not None or self.n<p["warmup_ticks"] or ts-self.last_entry<p["cooldown_sec"]*1e9 or not self.guard(bid,ask,ts):return
        a=self.atr(p["atr_period"])
        if a is None or a<=0:return
        mp=self.mom/.01;s=None
        if z>=p["entry_z"] and z<p["reversal_z"] and mp>=p["mom_min_points"]:s="BUY"
        elif z<=-p["entry_z"] and z>-p["reversal_z"] and mp<=-p["mom_min_points"]:s="SELL"
        if s and self.enter(s,ask if s=="BUY" else bid,a,ts):self.last_entry=ts

def positions(report):
    if report is None or report.empty:return []
    pc=next((c for c in report.columns if "pnl" in str(c).lower()),None);tc=next((c for c in report.columns if "closed" in str(c).lower()),None);out=[]
    for i,row in report.iterrows():
        try:ts=int(pd.Timestamp(row[tc]).value) if tc else i
        except:ts=i
        try:pnl=float(str(row[pc]).replace(",","").split()[0]) if pc else 0.
        except:pnl=0.
        out.append(dict(pnl=pnl,ts_closed=ts))
    return out
def metrics(t):
    a=np.array([x["pnl"] for x in t],float)
    if not len(a):return dict(N=0,WR_pct=0.,PF=0.,RF=None,Return_pct=0.,NetProfit=0.,MaxClosedDD_pct=0.,entries_per_day=0.)
    gp=float(a[a>0].sum());gl=abs(float(a[a<0].sum()));eq=peak=INITIAL;mdd=0.
    for x in a:eq+=x;peak=max(peak,eq);mdd=max(mdd,peak-eq)
    net=float(a.sum());return dict(N=len(a),WR_pct=float((a>0).mean()*100),PF=gp/gl if gl else None,RF=net/mdd if mdd else None,Return_pct=net/INITIAL*100,NetProfit=net,MaxClosedDD_pct=100*mdd/peak if peak else None,entries_per_day=len(a)/TRADING_DAYS)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--experiment-id",required=True);ap.add_argument("--raw-bidask-only",action="store_true");a=ap.parse_args()
    if not a.raw_bidask_only:raise SystemExit("raw-bidask-only mandatory")
    cp=Path(a.catalog);man=json.loads((cp/"catalog_manifest.json").read_text());cat=ParquetDataCatalog(str(cp));inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD");raw=cat.query_quote_ticks(identifiers=[inst.id.value]);ticks=[x for x in raw if START.value<=int(x.ts_event)<END_EXCL.value];ticks,replaced=ensure_executable_l1(ticks,1000)
    if not ticks:raise SystemExit("INVALID: no raw XAUUSD QuoteTicks in requested 21-business-day window")
    out=Path("results/ae-bt")/a.experiment_id;out.mkdir(parents=True,exist_ok=True);allm={}
    for code,cls in [("A",A),("B",B),("C",C)]:
        eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)));eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(INITIAL,USD)],default_leverage=Decimal("2000"));eng.add_instrument(inst);eng.add_data(ticks)
        bt=BarType.from_str(f"{inst.id.value}-1-MINUTE-BID-INTERNAL");st=cls(Cfg(instrument_id=inst.id,bar_type=bt,engine=code));eng.add_strategy(st);eng.run();pos=eng.trader.generate_positions_report();tr=positions(pos);m=metrics(tr);m.update(MaxFloatingDD_pct_internal=100*st.max_fdd/max(st.peak,1e-9),MaxFloatingDD_USD_internal=st.max_fdd,max_exposure_oz_internal=st.max_exp,max_basket_age_s_internal=st.max_age,submitted_entries_internal=st.n_entries)
        summary=dict(verification_level="NAUTILUS_BT_RAW_BIDASK_21D_PORT_V1",engine="NautilusTrader BacktestEngine",nautilus_version=getattr(nautilus_trader,"__version__","unknown"),engine_class={"A":"CORE_CANDIDATE","B":"CORE_CANDIDATE","C":"RANGE_CB_CANDIDATE"}[code],strategy=f"AMOS_{code}_Nautilus_Port_v1",data_kind="RAW_BIDASK QuoteTick prices/timestamps + synthetic nonzero L1 sizes",ohlc_resample_used=False,period=dict(start=str(START),end_exclusive=str(END_EXCL),trading_days=TRADING_DAYS),raw_tick_count=len(ticks),l1_size_replacements=replaced,account=dict(initial_usd=INITIAL,leverage=2000),params=PARAMS[code],metrics=m,wr5=dict(status="INVALID",present=["native_raw_bidask_spread","mark_to_market_floating_dd_internal"],missing=["broker_specific_round_trip_commission","probabilistic_slippage","execution_delay","swap_if_relevant","cashback_assumption","verified_margin_level_path","event_price_pitch_budget"]),limitations=["Port v1 uses one net position per engine; C MT5 source can allow multiple positions, so parity is incomplete.","Raw Bid/Ask prices/timestamps unchanged; zero quote sizes replaced by synthetic execution size 1000.","No broker-specific commission/slippage/delay/swap/cashback; WR5 remains INVALID."])
        d=out/code;d.mkdir(exist_ok=True);pd.DataFrame(tr).to_csv(d/"trades.csv",index=False);(d/"summary.json").write_text(json.dumps(summary,indent=2));allm[code]=summary;eng.dispose()
    (out/"summary_all.json").write_text(json.dumps(allm,indent=2));(out/"catalog_manifest.json").write_text(json.dumps(man,indent=2));print(json.dumps(allm,indent=2))
if __name__=="__main__":main()
