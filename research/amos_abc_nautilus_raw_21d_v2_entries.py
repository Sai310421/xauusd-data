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
from nautilus_trader.model.enums import AccountType,OmsType,BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from research.rangehunter_m1_trendfollow_v2_1_nautilus_raw_bt import ensure_executable_l1
from research.amos_abc_nautilus_raw_21d import (
    INITIAL,START,END_EXCL,TRADING_DAYS,Cfg,Base,positions,metrics
)

PARAMS_V2={
"A":dict(risk_pct=.60,max_dd_pct=4.,max_daily_dd_pct=2.,max_spread_points=35.,max_lot=.10,
         bands_period=24,rsi_period=14,adx_period=14,atr_period=14,max_adx=28.,
         min_z=1.45,buy_rsi_max=40.,sell_rsi_min=60.,max_mid_slope_atr=.18,
         stop_atr=1.35,rr=1.10,reject_wick_atr=.18,ema_stretch_atr=.85),
"B":dict(risk_pct=.50,max_dd_pct=4.,max_daily_dd_pct=2.,max_spread_points=35.,max_lot=.10,
         fast_ema=18,slow_ema=55,adx_period=14,atr_period=14,breakout=20,adx_min=20.,
         min_atr_exp=.98,min_breakout_atr=.03,stop_atr=1.50,rr=2.60,
         trigger=.16,add=.05,max_adds=2,pullback_atr=.30,impulse_atr=.75),
"C":dict(risk_pct=.15,max_dd_pct=15.,max_daily_dd_pct=4.,max_spread_points=35.,max_lot=.05,
         atr_period=14,warmup_ticks=250,ewma_alpha=.035,entry_z=.80,mom_alpha=.22,
         mom_min_points=.12,cooldown_sec=2,stop_atr=.48,rr=.90,reversal_z=3.0,
         fade_z=1.80,burst_points=.08),
}

class FamilyMixin:
    def _fam_init(self):
        self.family_counts={}
        self.last_family=None
    def enter_family(self,family,side,px,atr,ts):
        ok=self.enter(side,px,atr,ts)
        if ok:
            self.family_counts[family]=self.family_counts.get(family,0)+1
            self.last_family=family
        return ok

class A2(FamilyMixin,Base):
    def __init__(self,cfg):
        Base.__init__(self,cfg);self.p=PARAMS_V2["A"];self.armed=None;self._fam_init()
    def signal_bar(self):
        p=self.p
        if self.side is not None or len(self.b)<60:return
        c=self.arr("c");x=list(self.b);w=c[-p["bands_period"]:]
        mid=float(np.mean(w));sd=float(np.std(w));rsi=self.rsi(p["rsi_period"]);adx=self.adx(p["adx_period"]);atr=self.atr(p["atr_period"])
        if sd<=0 or rsi is None or adx is None or atr is None or atr<=0 or adx>p["max_adx"]:return
        prev=float(np.mean(c[-p["bands_period"]-1:-1]))
        if abs(mid-prev)>atr*p["max_mid_slope_atr"]:return
        z=(c[-1]-mid)/sd
        bar=x[-1];body=bar["c"]-bar["o"];lower_w=min(bar["o"],bar["c"])-bar["l"];upper_w=bar["h"]-max(bar["o"],bar["c"])
        ema20=self.ema(c,20)
        candidates=[]
        if z<=-p["min_z"] and rsi<=p["buy_rsi_max"]:candidates.append(("BUY","A_ZRSI"))
        if z>= p["min_z"] and rsi>=p["sell_rsi_min"]:candidates.append(("SELL","A_ZRSI"))
        if z<=-1.20 and lower_w>=p["reject_wick_atr"]*atr and body>0:candidates.append(("BUY","A_REJECT"))
        if z>= 1.20 and upper_w>=p["reject_wick_atr"]*atr and body<0:candidates.append(("SELL","A_REJECT"))
        if ema20 is not None:
            stretch=(bar["c"]-ema20)/atr
            if stretch<=-p["ema_stretch_atr"] and body>0 and rsi<48:candidates.append(("BUY","A_EMA_STRETCH"))
            if stretch>= p["ema_stretch_atr"] and body<0 and rsi>52:candidates.append(("SELL","A_EMA_STRETCH"))
        if candidates:
            side,fam=candidates[0];self.armed=(side,atr,int(bar["ts"]),fam)
    def on_quote_tick(self,t):
        bid=self.f(t.bid_price);ask=self.f(t.ask_price);ts=int(t.ts_event);self.manage(bid,ask,ts)
        if self.armed and self.side is None and self.guard(bid,ask,ts):
            s,a,z,fam=self.armed
            if ts-z<=60_000_000_000:self.enter_family(fam,s,ask if s=="BUY" else bid,a,ts)
            self.armed=None

class B2(FamilyMixin,Base):
    def __init__(self,cfg):
        Base.__init__(self,cfg);self.p=PARAMS_V2["B"];self.armed=None;self._fam_init()
    def signal_bar(self):
        p=self.p
        if self.side is not None or len(self.b)<p["slow_ema"]+p["breakout"]+5:return
        c=self.arr("c");x=list(self.b);bar=x[-1]
        f1=self.ema(c,p["fast_ema"]);f2=self.ema(c[:-1],p["fast_ema"]);s1=self.ema(c,p["slow_ema"]);s2=self.ema(c[:-1],p["slow_ema"])
        adx=self.adx(p["adx_period"]);a1=self.atr(p["atr_period"])
        if any(v is None for v in (f1,f2,s1,s2,adx,a1)) or a1<=0 or adx<p["adx_min"]:return
        save=self.b;self.b=deque(list(self.b)[:-1],maxlen=240);a2=self.atr(p["atr_period"]);self.b=save
        if a2 is None or a1/a2<p["min_atr_exp"]:return
        prior=x[:-1][-p["breakout"]:];hi=max(b["h"] for b in prior);lo=min(b["l"] for b in prior)
        trend_up=f1>s1 and f1>=f2 and s1>=s2;trend_dn=f1<s1 and f1<=f2 and s1<=s2
        cand=[]
        if trend_up and bar["c"]>hi+a1*p["min_breakout_atr"]:cand.append(("BUY","B_BREAKOUT"))
        if trend_dn and bar["c"]<lo-a1*p["min_breakout_atr"]:cand.append(("SELL","B_BREAKOUT"))
        if trend_up and bar["l"]<=f1+p["pullback_atr"]*a1 and bar["c"]>f1 and bar["c"]>bar["o"]:cand.append(("BUY","B_PULLBACK"))
        if trend_dn and bar["h"]>=f1-p["pullback_atr"]*a1 and bar["c"]<f1 and bar["c"]<bar["o"]:cand.append(("SELL","B_PULLBACK"))
        body=abs(bar["c"]-bar["o"])
        if trend_up and body>=p["impulse_atr"]*a1 and bar["c"]>bar["o"]:cand.append(("BUY","B_IMPULSE"))
        if trend_dn and body>=p["impulse_atr"]*a1 and bar["c"]<bar["o"]:cand.append(("SELL","B_IMPULSE"))
        if cand:
            side,fam=cand[0];self.armed=(side,a1,int(bar["ts"]),fam)
    def on_quote_tick(self,t):
        bid=self.f(t.bid_price);ask=self.f(t.ask_price);ts=int(t.ts_event);self.manage(bid,ask,ts)
        if self.armed and self.side is None and self.guard(bid,ask,ts):
            s,a,z,fam=self.armed
            if ts-z<=60_000_000_000:self.enter_family(fam,s,ask if s=="BUY" else bid,a,ts)
            self.armed=None
        if self.side is not None and self.adds<self.p["max_adds"] and self.guard(bid,ask,ts):
            px=bid if self.side=="BUY" else ask;need=self.p["trigger"] if self.adds==0 else self.p["add"]
            fav=px-self.last_add>=need if self.side=="BUY" else self.last_add-px>=need
            if fav:self.add(px)

class C2(FamilyMixin,Base):
    def __init__(self,cfg):
        Base.__init__(self,cfg);self.p=PARAMS_V2["C"];self.last=None;self.mean=0.;self.var=0.;self.mom=0.;self.n=0;self.last_entry=-10**30;self.deltas=deque(maxlen=4);self._fam_init()
    def on_quote_tick(self,t):
        bid=self.f(t.bid_price);ask=self.f(t.ask_price);ts=int(t.ts_event);self.manage(bid,ask,ts);mid=(ask+bid)/2
        if self.last is None:self.last=mid;return
        d=mid-self.last;self.last=mid;self.deltas.append(d);p=self.p
        sd=math.sqrt(max(self.var,0.)) if self.n>20 else 0.;z=(d-self.mean)/sd if sd>.0001 else 0.
        e=d-self.mean;self.mean+=p["ewma_alpha"]*e;self.var=(1-p["ewma_alpha"])*(self.var+p["ewma_alpha"]*e*e);self.mom=(1-p["mom_alpha"])*self.mom+p["mom_alpha"]*d;self.n+=1
        if self.side is not None or self.n<p["warmup_ticks"] or ts-self.last_entry<p["cooldown_sec"]*1e9 or not self.guard(bid,ask,ts):return
        a=self.atr(p["atr_period"])
        if a is None or a<=0:return
        mp=self.mom/.01;side=None;fam=None
        if z>=p["entry_z"] and z<p["reversal_z"] and mp>=p["mom_min_points"]:side,fam="BUY","C_IMPULSE"
        elif z<=-p["entry_z"] and z>-p["reversal_z"] and mp<=-p["mom_min_points"]:side,fam="SELL","C_IMPULSE"
        elif z>=p["fade_z"] and mp<0:side,fam="SELL","C_FADE"
        elif z<=-p["fade_z"] and mp>0:side,fam="BUY","C_FADE"
        elif len(self.deltas)>=3:
            s3=sum(list(self.deltas)[-3:])
            if all(v>0 for v in list(self.deltas)[-3:]) and s3/.01>=p["burst_points"]:side,fam="BUY","C_BURST"
            elif all(v<0 for v in list(self.deltas)[-3:]) and -s3/.01>=p["burst_points"]:side,fam="SELL","C_BURST"
        if side and self.enter_family(fam,side,ask if side=="BUY" else bid,a,ts):self.last_entry=ts

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--experiment-id",required=True);ap.add_argument("--engine",choices=("A","B","C"),required=True);ap.add_argument("--raw-bidask-only",action="store_true");a=ap.parse_args()
    if not a.raw_bidask_only:raise SystemExit("raw-bidask-only mandatory")
    cp=Path(a.catalog);man=json.loads((cp/"catalog_manifest.json").read_text());cat=ParquetDataCatalog(str(cp));inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD")
    raw=cat.query_quote_ticks(identifiers=[inst.id.value]);ticks=[x for x in raw if START.value<=int(x.ts_event)<END_EXCL.value];ticks,replaced=ensure_executable_l1(ticks,1000)
    if not ticks:raise SystemExit("INVALID: no raw XAUUSD QuoteTicks in requested 21-business-day window")
    code=a.engine;cls={"A":A2,"B":B2,"C":C2}[code];out=Path("results/ae-bt")/a.experiment_id;out.mkdir(parents=True,exist_ok=True)
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(INITIAL,USD)],default_leverage=Decimal("2000"))
    eng.add_instrument(inst);eng.add_data(ticks)
    bt=BarType.from_str(f"{inst.id.value}-1-MINUTE-BID-INTERNAL");st=cls(Cfg(instrument_id=inst.id,bar_type=bt,engine=code));eng.add_strategy(st);eng.run()
    pos=eng.trader.generate_positions_report();tr=positions(pos);m=metrics(tr)
    m.update(MaxFloatingDD_pct_internal=100*st.max_fdd/max(st.peak,1e-9),MaxFloatingDD_USD_internal=st.max_fdd,max_exposure_oz_internal=st.max_exp,max_basket_age_s_internal=st.max_age,submitted_entries_internal=st.n_entries)
    summary=dict(verification_level="NAUTILUS_BT_RAW_BIDASK_21D_ENTRY_EXPANSION_V2",engine="NautilusTrader BacktestEngine",nautilus_version=getattr(nautilus_trader,"__version__","unknown"),
      engine_class={"A":"CORE_CANDIDATE","B":"CORE_CANDIDATE","C":"RANGE_CB_CANDIDATE"}[code],strategy=f"AMOS_{code}_EntryExpansion_v2",data_kind="RAW_BIDASK QuoteTick prices/timestamps + synthetic nonzero L1 sizes",ohlc_resample_used=False,
      period=dict(start=str(START),end_exclusive=str(END_EXCL),trading_days=TRADING_DAYS),raw_tick_count=len(ticks),l1_size_replacements=replaced,account=dict(initial_usd=INITIAL,leverage=2000),
      params=PARAMS_V2[code],entry_family_counts=st.family_counts,metrics=m,
      research_inputs=dict(a17="entry/DD math architecture only; archive formulas not directly promoted",ml_builder="shadow meta-label/calibration/WFO role only; no direct orders",vision_ai="shadow visual-regime role only; no direct orders"),
      wr5=dict(status="INVALID",present=["native_raw_bidask_spread","mark_to_market_floating_dd_internal"],missing=["broker_specific_round_trip_commission","probabilistic_slippage","execution_delay","swap_if_relevant","cashback_assumption","verified_margin_level_path","event_price_pitch_budget"]),
      limitations=["Research IS iteration on the same 21-day window; not OOS or promotion evidence.","ML/Vision models are not used for direct execution in this v2.","No broker-specific commission/slippage/delay/swap/cashback; WR5 remains INVALID."])
    d=out/code;d.mkdir(exist_ok=True);pd.DataFrame(tr).to_csv(d/"trades.csv",index=False);(d/"summary.json").write_text(json.dumps(summary,indent=2));(out/"catalog_manifest.json").write_text(json.dumps(man,indent=2));print(json.dumps(summary,indent=2));eng.dispose()
if __name__=="__main__":main()
