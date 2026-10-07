from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

if not hasattr(ParquetDataCatalog, "query_quote_ticks"):
    def _qqt(self, identifiers=None, start=None, end=None):
        return self.query(data_cls=QuoteTick, identifiers=identifiers, start=start, end=end)
    ParquetDataCatalog.query_quote_ticks = _qqt

P = {
    "RSI_Period":14,"RSI_Low":10.0,"RSI_High":90.0,"RSI_Mid":50.0,"RSI_ExtremeLookback":12,
    "StochK":5,"StochSlowing":3,"StochLow":15.0,"StochHigh":85.0,"StochMid":50.0,
    "MACD_Fast":12,"MACD_Slow":26,"RequireMACDSlope":True,"RequireMACDAcceleration":False,
    "ADX_Period":14,"ADX_Range":20.0,"ADX_Trend":25.0,
    "ATR_Fast":14,"ATR_Slow":100,"ATR_RangeRatio":0.80,
    "BB_Period":20,"BB_Dev":2.0,"RangeLookback":24,"MinRangeATR":1.50,"SweepATRBuffer":0.15,
    "CRTThreshold":65,"CRTTimeoutMinutes":30,"TrendERMin":0.35,
    "BaseTrendLot":0.01,"MaxLegs":2,
    "MaxAccountDDPercent":10.0,"MaxBasketLossUSD":500.0,
    "EnableBasketTP":True,"BasketTakeProfitUSD":7.0,
    "EnableRecoveryExit":True,"RecoveryArmLossUSD":15.0,"RecoveryExitProfitUSD":0.50,"RecoveryMinHoldSeconds":30,
    "CloseOnTrendEnd":True,
    "DirectEntryEnabled":True,"DirectADXMin":18.0,"DirectERMin":0.15,
    "DirectCooldownMinutes":360,"DirectRequireMACDSlope":False,
    "MaxBasketHoldMinutes":180,
    "RangeEntryEnabled":True,"RangeEntryBandFrac":0.10,"RangeEntryADXMax":22.0,
    "RangeEntryCooldownMinutes":240,"RangeEntryRequireStochReclaim":True,
    "DDGovernorEnabled":True,"DDSoftCutPercent":9.50,"DDLegFreezePercent":99.0,
    "EnableG75Pursuit":True,"PursuitLot":0.01,"PursuitTriggerDistance":0.12,
    "PursuitAddDistance":0.025,"PursuitReversalDistance":0.20,"MaxPursuitLayers":10,
}

SIGNAL = {}

def fnum(x):
    try:
        return float(x.as_double())
    except Exception:
        return float(x)

def atr(df,n):
    pc=df.close.shift(1)
    tr=pd.concat([(df.high-df.low).abs(),(df.high-pc).abs(),(df.low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def rsi(s,n):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    return (100-100/(1+up/dn.replace(0,np.nan))).fillna(50)

def adx(df,n):
    up=df.high.diff(); dn=-df.low.diff()
    plus=np.where((up>dn)&(up>0),up,0.0); minus=np.where((dn>up)&(dn>0),dn,0.0)
    a=atr(df,n)
    pdi=100*pd.Series(plus,index=df.index).ewm(alpha=1/n,adjust=False,min_periods=n).mean()/a
    mdi=100*pd.Series(minus,index=df.index).ewm(alpha=1/n,adjust=False,min_periods=n).mean()/a
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False,min_periods=n).mean().fillna(0)

def resample_ohlc(m1,rule,minutes):
    x=m1.set_index("datetime")[["open","high","low","close","volume"]]
    y=x.resample(rule,label="left",closed="left").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    y.index=y.index+pd.Timedelta(minutes=minutes)
    return y

def build_signal_tables(ticks):
    rows=[]
    for t in ticks:
        ts=pd.Timestamp(int(t.ts_event),unit="ns",tz="UTC")
        bid=fnum(t.bid_price); ask=fnum(t.ask_price); mid=(bid+ask)/2.0
        rows.append((ts,mid))
    df=pd.DataFrame(rows,columns=["datetime","mid"]).drop_duplicates("datetime").sort_values("datetime")
    df["bucket"]=df.datetime.dt.floor("1min")
    m1=df.groupby("bucket",sort=True).agg(open=("mid","first"),high=("mid","max"),low=("mid","min"),close=("mid","last"),volume=("mid","size")).reset_index().rename(columns={"bucket":"datetime"})
    m1["datetime"]=m1["datetime"]+pd.Timedelta(minutes=1)
    m1=m1.set_index("datetime")
    m5=resample_ohlc(m1.reset_index().assign(datetime=lambda x:x.datetime-pd.Timedelta(minutes=1)),"5min",5)
    m15=resample_ohlc(m1.reset_index().assign(datetime=lambda x:x.datetime-pd.Timedelta(minutes=1)),"15min",15)

    m15["rsi"]=rsi(m15.close,P["RSI_Period"])
    ef=m15.close.ewm(span=P["MACD_Fast"],adjust=False).mean(); es=m15.close.ewm(span=P["MACD_Slow"],adjust=False).mean()
    m15["macd"]=ef-es; m15["macd_slope"]=m15.macd-m15.macd.shift(1); m15["macd_accel"]=m15.macd_slope-m15.macd_slope.shift(1)
    m15["rsi_reclaim_up"]=(m15.rsi.shift(1)<=P["RSI_Low"])&(m15.rsi>P["RSI_Low"])
    m15["rsi_reclaim_dn"]=(m15.rsi.shift(1)>=P["RSI_High"])&(m15.rsi<P["RSI_High"])

    m5["atr14"]=atr(m5,P["ATR_Fast"]); m5["atr100"]=atr(m5,P["ATR_Slow"]); m5["atr_ratio"]=m5.atr14/m5.atr100
    m5["adx"]=adx(m5,P["ADX_Period"])
    mid=m5.close.rolling(P["BB_Period"]).mean(); sd=m5.close.rolling(P["BB_Period"]).std(ddof=0)
    m5["bb_width"]=(2*P["BB_Dev"]*sd)/mid.abs()
    m5["bb_pct"]=m5.bb_width.rolling(50).apply(lambda a:np.mean(a[:-1]<a[-1]) if len(a)==50 else np.nan,raw=True)
    m5["adx_falling"]=(m5.adx<m5.adx.shift(1))&(m5.adx.shift(1)<m5.adx.shift(2)); m5["adx_expanding"]=m5.adx>m5.adx.shift(1)
    m5["range_state"]=(m5.bb_pct<=0.20)&(m5.adx<P["ADX_Range"])&m5.adx_falling&(m5.atr_ratio<P["ATR_RangeRatio"])
    m5["range_hi"]=m5.high.rolling(P["RangeLookback"]).max(); m5["range_lo"]=m5.low.rolling(P["RangeLookback"]).min(); m5["range_eq"]=(m5.range_hi+m5.range_lo)/2
    m5["range_width_ok"]=(m5.range_hi-m5.range_lo)>=P["MinRangeATR"]*m5.atr14
    direction=(m5.close-m5.close.shift(20)).abs(); noise=m5.close.diff().abs().rolling(20).sum(); m5["er20"]=(direction/noise.replace(0,np.nan)).fillna(0)

    ll=m1.low.rolling(P["StochK"]).min(); hh=m1.high.rolling(P["StochK"]).max()
    rawk=100*(m1.close-ll)/(hh-ll).replace(0,np.nan); m1["stoch"]=rawk.rolling(P["StochSlowing"]).mean()
    m1["stoch_reclaim_up"]=(m1.stoch.shift(1)<=P["StochLow"])&(m1.stoch>P["StochLow"])
    m1["stoch_reclaim_dn"]=(m1.stoch.shift(1)>=P["StochHigh"])&(m1.stoch<P["StochHigh"])
    m1["stoch50_up"]=(m1.stoch.shift(1)<P["StochMid"])&(m1.stoch>=P["StochMid"])
    m1["stoch50_dn"]=(m1.stoch.shift(1)>P["StochMid"])&(m1.stoch<=P["StochMid"])
    m1["m1_break_up"]=m1.close>m1.high.shift(1); m1["m1_break_dn"]=m1.close<m1.low.shift(1)
    m5a=m5.reindex(m1.index,method="ffill"); m15a=m15.reindex(m1.index,method="ffill")
    return m1,m5,m15,m5a,m15a,len(set(m1.index.date))

def macd_aligned(r15,d):
    if pd.isna(r15.macd): return False
    if d>0 and r15.macd<=0:return False
    if d<0 and r15.macd>=0:return False
    if P["RequireMACDSlope"]:
        if d>0 and r15.macd_slope<=0:return False
        if d<0 and r15.macd_slope>=0:return False
    if P["RequireMACDAcceleration"]:
        if d>0 and r15.macd_accel<=0:return False
        if d<0 and r15.macd_accel>=0:return False
    return True

def detect_m5_features(m5,i,d):
    if i is None or i<25:return dict(dis=False,cisd=False,br=False,fvg=False,ifvg=False,bpr=False)
    cur=m5.iloc[i]; avg=np.mean(np.abs(m5.close.iloc[i-20:i]-m5.open.iloc[i-20:i])); body=abs(cur.close-cur.open); brange=cur.high-cur.low
    dis=((d>0 and cur.close>cur.open) or (d<0 and cur.close<cur.open)) and body>1.8*avg and brange>1.2*cur.atr14
    cisd=False
    for k in range(i-1,max(-1,i-9),-1):
        bar=m5.iloc[k]
        if d>0 and bar.close<bar.open and cur.close>bar.open:cisd=True;break
        if d<0 and bar.close>bar.open and cur.close<bar.open:cisd=True;break
    prev=m5.iloc[i-3:i]; breaker=(cur.close>prev.high.max()) if d>0 else (cur.close<prev.low.min())
    fvg=(cur.low>m5.iloc[i-2].high) if d>0 else (cur.high<m5.iloc[i-2].low)
    ifvg=False
    for lag in range(3,9):
        a=i-lag;b=a-2
        if b<0:continue
        if d>0 and m5.iloc[a].high<m5.iloc[b].low and cur.close>m5.iloc[b].low:ifvg=True;break
        if d<0 and m5.iloc[a].low>m5.iloc[b].high and cur.close<m5.iloc[b].high:ifvg=True;break
    bulls=[];bears=[]
    for k in range(max(2,i-11),i+1):
        if m5.iloc[k].low>m5.iloc[k-2].high:bulls.append((m5.iloc[k-2].high,m5.iloc[k].low))
        if m5.iloc[k].high<m5.iloc[k-2].low:bears.append((m5.iloc[k].high,m5.iloc[k-2].low))
    bpr=any(max(a[0],b[0])<min(a[1],b[1]) for a in bulls for b in bears)
    return dict(dis=dis,cisd=cisd,br=breaker,fvg=fvg,ifvg=ifvg,bpr=bpr)

class HybridCfg(StrategyConfig, frozen=True):
    instrument_id: object
    initial_balance: float=1000.0
    pursuit_enabled: bool=True
    point_size: float=0.01

class HybridRawTickG75(Strategy):
    def __init__(self,config:HybridCfg):
        super().__init__(config)
        self.hybrid_state="SEARCH";self.range_locked=False;self.rh=np.nan;self.rl=np.nan;self.req=np.nan
        self.crt_dir=0;self.crt_score=0;self.crt_start=None;self.cycle_dir=0;self.legs=[False]*5
        self.parent=[];self.pursuit=[];self.balance=config.initial_balance;self.peak=config.initial_balance;self.maxdd=0.0;self.halted=False
        self.last_bid=None;self.last_ask=None;self.tick_i=0;self.current_dd=0.0
        self.last_signal_idx=-1;self.last_cycle_time=None;self.basket_open_time=None
        self.recovery_armed=False;self.recovery_armed_time=None
        self.entry_cycles=0;self.entry_days=set();self.parent_orders=0;self.range_orders=0;self.leg_counts=[0]*5
        self.basket_results=[];self.exit_counts=defaultdict(int);self.risk_reason=""
        self.pursuit_triggered=False;self.pursuit_done=False;self.pursuit_dir=0;self.pursuit_anchor=0.0;self.pursuit_last_add=0.0;self.pursuit_best=0.0
        self.pursuit_layers_opened=0;self.pursuit_reversal_exits=0;self.pursuit_parent_exits=0;self.pursuit_realized=0.0;self.cycle_pursuit_realized=0.0
        self.spread_sum=0.0;self.spread_max=0.0;self.spread_n=0;self.equity_samples=[]
        self.pursuit_spread_blocked=0

    def on_start(self): self.subscribe_quote_ticks(self.config.instrument_id)

    def mark(self,p,bid,ask):
        px=bid if p["dir"]>0 else ask
        return (px-p["entry"])*p["dir"]*100.0*p["lot"]

    def parent_float(self,bid,ask):return sum(self.mark(p,bid,ask) for p in self.parent)
    def pursuit_float(self,bid,ask):return sum(self.mark(p,bid,ask) for p in self.pursuit)
    def account_float(self,bid,ask):return self.parent_float(bid,ask)+self.pursuit_float(bid,ask)

    def open_parent(self,d,lot,kind,bid,ask,t):
        if self.halted or lot<=0:return False
        entry=ask if d>0 else bid
        was_empty=not self.parent
        self.parent.append({"dir":d,"lot":lot,"entry":entry,"kind":kind})
        self.parent_orders+=1
        if was_empty:
            self.entry_cycles+=1;self.entry_days.add(t.date());self.last_cycle_time=t;self.basket_open_time=t
            self.pursuit.clear();self.pursuit_triggered=False;self.pursuit_done=False;self.pursuit_dir=d
            self.pursuit_anchor=entry;self.pursuit_last_add=0.0;self.pursuit_best=(bid if d>0 else ask);self.cycle_pursuit_realized=0.0
        if kind=="RANGE":self.range_orders+=1
        return True

    def open_leg(self,d,k,bid,ask,t):
        if k<0 or k>=P["MaxLegs"] or self.legs[k]:return False
        if P["DDGovernorEnabled"] and k>=1 and self.current_dd>=P["DDLegFreezePercent"]:return False
        if self.open_parent(d,P["BaseTrendLot"],f"Leg{k+1}",bid,ask,t):
            self.legs[k]=True;self.leg_counts[k]+=1;return True
        return False

    def close_pursuit(self,bid,ask,reason):
        if not self.pursuit:return 0.0
        pnl=sum(self.mark(p,bid,ask) for p in self.pursuit);self.balance+=pnl;self.pursuit_realized+=pnl;self.cycle_pursuit_realized+=pnl;self.pursuit.clear()
        if reason=="REVERSAL":self.pursuit_reversal_exits+=1
        else:self.pursuit_parent_exits+=1
        return pnl

    def close_basket(self,bid,ask,reason):
        if not self.parent:return 0.0
        if self.pursuit:self.close_pursuit(bid,ask,"PARENT_EXIT")
        pnl=sum(self.mark(p,bid,ask) for p in self.parent);self.balance+=pnl;self.parent.clear()
        self.basket_results.append(pnl+self.cycle_pursuit_realized);self.exit_counts[reason]+=1
        self.range_locked=False;self.rh=self.rl=self.req=np.nan;self.crt_dir=0;self.crt_score=0;self.crt_start=None;self.cycle_dir=0;self.legs=[False]*5
        self.recovery_armed=False;self.recovery_armed_time=None;self.basket_open_time=None
        self.pursuit_triggered=False;self.pursuit_done=False;self.pursuit_dir=0;self.pursuit_anchor=0.0;self.pursuit_last_add=0.0;self.pursuit_best=0.0;self.cycle_pursuit_realized=0.0
        self.hybrid_state="SEARCH";return pnl

    def process_pursuit(self,bid,ask,t):
        # Exact parity with G75_追撃Pursuit_Overlay_ParentOnly_v1_00.mq5.
        # Trigger/add checks use executable px (Ask for long, Bid for short).
        # Best/reversal checks use liquidation mark (Bid for long, Ask for short).
        if (not self.config.pursuit_enabled) or self.pursuit_done or not self.parent or self.pursuit_dir==0:
            return
        d=self.pursuit_dir
        px=ask if d>0 else bid
        mark=bid if d>0 else ask
        spread_points=(ask-bid)/max(self.config.point_size,1e-12)

        if not self.pursuit_triggered:
            trigger=(px-self.pursuit_anchor>=P["PursuitTriggerDistance"]) if d>0 else (self.pursuit_anchor-px>=P["PursuitTriggerDistance"])
            if not trigger:
                return
            if spread_points>100.0:
                self.pursuit_spread_blocked+=1
                return
            self.pursuit.append({"dir":d,"lot":P["PursuitLot"],"entry":px,"kind":"G75"})
            self.pursuit_layers_opened+=1
            self.pursuit_triggered=True
            self.pursuit_last_add=px
            self.pursuit_best=mark
            # MT5 source returns immediately after first trigger layer.
            return

        self.pursuit_best=max(self.pursuit_best,mark) if d>0 else min(self.pursuit_best,mark)

        reversal_hit=(self.pursuit_best-bid>=P["PursuitReversalDistance"]) if d>0 else (ask-self.pursuit_best>=P["PursuitReversalDistance"])
        if reversal_hit:
            self.close_pursuit(bid,ask,"REVERSAL")
            self.pursuit_done=True
            return

        # Exact add semantics: multiple crossed levels may be filled on one tick.
        while len(self.pursuit)<P["MaxPursuitLayers"]:
            required=self.pursuit_last_add+d*P["PursuitAddDistance"]
            crossed=(px>=required) if d>0 else (px<=required)
            if not crossed:
                break
            if spread_points>100.0:
                self.pursuit_spread_blocked+=1
                break
            self.pursuit.append({"dir":d,"lot":P["PursuitLot"],"entry":px,"kind":"G75"})
            self.pursuit_layers_opened+=1
            self.pursuit_last_add=required

    def signal_step(self,i,bid,ask,t):
        m1=SIGNAL["m1"];m5=SIGNAL["m5"];m15=SIGNAL["m15"];m5a=SIGNAL["m5a"];m15a=SIGNAL["m15a"]
        if i<0 or i>=len(m1):return
        ts=m1.index[i];row1=m1.iloc[i];r5=m5a.iloc[i];r15=m15a.iloc[i]
        if pd.isna(r5.adx) or pd.isna(r15.rsi):return
        price=(bid+ask)/2.0
        dhtf=0
        if r15.rsi>P["RSI_Mid"] and macd_aligned(r15,1):dhtf=1
        elif r15.rsi<P["RSI_Mid"] and macd_aligned(r15,-1):dhtf=-1
        ddirect=0
        if r15.rsi>P["RSI_Mid"] and r15.macd>0:ddirect=1
        elif r15.rsi<P["RSI_Mid"] and r15.macd<0:ddirect=-1
        if ddirect and P["DirectRequireMACDSlope"]:
            if (ddirect>0 and r15.macd_slope<=0) or (ddirect<0 and r15.macd_slope>=0):ddirect=0
        mt=m5.index[m5.index<=ts][-1] if len(m5.index[m5.index<=ts]) else None
        mi=m5.index.get_loc(mt) if mt is not None else None
        prev_adx=m5.adx.iloc[mi-1] if mi is not None and mi>0 else np.nan
        expansion=lambda d:d!=0 and r5.adx>P["ADX_Trend"] and r5.adx>prev_adx and r5.atr_ratio>1.0
        trend_end=r5.adx<P["ADX_Range"] and r5.er20<0.25 and r5.atr_ratio<P["ATR_RangeRatio"]

        if self.hybrid_state=="SEARCH":
            cd=(self.last_cycle_time is None or (t-self.last_cycle_time).total_seconds()>=P["DirectCooldownMinutes"]*60)
            trig=False
            if ddirect>0:trig=bool(row1.stoch_reclaim_up) or bool(row1.stoch50_up) or bool(row1.m1_break_up)
            elif ddirect<0:trig=bool(row1.stoch_reclaim_dn) or bool(row1.stoch50_dn) or bool(row1.m1_break_dn)
            if P["DirectEntryEnabled"] and ddirect and cd and r5.adx>=P["DirectADXMin"] and r5.er20>=P["DirectERMin"] and trig and not bool(r5.range_state):
                self.cycle_dir=ddirect;self.crt_dir=ddirect;self.legs=[False]*5;self.open_leg(ddirect,0,bid,ask,t)
                if self.legs[0]:self.hybrid_state="TREND";return
            if bool(r5.range_state) and bool(r5.range_width_ok):
                self.rh=float(r5.range_hi);self.rl=float(r5.range_lo);self.req=float(r5.range_eq);self.range_locked=True;self.hybrid_state="RANGE";return
            if dhtf and expansion(dhtf) and r5.er20>=P["TrendERMin"]:
                self.cycle_dir=dhtf;self.crt_dir=dhtf;self.legs=[False]*5;self.hybrid_state="TREND"
        elif self.hybrid_state=="RANGE":
            cd=(self.last_cycle_time is None or (t-self.last_cycle_time).total_seconds()>=P["RangeEntryCooldownMinutes"]*60)
            if P["RangeEntryEnabled"] and self.range_locked and not self.parent and cd and r5.adx<=P["RangeEntryADXMax"]:
                half=max((self.rh-self.rl)/2.0,1e-9);z=(price-self.req)/half
                near_low=z<=-(1.0-P["RangeEntryBandFrac"]);near_high=z>=(1.0-P["RangeEntryBandFrac"])
                if near_low and bool(row1.stoch_reclaim_up):self.open_parent(1,P["BaseTrendLot"],"RANGE",bid,ask,t)
                elif near_high and bool(row1.stoch_reclaim_dn):self.open_parent(-1,P["BaseTrendLot"],"RANGE",bid,ask,t)
            sweep=0
            if self.range_locked:
                b=float(r5.atr14)*P["SweepATRBuffer"]
                if r5.low<self.rl-b and r5.close>self.rl:sweep=1
                elif r5.high>self.rh+b and r5.close<self.rh:sweep=-1
            if sweep:self.crt_dir=sweep;self.crt_start=t;self.crt_score=25;self.hybrid_state="TRANSITION"
        elif self.hybrid_state=="TRANSITION":
            if self.crt_start is None or (t-self.crt_start).total_seconds()>P["CRTTimeoutMinutes"]*60:
                self.crt_dir=0;self.crt_score=0;self.range_locked=False;self.hybrid_state="SEARCH";return
            z=detect_m5_features(m5,mi,self.crt_dir)
            self.crt_score=25+20*z["dis"]+15*z["cisd"]+10*z["br"]+10*z["ifvg"]+5*z["bpr"]+5*z["fvg"]+5*bool(r5.adx_expanding)+5*(r5.atr_ratio>1.0)
            if (self.crt_dir>0 and bool(r15.rsi_reclaim_up)) or (self.crt_dir<0 and bool(r15.rsi_reclaim_dn)):self.crt_score+=5
            eqbreak=(r5.close>self.req) if self.crt_dir>0 else (r5.close<self.req)
            if self.crt_score>=P["CRTThreshold"] and eqbreak:self.cycle_dir=self.crt_dir;self.legs=[False]*5;self.hybrid_state="EXPANSION"
        elif self.hybrid_state=="EXPANSION":
            if self.crt_dir==0:return
            if not self.legs[0]:
                initial=((self.crt_dir>0 and (bool(r15.rsi_reclaim_up) or bool(row1.stoch_reclaim_up))) or (self.crt_dir<0 and (bool(r15.rsi_reclaim_dn) or bool(row1.stoch_reclaim_dn))) or (macd_aligned(r15,self.crt_dir) and ((bool(row1.m1_break_up) if self.crt_dir>0 else bool(row1.m1_break_dn)) or expansion(self.crt_dir))))
                if initial or self.crt_score>=P["CRTThreshold"]:self.open_leg(self.crt_dir,0,bid,ask,t)
                if not self.legs[0]:return
            if expansion(self.crt_dir):self.hybrid_state="TREND"
        elif self.hybrid_state=="TREND":
            d=self.cycle_dir
            if d:
                initial=((d>0 and (bool(r15.rsi_reclaim_up) or bool(row1.stoch_reclaim_up))) or (d<0 and (bool(r15.rsi_reclaim_dn) or bool(row1.stoch_reclaim_dn))) or (macd_aligned(r15,d) and ((bool(row1.m1_break_up) if d>0 else bool(row1.m1_break_dn)) or expansion(d))))
                if not self.legs[0] and initial:self.open_leg(d,0,bid,ask,t)
                if not self.legs[1] and ((d>0 and bool(row1.stoch_reclaim_up)) or (d<0 and bool(row1.stoch_reclaim_dn))):self.open_leg(d,1,bid,ask,t)
            if trend_end:
                if P["CloseOnTrendEnd"] and self.parent:self.close_basket(bid,ask,"TREND_END")
                else:self.crt_dir=0;self.crt_score=0;self.range_locked=False;self.cycle_dir=0;self.legs=[False]*5;self.hybrid_state="SEARCH"

    def on_quote_tick(self,tick:QuoteTick):
        self.tick_i+=1;bid=fnum(tick.bid_price);ask=fnum(tick.ask_price);self.last_bid=bid;self.last_ask=ask
        t=pd.Timestamp(int(tick.ts_event),unit="ns",tz="UTC")
        sp=ask-bid;self.spread_sum+=sp;self.spread_max=max(self.spread_max,sp);self.spread_n+=1

        self.process_pursuit(bid,ask,t)
        eq=self.balance+self.account_float(bid,ask);self.peak=max(self.peak,eq);dd=100*(self.peak-eq)/self.peak if self.peak>0 else 0.0
        self.current_dd=dd;self.maxdd=max(self.maxdd,dd)
        if self.tick_i%10000==0:self.equity_samples.append((int(tick.ts_event),eq,dd,len(self.parent),len(self.pursuit)))

        if self.parent and P["DDGovernorEnabled"] and dd>=P["DDSoftCutPercent"]:
            self.close_basket(bid,ask,"DD_SOFT_CUT");return
        if dd>=P["MaxAccountDDPercent"] or self.parent_float(bid,ask)<=-P["MaxBasketLossUSD"]:
            self.risk_reason=f"DD {dd:.3f}%" if dd>=P["MaxAccountDDPercent"] else f"Basket {self.parent_float(bid,ask):.2f}"
            if self.parent:self.close_basket(bid,ask,"RISK_STOP")
            self.halted=True;return
        if self.halted:return

        if self.parent:
            cur=self.parent_float(bid,ask)
            if self.basket_open_time is not None and (t-self.basket_open_time).total_seconds()>=P["MaxBasketHoldMinutes"]*60:
                self.close_basket(bid,ask,"TIME_EXIT");return
            if P["EnableBasketTP"] and cur>=P["BasketTakeProfitUSD"]:
                self.close_basket(bid,ask,"BASKET_TP");return
            if P["EnableRecoveryExit"] and not self.recovery_armed and cur<=-P["RecoveryArmLossUSD"]:
                self.recovery_armed=True;self.recovery_armed_time=t
            if P["EnableRecoveryExit"] and self.recovery_armed and cur>=P["RecoveryExitProfitUSD"]:
                held=(t-self.recovery_armed_time).total_seconds() if self.recovery_armed_time is not None else 0
                if held>=P["RecoveryMinHoldSeconds"]:self.close_basket(bid,ask,"RECOVERY_EXIT");return

        idx=SIGNAL["m1"].index.searchsorted(t,side="right")-1
        while self.last_signal_idx<idx:
            self.last_signal_idx+=1
            self.signal_step(self.last_signal_idx,bid,ask,t)

    def on_stop(self):
        if self.parent and self.last_bid is not None:self.close_basket(self.last_bid,self.last_ask,"FORCED_EOT")

    def summary(self):
        gp=sum(x for x in self.basket_results if x>0);gl=-sum(x for x in self.basket_results if x<0)
        pf=gp/gl if gl>0 else (float("inf") if gp>0 else 0.0);wr=100*sum(x>0 for x in self.basket_results)/len(self.basket_results) if self.basket_results else 0.0
        td=SIGNAL["trading_days"]
        return {
            "initial_balance":self.config.initial_balance,"final_balance":round(self.balance,2),
            "return_pct":round(100*(self.balance/self.config.initial_balance-1),3),"max_equity_dd_pct":round(self.maxdd,3),
            "risk_halted":self.halted,"risk_reason":self.risk_reason,
            "basket_closes":len(self.basket_results),"basket_profit_factor":"inf" if math.isinf(pf) else round(pf,4),"basket_win_rate_pct":round(wr,3),
            "gross_profit":round(gp,2),"gross_loss":round(gl,2),"exit_counts":dict(self.exit_counts),
            "entry_cycles":self.entry_cycles,"entry_days":len(self.entry_days),"trading_days":td,
            "entries_per_trading_day":round(self.entry_cycles/td,3),"entry_day_coverage_pct":round(100*len(self.entry_days)/td,2),
            "parent_orders":self.parent_orders,"range_orders":self.range_orders,"leg_counts":self.leg_counts,
            "pursuit_layers_opened":self.pursuit_layers_opened,"pursuit_reversal_exits":self.pursuit_reversal_exits,
            "pursuit_enabled":self.config.pursuit_enabled,
            "pursuit_parent_exits":self.pursuit_parent_exits,"pursuit_realized_pnl":round(self.pursuit_realized,2),
            "pursuit_spread_blocked":self.pursuit_spread_blocked,
            "avg_raw_spread":round(self.spread_sum/max(self.spread_n,1),6),"max_raw_spread":round(self.spread_max,6),
        }

def run_one(inst,ticks,pursuit_enabled,point_size):
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"))
    eng.add_instrument(inst);eng.add_data(ticks)
    st=HybridRawTickG75(HybridCfg(instrument_id=inst.id,pursuit_enabled=pursuit_enabled,point_size=point_size))
    eng.add_strategy(st);eng.run()
    out=st.summary()
    samples=list(st.equity_samples)
    eng.dispose()
    return out,samples

def run(catalog_path,out_dir,scenario):
    global SIGNAL
    cat=ParquetDataCatalog(str(catalog_path))
    inst=next((x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD"),None)
    if inst is None:raise SystemExit("XAUUSD missing")
    ticks=cat.query_quote_ticks(identifiers=[inst.id.value])
    if not ticks:raise SystemExit("No Raw QuoteTicks")
    m1,m5,m15,m5a,m15a,td=build_signal_tables(ticks)
    SIGNAL={"m1":m1,"m5":m5,"m15":m15,"m5a":m5a,"m15a":m15a,"trading_days":td}
    try:
        point_size=fnum(inst.price_increment)
    except Exception:
        point_size=0.01

    pursuit_enabled=(scenario=="g75")
    result,samples=run_one(inst,ticks,pursuit_enabled,point_size)
    payload={
        "verification_level":"NAUTILUS_RAW_BIDASK_TICK_V040_EXACT_G75",
        "engine":"NautilusTrader BacktestEngine",
        "nautilus_version":getattr(nautilus_trader,"__version__","unknown"),
        "scenario":scenario,
        "raw_ticks":len(ticks),
        "ohlc_execution_used":False,
        "intrabar_ohlc_path_used":False,
        "execution":"All entries/exits, DD, TP/recovery and G75 pursuit are evaluated on chronological Raw Bid/Ask QuoteTicks. Long opens Ask/closes Bid; short opens Bid/closes Ask.",
        "signal_generation":"M1/M5/M15 indicator state is causally aggregated from the same Raw QuoteTicks; those bars generate signals only and are never used as execution prices or intrabar paths.",
        "point_size":point_size,
        "g75_source_parity":{
            "enabled":pursuit_enabled,
            "trigger_distance":0.12,"add_distance":0.025,"reversal_distance":0.20,
            "max_layers":10,"lot":0.01,"max_spread_points":100,
            "trigger_add_price":"Ask long / Bid short",
            "reversal_mark":"Bid long / Ask short",
            "multi_level_adds_per_tick":True,
            "first_trigger_returns_before_adds":True
        },
        "dd_governor":{"soft_cut_pct":9.5,"hard_stop_pct":10.0},
        **result,
    }
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    (out/f"{scenario}.json").write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8")
    pd.DataFrame(samples,columns=["ts_event","equity","dd_pct","parent_n","pursuit_n"]).to_csv(out/f"equity_{scenario}.csv",index=False)
    print(json.dumps(payload,indent=2,default=str))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",required=True)
    ap.add_argument("--out",default="bt_results/xau_hybrid_v040_exact_g75_rawtick")
    ap.add_argument("--scenario",choices=["parent","g75"],required=True)
    ap.add_argument("--raw-bidask-only",action="store_true")
    a=ap.parse_args()
    if not a.raw_bidask_only:raise SystemExit("--raw-bidask-only is mandatory")
    run(a.catalog,a.out,a.scenario)

if __name__=="__main__":main()
