from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import deque
from dataclasses import dataclass, asdict
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import nautilus_trader
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

from research.nautilus_catalog_compat import select_instrument_compat, query_quote_ticks_compat

SIM=Venue("SIM")
NS15=900*1_000_000_000
ROUTES=("NORMAL","AMD_ASIA","AMD_LONDON","AMD_NY")
VARIANTS=("CORE","VIDEO_OR")

def sha_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()

def js_hash(x) -> str:
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def fpx(x) -> float:
    return float(x.as_double()) if hasattr(x,"as_double") else float(x)

def jst_parts(ns:int):
    t=pd.Timestamp(ns,unit="ns",tz="UTC").tz_convert("Asia/Tokyo")
    return t, int(t.hour), int(t.minute), t.normalize()

def inside(h,a,b):
    return a<=h<b if a<b else (h>=a or h<b)

def route_at(ns:int)->str:
    _,h,_,_=jst_parts(ns)
    if inside(h,9,10): return "AMD_ASIA"
    if inside(h,16,18): return "AMD_LONDON"
    if inside(h,23,1): return "AMD_NY"
    return "NORMAL"

def macro_window(route:str,ns:int)->bool:
    _,h,m,_=jst_parts(ns)
    if route=="AMD_ASIA": return h==9 and m<30
    if route=="AMD_LONDON": return h==16 and m<30
    if route=="AMD_NY": return h==23 and m<30
    return False

@dataclass
class Rec:
    route:str
    variant:str
    side:int
    entry_time:str
    exit_time:str
    entry:float
    stop:float
    target:float
    exit_price:float
    R:float
    result:str
    pda:int
    macro:int
    volume:int
    cisd_level:float
    target_kind:str

class VideoParityConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar15: BarType
    bar30: BarType
    variant: str
    risk_pct: float = 0.35

class VideoParityStrategy(Strategy):
    def __init__(self, config:VideoParityConfig):
        super().__init__(config)
        self.b15=deque(maxlen=320)
        self.b30=deque(maxlen=180)
        self.tick_counts={}
        self.vol_hist=deque(maxlen=40)
        self.phase=0
        self.route="NORMAL"
        self.age=0
        self.di=0
        self.acc_hi=np.nan
        self.acc_lo=np.nan
        self.sweep_ext=np.nan
        self.cisd_level=np.nan
        self.pda=False
        self.macro=False
        self.volume=False
        self.armed=None
        self.active=None
        self.exit_pending=False
        self.trades:list[Rec]=[]
        self.sequence=[]
        self.raw_ticks=0
        self.last_bid=None
        self.last_ask=None
        self.last_ts=None
        self.display_equity=1000.0
        self.display_peak=1000.0
        self.display_mdd_pct=0.0
        self.active_risk_cash=0.0
        self.entries=0

    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id)
        self.subscribe_bars(self.config.bar15)
        self.subscribe_bars(self.config.bar30)

    def reset_setup(self):
        self.phase=0; self.age=0; self.di=0
        self.acc_hi=np.nan; self.acc_lo=np.nan
        self.sweep_ext=np.nan; self.cisd_level=np.nan
        self.pda=False; self.macro=False; self.volume=False

    def _atr15(self):
        if len(self.b15)<15:return None
        x=list(self.b15); trs=[]
        for i in range(-14,0):
            c=x[i];p=x[i-1]
            trs.append(max(c["h"]-c["l"],abs(c["h"]-p["c"]),abs(c["l"]-p["c"])))
        a=float(np.mean(trs))
        return a if math.isfinite(a) and a>0 else None

    def _session_range(self,route,ts):
        x=list(self.b15)
        if len(x)<20:return None
        if route=="NORMAL":
            q=x[-17:-1] if len(x)>=17 else x[:-1]
            return (max(z["h"] for z in q),min(z["l"] for z in q)) if q else None
        jt,h,m,day=jst_parts(ts)
        vals=[]
        for z in x[:-1]:
            zt,zh,zm,zday=jst_parts(z["ts"])
            mod=zh*60+zm
            use=False
            if route=="AMD_LONDON":
                use=(zday==day and 9*60<=mod<16*60)
            elif route=="AMD_NY":
                anchor=day-pd.Timedelta(days=1) if h<1 else day
                use=(zday==anchor and 16*60<=mod<23*60)
            elif route=="AMD_ASIA":
                prev=day-pd.Timedelta(days=1)
                use=(zday==prev and mod>=23*60) or (zday==day and mod<6*60)
            if use: vals.append(z)
        if not vals:return None
        return max(z["h"] for z in vals),min(z["l"] for z in vals)

    def _delivery(self,z,di):
        return z["c"]>z["o"] if di<0 else z["c"]<z["o"]

    def _cisd_origin(self,di):
        x=list(self.b15)
        if len(x)<4:return None
        j=len(x)-1
        if not self._delivery(x[j],di):j-=1
        if j<1 or not self._delivery(x[j],di):return None
        oldest=j
        limit=max(0,j-8)
        while oldest-1>=limit and self._delivery(x[oldest-1],di): oldest-=1
        return float(x[oldest]["o"])

    def _htf_pda(self,di,px,ts):
        q=[z for z in self.b30 if z["ts"]<ts]
        if len(q)<24:return False
        w=q[-24:];mid=(max(z["h"] for z in w)+min(z["l"] for z in w))/2
        return px>=mid if di<0 else px<=mid

    def _volume_influx(self):
        if len(self.vol_hist)<21:return False
        cur=self.vol_hist[-1];med=float(np.median(list(self.vol_hist)[-21:-1]))
        return med>0 and cur>=med*1.15

    def _candidate_targets(self,di,entry,sl):
        risk=abs(entry-sl)
        if risk<=0:return []
        out=[]
        x=list(self.b15)
        for k in range(max(1,len(x)-26),len(x)-2):
            if di<0 and x[k]["l"]<x[k-1]["l"] and x[k]["l"]<x[k+1]["l"] and x[k]["l"]<entry:
                out.append((x[k]["l"],"IRL_M15"))
            if di>0 and x[k]["h"]>x[k-1]["h"] and x[k]["h"]>x[k+1]["h"] and x[k]["h"]>entry:
                out.append((x[k]["h"],"IRL_M15"))
        q=list(self.b30)
        for k in range(max(2,len(q)-42),len(q)):
            if k<2:continue
            if q[k]["l"]>q[k-2]["h"] and di>0:
                lo,hi=q[k-2]["h"],q[k]["l"]
                if lo>entry: out.append((lo,"FVG_M30"))
                if hi>entry: out.append((hi,"FVG_M30"))
            if q[k]["h"]<q[k-2]["l"] and di<0:
                lo,hi=q[k]["h"],q[k-2]["l"]
                if hi<entry: out.append((hi,"FVG_M30"))
                if lo<entry: out.append((lo,"FVG_M30"))
        bound=self.acc_lo if di<0 else self.acc_hi
        if math.isfinite(bound):out.append((float(bound),"ACC_BOUND"))
        good=[]
        for px,kind in out:
            rew=(entry-px) if di<0 else (px-entry)
            rr=rew/risk
            if rew>0 and rr>=0.80:good.append((rew,px,rr,kind))
        good.sort(key=lambda z:z[0])
        return good

    def _context_ok(self):
        if self.config.variant=="CORE":return True
        return self.macro or (self.pda and self.volume)

    def _log(self,phase,ts,extra=None):
        x={"time":str(pd.Timestamp(ts,unit="ns",tz="UTC")),"route":self.route,"phase":phase,"dir":self.di,
           "acc_hi":None if not math.isfinite(self.acc_hi) else self.acc_hi,
           "acc_lo":None if not math.isfinite(self.acc_lo) else self.acc_lo,
           "cisd_level":None if not math.isfinite(self.cisd_level) else self.cisd_level,
           "pda":int(self.pda),"macro":int(self.macro),"volume":int(self.volume)}
        if extra:x.update(extra)
        self.sequence.append(x)

    def on_bar(self,bar:Bar):
        b={"o":fpx(bar.open),"h":fpx(bar.high),"l":fpx(bar.low),"c":fpx(bar.close),"ts":int(bar.ts_event)}
        s=str(bar.bar_type)
        if "30-MINUTE" in s:
            self.b30.append(b);return
        if "15-MINUTE" not in s:return

        self.b15.append(b)
        bucket=(b["ts"]-1)//NS15
        self.vol_hist.append(int(self.tick_counts.pop(bucket,0)))
        # keep tick-count map bounded
        for k in list(self.tick_counts):
            if k<bucket-4:self.tick_counts.pop(k,None)

        a=self._atr15()
        if a is None:return
        current=route_at(b["ts"])
        # AMD has priority over a NORMAL setup at the session boundary.
        if self.phase and self.route=="NORMAL" and current!="NORMAL":
            self.reset_setup()

        if self.phase==0:
            self.route=current
            rg=self._session_range(self.route,b["ts"])
            if rg is None:return
            self.acc_hi,self.acc_lo=rg
            self.phase=1;self.age=0
            self._log("ACCUMULATION_LOCK",b["ts"])
            return

        self.age+=1
        if self.phase==1:
            sh=b["h"]>self.acc_hi+a*.03 and b["c"]<self.acc_hi
            sl=b["l"]<self.acc_lo-a*.03 and b["c"]>self.acc_lo
            if sh or sl:
                self.di=-1 if sh else 1
                self.sweep_ext=b["h"] if sh else b["l"]
                c=self._cisd_origin(self.di)
                if c is None:
                    self.reset_setup();return
                self.cisd_level=c
                self.pda=self._htf_pda(self.di,b["c"],b["ts"])
                self.macro=macro_window(self.route,b["ts"])
                self.phase=2;self.age=0
                self._log("LIQUIDITY_SWEEP+CISD_CANDLE",b["ts"])
            elif self.age>8:self.reset_setup()
            return

        if self.phase==2:
            self.sweep_ext=max(self.sweep_ext,b["h"]) if self.di<0 else min(self.sweep_ext,b["l"])
            confirmed=b["c"]<self.cisd_level if self.di<0 else b["c"]>self.cisd_level
            if confirmed:
                self.volume=self._volume_influx()
                self.phase=3;self.age=0
                self._log("CISD_CONFIRMED",b["ts"])
            elif self.age>8:self.reset_setup()
            return

        if self.phase==3:
            touched=b["h"]>=self.cisd_level if self.di<0 else b["l"]<=self.cisd_level
            held=b["c"]<=self.cisd_level if self.di<0 else b["c"]>=self.cisd_level
            if touched and held and self.active is None and self.armed is None:
                stop=self.sweep_ext+a*.05 if self.di<0 else self.sweep_ext-a*.05
                # target candidates are frozen on confirmed M15 entry bar; executable choice uses next raw quote.
                self.armed={"route":self.route,"side":self.di,"stop":float(stop),"pda":self.pda,"macro":self.macro,
                            "volume":self.volume,"cisd_level":self.cisd_level,"ts":b["ts"]}
                self._log("CISD_ENTRY_ARMED",b["ts"])
                self.reset_setup()
            elif self.age>6:self.reset_setup()

    def _mark_dd(self,px):
        if self.active is None:return
        r=self.active["side"]*(px-self.active["entry"])/self.active["risk"]
        mark=self.active["equity0"]+self.active_risk_cash*r
        self.display_peak=max(self.display_peak,mark)
        if self.display_peak>0:self.display_mdd_pct=max(self.display_mdd_pct,(self.display_peak-mark)/self.display_peak*100)

    def _finish(self,px,ts,result):
        q=self.active
        if q is None:return
        r=q["side"]*(px-q["entry"])/q["risk"]
        eq0=q["equity0"];self.display_equity=eq0+self.active_risk_cash*r
        self.display_peak=max(self.display_peak,self.display_equity)
        if self.display_peak>0:self.display_mdd_pct=max(self.display_mdd_pct,(self.display_peak-self.display_equity)/self.display_peak*100)
        self.trades.append(Rec(q["route"],self.config.variant,q["side"],q["entry_time"],
            str(pd.Timestamp(ts,unit="ns",tz="UTC")),q["entry"],q["stop"],q["target"],px,float(r),result,
            int(q["pda"]),int(q["macro"]),int(q["volume"]),q["cisd_level"],q["target_kind"]))
        self.active=None;self.active_risk_cash=0.0;self.exit_pending=True

    def on_quote_tick(self,tick:QuoteTick):
        ts=int(tick.ts_event);bid=fpx(tick.bid_price);ask=fpx(tick.ask_price)
        self.raw_ticks+=1;self.last_bid=bid;self.last_ask=ask;self.last_ts=ts
        self.tick_counts[ts//NS15]=self.tick_counts.get(ts//NS15,0)+1
        if self.active is not None:
            px=bid if self.active["side"]>0 else ask
            self._mark_dd(px)
            stop=self.active["stop"];target=self.active["target"];side=self.active["side"]
            if (side>0 and px<=stop) or (side<0 and px>=stop):
                self._finish(px,ts,"LOSS");self.close_all_positions(self.config.instrument_id);return
            if (side>0 and px>=target) or (side<0 and px<=target):
                self._finish(px,ts,"WIN");self.close_all_positions(self.config.instrument_id);return
            return
        if self.exit_pending:return
        if self.armed is None:return
        a=self.armed
        if self.config.variant=="VIDEO_OR" and not (a["macro"] or (a["pda"] and a["volume"])):
            self.armed=None;return
        side=a["side"];entry=ask if side>0 else bid;stop=a["stop"];risk=abs(entry-stop)
        if risk<=0 or not math.isfinite(risk):self.armed=None;return
        cands=self._candidate_targets(side,entry,stop)
        if not cands:self.armed=None;return
        _,target,rr,kind=cands[0]
        inst=self.cache.instrument(self.config.instrument_id)
        order=self.order_factory.market(instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY if side>0 else OrderSide.SELL,
            quantity=inst.make_qty(Decimal("1")))
        self.submit_order(order)
        self.active={"route":a["route"],"side":side,"entry":entry,"stop":stop,"target":target,"risk":risk,
                     "entry_time":str(pd.Timestamp(ts,unit="ns",tz="UTC")),"equity0":self.display_equity,
                     "pda":a["pda"],"macro":a["macro"],"volume":a["volume"],"cisd_level":a["cisd_level"],"target_kind":kind}
        self.active_risk_cash=self.display_equity*(self.config.risk_pct/100.0)
        self.entries+=1;self.armed=None

    def on_position_closed(self,event):
        self.exit_pending=False

    def on_stop(self):
        if self.active is not None and self.last_ts is not None:
            px=self.last_bid if self.active["side"]>0 else self.last_ask
            self._finish(float(px),self.last_ts,"END")
            self.close_all_positions(self.config.instrument_id)

def summarize(trades:list[Rec],risk_pct:float,start:str,end:str,display_equity:float,mdd_pct:float):
    rs=np.array([x.R for x in trades],float)
    n=len(rs);wins=int((rs>0).sum());gp=float(rs[rs>0].sum()) if n else 0.;gl=float(-rs[rs<0].sum()) if n else 0.
    pf=gp/gl if gl>0 else (float("inf") if gp>0 else 0.)
    s=pd.Timestamp(start).date();e=pd.Timestamp(end).date();bd=max(1,int(np.busday_count(s,e)))
    ret=(display_equity/1000-1)*100
    monthly=((display_equity/1000)**(21/bd)-1)*100 if display_equity>0 else -100.
    daily=((1+monthly/100)**(1/21)-1)*100 if monthly>-100 else -100.
    return {"N":n,"wins":wins,"losses":n-wins,"WR_pct":100*wins/n if n else 0.,"PF_R":pf,
            "net_R":float(rs.sum()) if n else 0.,"Return_pct":ret,"MaxFloatingDD_pct":float(mdd_pct),
            "RF":ret/mdd_pct if mdd_pct>0 else None,"business_days":bd,"Monthly21_pct":monthly,"Daily_pct":daily,
            "N_21D":n*21/bd,"N_per_business_day":n/bd,"NetProfit_display_USD":display_equity-1000}

def route_metrics(trades,risk_pct,start,end):
    out={}
    for r in ROUTES:
        xs=[x for x in trades if x.route==r]
        # route-only equity path using closed R; route DD is closed-trade display, overall has tick-floating DD.
        eq=1000.;pk=1000.;dd=0.
        for x in xs:
            eq+=eq*(risk_pct/100)*x.R;pk=max(pk,eq);dd=max(dd,(pk-eq)/pk*100)
        out[r]=summarize(xs,risk_pct,start,end,eq,dd)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--catalog",required=True);ap.add_argument("--experiment-id",required=True)
    ap.add_argument("--variant",choices=VARIANTS,required=True);ap.add_argument("--risk-pct",type=float,default=.35);ap.add_argument("--start",default="2026-02-25");ap.add_argument("--days",type=int,default=90)
    a=ap.parse_args()
    cp=Path(a.catalog);manifest=json.loads((cp/"catalog_manifest.json").read_text())
    if manifest.get("status")!="COMPLETE" or manifest.get("data_kind")!="RAW_BIDASK" or manifest.get("ohlc_resample_used") is not False:
        raise SystemExit("RAW_BIDASK_FAIL_CLOSED")
    catalog=ParquetDataCatalog(str(cp));inst=select_instrument_compat(catalog,"XAUUSD")
    start_ts=pd.Timestamp(a.start,tz="UTC")
    end_ts=start_ts+pd.Timedelta(days=a.days)
    ticks=query_quote_ticks_compat(catalog,identifiers=[inst.id.value],start=start_ts,end=end_ts)
    if not ticks:raise SystemExit("NO_RAW_QUOTETICKS")
    cfg=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True))
    engine=BacktestEngine(config=cfg)
    engine.add_venue(venue=SIM,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=USD,
                     starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"))
    engine.add_instrument(inst);engine.add_data(ticks)
    b15=BarType.from_str(f"{inst.id.value}-15-MINUTE-BID-INTERNAL")
    b30=BarType.from_str(f"{inst.id.value}-30-MINUTE-BID-INTERNAL")
    strat=VideoParityStrategy(VideoParityConfig(instrument_id=inst.id,bar15=b15,bar30=b30,variant=a.variant,risk_pct=a.risk_pct))
    engine.add_strategy(strat);engine.run()
    out=Path("results/ae-bt")/a.experiment_id;out.mkdir(parents=True,exist_ok=True)
    trades=strat.trades
    pd.DataFrame([asdict(x) for x in trades]).to_csv(out/f"trades_{a.variant}.csv",index=False)
    try: engine.trader.generate_positions_report().to_csv(out/f"engine_positions_{a.variant}.csv")
    except Exception as exc:(out/f"engine_positions_{a.variant}.error.txt").write_text(str(exc))
    period_end=(pd.Timestamp(a.start)+pd.Timedelta(days=a.days)).date().isoformat()
    met=summarize(trades,a.risk_pct,a.start,period_end,strat.display_equity,strat.display_mdd_pct)
    result={"verification_level":"NAUTILUS_BT","raw_bidask_pass":True,"engine":"NautilusTrader BacktestEngine",
            "nautilus_version":getattr(nautilus_trader,"__version__","unknown"),"variant":a.variant,
            "symbol":"XAUUSD","signal_timeframe":"M15","htf_timeframe":"M30","raw_ticks":strat.raw_ticks,
            "period":{"start":a.start,"days":a.days,"end_exclusive":period_end,"parent_catalog_start":manifest["start"],"parent_catalog_days":manifest["days"]},
            "execution":"M15/M30 INTERNAL BID bars generated by Nautilus from raw Dukascopy QuoteTicks; market entry/SL/TP evaluated on raw Bid/Ask QuoteTicks; native observed spread included; no added commission or stochastic slippage",
            "risk_display":{"initial":1000.0,"risk_pct_per_trade":a.risk_pct,"engine_order_qty":1},
            "overall":met,"by_route":route_metrics(trades,a.risk_pct,a.start,period_end),
            "entries_submitted":strat.entries,"limitations":["No added broker commission or stochastic slippage beyond raw observed Bid/Ask spread.","Display equity is R-risk normalized at fixed percent risk; Nautilus engine order quantity is 1 XAU unit for execution-path validation."]}
    (out/f"result_{a.variant}.json").write_text(json.dumps(result,indent=2,ensure_ascii=False))
    pd.DataFrame(strat.sequence).to_csv(out/f"sequence_{a.variant}.csv",index=False)
    conf={"variant":a.variant,"risk_pct":a.risk_pct,"start":a.start,"days":a.days,"routes":"NORMAL + fixed-JST AMD priority","sweep_min_atr":.03,
          "cisd_confirm_ttl_bars":8,"entry_ttl_bars":6,"target_min_rr":.8,
          "context":"CORE=none; VIDEO_OR=Macro OR (HTF PDA and raw tick-volume influx)"}
    evidence={"experiment_id":a.experiment_id,"verification_level":"NAUTILUS_BT","git_sha":os.environ.get("GITHUB_SHA"),
              "run_id":os.environ.get("GITHUB_RUN_ID"),"nautilus_version":getattr(nautilus_trader,"__version__","unknown"),
              "dataset_id":f"dukascopy-XAUUSD-{a.start}-{a.days}d-slice",
              "dataset_sha256":manifest.get("catalog_sha256"),"strategy_sha256":sha_file(Path(__file__)),
              "config_sha256":js_hash(conf),"seed":None,"ohlc_resample_used":False,"config":conf}
    (out/f"manifest_{a.variant}.json").write_text(json.dumps(evidence,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False))
    engine.dispose()

if __name__=="__main__":main()
