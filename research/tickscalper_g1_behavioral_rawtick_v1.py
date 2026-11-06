#!/usr/bin/env python3
"""Behavioral Raw Bid/Ask measurement of TickScalper_G1_RiskGoverned.mq5.

This translates the supplied research EA's published proxy signal and risk rules.
It is not an MT5 terminal tick-model equivalence claim.
"""
from __future__ import annotations
import argparse, json, math
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import numpy as np
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, BookType
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
NS=1_000_000_000
def f(x): return float(x.as_double()) if hasattr(x,"as_double") else float(x)
def isots(t): return datetime.fromtimestamp(t//NS,timezone.utc).isoformat()

class Config(StrategyConfig,frozen=True): instrument_id:object

class G1(Strategy):
    # Inputs fixed to supplied EA defaults, except EnableTrading=true for research.
    BASE=.01; MAX_ADDS=2; MAX_RISK_LOTS=.02; STOP_ATR=1.; TARGET_R=2.; TRAIL_START=2.; TRAIL_ATR=1.; HOLD=45*60*NS
    def __init__(self,c):
        super().__init__(c)
        self.bucket=None; self.bar=None; self.closes=[]; self.ranges=[]; self.prevclose=None
        self.ef=self.es=None; self.avg_gain=self.avg_loss=None; self.atr=None
        self.last_bid=self.last_ask=None; self.buy=[]; self.sell=[]; self.last_add={1:0,-1:0}
        self.equity0=1000.; self.closed=0.; self.peak=1000.; self.day=None; self.day_start=1000.; self.halt=0; self.losses=0; self.last_halted_losses=0
        self.prevdd=self.prevvel=0.; self.trades=[]; self.ticks=0; self.entry_count=0

    def on_start(self): self.subscribe_quote_ticks(self.config.instrument_id)
    def _bar_close(self,b):
        close=b["c"]; n=len(self.closes)
        if self.prevclose is not None:
            tr=max(b["h"]-b["l"],abs(b["h"]-self.prevclose),abs(b["l"]-self.prevclose)); self.ranges.append(tr)
            if len(self.ranges)==14: self.atr=float(np.mean(self.ranges))
            elif len(self.ranges)>14: self.atr=(13*self.atr+tr)/14
            ch=close-self.prevclose; gain=max(ch,0.); loss=max(-ch,0.)
            if len(self.closes)==10:
                dif=np.diff(np.asarray(self.closes[-10:]+[close],dtype=float)); self.avg_gain=float(np.maximum(dif,0).mean()); self.avg_loss=float(np.maximum(-dif,0).mean())
            elif len(self.closes)>10:
                self.avg_gain=(8*self.avg_gain+gain)/9; self.avg_loss=(8*self.avg_loss+loss)/9
        self.closes.append(close); self.prevclose=close
        if len(self.closes)==12: self.ef=float(np.mean(self.closes[:12]))
        elif len(self.closes)>12: self.ef=(2/13)*close+(11/13)*self.ef
        if len(self.closes)==144: self.es=float(np.mean(self.closes[:144]))
        elif len(self.closes)>144: self.es=(2/145)*close+(143/145)*self.es

    def _ind(self,bid):
        if self.ef is None or self.es is None or self.atr is None or self.avg_gain is None or self.prevclose is None: return None
        fast=(2/13)*bid+(11/13)*self.ef; slow=(2/145)*bid+(143/145)*self.es
        change=bid-self.prevclose; ag=(8*self.avg_gain+max(change,0))/9; al=(8*self.avg_loss+max(-change,0))/9
        rsi=100. if al==0 else 100-100/(1+ag/al)
        curtr=max(self.bar["h"]-self.bar["l"],abs(self.bar["h"]-self.prevclose),abs(self.bar["l"]-self.prevclose))
        atr=(13*self.atr+curtr)/14
        return fast,slow,rsi,atr

    def _pnl(self,ps,bid,ask):
        return sum((bid-p["entry"] if p["side"]>0 else p["entry"]-ask)*p["vol"]*100 for p in ps)
    def _eq(self,bid,ask): return self.equity0+self.closed+self._pnl(self.buy,bid,ask)+self._pnl(self.sell,bid,ask)
    def _close(self,ps,bid,ask,why,ts):
        if not ps:return
        pnl=self._pnl(ps,bid,ask); x=ps[0]; risk=max(x["risk_usd"],1e-9)
        self.closed+=pnl; self.losses=(self.losses+1 if pnl<0 else 0)
        self.trades.append({"entry_ts":isots(x["opened"]),"exit_ts":isots(ts),"side":"BUY" if x["side"]>0 else "SELL","adds":len(ps)-1,"why":why,"pnl_usd":pnl,"r":pnl/risk,"mae_R":-x["mae"]/risk,"mfe_R":x["mfe"]/risk})
        ps.clear()

    def _hardhalt(self,bid,ask,ts):
        dt=datetime.fromtimestamp(ts//NS,timezone.utc).date()
        eq=self._eq(bid,ask)
        if dt!=self.day:self.day=dt;self.day_start=eq
        self.peak=max(self.peak,eq); dd=100*max(0,self.peak-eq)/self.peak; daydd=100*max(0,self.day_start-eq)/self.day_start
        # A loss-triggered pause is armed once per newly reached loss count.
        # Reassigning halt on every tick would turn a 24-hour pause permanent.
        if self.losses < 2: self.last_halted_losses=0
        if self.losses >= 2 and self.losses > self.last_halted_losses:
            self.halt=ts+24*3600*NS; self.last_halted_losses=self.losses
        if daydd>=5 or dd>=15 or ts<self.halt:
            self._close(self.buy,bid,ask,"hard_halt",ts);self._close(self.sell,bid,ask,"hard_halt",ts);return True
        return False

    def _fp(self,side,mean,bid,ask):
        v=self.closes[-30:]+([bid] if self.closes else [])
        if len(v)<31:return 0
        a=np.asarray(v[-31:]); rets=np.diff(a)/np.maximum(a[:-1],1e-9); mu=float(rets.mean()); sig=max(float(rets.std()),1e-6)
        price=bid if side>0 else ask; distance=(mean-price if side>0 else price-mean)/max(price,1e-9)
        if distance<=0:return 1
        z=((mu if side>0 else -mu)*30-distance)/(sig*math.sqrt(30))
        z=max(-60,min(60,z));return 1/(1+math.exp(-z))

    def _mult(self,bid,ask):
        eq=self._eq(bid,ask); dd=100*max(0,self.peak-eq)/max(self.peak,1e-9)
        if dd>=15:return 0
        reserve=(max(0,1-dd/15))**2; vel=max(0,dd-self.prevdd); acc=max(0,vel-self.prevvel); guard=math.exp(-2.5*vel-acc)
        self.prevdd,self.prevvel=dd,vel;return max(0,min(1,min(reserve,guard)))

    def _open(self,side,bid,ask,atr,ts,vol):
        entry=ask if side>0 else bid; sl=entry-side*self.STOP_ATR*atr; tp=entry+side*self.TARGET_R*self.STOP_ATR*atr
        ps=self.buy if side>0 else self.sell
        ps.append({"side":side,"entry":entry,"vol":vol,"sl":sl,"tp":tp,"opened":ts,"atr0":atr,"risk_usd":vol*100*self.STOP_ATR*atr,"mae":0.,"mfe":0.})
        self.entry_count+=1

    def _manage(self,side,bid,ask,atr,signal,spread,ts):
        ps=self.buy if side>0 else self.sell
        if not ps:return
        mark=bid if side>0 else ask; vol=sum(x["vol"] for x in ps); mean=sum(x["vol"]*x["entry"] for x in ps)/vol
        move=(mark-mean)*side; risk=max(self.STOP_ATR*atr,1e-9); basket_r=move/risk
        base=ps[0]; pnl=self._pnl(ps,bid,ask); base["mae"]=min(base["mae"],pnl);base["mfe"]=max(base["mfe"],pnl)
        hit=[]
        for p in ps:
            hit.append(mark<=p["sl"] if side>0 else mark>=p["sl"])
            hit.append(mark>=p["tp"] if side>0 else mark<=p["tp"])
        if any(hit):self._close(ps,bid,ask,"position_SL_or_TP",ts);return
        if basket_r<=-1 or ts-ps[0]["opened"]>=self.HOLD:self._close(ps,bid,ask,"basket_SL_or_time",ts);return
        if len(ps)-1<self.MAX_ADDS and vol<self.MAX_RISK_LOTS and basket_r<0 and signal and spread<=.15 and self._fp(side,mean,bid,ask)>=.55 and ts-self.last_add[side]>=60*NS:
            nextv=min(self.BASE*self._mult(bid,ask),self.MAX_RISK_LOTS-vol)
            if nextv>=.01:self._open(side,bid,ask,atr,ts,.01);self.last_add[side]=ts
        if basket_r>=self.TRAIL_START:
            newsl=mark-side*self.TRAIL_ATR*atr
            for p in ps:
                if (side>0 and newsl>p["sl"]) or (side<0 and newsl<p["sl"]):p["sl"]=newsl

    def on_quote_tick(self,t):
        self.ticks+=1;bid,ask,ts=f(t.bid_price),f(t.ask_price),t.ts_event;self.last_bid,self.last_ask=bid,ask
        buck=ts//(60*NS)
        if self.bucket is None:self.bucket=buck;self.bar={"o":bid,"h":bid,"l":bid,"c":bid,"ts":ts}
        elif buck!=self.bucket:
            self._bar_close(self.bar);self.bucket=buck;self.bar={"o":bid,"h":bid,"l":bid,"c":bid,"ts":ts}
        else:self.bar["h"]=max(self.bar["h"],bid);self.bar["l"]=min(self.bar["l"],bid);self.bar["c"]=bid
        if self._hardhalt(bid,ask,ts):return
        z=self._ind(bid)
        if z is None:return
        fast,slow,rsi,atr=z;spread=(ask-bid)/max(atr,1e-9)
        buy_sig=fast>slow and rsi>=50;sell_sig=fast<slow and rsi<=50
        self._manage(1,bid,ask,atr,buy_sig,spread,ts);self._manage(-1,bid,ask,atr,sell_sig,spread,ts)
        dt=datetime.fromtimestamp(ts//NS,timezone.utc)
        if dt.weekday()==4 and dt.hour>=21:return
        if spread>.15:return
        if not self.buy and buy_sig:self._open(1,bid,ask,atr,ts,.01)
        if not self.sell and sell_sig:self._open(-1,bid,ask,atr,ts,.01)

    def on_stop(self):
        if self.last_bid is not None:self._close(self.buy,self.last_bid,self.last_ask,"dataset_end",0);self._close(self.sell,self.last_bid,self.last_ask,"dataset_end",0)
    def result(self):
        pnl=np.asarray([x["pnl_usd"] for x in self.trades]);mae=np.asarray([x["mae_R"] for x in self.trades]);mfe=np.asarray([x["mfe_R"] for x in self.trades])
        win=pnl[pnl>0];loss=pnl[pnl<0]; eq=peak=1000.;dd=0
        for x in pnl:eq+=x;peak=max(peak,eq);dd=max(dd,peak-eq)
        return {"entry_orders":self.entry_count,"baskets":len(pnl),"WR_pct":float((pnl>0).mean()*100) if len(pnl) else 0,"PF":float(win.sum()/abs(loss.sum())) if len(loss) else None,"EV_usd":float(pnl.mean()) if len(pnl) else 0,"Net_usd":float(pnl.sum()),"MaxDD_pct":float(dd/peak*100),"MAE_MFE_R":{"avg_MAE_R":float(mae.mean()) if len(mae) else 0,"median_MAE_R":float(np.median(mae)) if len(mae) else 0,"avg_MFE_R":float(mfe.mean()) if len(mfe) else 0,"median_MFE_R":float(np.median(mfe)) if len(mfe) else 0,"MFE_ge_1R_pct":float((mfe>=1).mean()*100) if len(mfe) else 0},"trades":self.trades}

def l1(ts):
 q=Quantity.from_int(1);return [t if f(t.bid_size)>0 and f(t.ask_size)>0 else QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=t.bid_size if f(t.bid_size)>0 else q,ask_size=t.ask_size if f(t.ask_size)>0 else q,ts_event=t.ts_event,ts_init=t.ts_init) for t in ts]
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--catalog",action="append",required=True);ap.add_argument("--experiment-id",required=True);a=ap.parse_args();raw=[];inst=None
 for root in a.catalog:
  cat=ParquetDataCatalog(root);i=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD");inst=inst or i;raw+=cat.query(data_cls=QuoteTick,identifiers=[i.id.value])
 raw.sort(key=lambda x:x.ts_event);e=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)))
 e.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"));e.add_instrument(inst);e.add_data(l1(raw));s=G1(Config(instrument_id=inst.id));e.add_strategy(s);e.run();e.end()
 out={"status":"COMPLETE","engine":"NautilusTrader BacktestEngine","raw_ticks":len(raw),"ohlc_resample_used":False,"execution":"Raw QuoteTick BUY Ask / SELL Bid","source":"TickScalper_G1_RiskGoverned.mq5 supplied by user","translation_scope":"EMA12/EMA144 + RSI9 current-M1 proxy, ATR14, max two non-geometric adds, basket/time stop, TP2R/trail, DD/consecutive-loss halts; MT5 terminal fill model and commission are not reproduced.","results":{"tickscalper_g1_behavioral_v1":s.result()}}
 path=Path("results/tickscalper-g1")/a.experiment_id/"summary.json";path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=="__main__":main()
