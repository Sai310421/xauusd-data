from __future__ import annotations

"""OB∩FVG five-depth gate with dynamic ATR trailing exit.
Keeps Entry/POI logic identical to the baseline and changes only Exit.
"""
import argparse, json, math
from collections import Counter
from decimal import Decimal
from pathlib import Path
import numpy as np
import nautilus_trader
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, BookType
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from multiedge_ob_fvg_depth_gate import DEPTHS, GateConfig, DepthGate

if not hasattr(ParquetDataCatalog,'query_quote_ticks'):
    def _query_quote_ticks(self,identifiers=None,start=None,end=None):
        return self.query(data_cls=QuoteTick,identifiers=identifiers,start=start,end=end)
    ParquetDataCatalog.query_quote_ticks=_query_quote_ticks

class DynamicDepthGate(DepthGate):
    def _open_depth(self,d,side,px,atr):
        if self.active[d] is not None:return
        self._submit(side)
        self.active[d]={'side':side,'entry':px,'entry_i':self.m1_i,'atr':atr,'best':px,'worst':px,'trail_armed':False,'trail_stop':None}
        self.stats[d]['fills']+=1
    @staticmethod
    def _trail_distance_atr(mfe_atr):
        if mfe_atr>=2.0:return 0.75
        if mfe_atr>=1.5:return 1.00
        return 1.20
    def on_quote_tick(self,tick:QuoteTick):
        bid=self._f(tick.bid_price);ask=self._f(tick.ask_price);self.last_bid=bid;self.last_ask=ask
        if self.zone:
            side=self.zone['side'];probe=ask if side>0 else bid
            for d,level in self.zone['levels'].items():
                if d in self.zone['filled']:continue
                if (probe<=level if side>0 else probe>=level):
                    self.zone['filled'].add(d);self._open_depth(d,side,probe,self.zone['atr'])
            if (side>0 and bid<self.zone['lo']-0.35*self.zone['atr']) or (side<0 and ask>self.zone['hi']+0.35*self.zone['atr']):self.zone=None
        for d,a in list(self.active.items()):
            if a is None:continue
            side=a['side'];mark=bid if side>0 else ask;atr=max(a['atr'],1e-9)
            if side>0:a['best']=max(a['best'],mark);a['worst']=min(a['worst'],mark);mfe=max(0.0,a['best']-a['entry'])
            else:a['best']=min(a['best'],mark);a['worst']=max(a['worst'],mark);mfe=max(0.0,a['entry']-a['best'])
            mfe_atr=mfe/atr;move=(mark-a['entry'])*side
            if move<=-0.75*atr:self._close_depth(d,bid,ask,'HARD_SL');continue
            if mfe_atr>=0.75:
                a['trail_armed']=True;be=a['entry']+side*0.05*atr;dist=self._trail_distance_atr(mfe_atr)*atr
                candidate=a['best']-dist if side>0 else a['best']+dist
                if side>0:a['trail_stop']=max(be,candidate) if a['trail_stop'] is None else max(a['trail_stop'],be,candidate)
                else:a['trail_stop']=min(be,candidate) if a['trail_stop'] is None else min(a['trail_stop'],be,candidate)
            if a['trail_armed'] and a['trail_stop'] is not None and (mark<=a['trail_stop'] if side>0 else mark>=a['trail_stop']):
                self._close_depth(d,bid,ask,'DYNAMIC_ATR_TRAIL');continue
            if self.m1_i-a['entry_i']>=180:self._close_depth(d,bid,ask,'HORIZON_180M')
    def summary(self):
        out={'zones':self.zone_count,'reject_no_fvg':self.rejected_no_fvg,'reject_no_ob':self.rejected_no_ob,'reject_no_overlap':self.rejected_no_overlap,'depths':{}}
        for d,s in self.stats.items():
            t=s['trades'];n=len(t);net=sum(x['pnl'] for x in t);pf=s['gw']/s['gl'] if s['gl']>0 else (math.inf if s['gw']>0 else 0.0);reasons=Counter(x['reason'] for x in t)
            out['depths'][str(d)]={'N':n,'WR_pct':100*s['wins']/max(n,1),'PF':pf,'net_virtual':net,'expectancy':net/max(n,1),'MFE_mean':float(np.mean([x['mfe'] for x in t])) if n else 0.0,'MAE_mean':float(np.mean([x['mae'] for x in t])) if n else 0.0,'MFE_ATR_mean':float(np.mean([x['mfe_atr'] for x in t])) if n else 0.0,'MAE_ATR_mean':float(np.mean([x['mae_atr'] for x in t])) if n else 0.0,'exit_reasons':dict(reasons),'fills':s['fills']}
        return out

def ensure_l1(ticks):
    one=Quantity.from_int(1);out=[];rep=0
    for t in ticks:
        bs=float(t.bid_size.as_double()) if hasattr(t.bid_size,'as_double') else float(t.bid_size);az=float(t.ask_size.as_double()) if hasattr(t.ask_size,'as_double') else float(t.ask_size)
        if bs<=0 or az<=0:out.append(QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=one if bs<=0 else t.bid_size,ask_size=one if az<=0 else t.ask_size,ts_event=t.ts_event,ts_init=t.ts_init));rep+=1
        else:out.append(t)
    return out,rep

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args();catalog=ParquetDataCatalog(a.catalog)
    inst=next((x for x in catalog.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'),None)
    if inst is None:raise SystemExit('XAUUSD missing')
    raw=catalog.query_quote_ticks(identifiers=[inst.id.value]);
    if not raw:raise SystemExit('no raw XAUUSD QuoteTicks')
    ticks,replaced=ensure_l1(raw);engine=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    engine.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));engine.add_instrument(inst);engine.add_data(ticks)
    cfg=GateConfig(instrument_id=inst.id,m1=BarType.from_str(f'{inst.id.value}-1-MINUTE-BID-INTERNAL'),horizon_minutes=180);strat=DynamicDepthGate(cfg);engine.add_strategy(strat);engine.run();engine.end()
    result={'verification_level':'NAUTILUS_RAW_BIDASK_OB_FVG_DYNAMIC_ATR_EXIT_GATE','engine':'NautilusTrader BacktestEngine','nautilus_version':nautilus_trader.__version__,'raw_ticks':len(raw),'execution_ticks':len(ticks),'raw_zero_size_quotes_replaced':replaced,'ohlc_resample_used':False,'signal_bars':'Nautilus INTERNAL M1 from raw QuoteTicks','poi':'OB_INTERSECT_FVG','depths':[0,25,50,75,100],'exit_model':{'hard_sl_atr':0.75,'trail_arm_mfe_atr':0.75,'be_lock_atr':0.05,'trail_0_75_to_1_5_atr':1.20,'trail_1_5_to_2_atr':1.00,'trail_2plus_atr':0.75,'fixed_tp':None,'horizon_minutes':180},**strat.summary()}
    out=Path('results/multiedge')/a.experiment_id/'OB_FVG_DYNAMIC_ATR_EXIT.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
