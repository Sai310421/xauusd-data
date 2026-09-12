from __future__ import annotations

"""Independent OB∩IFVG / OB∩BPR five-depth Raw BidAsk gates.

Keeps the same XAUUSD M1, 0/25/50/75/100 depth geometry and Dynamic ATR exit
used by the OB∩FVG diagnostic. Only the POI construction changes.

IFVG definition (stateful, no lookahead):
1) Confirm a strict 3-candle FVG after candle close.
2) Keep the FVG active until price CLOSES through the distal boundary.
3) The failed FVG becomes an inversion FVG (IFVG) in the opposite direction.
4) Require overlap with the last opposite candle OB for the new IFVG direction.

BPR definition:
1) Maintain recent confirmed bullish and bearish FVG zones.
2) A BPR exists only when an opposing FVG pair overlaps in price.
3) Require BPR direction to match the current breakout/displacement direction.
4) Require overlap with the last opposite candle OB.

All POIs are armed only on closed M1 bars; fills occur on subsequent QuoteTicks.
"""

import argparse, json, math
from collections import Counter, deque
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiedge_ob_fvg_depth_gate import DEPTHS, GateConfig, DepthGate

if not hasattr(ParquetDataCatalog, 'query_quote_ticks'):
    def _query_quote_ticks(self, identifiers=None, start=None, end=None):
        return self.query(data_cls=QuoteTick, identifiers=identifiers, start=start, end=end)
    ParquetDataCatalog.query_quote_ticks = _query_quote_ticks

class DynamicExitMixin:
    def _open_depth(self,d,side,px,atr):
        if self.active[d] is not None:return
        self._submit(side)
        self.active[d]={'side':side,'entry':px,'entry_i':self.m1_i,'atr':atr,'best':px,'worst':px,'trail_armed':False,'trail_stop':None}
        self.stats[d]['fills']+=1
    def _trail_distance(self,m):
        if m>=2.0:return 0.75
        if m>=1.5:return 1.00
        return 1.20
    def on_quote_tick(self,tick:QuoteTick):
        bid=self._f(tick.bid_price); ask=self._f(tick.ask_price); self.last_bid=bid; self.last_ask=ask
        if self.zone:
            side=self.zone['side']; probe=ask if side>0 else bid
            for d,level in self.zone['levels'].items():
                if d in self.zone['filled']:continue
                touched=probe<=level if side>0 else probe>=level
                if touched:
                    self.zone['filled'].add(d);self._open_depth(d,side,probe,self.zone['atr'])
            if (side>0 and bid<self.zone['lo']-0.35*self.zone['atr']) or (side<0 and ask>self.zone['hi']+0.35*self.zone['atr']):self.zone=None
        for d,a in list(self.active.items()):
            if a is None:continue
            side=a['side']; mark=bid if side>0 else ask; atr=max(a['atr'],1e-9)
            if side>0:a['best']=max(a['best'],mark);a['worst']=min(a['worst'],mark);mfe=max(0,a['best']-a['entry'])
            else:a['best']=min(a['best'],mark);a['worst']=max(a['worst'],mark);mfe=max(0,a['entry']-a['best'])
            move=(mark-a['entry'])*side;m=mfe/atr
            if move<=-0.75*atr:self._close_depth(d,bid,ask,'HARD_SL');continue
            if m>=0.75:
                a['trail_armed']=True;be=a['entry']+side*0.05*atr;dist=self._trail_distance(m)*atr
                cand=a['best']-dist if side>0 else a['best']+dist
                if side>0:a['trail_stop']=max(be,cand) if a['trail_stop'] is None else max(a['trail_stop'],be,cand)
                else:a['trail_stop']=min(be,cand) if a['trail_stop'] is None else min(a['trail_stop'],be,cand)
            if a['trail_armed'] and a['trail_stop'] is not None:
                hit=mark<=a['trail_stop'] if side>0 else mark>=a['trail_stop']
                if hit:self._close_depth(d,bid,ask,'DYNAMIC_ATR_TRAIL');continue
            if self.m1_i-a['entry_i']>=180:self._close_depth(d,bid,ask,'HORIZON_180M')

class POIBase(DynamicExitMixin,DepthGate):
    def __init__(self,config,selected):
        super().__init__(config);self.selected=selected
        self.fvgs=deque(maxlen=80);self.ifvgs=deque(maxlen=80);self.bprs=deque(maxlen=80)
        self.poi_candidates=0;self.reject_no_poi=0;self.reject_no_ob_overlap=0
    def _strict_fvg_now(self):
        if len(self.c)<3:return None
        h=list(self.h);l=list(self.l)
        if l[-1]>h[-3]:return {'dir':1,'lo':float(h[-3]),'hi':float(l[-1]),'born':self.m1_i,'active':True}
        if h[-1]<l[-3]:return {'dir':-1,'lo':float(h[-1]),'hi':float(l[-3]),'born':self.m1_i,'active':True}
        return None
    def _update_fvgs_and_ifvgs(self):
        c=float(self.c[-1]);new=self._strict_fvg_now()
        if new:self.fvgs.append(new)
        for f in list(self.fvgs):
            if not f['active']:continue
            if f['dir']>0 and c<f['lo']:
                f['active']=False;self.ifvgs.append({'dir':-1,'lo':f['lo'],'hi':f['hi'],'born':self.m1_i})
            elif f['dir']<0 and c>f['hi']:
                f['active']=False;self.ifvgs.append({'dir':1,'lo':f['lo'],'hi':f['hi'],'born':self.m1_i})
        bulls=[f for f in self.fvgs if f['active'] and f['dir']>0 and self.m1_i-f['born']<=20]
        bears=[f for f in self.fvgs if f['active'] and f['dir']<0 and self.m1_i-f['born']<=20]
        for b in bulls[-5:]:
            for s in bears[-5:]:
                lo=max(b['lo'],s['lo']);hi=min(b['hi'],s['hi'])
                if lo<hi:
                    # Direction assigned by which FVG was formed most recently.
                    direction=1 if b['born']>s['born'] else -1
                    key=(round(lo,5),round(hi,5),direction,max(b['born'],s['born']))
                    if not any((round(x['lo'],5),round(x['hi'],5),x['dir'],x['born'])==key for x in self.bprs):
                        self.bprs.append({'dir':direction,'lo':lo,'hi':hi,'born':max(b['born'],s['born'])})
    def _break_side(self):
        n=self.config.breakout_lookback
        if len(self.c)<max(70,n+4) or self._atr()<=0:return 0
        o=np.asarray(self.o,float);h=np.asarray(self.h,float);l=np.asarray(self.l,float);c=np.asarray(self.c,float);atr=self._atr()
        prior_hi=float(h[-(n+1):-1].max());prior_lo=float(l[-(n+1):-1].min());body=abs(c[-1]-o[-1])
        if body<self.config.displacement_body_atr*atr:return 0
        return 1 if c[-1]>prior_hi else -1 if c[-1]<prior_lo else 0
    def _arm_from_poi(self,side,poi):
        ob=self._find_ob(side)
        if ob is None:self.rejected_no_ob+=1;return False
        zlo=max(ob[0],poi['lo']);zhi=min(ob[1],poi['hi'])
        if not zlo<zhi:self.reject_no_ob_overlap+=1;return False
        atr=self._atr();w=zhi-zlo
        levels={d:(zhi-d/100*w if side>0 else zlo+d/100*w) for d in DEPTHS}
        self.zone={'side':side,'lo':zlo,'hi':zhi,'levels':levels,'armed_i':self.m1_i,'atr':atr,'filled':set()};self.zone_count+=1;return True
    def _try_arm_zone(self):
        side=self._break_side()
        if not side:return
        source=self.ifvgs if self.selected=='OB_IFVG' else self.bprs
        cands=[x for x in source if x['dir']==side and self.m1_i-x['born']<=20]
        if not cands:self.reject_no_poi+=1;return
        self.poi_candidates+=1
        # Most recent structural POI first.
        self._arm_from_poi(side,sorted(cands,key=lambda x:x['born'],reverse=True)[0])
    def on_bar(self,bar):
        o,h,l,c=map(self._f,[bar.open,bar.high,bar.low,bar.close]);self.o.append(o);self.h.append(h);self.l.append(l);self.c.append(c);self.m1_i+=1
        tr=max(h-l,abs(h-self.prev_close) if self.prev_close is not None else 0,abs(l-self.prev_close) if self.prev_close is not None else 0);self.tr.append(tr);self.prev_close=c
        self._update_fvgs_and_ifvgs()
        if self.zone and self.m1_i-self.zone['armed_i']>self.config.max_wait_bars:self.zone=None
        if self.zone is None:self._try_arm_zone()
    def summary(self):
        out={'selected':self.selected,'zones':self.zone_count,'poi_candidates':self.poi_candidates,'reject_no_poi':self.reject_no_poi,'reject_no_ob':self.rejected_no_ob,'reject_no_ob_overlap':self.reject_no_ob_overlap,'depths':{}}
        for d,s in self.stats.items():
            t=s['trades'];n=len(t);net=sum(x['pnl'] for x in t);pf=s['gw']/s['gl'] if s['gl']>0 else (math.inf if s['gw']>0 else 0.0);reasons=Counter(x['reason'] for x in t)
            out['depths'][str(d)]={'N':n,'WR_pct':100*s['wins']/max(n,1),'PF':pf,'net_virtual':net,'expectancy':net/max(n,1),'MFE_ATR_mean':float(np.mean([x['mfe_atr'] for x in t])) if n else 0.0,'MAE_ATR_mean':float(np.mean([x['mae_atr'] for x in t])) if n else 0.0,'exit_reasons':dict(reasons),'fills':s['fills']}
        return out

def ensure_l1(ticks):
    one=Quantity.from_int(1);out=[];rep=0
    for t in ticks:
        bs=float(t.bid_size.as_double()) if hasattr(t.bid_size,'as_double') else float(t.bid_size);az=float(t.ask_size.as_double()) if hasattr(t.ask_size,'as_double') else float(t.ask_size)
        if bs<=0 or az<=0:out.append(QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,bid_size=one if bs<=0 else t.bid_size,ask_size=one if az<=0 else t.ask_size,ts_event=t.ts_event,ts_init=t.ts_init));rep+=1
        else:out.append(t)
    return out,rep

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--experiment-id',required=True);ap.add_argument('--selected',choices=['OB_IFVG','OB_BPR'],required=True);a=ap.parse_args()
    catalog=ParquetDataCatalog(a.catalog);inst=next((x for x in catalog.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD'),None)
    if inst is None:raise SystemExit('XAUUSD missing')
    raw=catalog.query_quote_ticks(identifiers=[inst.id.value]);
    if not raw:raise SystemExit('no raw XAUUSD QuoteTicks')
    ticks,replaced=ensure_l1(raw);engine=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    engine.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal('2000'));engine.add_instrument(inst);engine.add_data(ticks)
    cfg=GateConfig(instrument_id=inst.id,m1=BarType.from_str(f'{inst.id.value}-1-MINUTE-BID-INTERNAL'),horizon_minutes=180);strat=POIBase(cfg,a.selected);engine.add_strategy(strat);engine.run();engine.end()
    result={'verification_level':'NAUTILUS_RAW_BIDASK_OB_IFVG_BPR_DYNAMIC_ATR_GATE','engine':'NautilusTrader BacktestEngine','nautilus_version':nautilus_trader.__version__,'raw_ticks':len(raw),'execution_ticks':len(ticks),'raw_zero_size_quotes_replaced':replaced,'ohlc_resample_used':False,'signal_bars':'Nautilus INTERNAL M1 from raw QuoteTicks','depths_geometry':[0,25,50,75,100],'exit_model':'DYNAMIC_ATR_NO_FIXED_TP',**strat.summary()}
    out=Path('results/multiedge')/a.experiment_id/f'{a.selected}.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
