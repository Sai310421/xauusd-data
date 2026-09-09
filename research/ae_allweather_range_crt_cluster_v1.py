from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
import math
import statistics


class Regime(Enum):
    UNKNOWN = auto(); RANGE = auto(); TRANSITION = auto(); EXPANSION = auto(); TREND = auto(); SHOCK = auto()
class PoiType(Enum):
    FVG = auto(); IFVG = auto(); BPR = auto(); BREAKER = auto()
class PoiState(Enum):
    ACTIVE = auto(); MITIGATED = auto(); BROKEN = auto(); INVALID = auto()
class Side(Enum):
    LONG = 1; SHORT = -1

@dataclass
class Bar:
    ts:int; o:float; h:float; l:float; c:float
@dataclass
class RangeBox:
    high:float; low:float; mid:float; width:float; valid:bool=False
@dataclass
class RangeSignals:
    bb_squeeze:float=0.0; adx_low:float=0.0; atr_low:float=0.0
    @property
    def score(self)->float: return 100.0*(self.bb_squeeze+self.adx_low+self.atr_low)/3.0
@dataclass
class CRTSignals:
    sweep:float=0.0; displacement:float=0.0; cisd:float=0.0; breaker:float=0.0; ifvg:float=0.0; bpr:float=0.0; fvg:float=0.0; adx_expansion:float=0.0; atr_expansion:float=0.0
    def raw_score(self)->float:
        return 30*self.sweep+20*self.displacement+20*self.cisd+15*self.breaker+10*self.ifvg+10*self.bpr+5*self.fvg+5*self.adx_expansion+5*self.atr_expansion
    def score_100(self)->float: return 100.0*self.raw_score()/120.0
@dataclass
class POI:
    poi_type:PoiType; side:Side; low:float; high:float; created_ts:int; state:PoiState=PoiState.ACTIVE; freshness:float=1.0; mitigation_depth:float=0.0; structure_score:float=0.0; parent_id:Optional[int]=None; id:int=0
    def contains(self,px:float)->bool: return self.low<=px<=self.high
    def width(self)->float:return max(self.high-self.low,1e-9)
    def score(self)->float:
        tb={PoiType.FVG:1.0,PoiType.IFVG:1.10,PoiType.BPR:1.20,PoiType.BREAKER:1.15}[self.poi_type]
        sm={PoiState.ACTIVE:1.0,PoiState.MITIGATED:.8,PoiState.BROKEN:.5,PoiState.INVALID:0.0}[self.state]
        mp=max(0.0,min(1.0,1.0-self.mitigation_depth))
        return 100*tb*sm*self.freshness*mp*max(0.0,min(1.0,self.structure_score))
@dataclass
class Position:
    id:int; side:Side; entry:float; qty:float; opened_ts:int; cluster_id:int; alpha_score:float=0.0
@dataclass
class Cluster:
    id:int; side:Side; position_ids:List[int]=field(default_factory=list); peak_pnl:float=0.0; runner:bool=False; closed:bool=False
@dataclass
class EngineConfig:
    range_gate:float=70.0; transition_gate:float=55.0; booster_gate:float=75.0; strong_booster_gate:float=90.0
    bb_period:int=20; bb_std:float=2.0; squeeze_lookback:int=100; adx_period:int=14; atr_period:int=14; range_lookback:int=30
    sweep_penetration_atr:float=.15; sweep_reclaim_atr:float=.20; displacement_body_atr:float=1.2; displacement_range_atr:float=1.5
    max_positions_per_cluster:int=5; cluster_tp:float=5.0; cluster_trail_activation:float=7.0; cluster_trail_distance:float=2.0; runner_threshold_score:float=90.0
    base_entry_rate:int=1; booster_entry_rate:int=3; strong_booster_entry_rate:int=5; poi_min_score:float=55.0

class IndicatorMath:
    @staticmethod
    def sma(xs,n): return None if len(xs)<n else sum(xs[-n:])/n
    @staticmethod
    def std(xs,n): return None if len(xs)<n else statistics.pstdev(xs[-n:])
    @staticmethod
    def atr(bars,n):
        if len(bars)<n+1:return None
        trs=[]
        for i in range(-n,0):
            cur,prev=bars[i],bars[i-1]
            trs.append(max(cur.h-cur.l,abs(cur.h-prev.c),abs(cur.l-prev.c)))
        return sum(trs)/len(trs)
    @staticmethod
    def adx_proxy(bars,n):
        if len(bars)<n+1:return None
        directional=0.0; total=0.0
        for i in range(-n,0):
            d=bars[i].c-bars[i-1].c; directional+=d; total+=abs(d)
        return 0.0 if total<=1e-12 else 100.0*abs(directional)/total

class POIStateMachine:
    def __init__(self): self.pois={}; self._next_id=1
    def add(self,poi): poi.id=self._next_id; self._next_id+=1; self.pois[poi.id]=poi; return poi.id
    def active(self,side=None):
        xs=[p for p in self.pois.values() if p.state!=PoiState.INVALID]
        return xs if side is None else [p for p in xs if p.side==side]
    def update_price(self,px,ts):
        for poi in self.pois.values():
            if poi.state==PoiState.INVALID:continue
            if poi.contains(px): poi.state=PoiState.MITIGATED
            if poi.poi_type==PoiType.FVG:
                if poi.side==Side.LONG and px<poi.low: poi.state=PoiState.BROKEN
                elif poi.side==Side.SHORT and px>poi.high: poi.state=PoiState.BROKEN
    def promote_broken_fvg_to_ifvg(self,structure_score,ts):
        for poi in list(self.pois.values()):
            if poi.poi_type!=PoiType.FVG or poi.state!=PoiState.BROKEN:continue
            if any(p.poi_type==PoiType.IFVG and p.parent_id==poi.id and p.state!=PoiState.INVALID for p in self.pois.values()):continue
            ns=Side.SHORT if poi.side==Side.LONG else Side.LONG
            self.add(POI(PoiType.IFVG,ns,poi.low,poi.high,ts,structure_score=structure_score,parent_id=poi.id))
    def detect_bpr(self,ts,structure_score):
        longs=[p for p in self.active(Side.LONG) if p.poi_type in (PoiType.FVG,PoiType.IFVG)]
        shorts=[p for p in self.active(Side.SHORT) if p.poi_type in (PoiType.FVG,PoiType.IFVG)]
        for a in longs:
            for b in shorts:
                lo,hi=max(a.low,b.low),min(a.high,b.high)
                if lo>=hi:continue
                if any(p.poi_type==PoiType.BPR and abs(p.low-lo)<1e-9 and abs(p.high-hi)<1e-9 and p.state!=PoiState.INVALID for p in self.pois.values()):continue
                side=a.side if a.score()>=b.score() else b.side
                self.add(POI(PoiType.BPR,side,lo,hi,ts,structure_score=structure_score))
    def best(self,side,min_score):
        xs=[p for p in self.active(side) if p.score()>=min_score]
        return max(xs,key=lambda p:p.score()) if xs else None

class ClusterBook:
    def __init__(self,cfg): self.cfg=cfg; self.positions={}; self.clusters={}; self._next_pos_id=1; self._next_cluster_id=1
    def _new_cluster(self,side):
        c=Cluster(self._next_cluster_id,side); self._next_cluster_id+=1; self.clusters[c.id]=c; return c
    def add_position(self,side,px,qty,ts,alpha_score):
        cand=[c for c in self.clusters.values() if not c.closed and c.side==side and len(c.position_ids)<self.cfg.max_positions_per_cluster and not c.runner]
        c=cand[-1] if cand else self._new_cluster(side)
        p=Position(self._next_pos_id,side,px,qty,ts,c.id,alpha_score); self._next_pos_id+=1; self.positions[p.id]=p; c.position_ids.append(p.id); return p
    def cluster_pnl(self,c,bid,ask):
        pnl=0.0
        for pid in c.position_ids:
            p=self.positions.get(pid)
            if p is None:continue
            mark=bid if p.side==Side.LONG else ask
            pnl+=(mark-p.entry)*p.qty*p.side.value
        return pnl
    def evaluate_exits(self,bid,ask,crt_score):
        out=[]
        for c in self.clusters.values():
            if c.closed:continue
            pnl=self.cluster_pnl(c,bid,ask); c.peak_pnl=max(c.peak_pnl,pnl)
            if crt_score>=self.cfg.runner_threshold_score and pnl>0:c.runner=True
            if not c.runner and pnl>=self.cfg.cluster_tp: out.append((c.id,'CLUSTER_TP',pnl)); continue
            if c.peak_pnl>=self.cfg.cluster_trail_activation and pnl<=c.peak_pnl-self.cfg.cluster_trail_distance: out.append((c.id,'CLUSTER_TRAIL',pnl))
        return out
    def close_cluster(self,cid):
        c=self.clusters[cid]; c.closed=True
        for pid in c.position_ids:self.positions.pop(pid,None)

class AEAllWeatherEngine:
    def __init__(self,cfg=EngineConfig()):
        self.cfg=cfg; self.bars=deque(maxlen=500); self.regime=Regime.UNKNOWN; self.range_box=RangeBox(0,0,0,0,False); self.poi=POIStateMachine(); self.cluster_book=ClusterBook(cfg); self.last_range_signals=RangeSignals(); self.last_crt_signals=CRTSignals(); self.last_side_bias=None
    def _range_box(self):
        if len(self.bars)<self.cfg.range_lookback:return RangeBox(0,0,0,0,False)
        xs=list(self.bars)[-self.cfg.range_lookback:]; hi=max(b.h for b in xs); lo=min(b.l for b in xs); return RangeBox(hi,lo,(hi+lo)/2,max(hi-lo,1e-9),True)
    def _range_signals(self):
        bars=list(self.bars); closes=[b.c for b in bars]
        if len(bars)<max(self.cfg.squeeze_lookback,self.cfg.bb_period+1):return RangeSignals()
        ma=IndicatorMath.sma(closes,self.cfg.bb_period); sd=IndicatorMath.std(closes,self.cfg.bb_period); atr=IndicatorMath.atr(bars,self.cfg.atr_period); adx=IndicatorMath.adx_proxy(bars,self.cfg.adx_period)
        if None in (ma,sd,atr,adx):return RangeSignals()
        cur_bw=4*sd/max(ma,1e-9); widths=[]
        for i in range(self.cfg.bb_period,len(closes)+1):
            w=closes[i-self.cfg.bb_period:i]
            if len(w)==self.cfg.bb_period:
                m=sum(w)/len(w); s=statistics.pstdev(w); widths.append(4*s/max(m,1e-9))
        bw=widths[-self.cfg.squeeze_lookback:]; bb=sum(1 for w in bw if w>=cur_bw)/max(1,len(bw)); adxl=max(0,min(1,(25-adx)/15))
        atrs=[]
        for n in range(self.cfg.atr_period+1,len(bars)+1):
            a=IndicatorMath.atr(bars[:n],self.cfg.atr_period)
            if a is not None:atrs.append(a)
        ar=atrs[-self.cfg.squeeze_lookback:]; atrl=sum(1 for x in ar if x>=atr)/max(1,len(ar)) if ar else 0
        return RangeSignals(max(0,min(1,bb)),adxl,max(0,min(1,atrl)))
    def _detect_sweep(self,bar,box,atr):
        if not box.valid or atr<=0:return 0.0,None
        up=max(0,bar.h-box.high); dn=max(0,box.low-bar.l)
        if up>=self.cfg.sweep_penetration_atr*atr and bar.c<box.high:return min(1,up/atr+max(0,(box.high-bar.c)/atr)),Side.SHORT
        if dn>=self.cfg.sweep_penetration_atr*atr and bar.c>box.low:return min(1,dn/atr+max(0,(bar.c-box.low)/atr)),Side.LONG
        return 0.0,None
    def _detect_displacement(self,bar,atr,side):
        if atr<=0 or side is None:return 0.0
        body=abs(bar.c-bar.o); rng=bar.h-bar.l; ok=(side==Side.LONG and bar.c>bar.o) or (side==Side.SHORT and bar.c<bar.o)
        if not ok:return 0.0
        return .5*min(1,body/(self.cfg.displacement_body_atr*atr))+.5*min(1,rng/(self.cfg.displacement_range_atr*atr))
    def _detect_fvg(self,side,structure_score):
        if len(self.bars)<3:return
        a,b,c=list(self.bars)[-3:]
        if side==Side.LONG and a.h<c.l:self.poi.add(POI(PoiType.FVG,Side.LONG,a.h,c.l,c.ts,structure_score=structure_score))
        elif side==Side.SHORT and a.l>c.h:self.poi.add(POI(PoiType.FVG,Side.SHORT,c.h,a.l,c.ts,structure_score=structure_score))
    def _cisd(self,side):
        if side is None or len(self.bars)<5:return 0.0
        xs=list(self.bars)[-5:]; ds=[xs[i].c-xs[i-1].c for i in range(1,5)]
        return sum(1 for d in ds if d>0)/4 if side==Side.LONG else sum(1 for d in ds if d<0)/4
    def _mss(self,side):
        if side is None or len(self.bars)<10:return 0.0
        xs=list(self.bars); cur=xs[-1]; prev=xs[-10:-1]
        return float(cur.c>max(b.h for b in prev)) if side==Side.LONG else float(cur.c<min(b.l for b in prev))
    def _eq(self,side):
        if side is None or not self.range_box.valid or len(self.bars)<2:return 0.0
        prev,cur=list(self.bars)[-2:]; mid=self.range_box.mid
        return float(prev.c<=mid and cur.c>mid) if side==Side.LONG else float(prev.c>=mid and cur.c<mid)
    def on_bar(self,bar):
        self.bars.append(bar); bars=list(self.bars); atr=IndicatorMath.atr(bars,self.cfg.atr_period) or 0; adx=IndicatorMath.adx_proxy(bars,self.cfg.adx_period) or 0; adx0=IndicatorMath.adx_proxy(bars[:-1],self.cfg.adx_period) or adx; atr0=IndicatorMath.atr(bars[:-1],self.cfg.atr_period) or atr
        self.range_box=self._range_box(); self.last_range_signals=self._range_signals()
        if self.last_range_signals.score>=self.cfg.range_gate and self.regime in (Regime.UNKNOWN,Regime.RANGE):self.regime=Regime.RANGE
        sw,side=self._detect_sweep(bar,self.range_box,atr)
        if side is not None:self.last_side_bias=side
        disp=self._detect_displacement(bar,atr,self.last_side_bias); cisd=self._cisd(self.last_side_bias); mss=self._mss(self.last_side_bias); eq=self._eq(self.last_side_bias); struct=min(1,.35*cisd+.35*mss+.30*eq)
        if self.last_side_bias is not None and disp>.35:self._detect_fvg(self.last_side_bias,struct)
        self.poi.update_price(bar.c,bar.ts); self.poi.promote_broken_fvg_to_ifvg(struct,bar.ts); self.poi.detect_bpr(bar.ts,struct)
        best=self.poi.best(self.last_side_bias,self.cfg.poi_min_score) if self.last_side_bias is not None else None
        adxe=min(1,max(0,(adx-adx0)/10)) if adx>adx0 else 0; atre=min(1,max(0,(atr/max(atr0,1e-9)-1)/.5)) if atr>atr0 else 0
        self.last_crt_signals=CRTSignals(sw,disp,min(1,.5*cisd+.5*mss),float(bool(best and best.poi_type==PoiType.BREAKER)),float(bool(best and best.poi_type==PoiType.IFVG)),float(bool(best and best.poi_type==PoiType.BPR)),float(bool(best and best.poi_type==PoiType.FVG)),adxe,atre)
        crt=self.last_crt_signals.score_100()
        if self.regime==Regime.RANGE and crt>=self.cfg.transition_gate:self.regime=Regime.TRANSITION
        if self.regime==Regime.TRANSITION and crt>=self.cfg.booster_gate:self.regime=Regime.EXPANSION
        if self.regime==Regime.EXPANSION and crt<self.cfg.transition_gate:self.regime=Regime.UNKNOWN
        return {'range_score':self.last_range_signals.score,'crt_score':crt,'poi_score':best.score() if best else 0.0,'regime':self.regime.name}
    def desired_entry_side(self):
        if self.regime==Regime.RANGE and self.range_box.valid and self.bars:
            px=self.bars[-1].c; upper=self.range_box.mid+.25*self.range_box.width; lower=self.range_box.mid-.25*self.range_box.width
            return Side.LONG if px<=lower else Side.SHORT if px>=upper else None
        if self.regime==Regime.EXPANSION:return self.last_side_bias
        return None
