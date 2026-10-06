from __future__ import annotations
import argparse, json, math, os, hashlib, datetime
from dataclasses import dataclass, field
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
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, BookType
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

TF_MIN=(5,)
INITIAL=1000.0
LEVERAGE=2000.0
BASE_QTY=1.0
DEEP=0.804
TIMEOUT_NS=240*60*1_000_000_000
G75_TRIGGER=0.12
G75_ADD=0.025
G75_REVERSAL_DEFAULT=0.20
G75_MAX_LAYERS=10
M1_SETUP_TIMEOUT_NS=180*60*1_000_000_000
M1_TRADE_TIMEOUT_NS=240*60*1_000_000_000
M1_POI_TOL_ATR=0.10
M1_SL_BUFFER_ATR=0.10
COMMISSION_RT_PER_LOT=7.0
CASHBACK_RT_PER_LOT=6.0

def ff(x):
    return float(x.as_double()) if hasattr(x,'as_double') else float(x)

@dataclass
class BarState:
    start_ns:int=0; o:float=0.0; h:float=0.0; l:float=0.0; c:float=0.0
    hist:list=field(default_factory=list)

@dataclass
class Setup:
    tf:int; direction:int; origin:float; end:float; atr:float; signal_ns:int
    touched:bool=False; touch_ns:int=0; invalid:bool=False; entered:bool=False
    pending_ns:int=0; mss_close:float=0.0
    sl:float=0.0; tp:float=0.0

@dataclass
class Leg:
    tf:int; direction:int; qty:float; entry:float; entry_ns:int; sl:float; tp:float; expiry_ns:int
    active:bool=True
    leg_id:int=0
    kind:str='BASE'
    parent_id:int=0
    g75_triggered:bool=False
    g75_done:bool=False
    g75_last_add:float=0.0
    g75_extreme:float=0.0
    g75_layers:int=0


@dataclass
class M1POISetup:
    pattern:str
    direction:int
    poi:float
    sl:float
    target:float
    atr:float
    signal_ns:int
    key:str
    touched:bool=False
    touch_ns:int=0
    pending_ns:int=0
    invalid:bool=False
    entered:bool=False

@dataclass
class M1POILeg:
    pattern:str
    direction:int
    entry:float
    sl:float
    tp:float
    entry_ns:int
    expiry_ns:int
    active:bool=True

class Cfg(StrategyConfig, frozen=True):
    instrument_id: object
    mode: str='base'
    g75_reversal: float=G75_REVERSAL_DEFAULT

class M1LinePOIPlusM5G75OverlapV19(Strategy):
    def __init__(self,c):
        super().__init__(c)
        self.bars={m:BarState() for m in TF_MIN}
        self.setups={m:[] for m in TF_MIN}
        self.legs=[]; self.closed=[]
        self.stats={m:{'candidates':0,'touched':0,'invalid_before_entry':0,'mss':0,'entries':0} for m in TF_MIN}
        self.last_bid=None; self.last_ask=None; self.last_ns=0; self.submitted=0
        self.realized=0.0; self.peak_equity=INITIAL; self.max_dd=0.0; self.min_ml=math.inf; self.max_gross=0.0
        self.first_ns=None; self.last_seen_ns=None
        self.next_leg_id=1
        self.g75_stats={'triggered_bases':0,'layers_opened':0,'reversal_events':0,'base_exit_events':0,'max_layers_per_base':0}
        # M1 is a separate line-chart/POI engine. It does not gate or modify M5.
        self.m1_bar=BarState()
        self.m1_pivots=[]
        self.m1_seen=set()
        self.m1_setups=[]
        self.m1_legs=[]
        self.m1_closed=[]
        self.m1_stats={p:{'detected':0,'touched':0,'entries':0,'tp':0,'sl':0,'timeout':0,'invalid':0}
                       for p in ['CLASSIC_V_BUY','CLASSIC_A_SELL','QM_BUY','QM_SELL','OCL_BUY','OCL_SELL']}
        # Direction-overlap diagnostics. Durations are wall-clock time, not tick counts.
        self.ov_last_ns=None
        self.ov_prev_same=False
        self.ov_prev_opp=False
        self.ov_same_start=None
        self.ov_opp_start=None
        self.ov_same_ns=0
        self.ov_opp_ns=0
        self.ov_same_episodes=[]
        self.ov_opp_episodes=[]
        self.ov_same_context=[]
        self.ov_opp_context=[]

    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id)

    def _submit(self,d,qty):
        inst=self.cache.instrument(self.config.instrument_id)
        q=inst.make_qty(Decimal(str(qty)))
        o=self.order_factory.market(instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY if d>0 else OrderSide.SELL,quantity=q)
        self.submit_order(o); self.submitted+=1

    def _bar_update(self,m,ts,px):
        span=m*60*1_000_000_000; bucket=(ts//span)*span; b=self.bars[m]
        if b.start_ns==0:
            b.start_ns=bucket; b.o=b.h=b.l=b.c=px; return False
        if bucket==b.start_ns:
            b.h=max(b.h,px); b.l=min(b.l,px); b.c=px; return False
        b.hist.append((b.start_ns,b.o,b.h,b.l,b.c))
        if len(b.hist)>512:b.hist=b.hist[-512:]
        b.start_ns=bucket; b.o=b.h=b.l=b.c=px
        return True

    def _atr(self,hist):
        if len(hist)<16:return 0.0
        tr=[]
        for i in range(1,len(hist)):
            _,o,h,l,c=hist[i]; pc=hist[i-1][4]
            tr.append(max(h-l,abs(h-pc),abs(l-pc)))
        # Wilder-style EWM alpha 1/14 over available history, matching the prior unified diagnostic.
        a=tr[0]
        for x in tr[1:]: a=a+(x-a)/14.0
        return float(a)


    # ========================================================
    # M1 LINE-CHART / POI ENGINE (independent from M5)
    # ========================================================

    def _m1_bar_update(self,ts,px):
        span=60*1_000_000_000; bucket=(ts//span)*span; b=self.m1_bar
        if b.start_ns==0:
            b.start_ns=bucket; b.o=b.h=b.l=b.c=px; return False
        if bucket==b.start_ns:
            b.h=max(b.h,px); b.l=min(b.l,px); b.c=px; return False
        b.hist.append((b.start_ns,b.o,b.h,b.l,b.c))
        if len(b.hist)>1024:b.hist=b.hist[-1024:]
        b.start_ns=bucket; b.o=b.h=b.l=b.c=px
        return True

    def _m1_atr(self):
        h=self.m1_bar.hist
        if len(h)<16:return 0.0
        tr=[]
        for i in range(1,len(h)):
            _,o,hi,lo,c=h[i]; pc=h[i-1][4]
            tr.append(max(hi-lo,abs(hi-pc),abs(lo-pc)))
        a=tr[0]
        for x in tr[1:]:a=a+(x-a)/14.0
        return float(a)

    def _m1_update_pivot(self):
        h=self.m1_bar.hist
        if len(h)<5:return
        idx=len(h)-3
        if any(p['idx']==idx for p in self.m1_pivots):return
        c=h[idx][4]
        around=[h[j][4] for j in (idx-2,idx-1,idx+1,idx+2)]
        typ=None
        if c<min(around):typ='L'
        elif c>max(around):typ='H'
        if typ:
            self.m1_pivots.append({'type':typ,'idx':idx,'price':c,'bar':h[idx]})
            if len(self.m1_pivots)>80:self.m1_pivots=self.m1_pivots[-80:]

    def _m1_prior_pivot(self,typ,before_idx):
        xs=[p for p in self.m1_pivots if p['type']==typ and p['idx']<before_idx]
        return xs[-1] if xs else None

    def _m1_add_setup(self,pattern,direction,poi,sl,target,atr,signal_ns,key):
        if key in self.m1_seen:return
        if atr<=0 or not all(math.isfinite(x) for x in [poi,sl,target]):return
        if direction>0 and not sl<poi:return
        if direction<0 and not sl>poi:return
        self.m1_seen.add(key)
        self.m1_setups.append(M1POISetup(pattern,direction,poi,sl,target,atr,signal_ns,key))
        self.m1_stats[pattern]['detected']+=1

    def _m1_detect_patterns(self):
        h=self.m1_bar.hist
        if len(h)<25:return
        self._m1_update_pivot()
        i=len(h)-1
        ts,o,hi,lo,c=h[i]
        atr=self._m1_atr()
        if atr<=0:return
        prevc=h[i-1][4]
        piv=self.m1_pivots

        # Classic V/A: Close-line pivot plus two-bar line reversal.
        lows=[p for p in piv if p['type']=='L' and 2<=i-p['idx']<=8]
        if lows:
            p=lows[-1]
            if c>prevc and c>h[i-2][4]:
                ph=self._m1_prior_pivot('H',p['idx'])
                target=ph['price'] if ph else c+2.0*atr
                sl=min(x[3] for x in h[max(0,p['idx']-1):p['idx']+2])-M1_SL_BUFFER_ATR*atr
                self._m1_add_setup('CLASSIC_V_BUY',1,p['price'],sl,target,atr,ts,f"CVB:{p['idx']}")

        highs=[p for p in piv if p['type']=='H' and 2<=i-p['idx']<=8]
        if highs:
            p=highs[-1]
            if c<prevc and c<h[i-2][4]:
                pl=self._m1_prior_pivot('L',p['idx'])
                target=pl['price'] if pl else c-2.0*atr
                sl=max(x[2] for x in h[max(0,p['idx']-1):p['idx']+2])+M1_SL_BUFFER_ATR*atr
                self._m1_add_setup('CLASSIC_A_SELL',-1,p['price'],sl,target,atr,ts,f"CAS:{p['idx']}")

        # Quasimodo: L-H-lower-L then break H; mirrored for sell.
        if len(piv)>=3:
            a,b,d=piv[-3],piv[-2],piv[-1]
            if a['type']=='L' and b['type']=='H' and d['type']=='L' and d['price']<a['price']-0.05*atr and c>b['price']:
                sl=min(x[3] for x in h[max(0,d['idx']-1):d['idx']+2])-M1_SL_BUFFER_ATR*atr
                target=max(c+2.0*atr,b['price']+(b['price']-a['price']))
                self._m1_add_setup('QM_BUY',1,a['price'],sl,target,atr,ts,f"QMB:{a['idx']}:{d['idx']}")
            if a['type']=='H' and b['type']=='L' and d['type']=='H' and d['price']>a['price']+0.05*atr and c<b['price']:
                sl=max(x[2] for x in h[max(0,d['idx']-1):d['idx']+2])+M1_SL_BUFFER_ATR*atr
                target=min(c-2.0*atr,b['price']-(a['price']-b['price']))
                self._m1_add_setup('QM_SELL',-1,a['price'],sl,target,atr,ts,f"QMS:{a['idx']}:{d['idx']}")

        # Open-Close-Level: body-to-body reversal near a local extreme.
        # Structure is still read from Close; POI itself is an Open/Close body level.
        po,phh,pll,pc=h[i-1][1],h[i-1][2],h[i-1][3],h[i-1][4]
        avg_body=float(np.mean([abs(x[4]-x[1]) for x in h[i-20:i]]))
        body=abs(c-o)
        if avg_body>0 and body>=avg_body:
            local_lo=min(x[3] for x in h[i-5:i+1])
            local_hi=max(x[2] for x in h[i-5:i+1])
            # Last bearish body -> bullish body closes through its open, at/near local low.
            if pc<po and c>o and c>po and min(lo,pll)<=local_lo+0.15*atr:
                poi=max(po,pc)
                pp=self._m1_prior_pivot('H',i)
                target=pp['price'] if pp else c+2.0*atr
                sl=min(local_lo,pll)-M1_SL_BUFFER_ATR*atr
                self._m1_add_setup('OCL_BUY',1,poi,sl,target,atr,ts,f"OCLB:{i-1}")
            # Last bullish body -> bearish body closes through its open, at/near local high.
            if pc>po and c<o and c<po and max(hi,phh)>=local_hi-0.15*atr:
                poi=min(po,pc)
                pp=self._m1_prior_pivot('L',i)
                target=pp['price'] if pp else c-2.0*atr
                sl=max(local_hi,phh)+M1_SL_BUFFER_ATR*atr
                self._m1_add_setup('OCL_SELL',-1,poi,sl,target,atr,ts,f"OCLS:{i-1}")

    def _m1_after_close(self,ts):
        self._m1_detect_patterns()
        h=self.m1_bar.hist
        if len(h)<2:return
        c=h[-1][4]; pc=h[-2][4]
        for s in self.m1_setups:
            if s.invalid or s.entered or not s.touched or s.pending_ns:continue
            if ts-s.signal_ns>M1_SETUP_TIMEOUT_NS:continue
            confirm=(c>s.poi and c>pc) if s.direction>0 else (c<s.poi and c<pc)
            if confirm:s.pending_ns=ts

    def _m1_open(self,s,bid,ask,ts):
        px=ask if s.direction>0 else bid
        risk=(px-s.sl)*s.direction
        if risk<=0:return False
        tp=s.target
        # Image interpretation prefers prior structural target. If already behind fill,
        # use a transparent 2R fallback rather than inventing a different entry.
        if (tp-px)*s.direction<=0:
            tp=px+s.direction*2.0*risk
        self.m1_legs.append(M1POILeg(s.pattern,s.direction,px,s.sl,tp,ts,ts+M1_TRADE_TIMEOUT_NS,True))
        self.m1_stats[s.pattern]['entries']+=1
        s.entered=True
        return True

    def _m1_setup_tick(self,bid,ask,ts):
        mid=(bid+ask)/2.0
        keep=[]
        for s in self.m1_setups:
            if s.entered or s.invalid:continue
            if ts-s.signal_ns>M1_SETUP_TIMEOUT_NS:continue
            invalid=(bid<=s.sl) if s.direction>0 else (ask>=s.sl)
            if invalid:
                s.invalid=True; self.m1_stats[s.pattern]['invalid']+=1; continue
            if not s.touched and ts>s.signal_ns:
                tol=M1_POI_TOL_ATR*s.atr
                touch=(mid<=s.poi+tol) if s.direction>0 else (mid>=s.poi-tol)
                if touch:
                    s.touched=True; s.touch_ns=ts; self.m1_stats[s.pattern]['touched']+=1
            if s.pending_ns and ts>=s.pending_ns:
                if not self._m1_open(s,bid,ask,ts):
                    s.invalid=True; self.m1_stats[s.pattern]['invalid']+=1
                continue
            keep.append(s)
        self.m1_setups=keep

    def _m1_close(self,x,bid,ask,ts,reason):
        if not x.active:return
        px=bid if x.direction>0 else ask
        gross=(px-x.entry)*x.direction*BASE_QTY
        cost=(BASE_QTY/100.0)*COMMISSION_RT_PER_LOT
        cb=(BASE_QTY/100.0)*CASHBACK_RT_PER_LOT
        pnl=gross-cost+cb
        x.active=False
        self.m1_closed.append({'tf':1,'pattern':x.pattern,'direction':x.direction,'qty':BASE_QTY,
                               'entry':x.entry,'exit':px,'sl':x.sl,'tp':x.tp,'pnl':pnl,
                               'entry_ns':x.entry_ns,'exit_ns':ts,'reason':reason})
        k='tp' if reason=='TP' else ('sl' if reason=='SL' else 'timeout')
        self.m1_stats[x.pattern][k]+=1

    def _m1_manage(self,bid,ask,ts):
        for x in list(self.m1_legs):
            if not x.active:continue
            mark=bid if x.direction>0 else ask
            if (mark<=x.sl if x.direction>0 else mark>=x.sl):
                self._m1_close(x,bid,ask,ts,'SL')
            elif (mark>=x.tp if x.direction>0 else mark<=x.tp):
                self._m1_close(x,bid,ask,ts,'TP')
            elif ts>=x.expiry_ns:
                self._m1_close(x,bid,ask,ts,'TIMEOUT')

    def _overlap_tick(self,ts):
        if self.ov_last_ns is not None and ts>=self.ov_last_ns:
            dt=ts-self.ov_last_ns
            if self.ov_prev_same:self.ov_same_ns+=dt
            if self.ov_prev_opp:self.ov_opp_ns+=dt

        m1dirs=[x.direction for x in self.m1_legs if x.active]
        m5dirs=[x.direction for x in self.legs if x.active and x.kind=='BASE']
        same=any(a==b for a in m1dirs for b in m5dirs)
        opp=any(a!=b for a in m1dirs for b in m5dirs)

        if same and not self.ov_prev_same:
            self.ov_same_start=ts
            self.ov_same_context.append({'start_ns':ts,
                'm1':[{'pattern':x.pattern,'direction':x.direction,'entry_ns':x.entry_ns,'entry':x.entry}
                      for x in self.m1_legs if x.active],
                'm5':[{'leg_id':x.leg_id,'direction':x.direction,'entry_ns':x.entry_ns,'entry':x.entry}
                      for x in self.legs if x.active and x.kind=='BASE']})
        if not same and self.ov_prev_same and self.ov_same_start is not None:
            self.ov_same_episodes.append((self.ov_same_start,ts)); self.ov_same_start=None
        if opp and not self.ov_prev_opp:
            self.ov_opp_start=ts
            self.ov_opp_context.append({'start_ns':ts,
                'm1':[{'pattern':x.pattern,'direction':x.direction,'entry_ns':x.entry_ns,'entry':x.entry,
                       'sl':x.sl,'tp':x.tp}
                      for x in self.m1_legs if x.active],
                'm5':[{'leg_id':x.leg_id,'direction':x.direction,'entry_ns':x.entry_ns,'entry':x.entry,
                       'sl':x.sl,'tp':x.tp}
                      for x in self.legs if x.active and x.kind=='BASE']})
        if not opp and self.ov_prev_opp and self.ov_opp_start is not None:
            self.ov_opp_episodes.append((self.ov_opp_start,ts)); self.ov_opp_start=None

        self.ov_prev_same=same; self.ov_prev_opp=opp; self.ov_last_ns=ts

    def _overlap_result(self):
        same=list(self.ov_same_episodes); opp=list(self.ov_opp_episodes)
        if self.ov_prev_same and self.ov_same_start is not None and self.last_ns:
            same.append((self.ov_same_start,self.last_ns))
        if self.ov_prev_opp and self.ov_opp_start is not None and self.last_ns:
            opp.append((self.ov_opp_start,self.last_ns))
        def pack(xs,total_ns):
            durations=[max(0,b-a)/1e9 for a,b in xs]
            examples=[]
            for a,b in xs[:20]:
                examples.append({'start_utc':str(pd.Timestamp(a,unit='ns',tz='UTC')),
                                 'end_utc':str(pd.Timestamp(b,unit='ns',tz='UTC')),
                                 'duration_sec':(b-a)/1e9})
            return {'episodes':len(xs),'total_seconds':total_ns/1e9,
                    'max_episode_seconds':max(durations) if durations else 0.0,
                    'examples':examples}
        same_pack=pack(same,self.ov_same_ns); opp_pack=pack(opp,self.ov_opp_ns)
        for dst,ctx in [(same_pack,self.ov_same_context),(opp_pack,self.ov_opp_context)]:
            dst['contexts']=[]
            for z in ctx:
                q=dict(z)
                q['start_utc']=str(pd.Timestamp(q.pop('start_ns'),unit='ns',tz='UTC'))
                for side in ['m1','m5']:
                    for x in q[side]:
                        x['entry_utc']=str(pd.Timestamp(x['entry_ns'],unit='ns',tz='UTC'))
                dst['contexts'].append(q)
        return {'same_direction':same_pack,'opposite_direction':opp_pack}

    def _detect(self,m):
        hist=self.bars[m].hist
        if len(hist)<35:return
        i=len(hist)-1; ts,o,h,l,c=hist[i]
        prev20=hist[i-20:i]
        avg_body=float(np.mean([abs(x[4]-x[1]) for x in prev20]))
        atr=self._atr(hist)
        if avg_body<=0 or atr<=0 or abs(c-o)<1.6*avg_body or (h-l)<0.9*atr:return
        ph=max(x[2] for x in hist[i-3:i]); pl=min(x[3] for x in hist[i-3:i])
        bull=c>o and c>ph; bear=c<o and c<pl
        bull_origin=None; bear_origin=None
        for off in range(2,8):
            idx=len(hist)-off
            if idx<12:continue
            bar=hist[idx]; older=hist[idx-12:idx]
            ol=min(x[3] for x in older); oh=max(x[2] for x in older)
            if bull_origin is None and bar[3]<ol and bar[4]>ol: bull_origin=bar[3]
            if bear_origin is None and bar[2]>oh and bar[4]<oh: bear_origin=bar[2]
        s=None
        if bull and bull_origin is not None and h>bull_origin:
            s=Setup(m,1,bull_origin,h,atr,ts)
        elif bear and bear_origin is not None and bear_origin>l:
            s=Setup(m,-1,bear_origin,l,atr,ts)
        if s:
            span=abs(s.end-s.origin); s.sl=s.origin-s.direction*0.2*s.atr; s.tp=s.end+s.direction*0.618*span
            self.setups[m].append(s); self.stats[m]['candidates']+=1

    def _mss(self,m,s):
        hist=self.bars[m].hist
        if len(hist)<2:return False
        a,b=hist[-1],hist[-2]
        return a[4]>b[2] if s.direction>0 else a[4]<b[3]

    def _after_close(self,m,ts):
        self._detect(m)
        for s in self.setups[m]:
            if s.invalid or s.entered or not s.touched or s.pending_ns:continue
            if ts-s.signal_ns>TIMEOUT_NS:continue
            if self._mss(m,s):
                s.pending_ns=ts
                s.mss_close=self.bars[m].hist[-1][4]
                self.stats[m]['mss']+=1

    def _setup_tick(self,m,bid,ask,ts):
        keep=[]
        for s in self.setups[m]:
            if s.entered:continue
            if ts-s.signal_ns>TIMEOUT_NS:continue
            inv = bid<=s.sl if s.direction>0 else ask>=s.sl
            if inv:
                s.invalid=True; self.stats[m]['invalid_before_entry']+=1; continue
            level=s.end-s.direction*abs(s.end-s.origin)*DEEP
            if not s.touched:
                touch = bid<=level if s.direction>0 else ask>=level
                if touch:
                    s.touched=True; s.touch_ns=ts; self.stats[m]['touched']+=1
            if s.pending_ns and ts>=s.pending_ns:
                px=ask if s.direction>0 else bid
                # enforce logical stop relation at the actual fill
                if (s.direction>0 and px<=s.sl) or (s.direction<0 and px>=s.sl):
                    s.invalid=True; self.stats[m]['invalid_before_entry']+=1; continue
                self._submit(s.direction,BASE_QTY)
                lid=self.next_leg_id; self.next_leg_id+=1
                self.legs.append(Leg(m,s.direction,BASE_QTY,px,ts,s.sl,s.tp,s.signal_ns+TIMEOUT_NS,
                                     True,lid,'BASE',lid))
                s.entered=True; self.stats[m]['entries']+=1
                continue
            keep.append(s)
        self.setups[m]=keep

    def _close_leg(self,x,bid,ask,ts,reason):
        if not x.active:return
        px=bid if x.direction>0 else ask
        lots=x.qty/100.0
        gross=(px-x.entry)*x.direction*x.qty
        cost=lots*COMMISSION_RT_PER_LOT
        cb=lots*CASHBACK_RT_PER_LOT
        pnl=gross-cost+cb
        self._submit(-x.direction,x.qty); x.active=False; self.realized+=pnl
        self.closed.append({'tf':x.tf,'direction':x.direction,'qty':x.qty,'entry':x.entry,'exit':px,'gross':gross,
            'commission_rt':cost,'cashback_rt':cb,'pnl':pnl,'entry_ns':x.entry_ns,'exit_ns':ts,'reason':reason,
            'kind':x.kind,'leg_id':x.leg_id,'parent_id':x.parent_id})

    def _open_g75_layer(self,base,bid,ask,ts):
        if base.g75_layers>=G75_MAX_LAYERS:return False
        px=ask if base.direction>0 else bid
        lid=self.next_leg_id; self.next_leg_id+=1
        self._submit(base.direction,BASE_QTY)
        self.legs.append(Leg(base.tf,base.direction,BASE_QTY,px,ts,base.sl,base.tp,base.expiry_ns,
                             True,lid,'G75',base.leg_id))
        base.g75_layers+=1
        self.g75_stats['layers_opened']+=1
        self.g75_stats['max_layers_per_base']=max(self.g75_stats['max_layers_per_base'],base.g75_layers)
        return True

    def _close_g75_children(self,base,bid,ask,ts,reason):
        any_closed=False
        for x in list(self.legs):
            if x.active and x.kind=='G75' and x.parent_id==base.leg_id:
                self._close_leg(x,bid,ask,ts,reason); any_closed=True
        return any_closed

    def _manage_g75(self,bid,ask,ts):
        if self.config.mode!='g75':return
        for base in list(self.legs):
            if not base.active or base.kind!='BASE' or base.g75_done:continue
            mark=bid if base.direction>0 else ask

            if not base.g75_triggered:
                trigger_level=base.entry+base.direction*G75_TRIGGER
                crossed=mark>=trigger_level if base.direction>0 else mark<=trigger_level
                if not crossed:continue
                base.g75_triggered=True
                base.g75_extreme=mark
                if self._open_g75_layer(base,bid,ask,ts):
                    base.g75_last_add=ask if base.direction>0 else bid
                    self.g75_stats['triggered_bases']+=1
            else:
                base.g75_extreme=max(base.g75_extreme,mark) if base.direction>0 else min(base.g75_extreme,mark)

            while base.g75_triggered and base.g75_layers<G75_MAX_LAYERS:
                nxt=base.g75_last_add+base.direction*G75_ADD
                crossed=mark>=nxt if base.direction>0 else mark<=nxt
                if not crossed:break
                if not self._open_g75_layer(base,bid,ask,ts):break
                base.g75_last_add=nxt

            reversal_hit=(mark<=base.g75_extreme-self.config.g75_reversal) if base.direction>0 else (mark>=base.g75_extreme+self.config.g75_reversal)
            if base.g75_triggered and reversal_hit:
                self._close_g75_children(base,bid,ask,ts,'G75_REVERSAL')
                base.g75_done=True
                self.g75_stats['reversal_events']+=1

    def _manage_legs(self,bid,ask,ts):
        # Base M5 trade keeps its original SL/TP/TIMEOUT. G75 is an overlay only.
        for x in list(self.legs):
            if not x.active or x.kind!='BASE':continue
            mark=bid if x.direction>0 else ask
            reason=None
            if (mark<=x.sl if x.direction>0 else mark>=x.sl):
                reason='SL'
            elif (mark>=x.tp if x.direction>0 else mark<=x.tp):
                reason='TP'
            elif ts>=x.expiry_ns:
                reason='TIMEOUT'
            if reason is not None:
                self._close_g75_children(x,bid,ask,ts,'BASE_'+reason)
                if x.g75_triggered:self.g75_stats['base_exit_events']+=1
                self._close_leg(x,bid,ask,ts,reason)

    def _risk(self,bid,ask):
        floating=0.0; gross=0.0
        for x in self.legs:
            if not x.active:continue
            mark=bid if x.direction>0 else ask
            floating+=(mark-x.entry)*x.direction*x.qty; gross+=x.qty
        eq=INITIAL+self.realized+floating; self.peak_equity=max(self.peak_equity,eq); self.max_dd=max(self.max_dd,self.peak_equity-eq)
        self.max_gross=max(self.max_gross,gross)
        if gross>0:
            margin=gross*((bid+ask)/2)/LEVERAGE
            if margin>0:self.min_ml=min(self.min_ml,eq/margin*100.0)

    def on_quote_tick(self,t):
        bid,ask=ff(t.bid_price),ff(t.ask_price); ts=int(t.ts_event)
        self.last_bid,self.last_ask,self.last_ns=bid,ask,ts
        if self.first_ns is None:self.first_ns=ts
        self.last_seen_ns=ts

        # M1 and M5 are intentionally independent engines.
        m1_closed=self._m1_bar_update(ts,bid)
        self._m1_setup_tick(bid,ask,ts)
        self._m1_manage(bid,ask,ts)
        if m1_closed:
            self._m1_after_close(ts)
            self._m1_setup_tick(bid,ask,ts)

        closed=[]
        for m in TF_MIN:
            if self._bar_update(m,ts,bid):closed.append(m)
            self._setup_tick(m,bid,ask,ts)
        for m in closed:self._after_close(m,ts)
        for m in closed:self._setup_tick(m,bid,ask,ts)

        self._manage_legs(bid,ask,ts)
        self._manage_g75(bid,ask,ts)
        self._overlap_tick(ts)
        self._risk(bid,ask)

    def on_stop(self):
        if self.last_bid is None:return
        for x in list(self.m1_legs):
            if x.active:self._m1_close(x,self.last_bid,self.last_ask,self.last_ns,'TIMEOUT')
        for base in list(self.legs):
            if base.active and base.kind=='BASE':
                self._close_g75_children(base,self.last_bid,self.last_ask,self.last_ns,'BASE_EOD')
                self._close_leg(base,self.last_bid,self.last_ask,self.last_ns,'EOD')
        for x in list(self.legs):
            if x.active:self._close_leg(x,self.last_bid,self.last_ask,self.last_ns,'EOD_ORPHAN')
        self._overlap_tick(self.last_ns)

    def result(self):
        def met(rows):
            p=np.array([r['pnl'] for r in rows],float)
            gp=p[p>0].sum() if len(p) else 0.0; gl=-p[p<0].sum() if len(p) else 0.0
            return {'N':int(len(p)),'WR_pct':float((p>0).mean()*100) if len(p) else 0.0,
                'PF':float(gp/gl) if gl>0 else (math.inf if gp>0 else 0.0),'Net_USD':float(p.sum())}
        p=np.array([r['pnl'] for r in self.closed],float)
        base_rows=[r for r in self.closed if r.get('kind')=='BASE']
        g75_rows=[r for r in self.closed if r.get('kind')=='G75']
        base=met(self.closed); net=float(p.sum()) if len(p) else 0.0
        start=pd.Timestamp(self.first_ns,unit='ns',tz='UTC').date(); end=pd.Timestamp(self.last_seen_ns,unit='ns',tz='UTC').date()
        bdays=max(1,len(pd.bdate_range(start,end))); scale=21.0/bdays
        return {**base,'RF':float(net/self.max_dd) if self.max_dd>0 else None,
            'Return_pct':net/INITIAL*100.0,'MaxFloatingDD_USD':float(self.max_dd),'MaxFloatingDD_pct_initial':float(self.max_dd/INITIAL*100),
            'MinMarginLevel_pct_approx':None if math.isinf(self.min_ml) else float(self.min_ml),'MaxGrossQty_oz':float(self.max_gross),
            'MaxGrossLots_approx':float(self.max_gross/100.0),'BusinessDays':bdays,'Net21_USD_linearized':net*scale,
            'Monthly21_pct_linearized':net*scale/INITIAL*100.0,'N21_linearized':len(p)*scale,
            'per_tf':{str(m):met([r for r in self.closed if r['tf']==m]) for m in TF_MIN},
            'base_metrics':met(base_rows),'g75_layer_metrics':met(g75_rows),'g75_stats':self.g75_stats,
            'm1_metrics':met(self.m1_closed),'m1_pattern_stats':self.m1_stats,
            'direction_overlap':self._overlap_result(),
            'setup_stats':self.stats,'submitted_orders':self.submitted}

def executable(xs):
    one=Quantity.from_int(1); out=[]; repl=0
    for t in xs:
        b=ff(t.bid_size); a=ff(t.ask_size)
        if b<=0 or a<=0:
            out.append(QuoteTick(instrument_id=t.instrument_id,bid_price=t.bid_price,ask_price=t.ask_price,
                bid_size=one if b<=0 else t.bid_size,ask_size=one if a<=0 else t.ask_size,ts_event=t.ts_event,ts_init=t.ts_init)); repl+=1
        else:out.append(t)
    return out,repl

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',action='append',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--mode',choices=['base','g75'],required=True); ap.add_argument('--g75-reversal',type=float,default=G75_REVERSAL_DEFAULT)
    a=ap.parse_args(); cats=[ParquetDataCatalog(x) for x in a.catalog]
    inst=next(x for x in cats[0].instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    raw=[]
    for cat in cats:
        ci=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
        raw.extend(cat.query(data_cls=QuoteTick,identifiers=[ci.id.value]))
    raw.sort(key=lambda t:int(t.ts_event)); ticks,repl=executable(raw)
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR'),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,
        base_currency=USD,starting_balances=[Money(INITIAL,USD)],default_leverage=Decimal('2000'))
    eng.add_instrument(inst); eng.add_data(ticks)
    st=M1LinePOIPlusM5G75OverlapV19(Cfg(instrument_id=inst.id,mode=a.mode,g75_reversal=a.g75_reversal)); eng.add_strategy(st); eng.run(); eng.end()
    out=Path('results/m1-linepoi-m5-g75-overlap-v20')/a.experiment_id; out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(st.closed).to_csv(out/'trades.csv',index=False)
    pd.DataFrame(st.m1_closed).to_csv(out/'m1_trades.csv',index=False)
    result={'verification_level':'NAUTILUS_BT_M1_LINEPOI_M5_G75_OVERLAP_V20','engine':'NautilusTrader BacktestEngine',
        'nautilus_version':getattr(nautilus_trader,'__version__','unknown'),'raw_ticks':len(raw),'execution_ticks':len(ticks),
        'ohlc_input_used':False,'signal_bars':'M1 line-structure POI + M5 Elliott/Fib, both built online from raw BID QuoteTicks',
        'execution':'first raw Bid/Ask quote that reveals the closed M5 MSS bar; zero artificial delay','commission_rt_per_lot':COMMISSION_RT_PER_LOT,
        'cashback_rt_per_lot':CASHBACK_RT_PER_LOT,'deep_fib':DEEP,'pre_entry_invalidation':'ENFORCED','mode':a.mode,
        'g75_frozen':{'trigger':G75_TRIGGER,'add':G75_ADD,'max_layers':G75_MAX_LAYERS,'layer_qty_oz':BASE_QTY},'g75_reversal_tested':a.g75_reversal,
        'm1_rule_v1':{'line_structure':'Close-only causal pivots, 2 bars each side','patterns':['Classic V/A','Quasimodo Buy/Sell','Open-Close-Level Buy/Sell'],'entry':'POI raw-price revisit then favorable M1 close rejection','poi':'Classic/QM use Close pivot; OCL uses prior candle Open/Close body level','sl':'raw structural extreme plus 0.10 ATR buffer','tp':'prior structural opposite pivot; 2R fallback only when target is already behind actual fill'},
        'note':'Diagnostic v20: M1 and M5 remain independent. Adds exact M1 pattern and M5 parent context at each same/opposite overlap start, plus M1 trade CSV. No direction gate is added.',
        **st.result()}
    (out/'result.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    print(json.dumps(result,indent=2,default=str)); eng.dispose()

if __name__=='__main__':main()
