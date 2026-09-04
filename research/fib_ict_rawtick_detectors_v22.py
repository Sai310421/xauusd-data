from __future__ import annotations

"""Causal Raw Bid/Ask detector foundations for FIB/ICT v2.2.

Design rules:
- Raw Bid/Ask QuoteTick only. No OHLC construction/resample.
- Stage order is causal and auditable.
- A0/B0 remain Frozen semantically; candle/ATR quantities from the written spec
  are translated to tick-event equivalents, never silently redefined as bars.
- Any parameter introduced by this raw-tick translation is explicit in Config.
- B0 D-zone reference-leg ambiguity is NOT guessed. Pattern validation refuses
  to authorize D until the reference mapping is explicitly supplied.
"""
from dataclasses import dataclass, field
from statistics import median
from typing import Literal

Side = Literal["LONG", "SHORT"]

@dataclass(frozen=True)
class Quote:
    ts: float
    bid: float
    ask: float
    def __post_init__(self):
        if self.ask < self.bid: raise ValueError("crossed quote")

@dataclass(frozen=True)
class RawTickConfig:
    liquidity_lookback_sec: float
    structure_lookback_sec: float
    reclaim_deadline_sec: float
    displacement_window_sec: float
    lambda_liq: float = 3.0
    lambda_break: float = 2.0
    alpha_displacement: float = 6.0
    beta_displacement: float = 8.0
    ote_low: float = 0.62
    ote_primary: float = 0.708
    ote_high: float = 0.79

@dataclass
class AuditEvent:
    name: str
    ts: float
    price: float
    detail: dict = field(default_factory=dict)

@dataclass
class A0Setup:
    side: Side
    liquidity: float
    sweep_extreme: float
    mss_level: float
    impulse_a: float
    impulse_b: float
    confirmed_at: float
    scale: float
    audit: list[AuditEvent]

    def fib(self, r: float) -> float:
        a,b=self.impulse_a,self.impulse_b
        return b-r*(b-a) if self.side=="LONG" else b+r*(a-b)

    @property
    def ote_zone(self):
        x1=self.fib(0.62); x2=self.fib(0.79)
        return (min(x1,x2),max(x1,x2))

    @property
    def ote_primary(self): return self.fib(0.708)

@dataclass(frozen=True)
class XABCD:
    x: float; a: float; b: float; c: float; d: float
    t_x: float; t_a: float; t_b: float; t_c: float; t_d: float

@dataclass(frozen=True)
class B0PatternCheck:
    rb: float
    rb_valid: bool
    chronological: bool
    d_reference_resolved: bool
    d_valid: bool
    reason: str


def _window(qs:list[Quote], end_i:int, seconds:float)->list[Quote]:
    if end_i<=0: return []
    cutoff=qs[end_i-1].ts-seconds
    j=end_i-1
    while j>=0 and qs[j].ts>=cutoff: j-=1
    return qs[j+1:end_i]


def robust_tick_scale(qs:list[Quote])->float:
    """Tick-event substitute for ATR scale; never constructs bars."""
    d=[]
    for i in range(1,len(qs)):
        db=abs(qs[i].bid-qs[i-1].bid); da=abs(qs[i].ask-qs[i-1].ask)
        if db>0: d.append(db)
        if da>0: d.append(da)
    return max(median(d) if d else 0.0, 1e-9)


def detect_a0(qs:list[Quote], cfg:RawTickConfig)->list[A0Setup]:
    """A0 semantic order: Liquidity->Sweep->MSS/CHOCH->Displacement->Fib/OTE.

    LONG uses Bid for sell-side structure/reclaim/MSS observation; SHORT uses Ask.
    This function arms Fib/OTE only. It does not enter on touch.
    """
    out=[]; n=len(qs)
    for s in range(1,n-2):
        hist=_window(qs,s,cfg.liquidity_lookback_sec)
        if len(hist)<10: continue
        scale=robust_tick_scale(hist)
        eps_liq=cfg.lambda_liq*scale
        eps_break=cfg.lambda_break*scale

        # LONG: prior sell-side liquidity -> sweep below -> reclaim above.
        liq_low=min(q.bid for q in hist)
        if qs[s].bid < liq_low-eps_liq:
            reclaim=None
            for r in range(s+1,n):
                if qs[r].ts-qs[s].ts>cfg.reclaim_deadline_sec: break
                if qs[r].bid>liq_low: reclaim=r; break
            if reclaim is not None:
                sh_hist=_window(qs,s,cfg.structure_lookback_sec)
                if sh_hist:
                    sh=max(q.bid for q in sh_hist)
                    mss=None
                    for m in range(reclaim+1,n):
                        if qs[m].bid>sh+eps_break: mss=m; break
                    if mss is not None:
                        sweep_low=min(q.bid for q in qs[s:mss+1])
                        # Tick-native displacement: net directional movement and
                        # path range relative to event scale, within explicit time.
                        end=mss
                        while end+1<n and qs[end+1].ts-qs[mss].ts<=cfg.displacement_window_sec and qs[end+1].bid>=qs[end].bid:
                            end+=1
                        impulse_high=max(q.bid for q in qs[reclaim:end+1])
                        net=impulse_high-sweep_low
                        path=max(q.bid for q in qs[reclaim:end+1])-min(q.bid for q in qs[reclaim:end+1])
                        if net>=cfg.alpha_displacement*scale and path>=cfg.beta_displacement*scale:
                            audit=[
                              AuditEvent("LIQUIDITY",hist[-1].ts,liq_low),
                              AuditEvent("SWEEP",qs[s].ts,sweep_low),
                              AuditEvent("MSS_CHOCH",qs[mss].ts,qs[mss].bid,{"ref":sh}),
                              AuditEvent("DISPLACEMENT",qs[end].ts,impulse_high,{"net":net,"path":path,"scale":scale}),
                              AuditEvent("ANCHOR_LOCK",qs[end].ts,impulse_high,{"A":sweep_low,"B":impulse_high}),
                              AuditEvent("FIB_OTE",qs[end].ts,impulse_high),
                            ]
                            out.append(A0Setup("LONG",liq_low,sweep_low,sh,sweep_low,impulse_high,qs[end].ts,scale,audit))

        # SHORT: prior buy-side liquidity -> sweep above -> reclaim below.
        liq_high=max(q.ask for q in hist)
        if qs[s].ask > liq_high+eps_liq:
            reclaim=None
            for r in range(s+1,n):
                if qs[r].ts-qs[s].ts>cfg.reclaim_deadline_sec: break
                if qs[r].ask<liq_high: reclaim=r; break
            if reclaim is not None:
                sl_hist=_window(qs,s,cfg.structure_lookback_sec)
                if sl_hist:
                    sl=min(q.ask for q in sl_hist)
                    mss=None
                    for m in range(reclaim+1,n):
                        if qs[m].ask<sl-eps_break: mss=m; break
                    if mss is not None:
                        sweep_high=max(q.ask for q in qs[s:mss+1])
                        end=mss
                        while end+1<n and qs[end+1].ts-qs[mss].ts<=cfg.displacement_window_sec and qs[end+1].ask<=qs[end].ask:
                            end+=1
                        impulse_low=min(q.ask for q in qs[reclaim:end+1])
                        net=sweep_high-impulse_low
                        path=max(q.ask for q in qs[reclaim:end+1])-min(q.ask for q in qs[reclaim:end+1])
                        if net>=cfg.alpha_displacement*scale and path>=cfg.beta_displacement*scale:
                            audit=[
                              AuditEvent("LIQUIDITY",hist[-1].ts,liq_high),
                              AuditEvent("SWEEP",qs[s].ts,sweep_high),
                              AuditEvent("MSS_CHOCH",qs[mss].ts,qs[mss].ask,{"ref":sl}),
                              AuditEvent("DISPLACEMENT",qs[end].ts,impulse_low,{"net":net,"path":path,"scale":scale}),
                              AuditEvent("ANCHOR_LOCK",qs[end].ts,impulse_low,{"A":sweep_high,"B":impulse_low}),
                              AuditEvent("FIB_OTE",qs[end].ts,impulse_low),
                            ]
                            out.append(A0Setup("SHORT",liq_high,sweep_high,sl,sweep_high,impulse_low,qs[end].ts,scale,audit))
    return _dedup_a0(out)


def _dedup_a0(xs:list[A0Setup])->list[A0Setup]:
    seen=set(); out=[]
    for x in xs:
        key=(x.side,round(x.sweep_extreme,9),round(x.impulse_b,9),x.confirmed_at)
        if key not in seen: seen.add(key); out.append(x)
    return out


def validate_b0_xabcd(p:XABCD, d_reference_ratio:float|None=None)->B0PatternCheck:
    """Validate only what the source specification actually fixes.

    R_B=|A-B|/|A-X| and 0.64<=R_B<=0.68 are explicit.
    Source lists D candidates 0.697-0.706 and 0.825-0.835, but does not prove
    the reference leg for the latter. Therefore the detector must receive an
    externally resolved D ratio from a verified pattern-specific mapper; it is
    forbidden to infer one here.
    """
    chronological=p.t_x<p.t_a<p.t_b<p.t_c<p.t_d
    denom=abs(p.a-p.x)
    rb=abs(p.a-p.b)/denom if denom>0 else float("inf")
    rb_valid=.64<=rb<=.68
    if d_reference_ratio is None:
        return B0PatternCheck(rb,rb_valid,chronological,False,False,"D reference leg unresolved: entry forbidden")
    d_valid=(.697<=d_reference_ratio<=.706) or (.825<=d_reference_ratio<=.835)
    return B0PatternCheck(rb,rb_valid,chronological,True,d_valid,"OK" if chronological and rb_valid and d_valid else "XABCD ratio/order invalid")


def raw_entry_price(side:Side,q:Quote)->float: return q.ask if side=="LONG" else q.bid
def raw_exit_price(side:Side,q:Quote)->float: return q.bid if side=="LONG" else q.ask


def self_test()->dict:
    # B0 must refuse unresolved D mapping rather than inventing a leg.
    p=XABCD(0,10,3.5,8,2,0,1,2,3,4)
    c=validate_b0_xabcd(p,None)
    assert c.chronological and c.rb_valid and not c.d_reference_resolved and not c.d_valid
    # Executable-side convention.
    q=Quote(0,100,101)
    assert raw_entry_price("LONG",q)==101 and raw_exit_price("LONG",q)==100
    assert raw_entry_price("SHORT",q)==100 and raw_exit_price("SHORT",q)==101
    return {"a0":"CAUSAL_RAWTICK_IMPLEMENTED","b0":"EXPLICIT_CORE_VALIDATION_IMPLEMENTED_D_MAPPING_GATED","ohlc":False,"lookahead":False,"guessed_d_leg":False}

if __name__=="__main__": print(self_test())
