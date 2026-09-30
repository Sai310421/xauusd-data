from __future__ import annotations

"""FIB/ICT Source-of-Truth v2.2.

The purpose of this module is specification fidelity, not curve fitting.
Every strategy is represented as an ordered causal process. A Fibonacci level
is never sufficient by itself to authorize an entry.

Formal market input: genuine Raw Bid/Ask QuoteTick only. No OHLC, no resample,
no midpoint execution, no future/back-filled stage satisfaction.
"""
from dataclasses import dataclass
from enum import Enum
from math import sqrt
from statistics import mean, pstdev
from typing import Iterable


class Role(str, Enum):
    WHERE="WHERE"; WHEN="WHEN"; ENTRY="ENTRY"; STATE="STATE"; EXIT="EXIT"; ENGINE="ENGINE"


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    family: str
    roles: tuple[Role, ...]
    process: tuple[str, ...]
    ratios: tuple[float, ...] = ()
    frozen: bool = False
    standalone_entry: bool = True
    note: str = ""


A0_PROCESS=("LIQUIDITY","SWEEP","MSS_CHOCH","DISPLACEMENT","ANCHOR_LOCK","FIB_OTE","ENTRY_TRIGGER","RISK_OK","EXECUTION_GATE","ENTRY","EXIT")
B0_PROCESS=("X","A","B","C","D_PRZ","QML_HTF","LIQUIDITY_LTF","SWEEP_LTF","MSS_LTF","ENTRY_TRIGGER","RISK_OK","EXECUTION_GATE","ENTRY","EXIT")

SPECS: dict[str,StrategySpec] = {
 "A0": StrategySpec("A0","A",(Role.WHEN,Role.ENTRY),A0_PROCESS,(.50,.618,.705,.708,.786),True,True,"Frozen Liquidity/Sweep/MSS/Fib baseline"),
 "A1": StrategySpec("A1","A",(Role.WHEN,Role.ENTRY),A0_PROCESS[:-5]+("PD_ARRAY_RETEST",)+A0_PROCESS[-5:],(.50,.618,.705,.708,.786),False,True,"A0 + tick-native imbalance/IFVG/BPR-equivalent only after separately validated"),
 "A2": StrategySpec("A2","A",(Role.WHERE,Role.WHEN,Role.ENTRY),A0_PROCESS[:-5]+("FIB_GEOMETRY",)+A0_PROCESS[-5:],(.50,.618,.705,.708,.786)),
 "A3": StrategySpec("A3","A",(Role.WHERE,Role.WHEN,Role.ENTRY),A0_PROCESS[:-5]+("FIB_GEOMETRY","CIRCLE_TIME")+A0_PROCESS[-5:],(.50,.618,.705,.708,.786)),
 "B0": StrategySpec("B0","B",(Role.WHERE,Role.WHEN,Role.ENTRY),B0_PROCESS,(),True,True,"Frozen XABCD/D-PRZ/QML/Liquidity baseline"),
 "B1": StrategySpec("B1","B",(Role.WHERE,Role.WHEN,Role.ENTRY),B0_PROCESS[:-5]+("NATIVE_FIB_CONFLUENCE",)+B0_PROCESS[-5:],(.232,.25,.5,.559,.618,.669,.688,.705,.708,.718,.786,.822)),
 "B2": StrategySpec("B2","B",(Role.WHERE,Role.WHEN,Role.ENTRY),B0_PROCESS[:-5]+("HARMONIC_CIRCLE","TIME")+B0_PROCESS[-5:]),
 "B3": StrategySpec("B3","B",(Role.WHERE,Role.WHEN,Role.ENTRY),B0_PROCESS[:-5]+("LIQUIDITY_STRENGTH","PD_ARRAY_RETEST")+B0_PROCESS[-5:]),
 "C01": StrategySpec("C01","CIRCLE",(Role.WHERE,Role.WHEN),("ANCHOR_A","ANCHOR_B","MSNR_CIRCLE","CONFLUENCE"),(1.2,4.5,4.83),False,False,"Experimental MSNR geometry; not independent entry"),
 "C02": StrategySpec("C02","CIRCLE",(Role.WHERE,Role.WHEN),("ANCHOR_A","ANCHOR_B","CLASSIC_CIRCLE","CONFLUENCE"),(.236,.382,.5,.618,.786,1,1.618),False,False),
 "C03": StrategySpec("C03","CIRCLE",(Role.WHERE,Role.WHEN),("ANCHOR_A","ANCHOR_B","GOLDEN_CIRCLE","CONFLUENCE"),(.618,1.618),False,False),
 "C04": StrategySpec("C04","CIRCLE",(Role.WHEN,),("ANCHOR_A","ANCHOR_B","TIME_PROJECTION","TIME_WINDOW"),(.236,.382,.5,.618,1,1.618,2.618,4.236),False,False),
 "C05": StrategySpec("C05","CIRCLE",(Role.WHERE,Role.WHEN),("XABCD_VALID","D_PRZ","HARMONIC_CIRCLE","CONFLUENCE"),(.382,.5,.618,.707,.786,1,1.272,1.618,2.618,2.886),False,False),
 "C06": StrategySpec("C06","CIRCLE",(Role.WHERE,Role.WHEN),("LIQUIDITY_MAP","SWEEP","LIQUIDITY_CIRCLE","MSS"),(.5,.618,.786,1,1.618),False,False),
 "F01": StrategySpec("F01","FIB",(Role.ENTRY,Role.EXIT),("MONKEY_SETUP","MONKEY_RETRACE","MONKEY_TRIGGER","RISK_OK","EXECUTION_GATE","ENTRY","EXIT"),(.63,.78,-.33,-.66,-.99)),
 "F02": StrategySpec("F02","FIB",(Role.ENTRY,),("GS_SETUP","GS_NATIVE_ZONE","GS_TRIGGER","RISK_OK","EXECUTION_GATE","ENTRY","EXIT"),(.232,.25,.688,.705,.708,.718,.786,.822)),
 "F03": StrategySpec("F03","FIB",(Role.WHERE,),("SNR_SETUP","SNR_NATIVE_LEVEL","SNR_CONFIRM"),(-.21,-.255,-.29,1.55,2.47,2.64),False,False),
 "F04": StrategySpec("F04","FIB",(Role.ENTRY,),("VALID_DIRECTIONAL_SETUP","ANCHOR_LOCK","OTE_ZONE","OTE_TRIGGER","RISK_OK","EXECUTION_GATE","ENTRY","EXIT"),(.62,.705,.708,.79)),
 "F05": StrategySpec("F05","FIB",(Role.WHERE,Role.EXIT),("CRT_SETUP","CRT_DELIVERY","CRT_NATIVE_LEVEL","CRT_CONFIRM"),(-.40,-.29,-.255,-.21,0,1,1.47,1.55,2.56,2.60,2.64),False,False),
 "F06": StrategySpec("F06","STATS",(Role.STATE,),("ROLLING_DISTRIBUTION","ZSCORE_STATE"),(),False,False),
 "F07": StrategySpec("F07","EXIT",(Role.EXIT,),("VALID_POSITION","TARGET_PROJECTION"),(-1,-2,-2.5,-3,-4),False,False),
 "F08": StrategySpec("F08","STATE",(Role.STATE,),("ORDER_FLOW_SETUP","OF_POSITION_STATE"),(0,.25,.5,.75,1),False,False),
 "F09": StrategySpec("F09","FIB",(Role.ENTRY,),("POP_SETUP","POP_NATIVE_ZONE","POP_TRIGGER","RISK_OK","EXECUTION_GATE","ENTRY","EXIT"),(.5,.559,.669,.786)),
 "F10": StrategySpec("F10","STATE",(Role.STATE,),("DEALING_RANGE","EQ","PREMIUM_DISCOUNT_STATE"),(0,.5,1),False,False),
 "S25": StrategySpec("S25","REGIME",(Role.ENGINE,Role.ENTRY),("TREND_VALID","BOS","PULLBACK","NATIVE_PULLBACK_ZONE","CONTINUATION_TRIGGER","RISK_OK","EXECUTION_GATE","ENTRY","EXIT")),
 "S26": StrategySpec("S26","REGIME",(Role.ENGINE,Role.ENTRY),("RANGE_VALID","RANGE_EXTREME","LIQUIDITY_SWEEP","REJECTION","MEAN_REVERSION_TRIGGER","RISK_OK","EXECUTION_GATE","ENTRY","EXIT")),
 "S27": StrategySpec("S27","REGIME",(Role.ENGINE,Role.ENTRY),("COMPRESSION","EXPANSION","BREAKOUT","RETEST","CONTINUATION_TRIGGER","RISK_OK","EXECUTION_GATE","ENTRY","EXIT")),
}


class ProcessGate:
    def __init__(self,spec:StrategySpec): self.spec=spec; self.i=0; self.last_ts=float("-inf"); self.audit=[]
    @property
    def expected(self): return self.spec.process[self.i] if self.i<len(self.spec.process) else None
    def accept(self,event:str,ts:float,payload:dict|None=None)->bool:
        if event!=self.expected or ts<self.last_ts: return False
        self.audit.append((event,ts,payload or {})); self.last_ts=ts; self.i+=1; return True
    @property
    def complete(self): return self.i==len(self.spec.process)
    @property
    def entry_authorized(self): return self.complete and self.spec.standalone_entry and any(x[0]=="ENTRY" for x in self.audit)


def fib_price(a:float,b:float,r:float,bull:bool)->float:
    return b-r*(b-a) if bull else b+r*(a-b)

def range_position(p:float,l:float,h:float)->float:
    if h<=l: raise ValueError("invalid range")
    return (p-l)/(h-l)
def zscore(p:float,prices:Iterable[float])->float:
    xs=list(prices); s=pstdev(xs)
    return 0.0 if not xs or s==0 else (p-mean(xs))/s
def circle_error(t:float,p:float,t_a:float,p_a:float,t_b:float,p_b:float,r:float,v0:float)->float:
    t0=t_b-t_a
    if t0<=0 or v0<=0 or r<=0: raise ValueError("invalid circle anchors")
    x=(t-t_a)/t0; y=(p-p_a)/v0
    r0=sqrt(1.0+((p_b-p_a)/v0)**2)
    return abs(sqrt(x*x+y*y)-r*r0)/(r*r0)
def time_projection(t_a:float,t_b:float,r:float)->float: return t_a+r*(t_b-t_a)
def raw_fill(side:str,action:str,bid:float,ask:float)->float:
    if ask<bid: raise ValueError("crossed quote")
    if side=="LONG": return ask if action=="ENTRY" else bid
    if side=="SHORT": return bid if action=="ENTRY" else ask
    raise ValueError(side)


def invariant_tests()->dict:
    assert len(SPECS)==27
    assert SPECS["A0"].frozen and SPECS["B0"].frozen
    # Every independent entry strategy has explicit Risk + Execution gates before Entry.
    for s in SPECS.values():
        if s.standalone_entry:
            assert "ENTRY" in s.process, s.strategy_id
            j=s.process.index("ENTRY")
            assert "RISK_OK" in s.process[:j], s.strategy_id
            assert "EXECUTION_GATE" in s.process[:j], s.strategy_id
    # A Fib touch cannot skip the causal setup.
    g=ProcessGate(SPECS["A0"])
    assert not g.accept("FIB_OTE",1.0)
    assert not g.entry_authorized
    # Out-of-order/retroactive stages are rejected.
    g=ProcessGate(SPECS["F09"])
    assert g.accept("POP_SETUP",10)
    assert not g.accept("POP_TRIGGER",11)
    assert not g.accept("POP_NATIVE_ZONE",9)
    # Executable-side convention.
    assert raw_fill("LONG","ENTRY",100,101)==101
    assert raw_fill("LONG","EXIT",100,101)==100
    assert raw_fill("SHORT","ENTRY",100,101)==100
    assert raw_fill("SHORT","EXIT",100,101)==101
    return {"strategies":len(SPECS),"frozen":["A0","B0"],"ohlc":False,"mid_execution":False,"generic_fib_touch_entry":False,"process_order_required":True}

if __name__=="__main__": print(invariant_tests())
