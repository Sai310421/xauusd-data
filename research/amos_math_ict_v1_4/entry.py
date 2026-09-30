from dataclasses import dataclass
from .models import ICTState, IFVGState, BPRZone, Decision

@dataclass
class EntryContext:
    ict: ICTState
    ifvg: IFVGState | None = None
    bpr: BPRZone | None = None
    bpr_retested: bool = False
    bpr_rejected: bool = False

    def effective_groups(self):
        return {
            'structure': any([self.ict.mss,self.ict.choch,self.ict.bos]),
            'displacement': bool(self.ict.fvg),
            'ifvg_transition': bool(self.ifvg and self.ifvg.valid_transition and self.ifvg.retested),
            'location': any([self.ict.order_block,self.ict.ote]),
            'bpr_location': bool(self.bpr and self.bpr.valid_overlap and self.bpr_retested),
            'liquidity': any([self.ict.sweep,self.ict.smt]),
            'cycle': bool(self.ict.po3),
        }

    def directional_confirmation(self,direction:int)->bool:
        if self.ifvg and self.ifvg.valid_transition and self.ifvg.retested and self.ifvg.direction != direction:
            return False
        if self.bpr and self.bpr.valid_overlap and self.bpr_retested and self.bpr_rejected and self.bpr.rejection_direction not in (0,direction):
            return False
        return True

class LifecycleAwareEntryRouter:
    def __init__(self,enable_liquidity=True,enable_regime=True,enable_ai=True,enable_harmonic=False,min_effective_groups=3,min_liquidity_score=0.0,min_ai_confidence=0.60,max_spread=float('inf'),reject_regimes=('EXTREME',)):
        self.enable_liquidity=enable_liquidity; self.enable_regime=enable_regime; self.enable_ai=enable_ai; self.enable_harmonic=enable_harmonic
        self.min_effective_groups=min_effective_groups; self.min_liquidity_score=min_liquidity_score; self.min_ai_confidence=min_ai_confidence; self.max_spread=max_spread; self.reject_regimes=set(reject_regimes)

    def decide(self,market,ctx,ai=None,harmonic=None):
        ict=ctx.ict; g=ctx.effective_groups(); n=sum(bool(v) for v in g.values())
        c={f'group_{k}':float(bool(v)) for k,v in g.items()}; c.update({'effective_groups':float(n),'spread':market.spread,'liquidity_score':market.liquidity_score})
        if market.spread>self.max_spread: return Decision('HOLD',0.0,'SPREAD_REJECT',c)
        if ict.direction==0 or n<self.min_effective_groups: return Decision('HOLD',0.0,'ICT_GROUP_GATE',c)
        if not g['structure']: return Decision('HOLD',0.0,'STRUCTURE_REQUIRED',c)
        if not (g['location'] or g['bpr_location'] or g['liquidity'] or g['ifvg_transition'] or g['displacement']): return Decision('HOLD',0.0,'LOCATION_OR_DISPLACEMENT_REQUIRED',c)
        if not ctx.directional_confirmation(ict.direction): return Decision('HOLD',0.0,'ZONE_DIRECTION_CONFLICT',c)
        if self.enable_liquidity and market.liquidity_score<self.min_liquidity_score: return Decision('HOLD',0.0,'LIQUIDITY_REJECT',c)
        if self.enable_regime and market.regime in self.reject_regimes: return Decision('HOLD',0.0,'REGIME_REJECT',c)
        score=ict.score + (0.35 if g['ifvg_transition'] else 0) + (0.25 if g['bpr_location'] else 0)
        if self.enable_ai and ai is not None:
            c.update({'ai_confidence':ai.confidence,'ai_direction_prob':ai.direction_prob})
            if not ai.approve or ai.confidence<self.min_ai_confidence: return Decision('HOLD',0.0,ai.reject_reason or 'AI_REJECT',c)
            if (1 if ai.direction_prob>=.5 else -1)!=ict.direction: return Decision('HOLD',0.0,'AI_DIRECTION_DISAGREE',c)
            score += .25*ai.confidence
        if self.enable_harmonic and harmonic is not None and harmonic.direction:
            score += .20*harmonic.score if harmonic.direction==ict.direction else -.20*harmonic.score
        return Decision('BUY' if ict.direction>0 else 'SELL',score,'PASS',c)

@dataclass
class MTFState:
    states: dict[str, ICTState]
    def partial_alignment(self,entry_tf='M1',context_tfs=('M5','M15')):
        if entry_tf not in self.states or self.states[entry_tf].direction==0: return 0.0
        d=self.states[entry_tf].direction
        vals=[self.states[t].direction for t in context_tfs if t in self.states and self.states[t].direction!=0]
        return 0.0 if not vals else sum(1 for x in vals if x==d)/len(vals)

class MTFEntryRouter:
    def __init__(self,base_router=None,require_full_alignment=False,min_alignment=.5):
        self.base=base_router or LifecycleAwareEntryRouter(); self.require_full_alignment=require_full_alignment; self.min_alignment=min_alignment
    def decide(self,market,ctx,mtf,ai=None,harmonic=None):
        d=self.base.decide(market,ctx,ai,harmonic)
        if d.action=='HOLD': return d
        a=mtf.partial_alignment(); d.components['mtf_alignment']=a
        if self.require_full_alignment and a<1.0: return Decision('HOLD',0.0,'MTF_FULL_ALIGNMENT_REJECT',d.components)
        if a<self.min_alignment: return Decision('HOLD',0.0,'MTF_ALIGNMENT_REJECT',d.components)
        d.score += .20*a
        return d
