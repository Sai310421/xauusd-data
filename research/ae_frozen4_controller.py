from dataclasses import dataclass
from collections import Counter
import math, statistics

@dataclass
class Frozen4Config:
    level:str="v3"; dd_cap:float=18.0; ddr_lambda:float=2.5; min_scale:float=0.5
    fp_lo:float=0.60; fp_hi:float=0.80; rho:float=0.08; debt_lambda:float=1.0
    intervention_k:float=0.20; cvar_alpha:float=0.95; chance_eps_dd:float=0.10
    mpc_horizon:int=20; mpc_tail_lambda:float=2.0; mpc_cost_lambda:float=0.05

class AEFrozen4:
    """Executable low-dimensional realization of the frozen v1->v2->v3 contract.
    It does NOT claim an exact 16/32/64/100D HJB/BSDE solution.
    """
    def __init__(self,cfg):
        self.c=cfg; self.counts=Counter()
        self.last={"fallback":"SAFE_BOUNDARY"}

    def _root_y(self,k):
        lo,hi=0.0,max(2.0,k+4.0)
        for _ in range(80):
            m=(lo+hi)/2
            if m-math.tanh(m)-k<0:lo=m
            else:hi=m
        return (lo+hi)/2

    def _ddr(self,dd):
        cap=max(self.c.dd_cap,1e-9); lam=max(self.c.ddr_lambda,1e-9)
        raw=(1-math.exp(-lam*max(0,cap-dd)/cap))/max(1-math.exp(-lam),1e-12)
        return max(self.c.min_scale,min(1.0,raw))

    def _fp_scale(self,p_tail):
        if p_tail<=self.c.fp_lo:return 1.0
        if p_tail>=self.c.fp_hi:return self.c.min_scale
        return 1-(1-self.c.min_scale)*(p_tail-self.c.fp_lo)/max(self.c.fp_hi-self.c.fp_lo,1e-9)

    def v1(self,debt,sigma_d,p_nr):
        sig=max(sigma_d,1e-8);rho=max(self.c.rho,1e-9);lam=max(self.c.debt_lambda,1e-9)
        kappa=self.c.intervention_k*rho*math.sqrt(2*rho)/(2*lam*sig)
        y=self._root_y(max(0,kappa)); dstar=sig/math.sqrt(2*rho)*y
        delta=max(0,debt-dstar)
        if delta<=0:
            self.counts["v1_wait"]+=1; return 1.0,dstar,delta,"WAIT"
        # minimum reflected intervention, moderated by Natural Recovery probability
        severity=min(1.0,delta/max(debt+sig,1e-9))
        scale=max(self.c.min_scale,1-severity*(1-p_nr))
        action="MIN_INTERVENTION" if p_nr>=0.5 else "RISK_REDUCTION"
        self.counts["v1_"+action.lower()]+=1
        return scale,dstar,delta,action

    def _cvar(self,pnls):
        if not pnls:return 0.0
        losses=sorted((max(0,-x) for x in pnls),reverse=True)
        n=max(1,int(math.ceil((1-self.c.cvar_alpha)*len(losses))))
        return sum(losses[:n])/n

    def v2(self,vals,side,gross,add,dd,spread):
        if len(vals)<80:return 1.0,{"action":"WAIT","cvar":0.0,"p_dd":0.0,"p_ruin":0.0,"p_margin":0.0}
        xs=list(vals)[-400:]; mu=sum(xs)/len(xs); sd=statistics.pstdev(xs) or 1e-9
        # adverse ambiguity envelope: drift down, volatility/cost up
        mus=[mu,mu-0.5*sd*side,mu-1.0*sd*side]
        vols=[sd,1.25*sd]
        candidates=[self.c.min_scale,0.65,0.8,1.0]
        best=(self.c.min_scale,-1e99,None)
        head=max(self.c.dd_cap-dd,0.01)
        for qscale in candidates:
            q=gross+add*qscale; scenario=[]
            for m in mus:
                for v in vols:
                    adverse=side*m*self.c.mpc_horizon-2*v*self.c.mpc_horizon**0.5
                    scenario.append(adverse*q-spread*add*qscale)
            cv=self._cvar(scenario)
            p_dd=sum(max(0,-p)>head for p in scenario)/len(scenario)
            p_ruin=sum(max(0,-p)>(head+dd) for p in scenario)/len(scenario)
            p_margin=p_ruin # low-dimensional proxy; live margin mapping remains separate gate
            feasible=p_dd<=self.c.chance_eps_dd and p_ruin<=self.c.chance_eps_dd and p_margin<=self.c.chance_eps_dd
            if not feasible:continue
            robust=min(scenario)-cv
            if robust>best[1]:best=(qscale,robust,{"cvar":cv,"p_dd":p_dd,"p_ruin":p_ruin,"p_margin":p_margin})
        if best[2] is None:
            self.counts["v2_safe_mode"]+=1
            return self.c.min_scale,{"action":"SAFE_MODE","cvar":0.0,"p_dd":1.0,"p_ruin":1.0,"p_margin":1.0}
        action="WAIT" if best[0]>=0.999 else "REDUCE"
        self.counts["v2_"+action.lower()]+=1
        return best[0],{"action":action,**best[2]}

    def v3(self,vals,side,gross,add,spread,v2_terminal):
        if len(vals)<100:return 1.0,{"action":"WAIT","score":0.0,"horizon":0}
        xs=list(vals)[-400:];mu=sum(xs)/len(xs);sd=statistics.pstdev(xs) or 1e-9
        # distributional scenario tree; tail scenarios retained
        shocks=[-2.5,-1.5,-0.5,0,0.5,1.5,2.5]
        H=max(5,self.c.mpc_horizon//2) if sd>abs(mu)*4 else self.c.mpc_horizon
        cand=[self.c.min_scale,0.65,0.8,1.0];best=(self.c.min_scale,-1e99)
        for s in cand:
            q=gross+add*s
            pnl=[side*(mu+z*sd)*q*H-spread*add*s for z in shocks]
            cv=self._cvar(pnl);mean=sum(pnl)/len(pnl)
            terminal=0.05*v2_terminal
            score=mean-self.c.mpc_tail_lambda*cv-self.c.mpc_cost_lambda*spread*add*s+terminal
            if score>best[1]:best=(s,score)
        action="WAIT" if best[0]>=0.999 else "REDUCE"
        self.counts["v3_"+action.lower()]+=1
        return best[0],{"action":action,"score":best[1],"horizon":H}

    def decide(self,*,dd,p_adverse,recent_deltas,gross_lot,prospective_lot,spread,side):
        vals=list(recent_deltas);sig=statistics.pstdev(vals[-200:]) if len(vals)>=20 else 1e-8
        p_tail=max(0,min(1,p_adverse));p_nr=1-p_tail
        ddr=self._ddr(dd);fp=self._fp_scale(p_tail);base=min(ddr,fp)
        if self.c.level=="ddr_fp":
            out=base; fb="DDR_FP"
            self.last={"fallback":fb,"scale":out,"ddr":ddr,"fp":fp,"p_nr":p_nr};return out
        s1,dstar,delta,a1=self.v1(max(0,dd),sig,p_nr);out=min(base,s1);fb="V1"
        v2info=v3info={}
        if self.c.level in ("v2","v3"):
            s2,v2info=self.v2(vals,side,gross_lot,prospective_lot,dd,spread);out=min(out,s2);fb="V2"
            if v2info.get("action")=="SAFE_MODE":fb="V2_SAFE_MODE"
            if self.c.level=="v3":
                s3,v3info=self.v3(vals,side,gross_lot,prospective_lot,spread,s2);out=min(out,s3);fb="V3" if fb!="V2_SAFE_MODE" else fb
        self.counts["decisions"]+=1
        self.last={"fallback":fb,"scale":out,"ddr":ddr,"fp":fp,"D_star":dstar,"delta_R":delta,
                   "p_nr":p_nr,"p_tail":p_tail,"v1_action":a1,"v2":v2info,"v3":v3info}
        return max(self.c.min_scale,min(1,out))
