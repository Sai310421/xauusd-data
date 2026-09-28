from dataclasses import dataclass
import math, statistics

@dataclass
class MathStackConfig:
    level: str = "full"
    dd_cap: float = 18.0
    ddr_lambda: float = 2.5
    min_scale: float = 0.5
    fp_lo: float = 0.60
    fp_hi: float = 0.80
    rho: float = 0.08
    debt_lambda: float = 1.0
    intervention_k: float = 0.20
    chance_eps_dd: float = 0.10
    cvar_alpha: float = 0.95
    mpc_horizon: int = 20
    mpc_tail_lambda: float = 2.0
    mpc_cost_lambda: float = 0.05

class AEMathStack:
    """TickScalper lane numerical realization of Frozen v1->v2->v3 stack.
    It preserves the source contracts but uses low-dimensional online estimates,
    not a claimed exact 100D PDE/BSDE solution.
    """
    def __init__(self,cfg:MathStackConfig):
        self.c=cfg
        self.last={"fallback":"SAFE_BOUNDARY","D_star":0.0,"p_nr":0.5,"cvar":0.0,"mpc_score":0.0}

    def _ddr(self,dd):
        cap=max(self.c.dd_cap,1e-9); dist=max(0.0,cap-dd); lam=max(self.c.ddr_lambda,1e-9)
        den=1.0-math.exp(-lam)
        raw=(1.0-math.exp(-lam*dist/cap))/max(den,1e-12)
        return max(self.c.min_scale,min(1.0,raw))

    def _fp_scale(self,p):
        lo,hi=self.c.fp_lo,self.c.fp_hi
        if p<=lo:return 1.0
        if p>=hi:return self.c.min_scale
        return 1.0-(1.0-self.c.min_scale)*(p-lo)/max(hi-lo,1e-9)

    @staticmethod
    def _root_y(kappa):
        lo,hi=0.0,max(2.0,kappa+4.0)
        for _ in range(64):
            m=(lo+hi)/2; f=m-math.tanh(m)-kappa
            if f<0:lo=m
            else:hi=m
        return (lo+hi)/2

    def v1_boundary(self,debt,sigma_d,p_nr):
        sig=max(sigma_d,1e-6); rho=max(self.c.rho,1e-9); lam=max(self.c.debt_lambda,1e-9)
        kap=self.c.intervention_k*rho*math.sqrt(2*rho)/(2*lam*sig)
        y=self._root_y(max(0.0,kap))
        dstar=sig/math.sqrt(2*rho)*y
        excess=max(0.0,debt-dstar)
        # Preserve natural recovery when p_NR is high; otherwise minimum reflected reduction.
        if excess<=0:return 1.0,dstar
        severity=min(1.0,excess/max(dstar+sig,1e-9))
        nr=max(0.0,min(1.0,p_nr))
        scale=max(self.c.min_scale,1.0-severity*(1.0-nr))
        return scale,dstar

    def _cvar(self,vals):
        if len(vals)<20:return 0.0
        losses=sorted([max(0.0,-x) for x in vals],reverse=True)
        n=max(1,int(math.ceil((1-self.c.cvar_alpha)*len(losses))))
        return sum(losses[:n])/n

    def v2_robust(self,vals,gross_lot,prospective_lot,dd):
        if len(vals)<50:return 1.0,0.0,False
        cv=self._cvar(vals)
        projected=cv*(gross_lot+prospective_lot)*self.c.mpc_horizon
        headroom=max(self.c.dd_cap-dd,0.01)
        # Chance-constraint surrogate: reject only if tail projection overwhelms DD headroom.
        breach_ratio=projected/max(headroom,1e-9)
        hard=breach_ratio>10.0
        if hard:return self.c.min_scale,cv,True
        robust=max(self.c.min_scale,min(1.0,1.0/(1.0+0.35*breach_ratio)))
        return robust,cv,False

    def v3_mpc(self,vals,side,gross_lot,prospective_lot,spread):
        if len(vals)<80:return 1.0,0.0
        xs=list(vals)[-400:]
        mu=sum(xs)/len(xs); sd=statistics.pstdev(xs) if len(xs)>1 else 0.0
        # Deterministic scenario envelope (mean +/- sigma/tail), keeping tails explicit.
        scen=[mu-2*sd,mu-sd,mu,mu+sd,mu+2*sd]
        cand=[self.c.min_scale,0.65,0.8,1.0]
        best_s,best_q=self.c.min_scale,-1e99
        for s in cand:
            q=gross_lot+prospective_lot*s
            pnl=[side*z*q*self.c.mpc_horizon for z in scen]
            mean=sum(pnl)/len(pnl)
            losses=[max(0.0,-p) for p in pnl]
            tail=max(losses) if losses else 0.0
            cost=spread*prospective_lot*s*self.c.mpc_horizon
            score=mean-self.c.mpc_tail_lambda*tail-self.c.mpc_cost_lambda*cost
            if score>best_q:best_q,best_s=score,s
        return best_s,best_q

    def decide_scale(self,*,dd,p_adverse,recent_deltas,gross_lot,prospective_lot,spread,side):
        ddr=self._ddr(dd); fp=self._fp_scale(p_adverse)
        vals=list(recent_deltas)
        sigma_d=statistics.pstdev(vals[-200:]) if len(vals)>=20 else 1e-6
        p_nr=1.0-max(0.0,min(1.0,p_adverse))
        debt=max(0.0,dd)
        v1,dstar=self.v1_boundary(debt,sigma_d,p_nr)
        v2,cv,hard=self.v2_robust(vals,gross_lot,prospective_lot,dd)
        v3,score=self.v3_mpc(vals,side,gross_lot,prospective_lot,spread)
        level=self.c.level
        if level=="ddr": out=ddr; fb="DDR"
        elif level=="ddr_fp": out=min(ddr,fp); fb="DDR_FP"
        elif level=="v2": out=min(ddr,fp,v1,v2); fb="V2"
        elif level=="v3": out=min(ddr,fp,v1,v2,v3); fb="V3"
        else: out=min(ddr,fp,v1,v2,v3); fb="FULL"
        if hard: fb="V2_CHANCE_CONSTRAINT"
        self.last={"fallback":fb,"D_star":dstar,"p_nr":p_nr,"cvar":cv,"mpc_score":score,
                   "ddr":ddr,"fp":fp,"v1":v1,"v2":v2,"v3":v3,"scale":out}
        return max(self.c.min_scale,min(1.0,out))
