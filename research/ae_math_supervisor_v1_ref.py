from __future__ import annotations
from dataclasses import dataclass
from math import exp, log
from itertools import product

EPS=1e-12

def _mean(xs):
    xs=list(xs)
    if not xs: raise ValueError('empty sample')
    return sum(xs)/len(xs)

@dataclass(frozen=True)
class KellyConfig:
    enabled: bool=True
    lambda_risk: float=2.0
    max_fraction: float=1.0
    grid_points: int=200

@dataclass(frozen=True)
class RecoveryConfig:
    enabled: bool=True
    min_probability: float=.50
    full_probability: float=.80

@dataclass(frozen=True)
class WassersteinConfig:
    enabled: bool=True
    rho: float=.002
    min_robust_edge: float=-.0005
    full_robust_edge: float=.001

@dataclass(frozen=True)
class MPCConfig:
    enabled: bool=True
    horizon: int=3
    dd_penalty: float=2.0
    debt_penalty: float=1.0
    inventory_penalty: float=.25
    cost_penalty: float=1.0

@dataclass(frozen=True)
class AEState:
    equity: float
    peak_equity: float
    debt: float
    inventory: float
    margin_level: float
    recovery_coordinate: float
    @property
    def dd(self):
        return max(0.,(self.peak_equity-self.equity)/max(self.peak_equity,EPS))

@dataclass(frozen=True)
class MPCAction:
    name: str
    size_multiplier: float
    expected_profit: float=0.0
    dd_delta: float=0.0
    debt_delta: float=0.0
    inventory_delta: float=0.0
    cost: float=0.0

def risk_constrained_kelly_fraction(returns,cfg=KellyConfig()):
    if not cfg.enabled: return cfg.max_fraction
    rs=list(returns)
    if not rs: return 1.0
    best=(float('-inf'),0.0)
    for i in range(cfg.grid_points+1):
        f=cfg.max_fraction*i/cfg.grid_points
        ws=[1+f*r for r in rs]
        if min(ws)<=EPS: continue
        if _mean(w**(-cfg.lambda_risk) for w in ws)>1+1e-10: continue
        g=_mean(log(w) for w in ws)
        if g>=best[0]: best=(g,f)
    return best[1]

def brownian_first_passage_recovery_probability(x,lower,upper,drift,volatility):
    if upper<=lower: raise ValueError('upper<=lower')
    if x<=lower:return 0.0
    if x>=upper:return 1.0
    y=x-lower;b=upper-lower
    if volatility<=0:return 1.0 if drift>0 else (0.0 if drift<0 else y/b)
    if abs(drift)<1e-12:return y/b
    zy=max(-700,min(700,-2*drift*y/(volatility*volatility)))
    zb=max(-700,min(700,-2*drift*b/(volatility*volatility)))
    den=1-exp(zb)
    return y/b if abs(den)<EPS else max(0.,min(1.,(1-exp(zy))/den))

def probability_multiplier(p,cfg=RecoveryConfig()):
    if not cfg.enabled:return 1.0
    if p<=cfg.min_probability:return 0.0
    if p>=cfg.full_probability:return 1.0
    return (p-cfg.min_probability)/(cfg.full_probability-cfg.min_probability)

def wasserstein_linear_robust_edge(scenarios,exposure,rho):
    if not scenarios:return float('-inf')
    vals=[sum(w*x for w,x in zip(exposure,row)) for row in scenarios]
    norm=sum(w*w for w in exposure)**.5
    return _mean(vals)-rho*norm

def robust_multiplier(edge,cfg=WassersteinConfig()):
    if not cfg.enabled:return 1.0
    if edge<=cfg.min_robust_edge:return 0.0
    if edge>=cfg.full_robust_edge:return 1.0
    return (edge-cfg.min_robust_edge)/(cfg.full_robust_edge-cfg.min_robust_edge)

def mpc_multiplier(state,actions,cfg=MPCConfig()):
    if not cfg.enabled:return 1.0
    best=(float('inf'),1.0,'NORMAL')
    for seq in product(actions, repeat=cfg.horizon):
        eq=state.equity;peak=state.peak_equity;debt=state.debt;inv=state.inventory;cost=0.0
        for a in seq:
            eq=max(EPS,eq+a.expected_profit);peak=max(peak,eq)
            dd=max(0.,(peak-eq)/max(peak,EPS)+a.dd_delta)
            debt=max(0.,debt+a.debt_delta);inv=max(0.,inv+a.inventory_delta)
            cost += -a.expected_profit+cfg.dd_penalty*dd*dd+cfg.debt_penalty*debt*debt+cfg.inventory_penalty*inv*inv+cfg.cost_penalty*max(0.,a.cost)
        if cost<best[0]:best=(cost,max(0.,min(1.,seq[0].size_multiplier)),seq[0].name)
    return best[1],best[2]

def margin_multiplier(level,hard=500.,soft=650.):
    if level<=hard:return 0.0
    if level>=soft:return 1.0
    return (level-hard)/(soft-hard)
