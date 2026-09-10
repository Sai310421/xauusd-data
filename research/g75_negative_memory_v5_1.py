from __future__ import annotations
import numpy as np
import research.g75_negative_memory_v5 as base

# Patch v5 threshold selection to avoid class-imbalance collapse.
# Objective: preserve a meaningful share of winners while still rejecting
# high-loss-probability states. Threshold is learned only from each training fold.
def choose_threshold(y, p, pnl, coverage_min=0.30):
    y=np.asarray(y,int); p=np.asarray(p,float); pnl=np.asarray(pnl,float)
    wins=(y==0); losses=(y==1)
    best_t=None; best_score=-1e99
    qs=np.linspace(0.10,0.90,33)
    cands=np.unique(np.quantile(p,qs))
    for t in cands:
        keep=p<t
        n=int(keep.sum())
        if n<30: continue
        coverage=float(keep.mean())
        if coverage<0.15: continue
        win_ret=float((keep & wins).sum()/max(1,wins.sum()))
        loss_avoid=float(((~keep) & losses).sum()/max(1,losses.sum()))
        if win_ret<0.55: continue
        train_pf=base.pf(pnl[keep])
        # Prefer PF, but penalize winner destruction and near-total rejection.
        score=(2.0*min(train_pf,5.0) + 1.5*loss_avoid + 1.0*win_ret + 0.35*coverage)
        if score>best_score:
            best_score=score; best_t=float(t)
    if best_t is None:
        # Safety fallback: keep about 60% of training observations.
        best_t=float(np.quantile(p,0.60))
    return best_t

base.choose_threshold=choose_threshold

if __name__=='__main__':
    base.main()
