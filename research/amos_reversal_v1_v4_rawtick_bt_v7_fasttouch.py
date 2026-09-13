#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amos_reversal_v1_v4_rawtick_bt_v6_true_mtf as v6

_CACHE = {}

def _tick_arrays(t):
    key=id(t)
    c=_CACHE.get(key)
    if c is None:
        tv=np.fromiter((pd.Timestamp(x).value for x in t['time']),dtype=np.int64,count=len(t))
        bid=t['bid'].to_numpy(dtype=float,copy=False)
        ask=t['ask'].to_numpy(dtype=float,copy=False)
        c=(tv,bid,ask)
        _CACHE[key]=c
    return c

def fast_first_touch_tick(t,start_ns,end_ns,side,lo,hi):
    tv,bid,ask=_tick_arrays(t)
    a=int(np.searchsorted(tv,np.int64(start_ns),side='right'))
    b=int(np.searchsorted(tv,np.int64(end_ns),side='right'))
    if b<=a:
        return None
    px=ask[a:b] if side==1 else bid[a:b]
    hit=np.flatnonzero((px>=lo)&(px<=hi))
    return None if hit.size==0 else a+int(hit[0])

v6.first_touch_tick=fast_first_touch_tick
v6.m.simulate=v6.simulate_true_mtf

if __name__=='__main__':
    v6.m.main()
