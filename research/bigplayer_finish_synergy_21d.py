from __future__ import annotations
# Finish-stage external synergy screen. BigPlayer signal formulas remain untouched.
# Reuses persisted common-5m trades; tests external MTF confirmation as BOOST/routing, not hard formula changes.
import math, json
from pathlib import Path
import numpy as np
import pandas as pd

SRC=Path('results/bigplayer_synergy_supervisor_21d')
OUT=Path('results/bigplayer_finish_synergy_21d'); INITIAL=1000.0

def metrics(t):
    if t.empty:return dict(N=0,N_per_day=0.,WR=0.,PF=0.,RF=0.,net_profit=0.,return_pct=0.,daily_return_pct=0.,max_dd_pct=0.,max_dd_usd=0.,final_balance=INITIAL)
    t=t.sort_values(['entry_time','tf']); p=t['weighted_pnl'] if 'weighted_pnl' in t else t.pnl
    gp=p[p>0].sum(); gl=-p[p<0].sum(); net=p.sum(); eq=INITIAL+p.cumsum(); peak=np.maximum.accumulate(np.r_[INITIAL,eq.values])[1:]; dd=peak-eq.values; mdd=float(dd.max())
    return dict(N=len(t),N_per_day=len(t)/21.,WR=(p>0).mean()*100.,PF=gp/gl if gl>0 else math.inf,RF=net/mdd if mdd>0 else math.inf,net_profit=net,return_pct=net/INITIAL*100.,daily_return_pct=net/INITIAL*100./21.,max_dd_pct=float(np.max(np.where(peak>0,dd/peak*100.,0))),max_dd_usd=mdd,final_balance=INITIAL+net)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    base=pd.read_csv(SRC/'trades_TREND_REGIME.csv')
    # Persisted CSV can contain mixed timestamp string representations; normalize explicitly.
    base['entry_time']=pd.to_datetime(base['entry_time'],utc=True,format='mixed')
    base['exit_time']=pd.to_datetime(base['exit_time'],utc=True,format='mixed')
    # Exact-minute same-direction cross-TF confirmation. Signals remain independently tradeable.
    base['minute']=base.entry_time.dt.floor('min')
    g=base.groupby(['minute','direction']).size().rename('agree_count').reset_index()
    x=base.merge(g,on=['minute','direction'],how='left')
    rows=[]
    for boost in [1.0,1.25,1.5,2.0]:
        y=x.copy(); y['boost']=np.where(y.agree_count>=2,boost,1.0); y['weighted_pnl']=y.pnl*y.boost
        rows.append({'config':f'MTF_BOOST_{boost:g}','boost_when_same_direction_cross_tf':boost,'confirmed_trades':int((y.agree_count>=2).sum()),**metrics(y)})
    for name,mask in [('MTF_CONFIRMED_ONLY',x.agree_count>=2),('MTF_UNCONFIRMED_ONLY',x.agree_count<2)]:
        y=x.loc[mask].copy(); y['weighted_pnl']=y.pnl; rows.append({'config':name,'boost_when_same_direction_cross_tf':1.0,'confirmed_trades':int((y.agree_count>=2).sum()),**metrics(y)})
    r=pd.DataFrame(rows); r.to_csv(OUT/'summary_21d.csv',index=False); x.to_csv(OUT/'annotated_trades.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps({'source':'results/bigplayer_synergy_supervisor_21d/trades_TREND_REGIME.csv','formula_policy':'FROZEN_BIGPLAYER_NO_INTERNAL_CHANGES','execution':'persisted common fixed 5-minute raw-tick horizon','mtf_policy':'same-minute same-direction confirmation; single edges remain tradeable; boost only','timestamp_adapter_fix':'explicit pandas UTC mixed-format normalization; no signal or threshold changes','note':'This stage isolates MTF confirmation synergy before adding liquidity/exit branches.'},indent=2))
    print(r.to_string(index=False))
if __name__=='__main__': main()
