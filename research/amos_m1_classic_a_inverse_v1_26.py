#!/usr/bin/env python3
"""AMOS M1 Classic-A SELL inverse structure audit v1.26.

M1 ONLY. Video is primary. No M5/M15/M30/G75. No fixed RR and no HTF target.
Baseline is v1.25 Run 37694142813.

Purpose: isolate why CLASSIC_A_SELL underperforms without touching Classic V BUY.
All variants keep the v1.25 live Close-path ownership and the same pure-M1
same-setup opposite Close-pivot target policy.
"""
from __future__ import annotations
import argparse, json, importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import QuoteTick

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v125',HERE/'amos_m1_path_extreme_precision_v1_25.py')
v125=importlib.util.module_from_spec(spec);sys.modules['v125']=v125;spec.loader.exec_module(v125)
v24=v125.v24; v16=v125.v16

def load_raw(catalog):
    cat=ParquetDataCatalog(catalog)
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value]);ticks.sort(key=lambda x:int(x.ts_event))
    ts=np.fromiter((int(x.ts_event) for x in ticks),dtype=np.int64,count=len(ticks))
    bid=np.fromiter((v16.v14.fpx(x.bid_price) for x in ticks),float,count=len(ticks))
    ask=np.fromiter((v16.v14.fpx(x.ask_price) for x in ticks),float,count=len(ticks))
    return pd.DataFrame({'datetime':pd.to_datetime(ts,unit='ns'),'ns':ts,'bid':bid,'ask':ask})

def enrich_a(rows,z):
    if rows.empty:return rows.copy()
    x=rows.copy();sig=x.entry_bar_i.astype(int).to_numpy();anc=x.anchor_i.astype(int).to_numpy()
    x['a_age_bars']=sig-anc
    x['signal_open']=[float(z.open.iloc[i]) for i in sig]
    x['signal_close']=[float(z.close.iloc[i]) for i in sig]
    x['signal_high']=[float(z.high.iloc[i]) for i in sig]
    x['signal_low']=[float(z.low.iloc[i]) for i in sig]
    x['signal_atr']=[float(z.atr.iloc[i]) for i in sig]
    x['bear_body']=x.signal_close < x.signal_open
    x['body_atr']=(x.signal_open-x.signal_close)/x.signal_atr
    x['poi_gap_atr']=(x.poi-x.signal_close)/x.signal_atr
    x['anchor_wick_atr']=[(float(z.high.iloc[a])-float(z.close.iloc[a]))/max(float(z.atr.iloc[s]),1e-12) for a,s in zip(anc,sig)]
    return x

def apply_static(a,kind):
    if kind=='BASE':return a.copy()
    q=a.copy()
    if kind in ('FAST_A','FAST_BEAR','REVISIT6_FAST_BEAR'):q=q[q.a_age_bars<=4]
    if kind in ('BEAR_BODY','FAST_BEAR','REVISIT6_FAST_BEAR'):q=q[q.bear_body]
    if kind=='TIGHT_PATH':q=q[q.poi_gap_atr<=0.75]
    return q.copy()

def delayed_revisit_entry(row,z,raw,max_bars):
    si=int(row.entry_bar_i);poi=float(row.poi);atr=float(z.atr.iloc[si]);tol=.10*atr
    raw_ns=raw.ns.to_numpy();end=min(len(z)-1,si+max_bars)
    for i in range(si+1,end+1):
        b=z.iloc[i];prev=z.iloc[i-1]
        touched=float(b.high)>=poi-tol
        held=float(b.close)<poi and float(b.close)<float(prev.close)
        if touched and held:
            confirm_end=pd.Timestamp(b.datetime)+pd.Timedelta(minutes=1)
            j=int(np.searchsorted(raw_ns,int(confirm_end.value),side='left'))
            if j>=len(raw):return None
            first=raw.iloc[j];ep=float(first.bid)
            risk=(ep-float(row.sl))*-1
            if risk<=0:return None
            base=row.to_dict();base['entry_bar_i']=int(i);base['entry_ns']=int(first.ns);base['entry_time']=first.datetime;base['entry']=ep;base['risk']=risk
            tgt=v24.resolve_structure_target(pd.Series(base),z)
            if tgt is None:return None
            base.update(tgt);return base
    return None

def make_variant(base_ent,z,raw,kind):
    other=base_ent[base_ent.pattern!='CLASSIC_A_SELL'].copy()
    a=apply_static(enrich_a(base_ent[base_ent.pattern=='CLASSIC_A_SELL'].copy(),z),kind)
    if kind.startswith('REVISIT'):
        mb=6 if kind=='REVISIT6_FAST_BEAR' else 4;rows=[]
        for _,r in a.iterrows():
            q=delayed_revisit_entry(r,z,raw,mb)
            if q is not None:rows.append(q)
        if rows:
            a2=pd.DataFrame(rows);a2=a2[[c for c in base_ent.columns if c in a2.columns]]
        else:a2=base_ent.iloc[:0].copy()
    else:a2=a[base_ent.columns].copy()
    return pd.concat([other,a2],ignore_index=True).sort_values('entry_ns').reset_index(drop=True)

def metric_pack(ent,raw):
    tr=v16.simulate(ent,raw,1) if len(ent) else pd.DataFrame()
    allm=v16.metrics(tr)
    am=v16.metrics(tr[tr.pattern=='CLASSIC_A_SELL']) if len(tr) else v16.metrics(pd.DataFrame())
    vm=v16.metrics(tr[tr.pattern=='CLASSIC_V_BUY']) if len(tr) else v16.metrics(pd.DataFrame())
    return tr,{'ALL':allm,'CLASSIC_A_SELL':am,'CLASSIC_V_BUY':vm}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    raw=load_raw(a.catalog);z=v16.v14.feature(v16.v14.bars(raw,1));ctx,counts=v125.v22.owned_mssplus_context(z)
    base_ent,stats,audit=v125.path_extreme_entries(ctx,z,raw)
    kinds=['BASE','FAST_A','BEAR_BODY','FAST_BEAR','TIGHT_PATH','REVISIT4','REVISIT6_FAST_BEAR']
    res={};enrich_a(base_ent[base_ent.pattern=='CLASSIC_A_SELL'].copy(),z).to_csv(out/'classic_a_baseline_diagnostics.csv',index=False)
    for kind in kinds:
        ent=make_variant(base_ent,z,raw,kind);tr,mp=metric_pack(ent,raw)
        ent.to_csv(out/f'entries_{kind}.csv',index=False);tr.to_csv(out/f'trades_{kind}.csv',index=False)
        res[kind]={'selected_A_before_sim':int((ent.pattern=='CLASSIC_A_SELL').sum()),**mp}
    result={'version':'v1.26','verification':'PURE_M1_CLASSIC_A_INVERSE_STRUCTURE_AUDIT_RAW_BIDASK',
      'baseline_run':37694142813,'raw_ticks':len(raw),'m1_bars':len(z),'mssplus_counts':counts,'path_stats':dict(stats),
      'constraints':['M1 only','Classic V BUY unchanged','no M5/M15/M30/G75','no fixed RR','no HTF target','same v1.25 live-path ownership','same pure-M1 structure target policy'],
      'goal':{'classic_a_sell':{'N_min':120,'WR_pct_min':45,'PF_min':1.2},'all_m1':{'PF_min':1.2,'DD_pct_max_stage1':10,'DD_pct_max_final':5}},
      'variants':res}
    (out/'result.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8');print(json.dumps(result,indent=2,default=str))
if __name__=='__main__':main()
