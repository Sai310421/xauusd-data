from __future__ import annotations
import argparse, json, heapq
from dataclasses import asdict
from pathlib import Path
import pandas as pd
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from research.amos_reversal_v1_v4_rawtick_bt import TF_MIN, bars_from_ticks, simulate, metrics
from research.ae_multifiltration_eprocess import MultiFiltrationGovernor


def load_catalog_ticks(catalog_path:str):
    cat=ParquetDataCatalog(catalog_path)
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    ticks=cat.query(data_cls=QuoteTick, identifiers=[inst.id.value])
    rows=[]
    for t in ticks:
        f=lambda x: float(x.as_double()) if hasattr(x,'as_double') else float(x)
        rows.append((pd.to_datetime(t.ts_event,unit='ns',utc=True),f(t.bid_price),f(t.ask_price),1.0))
    return pd.DataFrame(rows,columns=['time','bid','ask','vol'])


def run_ab(ticks, mode='STANDARD', rr=2.0, min_e=1.05):
    all_trades=[]
    for tf in TF_MIN:
        b=bars_from_ticks(ticks,tf)
        all_trades.extend(simulate(ticks,b,'V4',tf,mode,rr))
    all_trades.sort(key=lambda t: pd.Timestamp(t.entry_time))
    gov=MultiFiltrationGovernor(fine_stream='M1',fusion_weight=0.5,min_evidence=min_e)
    pending=[]; accepted=[]; rows=[]
    for t in all_trades:
        et=pd.Timestamp(t.entry_time)
        while pending and pending[0][0] <= et:
            _,_,done=heapq.heappop(pending)
            gov.update(done.timeframe,1.0 if done.r>0 else 0.0,center=0.5,scale=0.5)
        allow=gov.allow_entry()
        fused=gov.fused_value()
        if allow: accepted.append(t)
        rows.append({**asdict(t),'eprocess_allow':allow,'fused_e_before':fused})
        heapq.heappush(pending,(pd.Timestamp(t.exit_time),len(rows),t))
    return all_trades,accepted,rows,gov.snapshot()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--catalog',required=True); p.add_argument('--out',required=True); p.add_argument('--mode',default='STANDARD'); p.add_argument('--rr',type=float,default=2.0); p.add_argument('--min-evidence',type=float,default=1.05); a=p.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    ticks=load_catalog_ticks(a.catalog)
    base,ep,rows,snap=run_ab(ticks,a.mode,a.rr,a.min_evidence)
    result={'verification_level':'RAW_BIDASK_SIGNAL_EXECUTION_AB','raw_ticks':len(ticks),'base':metrics(base),'eprocess':metrics(ep),'entry_reduction_pct':100*(1-len(ep)/max(len(base),1)),'final_evidence':snap,'mode':a.mode,'rr':a.rr,'min_evidence':a.min_evidence}
    pd.DataFrame(rows).to_csv(out/'trades_ab.csv',index=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    print(json.dumps(result,indent=2,default=str))

if __name__=='__main__': main()
