from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

from nautilus_trader.model.data import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from g75_ev_expectancy_rawtick_v1 import run, build_table, EVStats, TF_SEC


def table_to_json(table):
    out=[]
    for (ds,vb,mb),s in table.items():
        out.append({'ds':ds,'vb':vb,'mb':mb,'n':s.n,'wins':s.wins,'sum_win':s.sum_win,'sum_loss':s.sum_loss})
    return out

def table_from_json(rows):
    return {(int(r['ds']),int(r['vb']),int(r['mb'])):EVStats(int(r['n']),int(r['wins']),float(r['sum_win']),float(r['sum_loss'])) for r in rows}

def load_slice(catalog_path, start_frac, end_frac):
    cat=ParquetDataCatalog(str(Path(catalog_path)))
    inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
    ticks=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
    if not ticks: raise SystemExit('no ticks')
    n=len(ticks); i=int(n*start_frac); j=int(n*end_frac)
    return inst,ticks[i:j],n

def worker_train(a):
    inst,ticks,total=load_slice(a.catalog,0.0,0.40)
    res,rows=run(inst,ticks,a.tf,'PURE')
    table=build_table(rows)
    obj={'train_summary':res,'train_cycles':len(rows),'ev_cells':len(table),'table':table_to_json(table),'raw_ticks_total':total,'train_ticks':len(ticks)}
    Path(a.worker_out).write_text(json.dumps(obj),encoding='utf-8')

def worker_eval(a):
    inst,ticks,total=load_slice(a.catalog,0.40,1.0)
    table=table_from_json(json.loads(Path(a.table_file).read_text())['table'])
    res,_=run(inst,ticks,a.tf,a.variant,table)
    obj={**res,'test_ticks':len(ticks),'raw_ticks_total':total}
    Path(a.worker_out).write_text(json.dumps(obj),encoding='utf-8')

def orchestrate(a):
    if not a.raw_bidask_only: raise SystemExit('Raw BidAsk mandatory')
    work=Path('results/g75-ev')/a.experiment_id/'tmp'; work.mkdir(parents=True,exist_ok=True)
    train_file=work/f'{a.tf}_train.json'; eval_file=work/f'{a.tf}_{a.variant}_eval.json'
    base=[sys.executable,__file__,'--catalog',a.catalog,'--experiment-id',a.experiment_id,'--tf',a.tf,'--raw-bidask-only']
    subprocess.run(base+['--worker','train','--worker-out',str(train_file)],check=True,env=os.environ.copy())
    subprocess.run(base+['--worker','eval','--variant',a.variant,'--table-file',str(train_file),'--worker-out',str(eval_file)],check=True,env=os.environ.copy())
    tr=json.loads(train_file.read_text()); ev=json.loads(eval_file.read_text())
    out={**ev,'tf':a.tf,'variant_eval':a.variant,'train_ticks':tr['train_ticks'],'train_cycles':tr['train_cycles'],'ev_cells':tr['ev_cells'],'split':'40% chronological train / 60% causal OOS test','frozen_core':{'trigger':0.12,'add':0.025,'reversal':0.20,'max_layers':10},'verification_level':'CAUSAL_RAW_BIDASK_G75_EV_V2_ISOLATED_ENGINES','logger_lifecycle_fix':'train and OOS engines run in separate OS processes to avoid Nautilus global logger re-init panic','note':'EV layer changes candidate acceptance/layer budget only; price thresholds remain frozen.'}
    d=Path('results/g75-ev')/a.experiment_id; d.mkdir(parents=True,exist_ok=True)
    p=d/f'{a.tf}_{a.variant}.json'; p.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--catalog',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--tf',choices=TF_SEC,required=True); ap.add_argument('--variant',choices=['PURE','EV_GATE','EV_SIZER'],default='PURE'); ap.add_argument('--raw-bidask-only',action='store_true'); ap.add_argument('--worker',choices=['train','eval']); ap.add_argument('--worker-out'); ap.add_argument('--table-file')
    a=ap.parse_args()
    if a.worker=='train': return worker_train(a)
    if a.worker=='eval': return worker_eval(a)
    return orchestrate(a)

if __name__=='__main__': main()
