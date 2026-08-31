from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from research.hft_boost_raw_xau_bt import metrics


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--experiment-id',required=True); ap.add_argument('--initial',type=float,default=1000.0); ap.add_argument('--days',type=int,required=True)
    args=ap.parse_args(); root=Path(args.root); outroot=Path('results/ae-bt')/args.experiment_id; outroot.mkdir(parents=True,exist_ok=True)
    ranking=[]
    variants=sorted({json.loads(f.read_text()).get('variant') for f in root.rglob('summary.json') if json.loads(f.read_text()).get('verification_level')=='NAUTILUS_BT_RAW_BIDASK_RANGE_GRID'})
    for v in variants:
        sums=[]; trades=[]
        for f in root.rglob('summary.json'):
            s=json.loads(f.read_text())
            if s.get('verification_level')!='NAUTILUS_BT_RAW_BIDASK_RANGE_GRID' or s.get('variant')!=v: continue
            sums.append(s); tf=f.with_name('trades.csv')
            if tf.exists():
                df=pd.read_csv(tf)
                for _,r in df.iterrows(): trades.append({'pnl':float(r.pnl),'ts_closed':int(r.ts_closed)})
        trades.sort(key=lambda x:x['ts_closed'])
        k=metrics(trades,args.initial,args.days)
        ranking.append({'variant':v,'shards':len(sums),'empty_shards':sum(bool(x.get('empty_shard')) for x in sums),'raw_ticks':sum(int(x.get('raw_ticks',0)) for x in sums),'fills':sum(int(x.get('order_fills',0)) for x in sums),'rejects':sum(int(x.get('order_rejects',0)) for x in sums),'entries':sum(int(x.get('entries_submitted',0)) for x in sums),'adds':sum(int(x.get('adds_submitted',0)) for x in sums),'closed_baskets':sum(int(x.get('closed_baskets',0)) for x in sums),'max_layers_seen':max([int(x.get('max_layers_seen',0)) for x in sums] or [0]),'max_adverse_points':max([float(x.get('max_adverse_points',0)) for x in sums] or [0.0]),**k})
    ranking.sort(key=lambda x:(x.get('PF',0),x.get('NetProfit',0)),reverse=True)
    pd.DataFrame(ranking).to_csv(outroot/'ranking.csv',index=False)
    summary={'verification_level':'NAUTILUS_BT_RAW_BIDASK_RANGE_GRID_ROUND1','ohlc_resample_used':False,'days':args.days,'ranking':ranking}
    (outroot/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=True)); print(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=True))
if __name__=='__main__': main()
