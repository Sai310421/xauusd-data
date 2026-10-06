import argparse, json
from pathlib import Path
import pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument('--file',required=True)
ap.add_argument('--out',required=True)
a=ap.parse_args()

p=Path(a.file)
d=pd.read_parquet(p)
ts_col='ts' if 'ts' in d.columns else 'timestamp'
d['datetime']=pd.to_datetime(d[ts_col],utc=True)
d['bid']=pd.to_numeric(d['bid'],errors='coerce')
d['ask']=pd.to_numeric(d['ask'],errors='coerce')
d=d.dropna(subset=['datetime','bid','ask'])
d['day']=d.datetime.dt.floor('D')
d['spread']=d.ask-d.bid
g=d.groupby('day').agg(
    rows=('bid','size'),
    bid_min=('bid','min'),
    bid_median=('bid','median'),
    bid_max=('bid','max'),
    ask_median=('ask','median'),
    spread_median=('spread','median'),
).reset_index()
g['ratio_to_prev']=g.bid_median/g.bid_median.shift(1)
g['scale_break']=((g.ratio_to_prev>5)|(g.ratio_to_prev<0.2)).fillna(False)
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
g.to_csv(a.out,index=False)
print(g.to_string(index=False))
print(json.dumps({'file':p.name,'rows':len(d),'bid_min':float(d.bid.min()),'bid_median':float(d.bid.median()),'bid_max':float(d.bid.max()),'scale_break_days':g.loc[g.scale_break,'day'].astype(str).tolist()},indent=2))
