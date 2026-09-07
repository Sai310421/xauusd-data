from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

ap=argparse.ArgumentParser(); ap.add_argument('--experiment-id',required=True); args=ap.parse_args()
root=Path('results/ae-bt')/args.experiment_id; files=sorted((root/'cells').glob('*.json'))
if len(files)!=9: raise SystemExit(f'expected 9 cells, got {len(files)}')
rows=[json.loads(p.read_text(encoding='utf-8')) for p in files]
df=pd.DataFrame(rows).sort_values(['tf','variant']); df.to_csv(root/'kpi.csv',index=False)
rel=[]
for tf,g in df.groupby('tf'):
    a=g[g.variant=='A'].iloc[0]
    for _,r in g.iterrows():
        rel.append({'tf':tf,'variant':r.variant,'N_retention':float(r.cycles/max(a.cycles,1)),
                    'return_delta_vs_A':float(r.realized_virtual-a.realized_virtual),
                    'dd_delta_vs_A':float(r.max_floating_DD_pct-a.max_floating_DD_pct)})
summary={
 'verification_level':'NAUTILUS_BT_RAW_BIDASK_G75_TSUGI_ABC',
 'engine':'NautilusTrader BacktestEngine; isolated process per cell',
 'data_kind':'RAW_BIDASK QuoteTick','ohlc_resample_used':False,
 'signal_bars':'Nautilus INTERNAL BID bars built directly from Raw QuoteTicks',
 'results':rows,'relative':rel,
 'controller_semantics':{'A':'Frozen G75 baseline','B':'soft DD throttle + 4.5% hard stop','C':'B + 3.0% economic hedge-lock equivalent + Debt Recovery; 70% positive recovery PnL to debt'},
 'limitations':['C is flatten-and-debt economic lock, not simultaneous long/short HEDGING-mode execution.','Virtual basket metrics are synchronized to Raw Bid/Ask; native position-report PnL is kept as execution cross-check.','Observed spread is native; no added commission/slippage/reject stress model in this first gate.']
}
(root/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print(df.to_string(index=False)); print(json.dumps(summary,ensure_ascii=False))
