from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path('results/bigplayer_rawtick_5tf_21d/trades_all.csv')
OUT = Path('results/bigplayer_positive_edges_21d')
INITIAL = 1000.0
DAYS = 21
SELECTED = {('M1','IMBALANCE'), ('M5','ABSORPTION')}


def metrics(t: pd.DataFrame) -> dict:
    if t.empty:
        return dict(N=0,N_per_day=0.0,WR=0.0,PF=0.0,RF=0.0,net_profit=0.0,return_pct=0.0,max_dd_pct=0.0,max_dd_usd=0.0,final_balance=INITIAL)
    t = t.sort_values('entry_time').reset_index(drop=True)
    gp = float(t.loc[t.pnl > 0, 'pnl'].sum())
    gl = float(-t.loc[t.pnl < 0, 'pnl'].sum())
    net = float(t.pnl.sum())
    pf = gp / gl if gl > 0 else (math.inf if gp > 0 else 0.0)
    wr = float((t.pnl > 0).mean() * 100.0)
    eq = INITIAL + t.pnl.cumsum()
    peak = pd.concat([pd.Series([INITIAL]), eq], ignore_index=True).cummax().iloc[1:].to_numpy()
    dd = peak - eq.to_numpy()
    maxdd = float(dd.max()) if len(dd) else 0.0
    ddpct = np.where(peak > 0, dd / peak * 100.0, 0.0)
    maxddpct = float(ddpct.max()) if len(ddpct) else 0.0
    rf = net / maxdd if maxdd > 0 else (math.inf if net > 0 else 0.0)
    return dict(N=int(len(t)),N_per_day=float(len(t)/DAYS),WR=wr,PF=pf,RF=rf,net_profit=net,return_pct=net/INITIAL*100.0,max_dd_pct=maxddpct,max_dd_usd=maxdd,final_balance=INITIAL+net)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC)
    df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
    mask = df.apply(lambda r: (r['tf'], r['edge']) in SELECTED, axis=1)
    sel = df.loc[mask].copy().sort_values('entry_time').reset_index(drop=True)

    rows = []
    for tf, edge in sorted(SELECTED):
        x = sel[(sel.tf == tf) & (sel.edge == edge)].copy()
        rows.append({'scope':'EDGE','tf':tf,'edge':edge,**metrics(x)})
    rows.append({'scope':'PORTFOLIO','tf':'M1+M5','edge':'IMBALANCE+ABSORPTION',**metrics(sel)})

    # External-only overlap diagnostics. No edge formulas or thresholds are changed.
    z = sel[['entry_time','tf','edge','direction','pnl']].copy()
    z['minute'] = z['entry_time'].dt.floor('min')
    grouped = z.groupby(['minute','direction'])
    overlap = grouped.filter(lambda g: len(g) >= 2).sort_values(['minute','direction'])
    overlap_groups = int(overlap[['minute','direction']].drop_duplicates().shape[0]) if not overlap.empty else 0
    overlap_trades = int(len(overlap))

    pd.DataFrame(rows).to_csv(OUT/'summary_21d.csv', index=False)
    sel.to_csv(OUT/'selected_trades.csv', index=False)
    overlap.to_csv(OUT/'same_direction_overlap_candidates.csv', index=False)
    (OUT/'provenance.json').write_text(json.dumps({
        'source': str(SRC),
        'selection_policy': 'EXTERNAL_ONLY_NO_FORMULA_CHANGE',
        'selected_edges': [{'tf':'M1','edge':'IMBALANCE'},{'tf':'M5','edge':'ABSORPTION'}],
        'selection_basis': 'positive standalone 21d structural BT; treat as in-sample candidate, not out-of-sample proof',
        'apex_policy': 'NOT_INJECTED_UNTIL_EXACT_PERIODS_ARE_RECOVERED',
        'same_direction_overlap': {'groups': overlap_groups, 'trades': overlap_trades},
        'formula_changes': False,
    }, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps({'overlap_groups':overlap_groups,'overlap_trades':overlap_trades}, indent=2))


if __name__ == '__main__':
    main()
