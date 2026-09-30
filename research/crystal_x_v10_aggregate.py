from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    rows=[]
    for p in Path(args.root).rglob('result.json'):
        r=json.loads(p.read_text(encoding='utf-8'))
        rows.append({
            'policy':r['policy'],
            'threshold':r['base_threshold_points'],
            'locked':r['locked'],
            'lock_debt':r['lock_debt_points'],
            'recovered':r['economic_be_recovered'],
            'recovery_seconds':r['recovery_seconds'],
            'natural_recovery_after_lock':r['natural_recovery_after_lock'],
            'natural_recovery_seconds':r['natural_recovery_seconds'],
            'false_lock':r['score_components']['false_lock_flag'],
            'max_dd_prelock':r['max_dd_points_prelock'],
            'raw_ticks':r['raw_ticks'],
        })
    if not rows:
        raise SystemExit('no result.json files found')
    df=pd.DataFrame(rows).sort_values(['policy','threshold'])
    # Prefer successful recovery, then no false lock, then smaller debt and faster BE.
    df['recovery_penalty']=df['recovery_seconds'].fillna(1e18)
    df['score']=(df['recovered'].astype(int)*1_000_000
                 -df['false_lock'].astype(int)*250_000
                 -df['lock_debt'].fillna(1e9)
                 -df['recovery_penalty']/60.0)
    ranked=df.sort_values('score',ascending=False).reset_index(drop=True)
    ranked['rank']=ranked.index+1
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    ranked.to_csv(out/'frontier_ranked.csv',index=False)
    best=ranked.iloc[0].to_dict()
    summary={
        'verification_level':'RAW_BIDASK_LOCK_FRONTIER_RECONSTRUCTION',
        'selection_rule':'recovery success > avoid false lock > smaller frozen debt > faster Economic BE',
        'best':best,
        'candidates':ranked.to_dict(orient='records'),
        'important':'Do not promote a threshold to production until broker-cost reality and CRYSTAL parity gates pass.'
    }
    (out/'frontier_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    print(ranked.to_string(index=False))
    print(json.dumps(summary['best'],indent=2,default=str))

if __name__=='__main__':
    main()
