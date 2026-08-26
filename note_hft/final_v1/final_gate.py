from __future__ import annotations
import csv, json, sys
from pathlib import Path

BASELINE = {"N":176483,"BUY":88223,"SELL":88260,"WR":72.71,"PF":1.74,"MAX_DD":3.97}
MIN_N_RETENTION = 0.99


def load_signal(path: Path):
    rows=[]
    with path.open(newline='', encoding='utf-8') as f:
        r=csv.DictReader(f)
        need={'ts_ns','a_cond_num','b_cond_num'}
        if not need.issubset(set(r.fieldnames or [])):
            raise ValueError(f"signal csv must contain {sorted(need)}")
        for x in r:
            rows.append((int(x['ts_ns']), float(x['a_cond_num']), float(x['b_cond_num'])))
    return rows


def main():
    signal=Path(sys.argv[1]) if len(sys.argv)>1 else None
    result={"baseline":BASELINE,"status":None,"signal_rows":0,"n_retention":None}
    if signal is None or not signal.exists():
        result['status']='BLOCKED_SIGNAL'
        Path('final_gate_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))
        return 0
    rows=load_signal(signal)
    result['signal_rows']=len(rows)
    # Only classify authentic replay states. This is NOT a substitute alpha generator.
    buy=sum(1 for _,a,b in rows if b>0 and a<0)
    sell=sum(1 for _,a,b in rows if a>0 and b<0)
    n=buy+sell
    result.update({"candidate_buy_states":buy,"candidate_sell_states":sell,"candidate_n":n,
                   "n_retention":n/BASELINE['N'] if BASELINE['N'] else None})
    result['status']='READY_FOR_REALITY_BT' if result['n_retention']>=MIN_N_RETENTION else 'PARITY_FAIL'
    Path('final_gate_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
