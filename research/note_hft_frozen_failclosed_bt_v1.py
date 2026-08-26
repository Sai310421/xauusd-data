import json
from pathlib import Path
import pandas as pd

DATA=Path('csv/XAUUSD/XAUUSD_M1_2026Q1Q2.csv')
OUT=Path('results/note-hft-frozen-failclosed-v1')
OUT.mkdir(parents=True,exist_ok=True)

# Exact behavior of the supplied Frozen fragment where the original
# a_cond_num / b_cond_num generator is masked as 00000 and therefore
# falls through to a_cond_num=b_cond_num=0. No proxy signal is invented.
c_n_of=0
raw=pd.read_csv(DATA)
rows=len(raw)
a_cond_num=0
b_cond_num=0
s_ct_cond=(b_cond_num>c_n_of and a_cond_num<0)
l_ct_cond=(a_cond_num>c_n_of and b_cond_num<0)

trades=0
buy=0
sell=0

baseline={
  'N':176483,'WR_pct':72.71,'PF':1.74,'MaxDD_pct':3.97,
  'BUY':88223,'SELL':88260
}
result={
  'status':'FAIL_CLOSED_MISSING_SIGNAL_FRAGMENT',
  'data':str(DATA),
  'rows':rows,
  's_ct_cond':bool(s_ct_cond),
  'l_ct_cond':bool(l_ct_cond),
  'N':trades,'BUY':buy,'SELL':sell,
  'N_retention':trades/baseline['N'],
  'parity_pass':False,
  'reality_noise':'MANDATORY_BUT_NO_ORDERS_TO_NOISE',
  'baseline':baseline,
  'blocking_reason':'Original a_cond_num / b_cond_num generator is absent/masked as 00000 in supplied Frozen source; exact supplied behavior generates no entries. No substitute strategy used.',
  'next_required_input':'Original signal-generator fragment or source containing a_cond_num/b_cond_num assignments.'
}
(OUT/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
pd.DataFrame([result]).to_csv(OUT/'summary.csv',index=False)
print(json.dumps(result,indent=2,ensure_ascii=False))
