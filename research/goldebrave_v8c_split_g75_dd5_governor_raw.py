from __future__ import annotations
import argparse,json
from pathlib import Path
from goldebrave_v8c_split_g75_3branch_raw import V8CSplitG75,run as base_run

# DD5 follow-up contract for the winning FULL 3-branch structure.
# Parent signal/entry logic and N=96 configuration are frozen.
# The governor acts ONLY on new G75 ADD allocation, never deletes parent entries.
PROFILES={
 'g1': {'soft_dd':3.0,'hard_dd':5.0,'scales':(1.0,0.75,0.50,0.0)},
 'g2': {'soft_dd':3.5,'hard_dd':5.0,'scales':(1.0,0.80,0.40,0.0)},
 'g3': {'soft_dd':4.0,'hard_dd':5.0,'scales':(1.0,0.65,0.30,0.0)},
}

def governor_scale(dd_pct, profile):
 p=PROFILES[profile]; soft=p['soft_dd']; hard=p['hard_dd']; a,b,c,z=p['scales']
 if dd_pct>=hard:return z
 if dd_pct>=4.5:return c
 if dd_pct>=soft:return b
 return a

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--catalog',required=True);ap.add_argument('--governor',choices=PROFILES,required=True);ap.add_argument('--experiment-id',required=True);a=ap.parse_args()
 # Phase-1 executable audit: retain the verified FULL Raw engine and emit the governor contract.
 # Exact floating-equity order semantics are deliberately not faked here; this file is the handoff gate.
 base_run(a.catalog,'c','SPLIT_G75_3X10_FULL',a.experiment_id+'_'+a.governor)
 p=Path('results/goldebrave-v8c-split-g75-dd5')/a.experiment_id/f'{a.governor}_contract.json';p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps({'verification':'DD5_GOVERNOR_CONTRACT','parent_N':96,'parent_entry_frozen':True,'variant':'SPLIT_G75_3X10_FULL','governor':a.governor,**PROFILES[a.governor],'rule':'DD governor changes new G75 ADD sizing only; parent entries are never removed. Hard DD 5% => no new ADD until recovery.','next_gate':'Implement exact broker-order/floating-equity semantics before claiming DD<=5%.'},indent=2))
 print(p.read_text())
if __name__=='__main__':main()
