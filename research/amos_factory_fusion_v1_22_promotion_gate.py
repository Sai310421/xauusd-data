from __future__ import annotations
import argparse,json,math
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--summary",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    s=json.loads(Path(a.summary).read_text())
    modes=s.get("modes",{})
    out={"gate_version":"PHASE3_V122_PROMOTION_GATE","criteria":{
        "pf_min":1.2,"full_n_min":20,"half_n_min":10,"half_pf_min":1.0,
        "max_floating_dd_pct":5.0,"max_daily_loss_pct":3.5},"modes":{}}
    for name,r in modes.items():
        m=r.get("metrics",{});st=r.get("stability",{})
        checks={
          "pf": float(m.get("PF_legs",0))>=1.2,
          "full_n": int(m.get("N_legs",0))>=20,
          "first_half_n": int(st.get("first_half",{}).get("N",0))>=10,
          "second_half_n": int(st.get("second_half",{}).get("N",0))>=10,
          "first_half_pf": float(st.get("first_half",{}).get("PF",0))>1.0,
          "second_half_pf": float(st.get("second_half",{}).get("PF",0))>1.0,
          "floating_dd": float(r.get("max_floating_dd_pct",999))<=5.0,
          "daily_dd": float(r.get("max_daily_loss_pct",999))<=3.5,
        }
        out["modes"][name]={
          "decision":"PROMOTE" if all(checks.values()) else "REJECT",
          "checks":checks,
          "metrics":m,
          "stability":st,
          "max_floating_dd_pct":r.get("max_floating_dd_pct"),
          "max_daily_loss_pct":r.get("max_daily_loss_pct"),
        }
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()
