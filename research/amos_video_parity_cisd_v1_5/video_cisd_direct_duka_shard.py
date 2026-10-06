from __future__ import annotations
import argparse,json,math
from decimal import Decimal
from pathlib import Path
import numpy as np,pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig,RiskEngineConfig
from nautilus_trader.model import BarType,Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType,OmsType,BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from research.amos_video_parity_cisd_v1_5.video_cisd_nautilus_route_shard import VideoAMD,Cfg,extract_trades

ROUTES=["ASIA","LONDON","NY"]

def rmetrics(rows,risk_pct=.35):
 if not rows:return {"N":0,"wins":0,"WR_pct":0.,"PF_R":0.,"sum_R":0.,"Return_pct_risk":0.,"MaxDD_pct_risk":0.}
 x=np.array([r["R_realized"] for r in sorted(rows,key=lambda z:z["signal_ts"])],float)
 gp=x[x>0].sum();gl=-x[x<0].sum();eq=pk=100.;dd=0.
 for r in x:
  eq*=max(.0001,1+risk_pct*r/100);pk=max(pk,eq);dd=max(dd,100*(pk-eq)/pk)
 return {"N":len(x),"wins":int((x>0).sum()),"WR_pct":float((x>0).mean()*100),
  "PF_R":float(gp/gl) if gl>0 else (math.inf if gp>0 else 0.),"sum_R":float(x.sum()),
  "Return_pct_risk":float(eq-100),"MaxDD_pct_risk":float(dd)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--catalog",required=True);ap.add_argument("--out",required=True)
 ap.add_argument("--primary-start",required=True);ap.add_argument("--primary-end",required=True);ap.add_argument("--label",required=True)
 a=ap.parse_args();p=Path(a.out);p.mkdir(parents=True,exist_ok=True)
 ps=pd.Timestamp(a.primary_start,tz="UTC");pe=pd.Timestamp(a.primary_end,tz="UTC")
 cat=ParquetDataCatalog(a.catalog);inst=next(x for x in cat.instruments() if x.id.symbol.value.replace('/','')=='XAUUSD')
 raw=cat.query(data_cls=QuoteTick,identifiers=[inst.id.value])
 if not raw:raise RuntimeError("direct Duka catalog has no QuoteTicks")
 all_t=[];all_s=[];rrows=[];route_summaries={}
 for route in ROUTES:
  eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)))
  eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,
    base_currency=USD,starting_balances=[Money(300,USD)],default_leverage=Decimal("2000"))
  eng.add_instrument(inst);eng.add_data(raw)
  bt=BarType.from_str(f"{inst.id.value}-15-MINUTE-MID-INTERNAL")
  st=VideoAMD(Cfg(instrument_id=inst.id,bar_type=bt,signal_start_ns=int(ps.value),signal_end_ns=int(pe.value),route_name=route))
  eng.add_strategy(st);eng.run();eng.end()
  tr=extract_trades(eng.trader.generate_positions_report());sg=list(st.signal_meta)
  n=min(len(tr),len(sg));local=[]
  for i in range(n):
   pnl=float(tr[i]["pnl"]);entry=float(sg[i]["entry_ref"]);sl=float(sg[i]["sl"]);risk=abs(entry-sl)
   local.append({"shard":a.label,"route":route,"signal_ts":int(sg[i]["signal_ts"]),"signal_close_ts":int(sg[i].get("signal_close_ts",sg[i]["signal_ts"])),
    "pnl":pnl,"entry_ref":entry,"sl":sl,"tp":float(sg[i]["tp"]),"planned_rr":float(sg[i]["rr"]),"risk_ref":risk,
    "R_realized":pnl/risk if risk>0 else 0.})
  for x in tr:x.update({"shard":a.label,"route":route})
  for x in sg:x.update({"shard":a.label})
  all_t.extend(tr);all_s.extend(sg);rrows.extend(local)
  route_summaries[route]={"signals":st.signals,"closed":len(tr),**rmetrics(local)}
  eng.dispose()
 out={"verification":"NAUTILUS_DIRECT_DUKASCOPY_VIDEO_CISD_V1_11_SHARD","shard":a.label,
  "primary_start":str(ps),"primary_end_exclusive":str(pe),"raw_ticks":len(raw),"routes":route_summaries,**rmetrics(rrows)}
 (p/"summary.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
 pd.DataFrame(all_t,columns=["pnl","ts_closed","shard","route"]).to_csv(p/"trades.csv",index=False)
 pd.DataFrame(all_s).to_csv(p/"signals.csv",index=False)
 pd.DataFrame(rrows,columns=["shard","route","signal_ts","signal_close_ts","pnl","entry_ref","sl","tp","planned_rr","risk_ref","R_realized"]).to_csv(p/"risk_trades.csv",index=False)
 print(json.dumps(out,indent=2),flush=True)
if __name__=="__main__":main()
