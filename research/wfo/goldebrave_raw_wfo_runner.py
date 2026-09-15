from __future__ import annotations
import argparse, hashlib, json, math, sys
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

# Make repository/research importable regardless of whether this file is invoked
# from repository root or with working-directory=research.
RESEARCH_DIR = Path(__file__).resolve().parents[1]
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

from goldebrave_fasttf_parity_raw import Cfg, f, l1
from goldebrave_v8c_split_g75_cvar_reflect_raw import CVarReflect, PROFILES as CVAR_PROFILES

if not hasattr(ParquetDataCatalog, "query_quote_ticks"):
    def _q(self, identifiers=None, start=None, end=None):
        return self.query(data_cls=QuoteTick, identifiers=identifiers, start=start, end=end)
    ParquetDataCatalog.query_quote_ticks = _q


def canonical_hash(obj: dict) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def ts_to_pd(ts_ns: int) -> pd.Timestamp:
    return pd.Timestamp(int(ts_ns), unit="ns", tz="UTC")


def raw_bounds(raw) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not raw:
        raise RuntimeError("No Raw QuoteTick data found")
    return ts_to_pd(raw[0].ts_event), ts_to_pd(raw[-1].ts_event)


class AutoCVarReflect(CVarReflect):
    def __init__(self, c, enter_safe_dd=3.50, exit_safe_dd=2.25):
        super().__init__(c, "cv3")
        self.enter_safe_dd = enter_safe_dd
        self.exit_safe_dd = exit_safe_dd
        self.auto_safe = False
        self.auto_switch_to_safe = 0
        self.auto_switch_to_high = 0
        self.auto_safe_ticks = 0
        self.auto_high_ticks = 0

    def _risk_control(self, bid, ask, dd):
        if not self.auto_safe and dd >= self.enter_safe_dd:
            self.auto_safe = True
            self.auto_switch_to_safe += 1
        elif self.auto_safe and dd <= self.exit_safe_dd:
            self.auto_safe = False
            self.auto_switch_to_high += 1
        if self.auto_safe:
            self.auto_safe_ticks += 1
            return super()._risk_control(bid, ask, dd)
        self.auto_high_ticks += 1

    def summary_auto(self):
        s = self.summary_cvar()
        s.update({"wfo_mode":"AUTO","auto_enter_safe_dd":self.enter_safe_dd,"auto_exit_safe_dd":self.exit_safe_dd,"auto_switch_to_safe":self.auto_switch_to_safe,"auto_switch_to_high":self.auto_switch_to_high,"auto_safe_ticks":self.auto_safe_ticks,"auto_high_ticks":self.auto_high_ticks})
        return s


def strategy_cfg(inst):
    s = inst.id.value
    return Cfg(instrument_id=inst.id,m1=BarType.from_str(f"{s}-1-MINUTE-BID-INTERNAL"),m5=BarType.from_str(f"{s}-5-MINUTE-BID-INTERNAL"),m15=BarType.from_str(f"{s}-15-MINUTE-BID-INTERNAL"),h1=BarType.from_str(f"{s}-1-HOUR-BID-INTERNAL"),mode="m15")


def run_slice(cat, inst, start, end, mode, auto_enter=3.50, auto_exit=2.25):
    raw = cat.query_quote_ticks(identifiers=[inst.id.value],start=start.isoformat(),end=end.isoformat())
    if not raw: raise RuntimeError(f"No ticks in slice {start} -> {end}")
    eng=BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"),risk_engine=RiskEngineConfig(bypass=True)))
    eng.add_venue(venue=inst.id.venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,book_type=BookType.L1_MBP,base_currency=USD,starting_balances=[Money(1000,USD)],default_leverage=Decimal("2000"))
    eng.add_instrument(inst); eng.add_data(l1(raw))
    if mode=="HIGH": st=CVarReflect(strategy_cfg(inst),"off")
    elif mode=="SAFE": st=CVarReflect(strategy_cfg(inst),"cv3")
    elif mode=="AUTO": st=AutoCVarReflect(strategy_cfg(inst),auto_enter,auto_exit)
    else: raise ValueError(mode)
    eng.add_strategy(st); eng.run(); eng.end()
    summary=st.summary_auto() if mode=="AUTO" else st.summary_cvar()
    return {"mode":mode,"start":start.isoformat(),"end":end.isoformat(),"raw_ticks":len(raw),"ohlc_resample_used":False,"execution_semantics":"Raw Bid/Ask MTM mathematical sizing harness; not yet native broker order/fill semantics",**summary}


def overall(summary):
    x=summary.get("overall",{})
    pf=float(x.get("PF",0.0))
    return {"N":int(x.get("N",0)),"WR_pct":float(x.get("WR_pct",0.0)),"PF":pf if math.isfinite(pf) else 999999.0,"Net":float(x.get("Net",0.0)),"EV":float(x.get("EV",0.0)),"MaxDD_pct_virtual":float(x.get("MaxDD_pct_virtual",0.0)),"RF_virtual":x.get("RF_virtual"),"max_floating_dd_pct":float(summary.get("max_floating_dd_pct",0.0))}


def make_windows(data_start,data_end,is_days,oos_days,step_days):
    out=[]; anchor=data_start; i=0
    while True:
        is_start=anchor; is_end=is_start+timedelta(days=is_days); oos_start=is_end; oos_end=oos_start+timedelta(days=oos_days)
        if oos_end>data_end: break
        out.append((i,is_start,is_end,oos_start,oos_end)); i+=1; anchor=anchor+timedelta(days=step_days)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--catalog",required=True); ap.add_argument("--config",default="wfo/goldebrave_wfo_config.json"); ap.add_argument("--out",default="results/wfo/latest/wfo_records.json"); ap.add_argument("--modes",default="HIGH,SAFE,AUTO"); ap.add_argument("--max-windows",type=int,default=0); ap.add_argument("--manifest-only",action="store_true"); a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text()); cat=ParquetDataCatalog(a.catalog); inst=next(x for x in cat.instruments() if x.id.symbol.value.replace("/","")=="XAUUSD"); all_raw=cat.query_quote_ticks(identifiers=[inst.id.value]); d0,d1=raw_bounds(all_raw)
    rw=cfg["rolling_windows"]; windows=make_windows(d0,d1,rw["is_days"],rw["oos_days"],rw["step_days"])
    if a.max_windows>0: windows=windows[:a.max_windows]
    coverage_days=(d1-d0).total_seconds()/86400.0; required_min_days=rw["is_days"]+rw["oos_days"]+(rw["minimum_oos_windows"]-1)*rw["step_days"]
    manifest={"schema":"goldebrave-raw-wfo-records-v1","catalog":a.catalog,"instrument":inst.id.value,"data_start":d0.isoformat(),"data_end":d1.isoformat(),"coverage_days":coverage_days,"required_min_days_for_gate":required_min_days,"available_windows":len(windows),"minimum_oos_windows":rw["minimum_oos_windows"],"preferred_oos_windows":rw["preferred_oos_windows"],"window_spec":rw,"warning":None if len(windows)>=rw["minimum_oos_windows"] else "INSUFFICIENT_CATALOG_COVERAGE_FOR_ADOPTION_GATE","windows":[{"window_id":i,"is_start":s.isoformat(),"is_end":e.isoformat(),"oos_start":os.isoformat(),"oos_end":oe.isoformat()} for i,s,e,os,oe in windows]}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    if a.manifest_only: out.write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2)); return
    modes=[x.strip().upper() for x in a.modes.split(",") if x.strip()]; records=[]
    for mode in modes:
        frozen=cfg["frozen_profiles"].get(mode,{}) if mode!="AUTO" else {"HIGH":cfg["frozen_profiles"]["HIGH"],"SAFE":cfg["frozen_profiles"]["SAFE"],"auto_enter_safe_dd":cfg.get("auto",{}).get("enter_safe_dd",3.50),"auto_exit_safe_dd":cfg.get("auto",{}).get("exit_safe_dd",2.25)}; ph=canonical_hash(frozen)
        for wid,is_start,is_end,oos_start,oos_end in windows:
            is_res=run_slice(cat,inst,is_start,is_end,mode,cfg.get("auto",{}).get("enter_safe_dd",3.50),cfg.get("auto",{}).get("exit_safe_dd",2.25)); oos_res=run_slice(cat,inst,oos_start,oos_end,mode,cfg.get("auto",{}).get("enter_safe_dd",3.50),cfg.get("auto",{}).get("exit_safe_dd",2.25)); ik=overall(is_res); ok=overall(oos_res); wfe=(100.0*ok["Net"]/ik["Net"]) if ik["Net"]>0 else None; n_ret=(ok["N"]/max(1,ik["N"]*(rw["oos_days"]/rw["is_days"]))) if ik["N"]>0 else 0.0
            records.append({"experiment_id":f"raw-wfo-{mode.lower()}-w{wid:02d}","profile":mode,"parameter_hash":ph,"parameters_frozen_before_oos":True,"window_id":wid,"is":{**ik,"start":is_start.isoformat(),"end":is_end.isoformat()},"oos":{**ok,"start":oos_start.isoformat(),"end":oos_end.isoformat()},"wfe_pct":wfe,"n_retention":n_ret,"raw_ticks_is":is_res["raw_ticks"],"raw_ticks_oos":oos_res["raw_ticks"],"ohlc_resample_used":False})
    payload={**manifest,"records":records}; out.write_text(json.dumps(payload,indent=2)); print(json.dumps(payload,indent=2))

if __name__=="__main__": main()
