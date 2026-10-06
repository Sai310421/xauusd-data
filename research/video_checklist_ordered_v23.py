# v23 research wrapper: ordered/latching checklist audit over v22.
# This file is backtest-only. It does not connect to a broker or place live orders.
import importlib.util, json, shutil
from pathlib import Path

P=Path(__file__).with_name("video_checklist_core_v22.py")
spec=importlib.util.spec_from_file_location("v22",P)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

# Video review shows a sequence, not a simultaneous all-green AND.
# Remove v22's arbitrary 60m window; retain the pre-existing M5 setup lifetime.
m.FRAMEWORK_ACTIVE_NS=m.TIMEOUT_NS
C=m.M1LinePOIPlusM5G75OverlapV19
orig_after=C._after_close
orig_open=C._m1_open
orig_result=C.result

def after_close_v23(self,tf,ts):
    n=len(self.m5_mss_events)
    orig_after(self,tf,ts)
    for ev in self.m5_mss_events[n:]:
        match=None
        for x in reversed(self.setups[tf]):
            if x.pending_ns==ev["ns"] and x.direction==ev["direction"]:
                match=x; break
        if match is None: continue
        ev["liquidity_sweep_ns"]=int(match.signal_ns)
        ev["htf_pda_delivery_proxy_ns"]=int(match.touch_ns)
        ev["mss_confirmed_ns"]=int(ev["ns"])
        ev["gate_order_valid"]=bool(
            match.touched and match.signal_ns<=match.touch_ns<=ev["ns"]
        )
        ev["expires_ns"]=int(match.signal_ns+m.TIMEOUT_NS)

def open_v23(self,s,bid,ask,ts):
    latest=next((e for e in reversed(self.m5_mss_events) if e["ns"]<=ts),None)
    if latest is not None:
        if not latest.get("gate_order_valid",False):
            self.framework_stats["reject_bad_gate_order"]=self.framework_stats.get("reject_bad_gate_order",0)+1
            return False
        if not (latest["liquidity_sweep_ns"]<=latest["htf_pda_delivery_proxy_ns"]<=latest["mss_confirmed_ns"]<=ts):
            self.framework_stats["reject_bad_gate_order"]=self.framework_stats.get("reject_bad_gate_order",0)+1
            return False
    ok=orig_open(self,s,bid,ask,ts)
    if ok and latest is not None:
        latest["entry_model_ns"]=int(ts)
        latest["entry_model"]=s.pattern
        latest["clear_target_checked_ns"]=int(ts)
        latest["armed"]=True
    return ok

def result_v23(self):
    r=orig_result(self)
    r.pop("framework_core_v1",None)
    r["framework_ordered_v23"]={
        "setup_lifetime_minutes":240,
        "latched_order":[
            "1 Liquidity Sweep",
            "2 HTF PDA Delivery proxy",
            "3 MSS Confirmed",
            "4 same-direction M1 Entry Model",
            "5 Clear Target >= 1R"
        ],
        "important":"Gates are latched in order. A row turning red later does not erase an earlier satisfied gate.",
        "proxy_disclosure":"Exact HTF PDA Delivery formula is not visible in the supplied videos. The current causal 0.804 deep-revisit is explicitly a proxy, not claimed exact parity.",
        "not_hard_gated_yet":["Macro Window","Volume Influx","IFVG","EQ Rebalance","CISD preset"],
        "stats":self.framework_stats,
        "ordered_events":self.m5_mss_events
    }
    return r

C._after_close=after_close_v23
C._m1_open=open_v23
C.result=result_v23

if __name__=="__main__":
    m.main()
    # v22 main writes its evidence first; relabel/copy it as v23 evidence.
    import sys
    exp=sys.argv[sys.argv.index("--experiment-id")+1]
    src=Path("results/video-checklist-core-v22")/exp
    dst=Path("results/video-checklist-ordered-v23")/exp
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(src,dst)
    p=dst/"result.json"
    r=json.loads(p.read_text())
    r["verification_level"]="NAUTILUS_BT_VIDEO_CHECKLIST_ORDERED_V23"
    r["note"]="v23 enforces a latched ordered MSS checklist. M5 parent/G75 are unchanged. M1 is entry refinement only. Exact HTF PDA formula remains an explicit proxy; Macro/Volume/IFVG/EQ/CISD are not silently invented."
    p.write_text(json.dumps(r,indent=2,default=str))

# workflow trigger
