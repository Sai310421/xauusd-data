from __future__ import annotations
# GitHub Actions parity smoke trigger
import argparse, json
from decimal import Decimal
from pathlib import Path

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import BarType, Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from research.goldebrave_fasttf_parity_raw import Cfg, l1
from research.goldebrave_v8c_split_g75_cvar_reflect_raw import CVarReflect

if not hasattr(ParquetDataCatalog, "query_quote_ticks"):
    def _q(self, identifiers=None, start=None, end=None):
        return self.query(data_cls=QuoteTick, identifiers=identifiers, start=start, end=end)
    ParquetDataCatalog.query_quote_ticks = _q

# Frozen from uploaded GoldeBrave_V8C_G75_DualMode_Standalone.mq5 v1.10.
MQ5 = {
    "parent_scale": 0.70,
    "max_parent_entries_day": 3,
    "max_spread_price": 2.50,
    "g75_trigger": 0.12,
    "g75_add": 0.025,
    "g75_max_layers": 10,
    "strong_layer_start": 7,
    "strong_abs_move": 0.10,
    "strong_vol_mult": 0.15,
    "strong_spread_max": 1.00,
    "strong_boost": 3.25,
    "child_partial_frac": 0.35,
    "child_trail_r": 0.15,
    "safe_boundary_pct": 5.00,
    "cvar_alpha": 0.99,
    "cvar_vol_floor": 0.60,
    "cvar_horizon": 16.0,
    "cvar_reserve": 1.15,
    "cvar_window": 4000,
    "cvar_min_samples": 128,
    "auto_enter_safe_dd": 3.50,
    "auto_exit_safe_dd": 2.25,
}

class AutoStandalone(CVarReflect):
    def __init__(self, c):
        super().__init__(c, "cv3")
        self.auto_safe = False
        self.switch_to_safe = 0
        self.switch_to_high = 0
        self.safe_ticks = 0
        self.high_ticks = 0

    def _risk_control(self, bid, ask, dd):
        if not self.auto_safe and dd >= MQ5["auto_enter_safe_dd"]:
            self.auto_safe = True
            self.switch_to_safe += 1
        elif self.auto_safe and dd <= MQ5["auto_exit_safe_dd"]:
            self.auto_safe = False
            self.switch_to_high += 1
        if self.auto_safe:
            self.safe_ticks += 1
            return super()._risk_control(bid, ask, dd)
        self.high_ticks += 1

    def summary_auto(self):
        s = self.summary_cvar()
        s.update({
            "mode": "AUTO",
            "switch_to_safe": self.switch_to_safe,
            "switch_to_high": self.switch_to_high,
            "safe_ticks": self.safe_ticks,
            "high_ticks": self.high_ticks,
        })
        return s

def cfg(inst):
    s = inst.id.value
    return Cfg(
        instrument_id=inst.id,
        m1=BarType.from_str(f"{s}-1-MINUTE-BID-INTERNAL"),
        m5=BarType.from_str(f"{s}-5-MINUTE-BID-INTERNAL"),
        m15=BarType.from_str(f"{s}-15-MINUTE-BID-INTERNAL"),
        h1=BarType.from_str(f"{s}-1-HOUR-BID-INTERNAL"),
        mode="m15",
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--mode", choices=["HIGH", "SAFE", "AUTO"], required=True)
    ap.add_argument("--experiment-id", required=True)
    a = ap.parse_args()

    cp = Path(a.catalog)
    manifest = json.loads((cp / "catalog_manifest.json").read_text()) if (cp / "catalog_manifest.json").exists() else {}
    cat = ParquetDataCatalog(str(cp))
    inst = next(x for x in cat.instruments() if x.id.symbol.value.replace("/", "") == "XAUUSD")
    raw = cat.query_quote_ticks(identifiers=[inst.id.value])

    eng = BacktestEngine(config=BacktestEngineConfig(
        logging=LoggingConfig(log_level="ERROR"),
        risk_engine=RiskEngineConfig(bypass=True),
    ))
    eng.add_venue(
        venue=inst.id.venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        book_type=BookType.L1_MBP,
        base_currency=USD,
        starting_balances=[Money(1000, USD)],
        default_leverage=Decimal("2000"),
    )
    eng.add_instrument(inst)
    eng.add_data(l1(raw))

    if a.mode == "HIGH":
        st = CVarReflect(cfg(inst), "off")
    elif a.mode == "SAFE":
        st = CVarReflect(cfg(inst), "cv3")
    else:
        st = AutoStandalone(cfg(inst))

    eng.add_strategy(st)
    eng.run()
    eng.end()

    summary = st.summary_auto() if a.mode == "AUTO" else st.summary_cvar()
    out = {
        "verification": "GOLDEBRAVE_STANDALONE_MQ5_V110_LOGIC_PARITY_RAW_MTM",
        "source_bot": "GoldeBrave_V8C_G75_DualMode_Standalone.mq5 v1.10",
        "mode": a.mode,
        "engine": "NautilusTrader BacktestEngine",
        "starting_balance_usd": 1000,
        "leverage": 2000,
        "raw_ticks": len(raw),
        "ohlc_resample_used": False,
        "catalog_manifest": manifest,
        "mq5_frozen_inputs": MQ5,
        "important_parity_notes": [
            "Stage-1 BOT logic parity run on Raw Bid/Ask.",
            "HIGH maps to tested V8-C FULL/off control; SAFE maps to tested cv3 reflected-CVaR; AUTO uses MQ5 3.50/2.25 DD hysteresis.",
            "The uploaded MQ5 additionally pre-compresses new SAFE child lots near the 5% boundary; the research cv3 reference does not. This run intentionally preserves the validated cv3 sizing semantics rather than silently claiming exact native MT5 parity.",
            "This stage is still Raw Bid/Ask MTM sizing semantics, not final native broker order/fill parity. Native order/fill conversion remains the next gate.",
        ],
        **summary,
    }
    p = Path("results/goldebrave-standalone") / a.experiment_id / f"{a.mode}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    eng.dispose()

if __name__ == "__main__":
    main()
