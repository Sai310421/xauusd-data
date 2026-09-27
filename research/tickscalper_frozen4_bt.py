from __future__ import annotations
import argparse, json, math
from decimal import Decimal
from pathlib import Path

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType, BookType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from research.tickscalper_nautilus_raw_bt import Cfg, TickScalperCandidate, fix_sizes

class FrozenFourPartStrategy(TickScalperCandidate):
    """
    Four-part AE integration:
      1) v1 analytical boundary / Natural Recovery
      2) v2 robust feasibility / tail screen
      3) v3 tail-aware short-horizon score
      4) Frozen Integration Charter fallback policy
    The validated FAST rescue proposal is preserved as the candidate action.
    """

    def __init__(self, c):
        super().__init__(c)
        self.frozen_started = 0
        self.frozen_completed = 0
        self.frozen_hard_closed = 0
        self.frozen_denied = 0
        self.frozen_reduced = 0
        self.frozen_paths = {"v3": 0, "v2": 0, "v1": 0, "wait": 0, "safe": 0, "nonfast": 0}
        self.frozen_last = {}

    def _four_part_fraction(self):
        cls = self._eligibility_class()
        proposal = float(self.config.eligibility_fast_frac)
        if cls != "FAST":
            self.frozen_paths["nonfast"] += 1
            self.frozen_last = {"class": cls, "decision": "DENY_NONFAST", "fraction": 0.0}
            return 0.0

        D, Dstar, mu, sigma, gross = self._ae_v1_boundary()
        pnr = self._ae_pnr(D, mu, sigma, gross)
        cdar = self._cdar_estimate()
        edar = self._edar_estimate()
        regime = self._rescue_regime()
        fdd = self._floating_dd_pct()

        # v1: boundary + Natural Recovery.
        v1_allow = (D >= Dstar and pnr < 0.95 and fdd < self.config.ae_hard_fdd)

        # v2: robustize recovery probability and enforce tail/shock feasibility.
        uncertainty = min(0.20, 0.04 + sigma / max(abs(mu) + sigma, 1e-9) * 0.10)
        pnr_rob = max(0.0, pnr - uncertainty)
        tail_prob = 1.0 - pnr_rob
        tail_max = float(self.config.fp_block_threshold)
        v2_allow = v1_allow and regime != "SHOCK" and tail_prob <= tail_max and fdd < self.config.ae_hard_fdd

        # v3: executable short-horizon distributional approximation.
        benefit = pnr_rob * D * proposal
        tail_cost = tail_prob * (D + gross * self.config.ae_tail_distance) * proposal
        risk_cost = (0.01 * cdar + 0.005 * edar) * proposal
        score = benefit - tail_cost - risk_cost
        v3_allow = v2_allow and score >= 0.0

        frac = 0.0
        decision = "SAFE"
        fallback = "safe"
        if v3_allow:
            frac = proposal
            decision = "ALLOW_V3"
            fallback = "v3"
            self.frozen_paths["v3"] += 1
        elif v2_allow:
            # Charter fallback v3 -> v2.
            frac = proposal
            decision = "ALLOW_V2_FALLBACK"
            fallback = "v2"
            self.frozen_paths["v2"] += 1
        elif v1_allow and tail_prob < 0.95 and regime != "SHOCK":
            # Charter fallback v2 -> v1 with minimum intervention.
            frac = min(proposal, 0.03)
            decision = "REDUCE_V1_FALLBACK"
            fallback = "v1"
            self.frozen_reduced += 1
            self.frozen_paths["v1"] += 1
        elif pnr >= 0.95 and fdd < self.config.ae_hard_fdd:
            decision = "WAIT_NATURAL_RECOVERY"
            fallback = "wait"
            self.frozen_paths["wait"] += 1
        else:
            self.frozen_paths["safe"] += 1

        self.frozen_last = {
            "class": cls, "D": D, "Dstar": Dstar, "deltaR": max(0.0, D-Dstar),
            "p_NR": pnr, "p_NR_robust": pnr_rob, "tail_prob": tail_prob,
            "cdar": cdar, "edar": edar, "regime": regime, "fdd": fdd,
            "v3_score": score, "proposal": proposal, "decision": decision,
            "fraction": frac, "fallback": fallback,
        }
        return frac

    def _start_eligibility_rescue(self):
        if self.config.eligibility_mode != "conservative" or self.rescue_active or not self.entries:
            return False
        frac = self._four_part_fraction()
        if frac <= 0:
            self.frozen_denied += 1
            self.eligibility_denied += 1
            return False
        gross = sum(l for _, l in self.entries)
        lot = max(0.01, math.floor((gross * frac + 1e-12) / 0.01) * 0.01)
        rs = self.side
        px = self.ask if rs > 0 else self.bid
        self._submit(rs, lot)
        self.rescue_legs = [(rs, px, lot)]
        self.rescue_active = True
        self.rescue_turns = 1
        self.frozen_started += 1
        self.eligibility_started += 1
        self.rescue_started_count += 1
        self.max_lots = max(self.max_lots, gross + lot)
        return True

    def _handle_eligibility_rescue(self):
        if not self.rescue_active or self.config.eligibility_mode != "conservative":
            return False
        if self._zr_combined_pnl() >= 0.0:
            self.frozen_completed += 1
            self.eligibility_completed += 1
            self.rescue_completed_count += 1
            self._close("FROZEN4_COMPLETE")
            return True
        if self._floating_dd_pct() >= self.config.ae_hard_fdd:
            self.frozen_hard_closed += 1
            self.eligibility_hard_closed += 1
            self.rescue_hard_close_count += 1
            self._close("FROZEN4_HARD_DD")
            return True
        return True

    def summary(self):
        out = super().summary()
        out.update({
            "frozen_started": self.frozen_started,
            "frozen_completed": self.frozen_completed,
            "frozen_hard_closed": self.frozen_hard_closed,
            "frozen_denied": self.frozen_denied,
            "frozen_reduced": self.frozen_reduced,
            "frozen_paths": self.frozen_paths,
            "frozen_last": self.frozen_last,
        })
        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--fast-frac", type=float, default=0.06)
    ap.add_argument("--tail-max", type=float, default=0.85)
    ap.add_argument("--ae-k", type=float, default=0.20)
    ap.add_argument("--ae-rho", type=float, default=0.08)
    ap.add_argument("--ae-hard-fdd", type=float, default=45.0)
    ap.add_argument("--raw-bidask-only", action="store_true")
    a = ap.parse_args()
    if not a.raw_bidask_only:
        raise SystemExit("raw-bidask-only mandatory")

    cat = ParquetDataCatalog(a.catalog)
    inst = next((x for x in cat.instruments() if x.id.symbol.value.replace("/", "") == "XAUUSD"), None)
    if inst is None:
        raise SystemExit("XAUUSD missing")
    ticks = fix_sizes(cat.query_quote_ticks(identifiers=[inst.id.value]))

    eng = BacktestEngine(config=BacktestEngineConfig(
        logging=LoggingConfig(log_level="ERROR"),
        risk_engine=RiskEngineConfig(bypass=True),
    ))
    eng.add_venue(
        venue=inst.id.venue, oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
        book_type=BookType.L1_MBP, base_currency=USD,
        starting_balances=[Money(1000, USD)], default_leverage=Decimal("2000"),
    )
    eng.add_instrument(inst)
    eng.add_data(ticks)

    cfg = Cfg(
        instrument_id=inst.id,
        base_qty=Decimal("0.30"),
        direction_sign=1,
        session_start_hour=7,
        session_end_hour=17,
        basket_offset=0.8575,
        max_layers=10,
        entry_mode="ticksmoother",
        ts_ticks_per_bar=5,
        ts_fast=3, ts_slow=5, ts_conf1=8, ts_conf2=13,
        ts_cross_only=True,
        risk_mode="none",
        rescue_mode="none",
        math_rescue_mode="none",
        eligibility_mode="conservative",
        eligibility_fast_frac=a.fast_frac,
        eligibility_deep_frac=0.0,
        eligibility_hard_fdd=a.ae_hard_fdd,
        ae_stack_mode="none",
        ae_k=a.ae_k,
        ae_rho=a.ae_rho,
        ae_lambda=1.0,
        ae_tail_distance=8.0,
        ae_rescue_cap=a.fast_frac,
        ae_hard_fdd=a.ae_hard_fdd,
        fp_block_threshold=a.tail_max,
    )
    st = FrozenFourPartStrategy(cfg)
    eng.add_strategy(st)
    eng.run()
    fills = eng.trader.generate_order_fills_report()
    out = {
        "verification_level": "NAUTILUS_RAW_BIDASK_FROZEN4_CANDIDATE",
        "raw_ticks": len(ticks),
        "native_fills": len(fills) if fills is not None else 0,
        "ohlc_resample_used": False,
        "architecture": "FAST_PROPOSAL -> v1 -> v2 -> v3 -> Frozen Charter fallback",
        "fast_frac": a.fast_frac,
        "tail_max": a.tail_max,
        "ae_k": a.ae_k,
        "ae_rho": a.ae_rho,
        "ae_hard_fdd": a.ae_hard_fdd,
        **st.summary(),
    }
    outdir = Path("results/tickscalper-nautilus") / a.experiment_id
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "kpi.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    eng.dispose()

if __name__ == "__main__":
    main()
