from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Emit standardized AMOS Arena result JSON")
    p.add_argument("--out", required=True)
    p.add_argument("--candidate-id", required=True)
    p.add_argument("--agent", required=True)
    p.add_argument("--strategy", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--initial-equity", type=float, default=1000.0)
    p.add_argument("--final-equity", type=float, required=True)
    p.add_argument("--max-dd-pct", type=float, required=True)
    p.add_argument("--profit-factor", type=float, required=True)
    p.add_argument("--win-rate-pct", type=float, default=0.0)
    p.add_argument("--trades", type=int, required=True)
    p.add_argument("--ruin-probability", type=float, default=0.0)
    p.add_argument("--oos-retention", type=float, default=0.0)
    p.add_argument("--cost-retention", type=float, default=0.0)
    p.add_argument("--regime-pass-rate", type=float, default=0.0)
    p.add_argument("--discovery-efficiency", type=float, default=0.0)
    p.add_argument("--verification-level", default="BACKTEST")
    args = p.parse_args()

    initial = args.initial_equity
    result = {
        "schema_version": "amos.arena.v1",
        "candidate_id": args.candidate_id,
        "agent": args.agent,
        "strategy": args.strategy,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "initial_equity": initial,
        "final_equity": args.final_equity,
        "return_pct": ((args.final_equity - initial) / initial * 100.0) if initial else 0.0,
        "max_dd_pct": args.max_dd_pct,
        "profit_factor": args.profit_factor,
        "win_rate_pct": args.win_rate_pct,
        "trades": args.trades,
        "ruin_probability": args.ruin_probability,
        "oos_retention": args.oos_retention,
        "cost_retention": args.cost_retention,
        "regime_pass_rate": args.regime_pass_rate,
        "discovery_efficiency": args.discovery_efficiency,
        "verification_level": args.verification_level,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
