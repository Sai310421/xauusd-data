# MSS v1.5 Cloud-Only Verification Contract

Status: ACTIVE / fail-closed

## Goal
Finish MSS_MultiStrat_v1_5 research, backtest, robustness and KPI evidence without any user PC or self-hosted runner.

## Authoritative path
GitHub-hosted Ubuntu -> pinned NautilusTrader -> raw Bid/Ask QuoteTick catalog -> MSS v1.5 Python/Nautilus parity port -> WR1-WR5 -> WFO/Monte Carlo/parameter stability -> artifact.

## Source identity
Upstream MQ5: Sai310421/my-ea-factory-ICT @ research/mss-v15-strategy-ablation-v1
Path: ea/src/MSS_MultiStrat_v1_5/MSS_MultiStrat_v1_5.mq5
User-file SHA-256: 7fb3065e4a9a3110c4aaaa594705893c3f96b47bf8df3b441bdcab396c0caa2d
Strategies: S01-S15.
Core risk controls: confluence tiers, split entries, anti-Martingale, DD Lv1/Lv2/Lv3/Kill, daily DD, spread/session gates, hedge controls.

## Canonical data
RAW Bid/Ask only. OHLC proxy results may be used only for historical parity and must be labeled PROXY_BT.
Default first canonical cell: XAUUSD, 91 days from 2026-02-25, NautilusTrader 1.230.0.
The existing amos-complete raw catalog builder/cache is the canonical cloud data source.

## Fail-closed rules
No silent OHLC fallback.
No fabricated broker costs.
No PASS when required WR5 fields are absent.
No CORE/BOOST promotion from MT5 historical proxy results.
No dependency on C:\ paths, local MT5, self-hosted runners, or a powered-on user PC.

## WR5 required evidence
raw spread; round-trip commission; slippage; latency/delay; swap where relevant; cashback assumption; mark-to-market equity; floating DD; open-position state; aggregate exposure; margin level; basket age; unresolved inventory; event/price-pitch budget.

Missing any required field => WR5 INVALID.

## Promotion
Compare under identical data/cost/sizing/execution assumptions.
CORE target: about 6 quality entries/day, WR >=80%, PF >=2.5, MaxDD <=4%, materially higher monthly return.
Final acceptance: monthly trading return >=40%, cashback >=10%, combined >=50%, MaxDD <=15% (target <=10%), PF >=1.8, estimated ruin <1%, rescue <=5%, ZR dependency 2-5%, BE-or-better exits >=90%.
