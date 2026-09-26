# TickScalper Original Lineage Reconstruction

Mainline reconstruction branch.

## Evidence order
1. Igor Durkin / igorad original line (v3.42)
2. v3.44 / v3.44.7
3. v3.45.x / v3.45.7a_nmc
4. v4_nmc fixed line (2025)
5. reofx526-eng 2026 implementation as downstream comparison target

## Required source set
- tickscalper_v4_nmc.mq4
- tickScalper_moemoresafe.set
- spreadometer_v1.mq4
- ticksmoother_v2.1.mq4
- tickscalper_v3.45.7a_nmc.mq4
- tickscalper_v3.44_newopen_nmc.mq4
- tickscalper_v3.44.mq4 / v3.42 where obtainable

## Reconstruction order
TickSmoother parity -> Entry -> Timing/Spread -> Add/Sizing -> Exit/Basket -> SL/Protection -> Recovery -> reofx delta.

## Gates
- Original source is read-only.
- No OHLC-only acceptance for tick mode.
- Raw bid/ask tick is primary parity gate.
- Recovery/sizing EDGE is measured separately from Direction/Timing EDGE.
- No release artifact until statement/runtime parity is materially achieved.

## Public source anchors
- https://www.mql5.com/en/forum/179720
- https://www.mql5.com/en/forum/179720/page47
- https://www.mql5.com/en/forum/179720/page64
- https://www.mql5.com/en/forum/179720/page72
- https://www.mql5.com/en/forum/179720/page78
- https://github.com/reofx526-eng/-TickScalper
