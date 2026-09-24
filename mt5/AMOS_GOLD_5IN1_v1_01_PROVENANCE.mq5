#property strict
#property version   "1.01"
#property description "AMOS GOLD 5IN1 - clean-room unified XAUUSD multi-engine EA"
#include <Trade/Trade.mqh>
CTrade trade;
// Canonical source is mirrored from the ChatGPT build artifact AMOS_GOLD_5IN1_v1_01.mq5.
// Five engines: GrandLine / AITrader proxy / FlashScalper / LimitBreak / NineLayer.
// Full source artifact SHA256 is tracked by the BT harness; this repo copy is intentionally a provenance marker.
