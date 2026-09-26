# Preliminary State / Function Map

| Layer | Evidence-backed elements | Status |
|---|---|---|
| Tick preprocessing | TickSmoother v2.1, tick bars/state arrays | SOURCE-CONFIRMED |
| Fast direction | Fast tick MA | SOURCE-CONFIRMED |
| Slow confirmation | Slow tick MA + confirm MAs | SOURCE-CONFIRMED |
| MACD confirmation | averages MACD in some MC modes | SOURCE-CONFIRMED |
| Spread | Spreadometer present in official packages | SOURCE-CONFIRMED dependency; exact trading use pending source body |
| Timing | Timing parameter group; later expanded time filter | SOURCE-CONFIRMED |
| MM | dedicated MM parameter group | SOURCE-CONFIRMED |
| Recovery | Martingale ExitMode 11/12; later Salvage | SOURCE-CONFIRMED |
| Basket target | ProfitTarget / Total Pips / Total Profit modes | SOURCE-CONFIRMED |
| Equity protection | EquityTrailing in later versions | SOURCE-CONFIRMED |
| Max-order protection | MaxOrders close safety in later versions | SOURCE-CONFIRMED |
| reofx ER/L1/L2/SideLock | 2026 public spec only | REOFX-ONLY until lineage comparison |
