"""Native Nautilus matching smoke test for A01/B01/C01, independent per strategy.
Raw quote prices/timestamps preserved; zero L1 sizes replaced for matching only.
Fails closed if no actual Nautilus fills. No live broker connection.
"""
import argparse
import json
from collections import Counter, deque
from decimal import Decimal
from pathlib import Path

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, BookType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

from amos_abc_nautilus_rawtick_bt import Candle, signals


class NativeConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    edge: str
    quantity: Decimal = Decimal("1")


class NativeEdge(Strategy):
    def __init__(self, config):
        super().__init__(config)
        self.bars = deque(maxlen=64)
        self.bucket = None
        self.cur = None
        self.age = 0
        self.pending = False
        self.active_side = 0
        self.events = Counter()
        self.submitted = 0

    def on_start(self):
        self.subscribe_quote_ticks(self.config.instrument_id)

    def on_order_filled(self, event):
        self.events["filled"] += 1
        self.pending = False

    def on_order_rejected(self, event):
        self.events["rejected"] += 1
        self.pending = False

    def on_order_denied(self, event):
        self.events["denied"] += 1
        self.pending = False

    def _order(self, side):
        instrument = self.cache.instrument(self.config.instrument_id)
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY if side > 0 else OrderSide.SELL,
            quantity=instrument.make_qty(self.config.quantity),
        )
        self.pending = True
        self.submitted += 1
        self.submit_order(order)

    def on_quote_tick(self, tick):
        bid = float(tick.bid_price)
        bucket = int(tick.ts_event) // 60_000_000_000
        if self.bucket is not None and bucket != self.bucket:
            self.bars.append(self.cur)
            if self.active_side and not self.pending:
                self.age += 1
                if self.age >= 5:
                    self._order(-self.active_side)
                    self.active_side = 0
            if not self.active_side and not self.pending:
                choices = [side for name, side in signals(self.bars) if name == self.config.edge]
                if choices:
                    self._order(choices[0])
                    self.active_side = choices[0]
                    self.age = 0
            self.cur = None
        self.bucket = bucket
        if self.cur is None:
            self.cur = Candle(bid, bid, bid, bid)
        else:
            self.cur.high = max(self.cur.high, bid)
            self.cur.low = min(self.cur.low, bid)
            self.cur.close = bid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--out", default="results/amos-abc/native-matching.json")
    args = ap.parse_args()
    catalog = ParquetDataCatalog(args.catalog)
    instrument = next(i for i in catalog.instruments() if i.id.symbol.value.replace("/", "") == "XAUUSD")
    raw = catalog.query(data_cls=QuoteTick, identifiers=[instrument.id.value])
    if len(raw) < 1000:
        raise SystemExit("Insufficient genuine raw QuoteTicks")
    corrected = 0
    ticks = []
    for t in raw:
        if float(t.bid_size) <= 0 or float(t.ask_size) <= 0:
            corrected += 1
            t = QuoteTick(
                instrument_id=t.instrument_id, bid_price=t.bid_price,
                ask_price=t.ask_price,
                bid_size=Quantity.from_int(1) if float(t.bid_size) <= 0 else t.bid_size,
                ask_size=Quantity.from_int(1) if float(t.ask_size) <= 0 else t.ask_size,
                ts_event=t.ts_event, ts_init=t.ts_init,
            )
        ticks.append(t)
    report = {
        "verification": "NAUTILUS_NATIVE_MATCHING_RAW_QUOTETICK",
        "raw_ticks": len(raw),
        "L1_zero_size_replacements": corrected,
        "limitation": "Raw Bid/Ask prices and timestamps unchanged; missing quote sizes replaced with 1 for L1 matching. Broker slippage/latency not calibrated.",
        "edges": {},
    }
    for edge in ("A01", "B01", "C01"):
        engine = BacktestEngine(config=BacktestEngineConfig(
            logging=LoggingConfig(log_level="ERROR"),
            risk_engine=RiskEngineConfig(bypass=True),
        ))
        engine.add_venue(
            venue=instrument.id.venue, oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN, book_type=BookType.L1_MBP,
            base_currency=USD, starting_balances=[Money(1000, USD)],
            default_leverage=Decimal("2000"),
        )
        engine.add_instrument(instrument)
        engine.add_data(ticks)
        strategy = NativeEdge(NativeConfig(instrument_id=instrument.id, edge=edge))
        engine.add_strategy(strategy)
        engine.run()
        fills = engine.trader.generate_order_fills_report()
        orders = engine.trader.generate_orders_report()
        count = 0 if fills is None else len(fills)
        report["edges"][edge] = {
            "submitted": strategy.submitted,
            "native_fills": count,
            "native_orders": 0 if orders is None else len(orders),
            "events": dict(strategy.events),
        }
        engine.dispose()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if any(e["native_fills"] == 0 for e in report["edges"].values()):
        raise SystemExit("Native fill gate failed for one or more ABC edges")


if __name__ == "__main__":
    main()
