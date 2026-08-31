from __future__ import annotations

"""Raw Bid/Ask catalog builder v3.

Preserves Dukascopy fractional top-of-book volume instead of rounding it to
integer size. Price/timestamps remain identical to the canonical builder.
"""

from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Currency, Price, Quantity

import research.build_raw_bidask_catalog_duka as base


def make_instrument_v3(symbol: str, meta: dict) -> CurrencyPair:
    base_ccy, quote_ccy = meta["pair"].split("/")
    precision = int(meta["price_precision"])
    price_increment = "0." + ("0" * (precision - 1)) + "1"
    return CurrencyPair(
        instrument_id=InstrumentId(Symbol(symbol), base.SIM),
        raw_symbol=Symbol(symbol),
        base_currency=Currency.from_str(base_ccy),
        quote_currency=Currency.from_str(quote_ccy),
        price_precision=precision,
        size_precision=6,
        price_increment=Price.from_str(price_increment),
        size_increment=Quantity.from_str("0.000001"),
        ts_event=0,
        ts_init=0,
    )


base.make_instrument = make_instrument_v3


if __name__ == "__main__":
    base.main()
