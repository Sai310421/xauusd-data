from __future__ import annotations

from nautilus_trader.model.enums import BookType

import research.amos_allweather_raw_bidask_bt_compat as compat


_ORIG_ADD_VENUE = compat.base.BacktestEngine.add_venue


def _add_venue_l1(self, *args, **kwargs):
    # Raw Bid/Ask QuoteTick execution requires an L1 market to exist on the
    # simulated venue. Without this Nautilus rejects every market order with
    # `no market for XAUUSD.SIM` even though QuoteTicks are present.
    kwargs.setdefault("book_type", BookType.L1_MBP)
    return _ORIG_ADD_VENUE(self, *args, **kwargs)


compat.base.BacktestEngine.add_venue = _add_venue_l1
print("SIM_MARKET_BOOK_TYPE=L1_MBP")


if __name__ == "__main__":
    compat.main()
