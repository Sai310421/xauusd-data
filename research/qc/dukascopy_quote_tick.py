from __future__ import annotations

from datetime import datetime, timedelta

from AlgorithmImports import *


class DukascopyQuoteTick(PythonData):
    """Custom data reader for Lean-local / Object Store exports.

    Expected CSV line:
    ms_since_midnight,bid,ask,bid_size,ask_size,exchange,condition,suspicious
    """

    def get_source(self, config, date, is_live_mode):
        symbol = config.symbol.value.lower()
        stamp = date.strftime("%Y%m%d")
        if is_live_mode:
            return SubscriptionDataSource("", SubscriptionTransportMedium.STREAMING)
        path = (
            f"{Globals.data_folder}/cfd/dukascopy/tick/{symbol}/{stamp}_quote.zip"
            if symbol == "xauusd"
            else f"{Globals.data_folder}/forex/dukascopy/tick/{symbol}/{stamp}_quote.zip"
        )
        return SubscriptionDataSource(path, SubscriptionTransportMedium.LOCAL_FILE)

    def reader(self, config, line, date, is_live_mode):
        if not line or line.startswith("#") or line.startswith("ms"):
            return None
        parts = line.split(",")
        if len(parts) < 5:
            return None
        try:
            ms = int(float(parts[0]))
            bid = float(parts[1])
            ask = float(parts[2])
        except ValueError:
            return None
        if ask <= 0 or bid <= 0 or ask < bid:
            return None
        row = DukascopyQuoteTick()
        row.symbol = config.symbol
        row.time = datetime(date.year, date.month, date.day) + timedelta(milliseconds=ms)
        row["bid"] = bid
        row["ask"] = ask
        row.value = (bid + ask) / 2.0
        return row
