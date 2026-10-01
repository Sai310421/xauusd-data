"""Expected Dukascopy trading hours for supported symbols.

This models regular sessions only. An exceptional holiday closure remains
unresolved until independently documented; it is never called COMPLETE.
"""
import datetime as dt
from zoneinfo import ZoneInfo

SCHEDULE_SOURCE = 'https://www.dukascopy.com/swiss/english/forex/forex-trading-accounts/link/'


def expected_trading_hour(symbol: str, hour_utc: dt.datetime) -> bool:
    if hour_utc.tzinfo is None or hour_utc.utcoffset() != dt.timedelta(0):
        raise ValueError('UTC-aware hour required')
    local = hour_utc.astimezone(ZoneInfo('America/New_York'))
    weekday = local.weekday()  # Monday=0, Sunday=6.
    if weekday == 5 or (weekday == 6 and local.hour < 17) or (weekday == 4 and local.hour >= 17):
        return False
    if symbol.upper() == 'XAUUSD' and local.hour == 17:
        return False
    return True
