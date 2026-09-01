from .models import FVGZone, IFVGState, BPRZone, ZoneLifecycle

def activate_ifvg(source: FVGZone, ts_ns: int) -> IFVGState:
    if source.lifecycle != ZoneLifecycle.BROKEN or source.break_direction == 0:
        return IFVGState(source_fvg=source, active=False)
    return IFVGState(source_fvg=source,active=True,direction=-source.direction,activated_ts_ns=ts_ns)

def make_bpr(bullish: FVGZone, bearish: FVGZone, ts_ns: int) -> BPRZone | None:
    if bullish.direction != 1 or bearish.direction != -1:
        return None
    if bullish.lifecycle == ZoneLifecycle.INVALIDATED or bearish.lifecycle == ZoneLifecycle.INVALIDATED:
        return None
    lo=max(bullish.lower,bearish.lower)
    hi=min(bullish.upper,bearish.upper)
    if hi <= lo:
        return None
    return BPRZone(bullish_fvg=bullish,bearish_fvg=bearish,lower=lo,upper=hi,created_ts_ns=ts_ns,lifecycle=ZoneLifecycle.ACTIVE)
