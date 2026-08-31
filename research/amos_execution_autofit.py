from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from statistics import median


@dataclass(frozen=True)
class AutoFitDecision:
    market_ready: bool
    spread_ok: bool
    spread: float
    spread_median: float
    spread_limit: float
    volatility: float
    size_multiplier: float
    tf_weight: float
    execution_quality: float
    reason: str


class ExecutionAutoFit:
    """Causal execution governor for Raw Bid/Ask backtests.

    Uses only information available up to the current quote. It does not optimize
    against future PnL, so the same component can be used in walk-forward/live mode.
    """

    def __init__(
        self,
        tf_minutes: int,
        spread_window: int = 2048,
        return_window: int = 2048,
        warmup_quotes: int = 128,
        spread_mult: float = 3.0,
        min_size_multiplier: float = 0.20,
    ) -> None:
        self.tf_minutes = int(tf_minutes)
        self.spreads = deque(maxlen=int(spread_window))
        self.abs_returns = deque(maxlen=int(return_window))
        self.warmup_quotes = int(warmup_quotes)
        self.spread_mult = float(spread_mult)
        self.min_size_multiplier = float(min_size_multiplier)
        self.last_mid = None
        self.quotes_seen = 0

    @staticmethod
    def _f(value) -> float:
        try:
            return float(value)
        except Exception:
            return float('nan')

    @staticmethod
    def _quantile(values, q: float) -> float:
        if not values:
            return 0.0
        xs = sorted(values)
        if len(xs) == 1:
            return float(xs[0])
        pos = max(0.0, min(1.0, q)) * (len(xs) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        w = pos - lo
        return float(xs[lo] * (1.0 - w) + xs[hi] * w)

    def observe(self, bid, ask, bid_size=None, ask_size=None) -> AutoFitDecision:
        bid_f = self._f(bid)
        ask_f = self._f(ask)
        bid_sz = self._f(bid_size) if bid_size is not None else 1.0
        ask_sz = self._f(ask_size) if ask_size is not None else 1.0
        self.quotes_seen += 1

        basic_ready = (
            isfinite(bid_f)
            and isfinite(ask_f)
            and ask_f > bid_f > 0.0
            and isfinite(bid_sz)
            and isfinite(ask_sz)
            and bid_sz > 0.0
            and ask_sz > 0.0
        )
        if not basic_ready:
            return AutoFitDecision(False, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'invalid_quote')

        mid = (bid_f + ask_f) / 2.0
        spread = ask_f - bid_f
        if self.last_mid is not None and self.last_mid > 0.0:
            self.abs_returns.append(abs(mid - self.last_mid))
        self.last_mid = mid
        self.spreads.append(spread)

        spread_med = median(self.spreads) if self.spreads else spread
        spread_p90 = self._quantile(self.spreads, 0.90)
        # The p90 floor prevents the gate from becoming unrealistically tight in
        # quiet periods; the median multiple protects against spread explosions.
        spread_limit = max(spread_p90, spread_med * self.spread_mult)
        spread_ok = spread <= spread_limit if spread_limit > 0.0 else True
        warmed = self.quotes_seen >= self.warmup_quotes

        volatility = median(self.abs_returns) if self.abs_returns else 0.0
        vol_p90 = self._quantile(self.abs_returns, 0.90)
        vol_ratio = 1.0 if volatility <= 0.0 else max(1.0, vol_p90 / max(volatility, 1e-12))
        spread_ratio = spread / max(spread_med, 1e-12)

        # Effective leverage is implemented through quantity scaling rather than
        # mutating broker/account leverage during a run.
        spread_scale = min(1.0, 1.0 / max(1.0, spread_ratio))
        vol_scale = min(1.0, 1.5 / max(1.0, vol_ratio))

        # M1 is most sensitive to execution noise; higher TFs receive a small
        # stability premium when spread/volatility deteriorate.
        if self.tf_minutes <= 1:
            tf_weight = 1.00
        elif self.tf_minutes <= 5:
            tf_weight = 1.05
        else:
            tf_weight = 1.10
        tf_weight *= min(1.0, 1.25 / max(1.0, spread_ratio))

        size_mult = max(
            self.min_size_multiplier,
            min(1.0, spread_scale * vol_scale * tf_weight),
        )
        quality = max(0.0, min(1.0, 0.5 * spread_scale + 0.5 * vol_scale))

        ready = basic_ready and warmed and spread_ok
        if not warmed:
            reason = 'warmup'
        elif not spread_ok:
            reason = 'spread_gate'
        else:
            reason = 'ready'

        return AutoFitDecision(
            market_ready=ready,
            spread_ok=spread_ok,
            spread=spread,
            spread_median=float(spread_med),
            spread_limit=float(spread_limit),
            volatility=float(volatility),
            size_multiplier=float(size_mult),
            tf_weight=float(tf_weight),
            execution_quality=float(quality),
            reason=reason,
        )
