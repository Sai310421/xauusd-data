from __future__ import annotations

import sys
from research import hft_boost_bot_tournament_v1 as tour


class GoldUpgrade(tour.BotTournament):
    def __init__(self, *args, variant="base", **kwargs):
        super().__init__(*args, **kwargs)
        self.variant = variant

    def _gold(self, m):
        sig = super()._gold(m)
        if sig is None:
            return None
        side, score = sig
        spread = m[5]

        # Explicit spread ablations using the same unit as tournament v1
        if self.variant in {"spread760", "midas_htf_760"} and spread > 760:
            return None
        if self.variant in {"spread630", "midas_htf_630"} and spread > 630:
            return None

        # Borrow only the low-DD HTF structure filter from Midas.
        # Do not require a full Midas entry signal: this preserves GOLD-HFT as the main BOT.
        if self.variant.startswith("midas_htf"):
            h = self.m15.data()
            if len(h) >= 30:
                htf, _, _ = tour.swing_structure(h)
                if side == "buy" and htf == "DOWN":
                    return None
                if side == "sell" and htf == "UP":
                    return None
                if (side == "buy" and htf == "UP") or (side == "sell" and htf == "DOWN"):
                    score += 8

        return side, score


def main():
    variant = "base"
    argv = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--variant":
            variant = sys.argv[i + 1]
            i += 2
        else:
            argv.append(sys.argv[i])
            i += 1

    allowed = {"base", "spread760", "spread630", "midas_htf", "midas_htf_760", "midas_htf_630"}
    if variant not in allowed:
        raise SystemExit(f"bad variant {variant}")

    original = tour.BotTournament

    class Injected(GoldUpgrade):
        def __init__(self, config, bot):
            super().__init__(config, bot, variant=variant)

    tour.BotTournament = Injected
    old_argv = sys.argv
    try:
        # Reuse proven tournament main and force GOLD as the main profile.
        sys.argv = [old_argv[0]] + argv + ["--bot", "gold"]
        tour.main()
    finally:
        sys.argv = old_argv
        tour.BotTournament = original


if __name__ == "__main__":
    main()
