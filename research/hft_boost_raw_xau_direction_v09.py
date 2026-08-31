from __future__ import annotations

import sys

from research import hft_boost_raw_xau_bt_v08 as v08


class HFTDirectionV09(v08.HFTBaseV08):
    """Direction ablation over the same v0.8 execution/cost model.

    continuation: keep original signal side.
    reversal: invert every accepted signal side.
    exhaustion: invert only when microstructure reports exhaustion; otherwise skip.
    """

    def __init__(self, *args, direction_mode="continuation", **kwargs):
        super().__init__(*args, **kwargs)
        self.direction_mode = direction_mode

    def _signal(self, m):
        s = super()._signal(m)
        if s is None:
            return None
        side, score = s
        exhaustion = bool(m[6])
        if self.direction_mode == "continuation":
            return side, score
        if self.direction_mode == "reversal":
            return ("sell" if side == "buy" else "buy"), score
        if self.direction_mode == "exhaustion":
            if not exhaustion:
                return None
            return ("sell" if side == "buy" else "buy"), score
        raise ValueError(self.direction_mode)


def main():
    # Reuse the proven Raw QuoteTick v0.8 runner while injecting direction mode.
    mode = "continuation"
    argv = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--direction-mode":
            mode = sys.argv[i + 1]
            i += 2
        else:
            argv.append(sys.argv[i])
            i += 1

    if mode not in {"continuation", "reversal", "exhaustion"}:
        raise SystemExit(f"bad direction mode: {mode}")

    original = v08.HFTBaseV08

    class Injected(HFTDirectionV09):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, direction_mode=mode, **kwargs)

    v08.HFTBaseV08 = Injected
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + argv
        v08.main()
    finally:
        sys.argv = old_argv
        v08.HFTBaseV08 = original


if __name__ == "__main__":
    main()
