from __future__ import annotations

import numpy as np
import multiedge_nautilus_raw_bt as base

# Inversion-check wrapper: preserves execution logic and flips only signal direction.

def _atr_fixed(self):
    return float(np.mean(list(self.tr)[-20:])) if self.tr else 0.0


_original_signal = base.MultiEdgeRawStrategy._signal


def _signal_inverted(self):
    s = _original_signal(self)
    return -s if s else 0


base.MultiEdgeRawStrategy._atr = _atr_fixed
base.MultiEdgeRawStrategy._signal = _signal_inverted

if __name__ == '__main__':
    base.main()
