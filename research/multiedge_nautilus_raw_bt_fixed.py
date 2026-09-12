from __future__ import annotations

import numpy as np
import multiedge_nautilus_raw_bt as base


def _atr_fixed(self):
    return float(np.mean(list(self.tr)[-20:])) if self.tr else 0.0


base.MultiEdgeRawStrategy._atr = _atr_fixed

if __name__ == '__main__':
    base.main()
