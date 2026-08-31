from __future__ import annotations

from collections import Counter

import research.amos_allweather_raw_bidask_bt_compat as compat


_orig_init = compat.CompatStrat.__init__
_orig_rejected = compat.CompatStrat.on_order_rejected


def _init(self, cfg):
    _orig_init(self, cfg)
    self.reject_reasons = Counter()
    self.reject_samples = []


def _reason(event) -> str:
    for name in ('reason', 'rejection_reason', 'message'):
        v = getattr(event, name, None)
        if v is not None:
            return str(v)
    return repr(event)


def _rejected(self, event):
    r = _reason(event)
    self.reject_reasons[r] += 1
    if len(self.reject_samples) < 5:
        self.reject_samples.append(r)
        print(f'ORDER_REJECT_SAMPLE={r}')
    _orig_rejected(self, event)


compat.CompatStrat.__init__ = _init
compat.CompatStrat.on_order_rejected = _rejected

_orig_main = compat.main


def main():
    print('REJECTION_REASON_PROBE=1')
    _orig_main()


if __name__ == '__main__':
    main()
