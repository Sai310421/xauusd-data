from __future__ import annotations

"""HFT Boost Raw BT v0.7 execution wrapper.

Restores normal reject handling after the v0.6 fail-fast diagnostic. The
underlying fill-driven state machine remains unchanged.
"""

from research.hft_boost_raw_xau_bt_event import HFTBaseEventStrategy, main


def _normal_reject(self, event):
    self.order_rejects += 1
    self._clear_entry_pending()
    self.exit_pending = False


HFTBaseEventStrategy.on_order_rejected = _normal_reject


if __name__ == "__main__":
    main()
