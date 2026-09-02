# region imports
from AlgorithmImports import *
# endregion

"""
QuantConnect cloud algorithm for Sai310421/xauusd-data.

Verification label: QC_CLOUD_BT
Never claim NAUTILUS_BT / RAW_BIDASK_PASS / VERIFIED from this file.

Data: QuantConnect CFD XAUUSD QuoteBars (OANDA-sourced in QC cloud).
This is bid/ask quote bars, not Dukascopy BI5 QuoteTick.
"""

import json


class XauusdHourCloud(QCAlgorithm):
    def initialize(self) -> None:
        start = str(self.get_parameter("start_date", "2025-09-01"))
        end = str(self.get_parameter("end_date", "2026-08-28"))
        sy, sm, sd = (int(x) for x in start.split("-"))
        ey, em, ed = (int(x) for x in end.split("-"))
        self.set_start_date(sy, sm, sd)
        self.set_end_date(ey, em, ed)
        self.set_cash(float(self.get_parameter("cash", "1000")))
        self.set_brokerage_model(BrokerageName.OANDA_BROKERAGE, AccountType.MARGIN)
        self.set_time_zone("UTC")

        resolution_name = str(self.get_parameter("resolution", "Hour"))
        resolution = {
            "Minute": Resolution.MINUTE,
            "Hour": Resolution.HOUR,
            "Daily": Resolution.DAILY,
        }.get(resolution_name, Resolution.HOUR)

        self.xau = self.add_cfd("XAUUSD", resolution).symbol
        self.fast_len = int(self.get_parameter("fast", "20"))
        self.slow_len = int(self.get_parameter("slow", "50"))
        self.atr_len = int(self.get_parameter("atr", "14"))
        self.atr_stop = float(self.get_parameter("atr_stop", "1.5"))
        self.rr = float(self.get_parameter("rr", "2.0"))
        self.holdings = float(self.get_parameter("holdings", "0.25"))
        self.experiment_id = str(self.get_parameter("experiment_id", "qc-local"))

        self.fast = self.ema(self.xau, self.fast_len, resolution)
        self.slow = self.ema(self.xau, self.slow_len, resolution)
        self.atr = self.atr(self.xau, self.atr_len, resolution)
        self.set_warm_up(max(self.slow_len, self.atr_len) + 5, resolution)

        self.entry = None
        self.stop = None
        self.target = None
        self.wins = 0
        self.losses = 0
        self.gross_win = 0.0
        self.gross_loss = 0.0
        self.closed = 0
        self.peak_equity = float(self.portfolio.cash)
        self.max_dd = 0.0
        self.last_spread = None

    def on_data(self, data: Slice) -> None:
        if self.is_warming_up or not data.quote_bars.contains_key(self.xau):
            return
        if not self.fast.is_ready or not self.slow.is_ready or not self.atr.is_ready:
            return

        bar = data.quote_bars[self.xau]
        bid = float(bar.bid.close)
        ask = float(bar.ask.close)
        self.last_spread = ask - bid
        equity = float(self.portfolio.total_portfolio_value)
        if equity > self.peak_equity:
            self.peak_equity = equity
        if self.peak_equity > 0:
            dd = (self.peak_equity - equity) / self.peak_equity * 100.0
            if dd > self.max_dd:
                self.max_dd = dd

        invested = self.portfolio[self.xau].invested
        if invested and self.entry is not None:
            holdings = self.portfolio[self.xau].quantity
            hit_stop = (holdings > 0 and bid <= self.stop) or (holdings < 0 and ask >= self.stop)
            hit_tp = (holdings > 0 and bid >= self.target) or (holdings < 0 and ask <= self.target)
            if hit_stop or hit_tp:
                self.liquidate(self.xau, tag="exit")
                return

        if invested:
            return

        atr = float(self.atr.current.value)
        if atr <= 0:
            return
        if self.fast.current.value > self.slow.current.value:
            self.set_holdings(self.xau, self.holdings)
            self.entry = ask
            self.stop = ask - self.atr_stop * atr
            self.target = ask + self.rr * self.atr_stop * atr
        elif self.fast.current.value < self.slow.current.value:
            self.set_holdings(self.xau, -self.holdings)
            self.entry = bid
            self.stop = bid + self.atr_stop * atr
            self.target = bid - self.rr * self.atr_stop * atr

    def on_order_event(self, order_event: OrderEvent) -> None:
        if order_event.status != OrderStatus.FILLED:
            return
        if order_event.ticket is None:
            return
        if "exit" not in str(order_event.ticket.tag):
            return
        closed_profit = float(self.portfolio[self.xau].last_trade_profit)
        self.closed += 1
        if closed_profit >= 0:
            self.wins += 1
            self.gross_win += closed_profit
        else:
            self.losses += 1
            self.gross_loss += abs(closed_profit)
        self.entry = None
        self.stop = None
        self.target = None

    def on_end_of_algorithm(self) -> None:
        if self.portfolio[self.xau].invested:
            self.liquidate(self.xau, tag="eod-flat")
        equity = float(self.portfolio.total_portfolio_value)
        initial = 1000.0
        net21 = equity - initial
        monthly21 = net21 / initial * 100.0 if initial else 0.0
        daily = ((1.0 + monthly21 / 100.0) ** (1.0 / 21.0) - 1.0) * 100.0 if monthly21 > -100 else None
        wr = (self.wins / self.closed * 100.0) if self.closed else None
        pf = (self.gross_win / self.gross_loss) if self.gross_loss > 0 else (None if self.gross_win == 0 else 999.0)
        rf = (net21 / self.max_dd) if self.max_dd else None
        payload = {
            "edge": "QC_XAUUSD_HOUR_EMA",
            "level": "QC_CLOUD_BT",
            "verification_level": "QC_CLOUD_BT",
            "initial": initial,
            "monthly21": monthly21,
            "daily": daily,
            "wr": wr,
            "n21": float(self.closed),
            "pf": pf,
            "rf": rf,
            "dd": self.max_dd,
            "net21": net21,
            "experiment_id": self.experiment_id,
            "dataset": "QC_CFD_XAUUSD_QUOTEBAR",
            "last_spread": self.last_spread,
            "not_nautilus": True,
            "not_raw_bidask_tick": True,
        }
        line = "QC_KPI_JSON=" + json.dumps(payload, separators=(",", ":"))
        self.debug(line)
        self.log(line)
