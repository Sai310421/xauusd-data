import csv
import tempfile
import unittest
from pathlib import Path

from xau_tick_research import run
from xau_oos_gate import evaluate
from export_mt5_ticks import export


class TickReplayTests(unittest.TestCase):
    def replay(self, quotes, **params):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'ticks.csv'
            with file.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['symbol', 'available_at', 'bid', 'ask'])
                for n, (bid, ask) in enumerate(quotes):
                    writer.writerow(['XAUUSD', f'2026-01-01T00:00:{n:02d}Z', bid, ask])
            return run(file, lookback=params.pop('lookback',2), threshold_usd=.5,
                       execution_delay_ticks=params.pop('execution_delay_ticks',0),
                       max_spread_usd=.35, **params)

    def test_buy_uses_ask_entry_bid_exit_and_commission(self):
        result = self.replay([(100,100.2),(100.3,100.5),(100.6,100.8),
                              (104.9,105.1)], commission_round_usd=.25)
        self.assertEqual(result['closed_trades'], 1)
        trade = result['trade_log'][0]
        self.assertEqual((trade['entry'],trade['exit'],trade['reason']),
                         (100.8,104.9,'TAKE'))
        self.assertAlmostEqual(trade['net_pnl'],3.85)
        self.assertEqual(result['promotion'], 'BLOCKED_DATA_AND_INDEPENDENT_OOS')

    def test_wide_spread_rejects_signal(self):
        result = self.replay([(100,100.2),(100.3,100.5),(100.6,101.1)])
        self.assertEqual(result['closed_trades'], 0)
        self.assertEqual(result['spread_blocked_ticks'], 1)

    def test_open_position_is_exited_at_end_of_file(self):
        result = self.replay([(100,100.2),(100.3,100.5),(100.6,100.8)])
        self.assertEqual(result['trade_log'][0]['reason'], 'END_OF_DATA')
        self.assertAlmostEqual(result['cash'], 999.8)
        self.assertFalse(result['open_position'])

    def test_next_tick_entry_and_adverse_slippage(self):
        result=self.replay([(100,100.2),(100.3,100.5),(100.6,100.8),
                            (101,101.2),(106,106.2)],
                           execution_delay_ticks=1,slippage_usd_per_side=.1)
        self.assertEqual(result['closed_trades'],1)
        self.assertAlmostEqual(result['trade_log'][0]['entry'],101.3)
        self.assertAlmostEqual(result['trade_log'][0]['exit'],105.9)
        self.assertAlmostEqual(result['trade_log'][0]['net_pnl'],4.6)

    def test_floating_risk_limit_stops_and_liquidates(self):
        result = self.replay([(100,100.2),(100.3,100.5),(100.6,100.8),
                              (47,47.2)], stop_usd=100)
        self.assertTrue(result['risk_halted'])
        self.assertEqual(result['trade_log'][0]['reason'], 'RISK_LIMIT')
        self.assertLessEqual(result['cash'], 950)

    def test_invalid_order_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'ticks.csv'
            file.write_text('symbol,available_at,bid,ask\n'
                            'XAUUSD,2026-01-01T00:00:01Z,100,100.2\n'
                            'XAUUSD,2026-01-01T00:00:00Z,101,101.2\n')
            with self.assertRaisesRegex(ValueError,'non-decreasing'):
                run(file)

    def test_adaptive_feature_rejects_choppy_path(self):
        quotes=[(100+i*.1,100.2+i*.1) for i in range(21)]
        trend=self.replay(quotes,signal_mode='adaptive_trend',lookback=20)
        self.assertGreater(trend['closed_trades'],0)
        choppy=[(100+(.2 if i%2 else 0),100.2+(.2 if i%2 else 0)) for i in range(21)]
        noise=self.replay(choppy,signal_mode='adaptive_trend',lookback=20)
        self.assertEqual(noise['closed_trades'],0)

    def test_oos_data_cannot_change_calibration_selection(self):
        from datetime import datetime,timedelta,timezone
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'ticks.csv'
            origin=datetime(2026,1,1,tzinfo=timezone.utc)
            def write(oos_shift):
                with file.open('w',newline='') as f:
                    writer=csv.writer(f);writer.writerow(['symbol','available_at','bid','ask'])
                    for i in range(200):
                        bid=100+i*.05 if i<100 else 105+(i-100)*oos_shift
                        writer.writerow(['XAUUSD',(origin+timedelta(seconds=i)).isoformat(),bid,bid+.2])
            args=(origin+timedelta(seconds=99)).isoformat(),(origin+timedelta(seconds=110)).isoformat()
            write(.05)
            one=evaluate(file,*args,'test_fixture',min_cal_trades=1,min_oos_trades=1)
            write(-.05)
            two=evaluate(file,*args,'test_fixture',min_cal_trades=1,min_oos_trades=1)
            self.assertEqual(one['calibration_trials'],two['calibration_trials'])
            self.assertEqual(one['selected'],two['selected'])
            self.assertEqual(one['split_counts'],{'calibration':100,'embargo':10,'oos':90})

    def test_mt5_export_keeps_same_millisecond_tick_order(self):
        from types import SimpleNamespace
        class FakeMT5:
            COPY_TICKS_INFO=1
            def initialize(self):return True
            def shutdown(self):pass
            def symbol_info(self,symbol):return SimpleNamespace(trade_contract_size=100,volume_min=.01,volume_step=.01)
            def symbol_select(self,symbol,enable):return True
            def copy_ticks_range(self,symbol,start,end,flag):
                ms=int(start.timestamp()*1000)+1
                return [{'time_msc':ms,'bid':100,'ask':100.2},
                        {'time_msc':ms,'bid':100.1,'ask':100.3}]
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'ticks.csv'
            export('XAUUSD','2026-01-01T00:00:00Z','2026-01-01T00:00:02Z',file,FakeMT5())
            result=run(file,lookback=2)
            self.assertEqual(result['ticks'],2)


if __name__ == '__main__':
    unittest.main()
