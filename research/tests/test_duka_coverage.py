"""Regression for empty daily pause and failed open-hour acquisition."""
import contextlib
import datetime as dt
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(RESEARCH / 'shared_bidask'))
import build_raw_bidask_catalog_duka as builder
from verify_builder66_catalog import audit as verify_catalog
import source
from market_calendar import expected_trading_hour


class FakeResponse:
    status = 200
    def __enter__(self): return self
    def __exit__(self, *_): return False
    def read(self): return b''


class CoverageTests(unittest.TestCase):
    def test_empty_http_200_is_preserved_in_both_fetchers(self):
        stamp=dt.datetime(2026,5,26,21,tzinfo=dt.timezone.utc)
        with mock.patch.object(builder.urllib.request,'urlopen',return_value=FakeResponse()):
            self.assertEqual(builder.fetch_hour('XAUUSD',1000,stamp),([], '200_EMPTY'))
            self.assertEqual(source.fetch_hour('XAUUSD',1000,stamp),([], '200_EMPTY'))

    def generate(self, failed_hour=None):
        day=dt.datetime(2026,5,26,tzinfo=dt.timezone.utc)
        def fake_fetch(symbol,scale,hour):
            if hour.hour==failed_hour: return [],503
            if not expected_trading_hour('XAUUSD',hour): return [],'200_EMPTY'
            return [(hour+dt.timedelta(milliseconds=1),3300.0,3300.6,1.0,1.0)],200
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'catalog'
            args=['builder','--start','2026-05-26','--days','1','--symbols','XAUUSD',
                  '--workers','2','--catalog',str(path)]
            with mock.patch.object(sys,'argv',args), mock.patch.object(builder,'fetch_hour',side_effect=fake_fetch), contextlib.redirect_stdout(io.StringIO()):
                if failed_hour is None:
                    builder.main()
                else:
                    with self.assertRaises(SystemExit):
                        builder.main()
            return json.loads((path/'catalog_manifest.json').read_text()), verify_catalog(path,'2026-05-26',1)

    def test_daily_pause_complete_and_missing_open_hour_incomplete(self):
        complete, verified=self.generate()
        self.assertEqual(complete['status'],'COMPLETE')
        self.assertEqual(verified['status'],'SHARD_HOURLY_COVERAGE_VERIFIED')
        self.assertEqual(verified['quote_count'],23)
        self.assertEqual(complete['coverage_schema_version'],2)
        self.assertEqual(complete['stats']['XAUUSD']['http_status_counts']['200_EMPTY'],1)
        self.assertEqual(complete['unresolved_open_hours'],[])
        incomplete, blocked=self.generate(failed_hour=19)
        self.assertEqual(incomplete['status'],'INCOMPLETE')
        self.assertEqual(blocked['status'],'BLOCKED')
        self.assertEqual(incomplete['unresolved_open_hours'][0]['hour_utc'],'2026-05-26T19:00:00+00:00')

    def test_shared_fetch_keeps_hour_provenance(self):
        day=dt.datetime(2026,5,26,tzinfo=dt.timezone.utc)
        def fake_fetch(symbol,scale,hour):
            if hour.hour==19: return [],503
            if hour.hour==21: return [],'200_EMPTY'
            return [(hour,3300.0,3300.6,1.0,1.0)],200
        with mock.patch.object(source,'fetch_hour',side_effect=fake_fetch):
            rows,counts,hours=source.fetch_day('XAUUSD',1000,day,workers=2,include_hour_status=True)
        self.assertEqual(len(rows),22)
        self.assertEqual(counts['200_EMPTY'],1)
        self.assertTrue(hours[19]['expected_open'])
        self.assertFalse(hours[21]['expected_open'])
        self.assertEqual(hours[19]['http_status'],503)


if __name__=='__main__':
    unittest.main()
