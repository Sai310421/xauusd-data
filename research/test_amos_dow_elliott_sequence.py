"""Small geometry cases for the Dow -> Elliott admission stage."""
import unittest

from research.amos_dow_elliott_video_nautilus import wave_context, wave_phase, confirmed_c_breakout, d_retest_expired


class DecisionOrderTest(unittest.TestCase):
    def test_dow_must_be_known_before_wave_context(self):
        _, allowed = wave_context([], 0, 'observe')
        self.assertEqual(allowed, set())

    def test_confirmed_p2_enables_wave3_without_future_p3(self):
        pivots=[(0,-1,100),(1,1,110),(2,-1,104)]
        waves,allowed=wave_context(pivots,1,'structural')
        self.assertEqual(allowed,{1})
        self.assertEqual(waves[1][3:],(3,2))
        self.assertEqual(wave_context(pivots,1,'fib2')[1],set())

    def test_confirmed_p4_enables_wave5_without_future_p5(self):
        # The wave is structurally sound, but only F4 satisfies its tight band.
        pivots = [(i,k,p) for i,(k,p) in enumerate(
            [(-1,100),(1,110),(-1,103),(1,116),(-1,112)])]
        waves, structural = wave_context(pivots, 1, 'structural')
        _, strict = wave_context(pivots, 1, 'fib2')
        self.assertEqual(structural, {1})
        self.assertEqual(strict, set())
        self.assertTrue(waves[1][0])
        self.assertEqual(waves[1][1], 1.0)
        self.assertEqual(waves[1][3:],(5,4))

    def test_p3_peak_and_broken_p4_do_not_open_new_wave(self):
        base=[(0,-1,100),(1,1,110),(2,-1,104),(3,1,122)]
        self.assertEqual(wave_phase(base,1)[3],0)
        self.assertEqual(wave_phase(base+[(4,-1,108)],1)[3],0)

    def test_no_wave2_below_origin(self):
        self.assertEqual(wave_phase([(0,-1,100),(1,1,110),(2,-1,99)],1)[3],0)

    def test_c_breakout_requires_preexisting_confirmed_high(self):
        bars=[{'c':100},{'c':101},{'c':105},{'c':111}]
        self.assertTrue(confirmed_c_breakout(bars,1,110))
        self.assertFalse(confirmed_c_breakout(bars,2,110))
        self.assertFalse(confirmed_c_breakout(bars,1,111))

    def test_d_flip_waits_for_confirmed_m15_pivots(self):
        t=16*60*1_000_000_000
        self.assertTrue(d_retest_expired(0,t,'retest'))
        self.assertFalse(d_retest_expired(0,t,'flip'))
        self.assertTrue(d_retest_expired(0,91*60*1_000_000_000,'flip'))

    def test_directional_wave_can_admit_dow_reversal_d(self):
        pivots = [(i,k,p) for i,(k,p) in enumerate(
            [(-1,100),(1,110),(-1,104),(1,121),(-1,114)])]
        _, allowed = wave_context(pivots, -1, 'structural')
        self.assertEqual(allowed, {1})


if __name__ == '__main__':
    unittest.main()
