"""Small geometry cases for the Dow -> Elliott admission stage."""
import unittest

from research.amos_dow_elliott_video_nautilus import wave_context


class DecisionOrderTest(unittest.TestCase):
    def test_dow_must_be_known_before_wave_context(self):
        _, allowed = wave_context([], 0, 'observe')
        self.assertEqual(allowed, set())

    def test_structural_wave_precedes_video_even_without_fib_score(self):
        # The wave is structurally sound, but only F4 satisfies its tight band.
        pivots = [(i,k,p) for i,(k,p) in enumerate(
            [(-1,100),(1,110),(-1,103),(1,116),(-1,112)])]
        waves, structural = wave_context(pivots, 1, 'structural')
        _, strict = wave_context(pivots, 1, 'fib2')
        self.assertEqual(structural, {1})
        self.assertEqual(strict, set())
        self.assertTrue(waves[1][0])
        self.assertEqual(waves[1][1], 1.0)

    def test_directional_wave_can_admit_dow_reversal_d(self):
        pivots = [(i,k,p) for i,(k,p) in enumerate(
            [(-1,100),(1,110),(-1,104),(1,121),(-1,114)])]
        _, allowed = wave_context(pivots, -1, 'structural')
        self.assertEqual(allowed, {1})


if __name__ == '__main__':
    unittest.main()
