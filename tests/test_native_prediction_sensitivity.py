import unittest
from experiments.native_prediction_sensitivity import corrected_math


class NativePredictionSensitivityTests(unittest.TestCase):
    def test_zero_null_sentinel_maps_to_intended_indicator_without_mutating_frozen_features(self):
        frozen = {'math': [5., 0., .5, .25, .25]}
        self.assertEqual(corrected_math(frozen), [5., 1., .5, .25, .25])
        self.assertEqual(frozen['math'][1], 0.)
        self.assertEqual(corrected_math({'math': [3., 0., .5, .25, .25]})[1], 0.)


if __name__ == '__main__': unittest.main()
