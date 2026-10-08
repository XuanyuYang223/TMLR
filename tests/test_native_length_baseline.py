import unittest
import numpy as np
from experiments.native_length_baseline import smooth_prediction, majority_predictions


class NativeLengthBaselineTests(unittest.TestCase):
    def test_far_lengths_use_nearest_support_without_underflow(self):
        result = smooth_prediction(np.array([10, 11]), np.array([2, 3]), np.array([1000]), .5, 5)
        self.assertEqual(result.tolist(), [3])

    def test_exact_length_majority_falls_back_to_support_global(self):
        global_prediction, conditional = majority_predictions(np.array([10, 10, 11]), np.array([1, 1, 2]), np.array([10, 11, 12]))
        self.assertEqual(global_prediction.tolist(), [1, 1, 1])
        self.assertEqual(conditional.tolist(), [1, 2, 1])


if __name__ == '__main__': unittest.main()
