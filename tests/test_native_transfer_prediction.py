import unittest
import numpy as np
from experiments.native_transfer_prediction import ridge_predict, label_statistics


class NativeTransferPredictionTests(unittest.TestCase):
    def test_train_only_standardization_ignores_constant_indicator(self):
        x = np.column_stack([np.arange(6.), np.zeros(6)])
        test = np.array([[7., 0.], [7., 1.]])
        prediction = ridge_predict(x, np.arange(6.), test)
        self.assertAlmostEqual(prediction[0], prediction[1])

    def test_conditioning_removes_length_only_similarity(self):
        n = np.repeat([10, 11], 20)
        y = np.tile(np.repeat([0, 1], 10), 2)
        source = np.column_stack([y, 1-y])
        values = label_statistics(source, n, n)
        self.assertAlmostEqual(values[3], 0)
        self.assertAlmostEqual(values[4], 0)


if __name__ == '__main__': unittest.main()
