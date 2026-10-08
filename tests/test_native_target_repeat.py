import unittest
import numpy as np
from experiments.native_target_repeat import fit_weights, apply_weights, feature_matrix


class NativeTargetRepeatTests(unittest.TestCase):
    def test_serialized_forecaster_matches_train_standardized_ridge(self):
        x = np.column_stack([np.arange(6.), np.ones(6)])
        y = np.arange(6.)
        weights = fit_weights(x, y)
        expected = 2.5+((np.array([1., 4.])-2.5)/x[:, 0].std())*(6/7*x[:, 0].std())
        np.testing.assert_allclose(apply_weights(np.array([[1., 1.], [4., 1.]]), weights), expected)

    def test_sensitivity_changes_only_intended_math_column(self):
        rows = [{'baseline': [2., 3.], 'math': [5., 0., .5, .25, .25], 'geometry': [.8, .7]}]
        np.testing.assert_array_equal(feature_matrix(rows, 'baseline'), [[2., 3.]])
        self.assertEqual(feature_matrix(rows, 'plus_both', True)[0, 3], 1.)
        np.testing.assert_array_equal(feature_matrix(rows, 'plus_measured_geometry'), [[2., 3., .8, .7]])


if __name__ == '__main__': unittest.main()
