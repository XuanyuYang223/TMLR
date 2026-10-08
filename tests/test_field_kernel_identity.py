from itertools import product
import unittest
import numpy as np
from experiments.analysis import linear_cka
from experiments.field_kernel_identity import directional_cka, categorical_features


class FieldKernelIdentityTests(unittest.TestCase):
    def test_direction_multiplicity_matches_full_categorical_kernel(self):
        x = np.array(list(product(range(3), repeat=2)))
        left = np.array([[1, 0], [1, 0], [0, 1]])
        right = np.array([[1, 0], [0, 1], [0, 1]])
        expected = directional_cka(left, right, 3)
        self.assertAlmostEqual(expected, .8)
        self.assertAlmostEqual(linear_cka(categorical_features(x, left, 3), categorical_features(x, right, 3)), expected)

    def test_affine_label_relabeling_and_constant_task_do_not_change_kernel(self):
        x = np.array(list(product(range(5), repeat=2)))
        left = np.array([[1, 0], [0, 1], [0, 0]])
        right = np.array([[2, 0], [0, 3], [0, 0]])
        a = categorical_features(x, left, 5)
        b = categorical_features(x, right, 5, [1, 4, 2])
        self.assertAlmostEqual(directional_cka(left, right, 5), 1)
        self.assertAlmostEqual(linear_cka(a, b), 1)


if __name__ == '__main__': unittest.main()
