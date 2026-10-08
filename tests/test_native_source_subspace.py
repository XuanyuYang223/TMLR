import unittest
import numpy as np
from experiments.native_source_subspace import contrast_basis, kernels


class NativeSourceSubspaceTests(unittest.TestCase):
    def test_projection_preserves_numeric_margins_and_complementary_kernels_add(self):
        rng = np.random.default_rng(124)
        weights = rng.normal(size=(7, 16)); hidden = rng.normal(size=(20, 32))
        basis, centered = contrast_basis(weights)
        self.assertEqual(basis.shape, (16, 6))
        blocks = hidden.reshape(20, 2, 16)
        np.testing.assert_allclose(((blocks@basis)@basis.T)@centered.T, blocks@centered.T, atol=1e-12)
        result = kernels(hidden, basis, centered, np.repeat([10, 11], 10), 'within_length')
        np.testing.assert_allclose(result['numeric_contrast_space']+result['numeric_null_space'], result['full'], atol=1e-12)


if __name__ == '__main__': unittest.main()
