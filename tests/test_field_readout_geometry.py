import unittest
import numpy as np
from experiments.field_readout_geometry import class_contrast_basis, standardized


class SourceReadoutSubspaceTests(unittest.TestCase):
    def test_common_class_logit_direction_does_not_enter_contrast_subspace(self):
        rng=np.random.default_rng(19)
        original=rng.normal(size=(4,5,64))
        common=rng.normal(size=(4,1,64))*100
        a,_=class_contrast_basis(original,5);b,_=class_contrast_basis(original+common,5)
        np.testing.assert_allclose(a.T@a,b.T@b,atol=1e-12)
        self.assertEqual(len(a),16)

    def test_positive_diagonal_scale_cancels_under_support_standardization(self):
        rng=np.random.default_rng(17);h=rng.normal(size=(625,64))
        scale=2**rng.uniform(-3,3,64);support=np.arange(25)
        np.testing.assert_allclose(standardized(h,support),standardized(h*scale,support),atol=1e-12)


if __name__=='__main__':unittest.main()
