import unittest
import numpy as np
from experiments.factor_kernel_readout import interaction_kernel


class FactorKernelTests(unittest.TestCase):
    def test_independent_class_relabelings_leave_the_kernel_unchanged(self):
        rng=np.random.default_rng(19);values=rng.uniform(size=(30,4,5));values/=values.sum(-1,keepdims=True)
        changed=values.copy()
        for task in range(4):changed[:,task]=changed[:,task,rng.permutation(5)]
        for degree in (1,2,3,4):
            np.testing.assert_allclose(interaction_kernel(values,values,degree),interaction_kernel(changed,changed,degree),atol=1e-14)

    def test_interaction_kernel_is_positive_semidefinite(self):
        rng=np.random.default_rng(42);values=np.eye(5)[rng.integers(5,size=(40,4))]
        for degree in (1,2,3,4):
            kernel=interaction_kernel(values,values,degree)
            self.assertGreaterEqual(np.linalg.eigvalsh(kernel).min(),-1e-12)
            np.testing.assert_allclose(np.diag(kernel),1.)


if __name__=='__main__':unittest.main()
