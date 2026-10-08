from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.algebra_structure_replication import collect_prior_inputs,extra_metrics
from experiments.algebra_structure_direct import direct_probe
from experiments.representation_algebra import permutation_action_table
from experiments.algebra_noninvertible import coordinate_zero_projection,polynomial_operators,polynomial_law_audit


class AlgebraReplicationTests(unittest.TestCase):
    def test_prior_orbit_archive_excludes_every_transformed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'prior.npz'
            raw=np.array([[[1,2,3,0],[3,1,2,0]]])
            np.savez(path,permutations=raw,lengths=[3],input=np.zeros((1,2,10)))
            self.assertEqual(collect_prior_inputs([path]),{(1,2,3),(3,1,2)})

    def test_full_space_two_inverse_metric_recovers_an_independent_regular_action(self):
        table=permutation_action_table();rng=np.random.default_rng(175);base=rng.normal(size=(80,8))
        h=np.array([[row[table[g]] for g in range(8)] for row in base]);split=np.array([0]*50+[1]*15+[2]*15)
        result,arrays=direct_probe(h,split,np.zeros(80),table,{'c':1,'r':2,'i':4},[('c','i')],8,(1e-9,),33)
        extra=extra_metrics(h,{'split':split,'lengths':np.zeros(80)},arrays,result)
        self.assertEqual(extra['identity_displacement_nmse'],1.)
        self.assertLess(extra['two_inverse_full_space_reconstruction_nmse'],1e-12)

    def test_projection_keeps_dimension_but_deletion_does_not(self):
        x=np.array([4,7,2,9]);p=coordinate_zero_projection(4,2)
        np.testing.assert_array_equal(p @ p @ x,p @ x)
        self.assertEqual((p @ x).shape,(4,))
        once=np.delete(x,2);twice=np.delete(once,2)
        self.assertEqual(once.shape,(3,));self.assertEqual(twice.shape,(2,))

    def test_derivative_truncation_cross_law_and_wrong_commutation(self):
        derivative,projections=polynomial_operators(5);p=projections[3]
        coefficients=np.array([2,3,5,7,11,13])
        np.testing.assert_array_equal(derivative @ p @ coefficients,projections[2] @ derivative @ coefficients)
        self.assertFalse(np.array_equal(derivative @ p,p @ derivative))
        self.assertFalse(np.linalg.matrix_power(derivative,6).any())
        self.assertGreater(polynomial_law_audit()['exact_matrix_laws_checked'],300)


if __name__=='__main__':unittest.main()
