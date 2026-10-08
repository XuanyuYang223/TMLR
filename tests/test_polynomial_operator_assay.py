from itertools import product
import unittest
import numpy as np

from experiments.polynomial_operator_assay import operators,input_word,context_split,probe


class PolynomialOperatorAssayTests(unittest.TestCase):
    def test_cross_law_preserves_context_and_has_distinct_wrong_order(self):
        latent=np.array(list(product(range(5),repeat=4)))
        correct=input_word(latent,('p1','d'));right=input_word(latent,('d','p0'));wrong=input_word(latent,('d','p1'))
        np.testing.assert_array_equal(correct,right)
        self.assertFalse(np.array_equal(correct,wrong))
        np.testing.assert_array_equal(correct[:,3],latent[:,3])

    def test_heldout_context_images_disjoint_and_unfitted_cross_recovered(self):
        latent=np.array(list(product(range(5),repeat=4)));lookup={tuple(z):i for i,z in enumerate(latent)}
        words=[('d',),('p1',),('p0',),('p1','d'),('d','p0'),('p1','p1'),('p0','p0'),('d','p1')]
        images={'_'.join(w):np.array([lookup[tuple(z)] for z in input_word(latent,w)]) for w in words}
        split,_=context_split(latent,942)
        for ids in images.values():np.testing.assert_array_equal(split[ids],split)
        hidden=np.eye(5)[latent].reshape(625,-1)
        result,arrays=probe(hidden,images,split,latent[:,3],20,99)
        self.assertLess(max(r['displacement_nmse'] for r in result['generators']),1e-8)
        self.assertLess(max(r['displacement_nmse'] for r in result['composites']),1e-8)
        self.assertLess(result['laws'][0]['full_space_consistency_nmse'],1e-8)
        self.assertGreater(result['wrong_order_gap'],.1)
        self.assertEqual(set(k for k in arrays if k.startswith('rho_')),{'rho_d','rho_p1','rho_p0'})


if __name__=='__main__':unittest.main()
