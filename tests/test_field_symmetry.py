import unittest
import numpy as np
from experiments.field_symmetry import audit_group, group_sources, spectral_power


class FieldSymmetryTests(unittest.TestCase):
    def test_exact_statistics_and_three_predicted_similarity_levels(self):
        expected={'P':1.,'M1':.75,'M4':.75,'M2':.5,'M3':.5}
        for group,value in expected.items():
            audit=audit_group(group,5)
            self.assertEqual(audit['rank'],4)
            self.assertEqual(audit['joint_support'],625)
            self.assertEqual(audit['predicted_categorical_cka'],value)
            self.assertAlmostEqual(audit['exact_categorical_cka'],value)

    def test_four_dimensional_parseval(self):
        from itertools import product
        latent=np.array(list(product(range(5),repeat=4)))
        h=np.column_stack([np.cos(2*np.pi*latent[:,0]/5),np.sin(2*np.pi*(latent[:,2]+2*latent[:,3])/5)])
        self.assertAlmostEqual(spectral_power(h,latent,5).sum(),np.square(h-h.mean(0)).sum(1).mean())


if __name__=='__main__':unittest.main()
