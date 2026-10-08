from itertools import product
import unittest
import numpy as np

from experiments.field_mechanism import categorical_decoder, spectral_decomposition, state_splits


class FieldMechanismTests(unittest.TestCase):
    def test_generic_decoder_uses_categorical_pairs_and_survives_value_relabeling(self):
        z=np.array(list(product(range(5),repeat=3)))
        codes=np.column_stack([z,(z[:,0]+z[:,1])%5])
        labels=(z[:,0]+2*z[:,1])%5
        supports,test=state_splits(z,labels,5,71)
        train=supports[100]
        prediction,seen,selected=categorical_decoder(codes[train],labels[train],codes[test],5)
        self.assertEqual(len(selected),2)
        self.assertTrue(np.all(seen))
        np.testing.assert_array_equal(prediction,labels[test])
        rng=np.random.default_rng(7)
        renamed=np.column_stack([rng.permutation(5)[column] for column in codes.T])
        rename_y=rng.permutation(5)
        prediction,seen,selected=categorical_decoder(renamed[train],rename_y[labels[train]],renamed[test],5)
        np.testing.assert_array_equal(prediction,rename_y[labels[test]])

    def test_three_component_target_cannot_be_recovered_by_full_state_lookup(self):
        z=np.array(list(product(range(5),repeat=3)))
        codes=np.column_stack([z,(z[:,0]+z[:,1])%5])
        labels=(z[:,0]+2*z[:,1]+z[:,2])%5
        supports,test=state_splits(z,labels,5,71)
        prediction,seen,selected=categorical_decoder(codes[supports[100]],labels[supports[100]],codes[test],5)
        self.assertEqual(len(selected),3)
        self.assertFalse(np.any(seen))
        self.assertAlmostEqual(np.mean(prediction==labels[test]),.2)

    def test_parseval_separates_true_nuisance_and_source_character_energy(self):
        grid=np.array(list(product(range(5),repeat=4)))
        latent=grid[:,:3]
        source=np.eye(3,dtype=int)
        representation=np.column_stack([np.cos(2*np.pi*grid[:,0]/5),np.sin(2*np.pi*grid[:,3]/5)])
        power,stats,means=spectral_decomposition(representation,latent,source,5)
        self.assertAlmostEqual(stats['nuisance_fraction'],.5)
        self.assertAlmostEqual(stats['order_1_energy_fraction'],.5)
        self.assertAlmostEqual(stats['order_2_energy_fraction'],0)
        self.assertAlmostEqual(stats['order_3_energy_fraction'],0)


if __name__=='__main__':unittest.main()
