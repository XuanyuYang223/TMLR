import unittest
import numpy as np

from experiments.algebra import rank_mod
from experiments.field_symmetry import group_sources
from experiments.matched_field import controlled_world


class MatchedFieldTests(unittest.TestCase):
    def test_independent_bases_keep_all_source_functions_equally_dense(self):
        bases = []
        for seed in (271, 314, 593):
            inputs, _, encoded, basis = controlled_world(5, 4, seed)
            self.assertEqual(rank_mod(basis, 5), 4)
            self.assertEqual(encoded.shape, (625, 20))
            bases.append(basis)
            for g in ('P', 'M1', 'M2'):
                physical = group_sources(g, 5)@basis % 5
                self.assertTrue(np.all(physical != 0))
                self.assertEqual(len(np.unique(inputs@physical.T % 5, axis=0)), 625)
                for coefficient in physical:
                    changed = inputs*coefficient % 5
                    self.assertEqual(len(np.unique(changed, axis=0)), 625)
                    np.testing.assert_array_equal(changed.sum(1) % 5, inputs@coefficient % 5)
        self.assertFalse(np.array_equal(bases[0], bases[1]))


if __name__ == '__main__': unittest.main()
