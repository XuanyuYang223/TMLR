from itertools import product
import unittest
import numpy as np

from experiments.representation_algebra import (permutation_action_table, orbit_residuals,
    group_probe, word_action, apply_word)
from experiments.field_algebra_structure import (dihedral_table, dihedral_actions,
    field_orbits, projection_split, projection_probe)


class RepresentationAlgebraTests(unittest.TestCase):
    def test_actual_noncommuting_words_and_associative_table(self):
        t = permutation_action_table()
        self.assertNotEqual(t[1, 4], t[4, 1])
        self.assertEqual(t[1, 2], t[2, 1])
        self.assertEqual(t[t[4, 1], 4], 2)
        for a, b, c in product(range(8), repeat=3):
            self.assertEqual(t[t[a, b], c], t[a, t[b, c]])

    def test_independent_probes_recover_regular_action_on_new_orbits(self):
        table = permutation_action_table(); rng = np.random.default_rng(45)
        # Eight arbitrary coordinates, acted on by the left regular action.
        # Hidden maps are not supplied to group_probe; fitting uses 40 whole
        # random orbits and prediction uses 12 new complete orbits.
        base = rng.normal(size=(64, 8))
        h = np.array([[row[table[g]] for g in range(8)] for row in base])
        split = np.array([0]*40+[1]*12+[2]*12)
        result, _ = group_probe(h, split, np.zeros(64), table, {'c': 1, 'r': 2, 'i': 4},
            [('c', 'i'), ('r', 'c', 'i')], dimension=8, ridge_grid=(1e-9,), seed=42)
        self.assertLess(max(r['test_nmse'] for r in result['generators']), 1e-14)
        self.assertLess(max(r['test_nmse'] for r in result['composites']), 1e-13)
        self.assertGreater(min(r['shuffled_fit_nmse'] for r in result['generators']), .8)
        self.assertGreater(result['wrong_order'][0]['gap_wrong_minus_correct'], .5)
        self.assertLess(max(r['consistency_nmse'] for r in result['laws']), 1e-13)

    def test_invariant_features_do_not_pass_action_probe(self):
        h = np.tile(np.random.default_rng(7).normal(size=(30, 1, 9)), (1, 8, 1))
        result, _ = group_probe(h, np.array([0]*18+[1]*6+[2]*6), np.zeros(30), permutation_action_table(),
            {'c': 1, 'r': 2, 'i': 4}, [('c', 'i')], 8)
        self.assertEqual(result['status'], 'zero_action_variance')
        self.assertEqual(result['generators'], [])

    def test_dihedral_laws_and_whole_orbit_splits(self):
        table = dihedral_table()
        self.assertEqual(word_action('aaaa', table, {'a': 1, 's': 4}), 0)
        self.assertEqual(word_action('sas', table, {'a': 1, 's': 4}), 3)
        self.assertNotEqual(table[1, 4], table[4, 1])
        orbits, split = field_orbits(5, 82)
        sets = [set(map(tuple, orbits[split == k].reshape(-1, 4))) for k in range(3)]
        self.assertEqual(len(set.union(*sets)), 625)
        for a, b in ((0, 1), (0, 2), (1, 2)): self.assertFalse(sets[a] & sets[b])

    def test_noninvertible_projection_has_disjoint_probe_images_and_is_recovered(self):
        latent = np.array(list(product(range(5), repeat=4)))
        lookup = {tuple(z): i for i, z in enumerate(latent)}
        projected = latent.copy(); projected[:, 2] = 0
        images = np.array([lookup[tuple(z)] for z in projected])
        split = projection_split(latent, 129)
        for a, b in ((0, 1), (0, 2), (1, 2)):
            self.assertFalse(set(images[split == a]) & set(images[split == b]))
        result, _ = projection_probe(latent, images, split, 4, (1e-9,), 67)
        self.assertLess(result['test_nmse'], 1e-14)
        self.assertLess(result['idempotence_consistency_nmse'], 1e-14)
        self.assertGreater(result['identity_nmse'], .1)


if __name__ == '__main__': unittest.main()
