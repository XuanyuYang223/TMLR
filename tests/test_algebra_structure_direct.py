import unittest
import numpy as np

from experiments.algebra_structure_direct import direct_probe
from experiments.algebra_structure_controls import relabel_orbits
from experiments.representation_algebra import group_probe, permutation_action_table, orbit_residuals


class DirectAlgebraTests(unittest.TestCase):
    def fixture(self):
        table = permutation_action_table()
        rng = np.random.default_rng(222)
        base = rng.normal(size=(80, 8))
        h = np.array([[x[table[g]] for g in range(8)] for x in base])
        split = np.array([0]*50+[1]*15+[2]*15)
        return table, h, split

    def test_direct_regular_action_and_no_heldout_fit_leak(self):
        table, h, split = self.fixture()
        def probe(x): return direct_probe(x, split, np.zeros(80), table, {'c': 1, 'r': 2, 'i': 4},
            [('c', 'i'), ('r', 'c', 'i')], 8, (1e-9,), 78)
        r, a = probe(h)
        self.assertLess(max(x['full_space_displacement_nmse'] for x in r['generators']), 1e-14)
        self.assertLess(max(x['full_space_displacement_nmse'] for x in r['composites']), 1e-13)
        self.assertLess(max(x['consistency_nmse'] for x in r['laws']), 1e-13)
        changed = h.copy(); changed[split == 2] += np.arange(8)*9
        _, b = probe(changed)
        np.testing.assert_array_equal(a['basis'], b['basis'])
        for key in ('map_c', 'map_r', 'map_i'): np.testing.assert_array_equal(a[key], b[key])

    def test_orbit_centering_can_create_an_unearned_sign_action(self):
        table = permutation_action_table(); rng = np.random.default_rng(99)
        # A and B are independent: knowing raw A cannot predict raw B.
        # But orbit-mean subtraction turns them into +/- (A-B)/2 exactly.
        a, b = rng.normal(size=(800, 2, 5)).transpose(1, 0, 2)
        signs = np.array([1, -1, -1, 1, 1, -1, -1, 1])
        h = np.where(signs[None, :, None] == 1, a[:, None, :], b[:, None, :])
        split = np.array([0]*500+[1]*150+[2]*150)
        centered, _ = group_probe(h, split, np.zeros(800), table, {'c': 1, 'r': 2, 'i': 4}, [('c', 'i')], 5, (1e-9,), 42)
        direct, _ = direct_probe(h, split, np.zeros(800), table, {'c': 1, 'r': 2, 'i': 4}, [('c', 'i')], 5, (1e-9,), 42)
        self.assertLess(centered['generators'][0]['test_nmse'], 1e-14)
        self.assertGreater(direct['generators'][0]['full_space_displacement_nmse'], .4)
        self.assertGreater(direct['laws'][0]['consistency_nmse'], .8)
        self.assertEqual(direct['generators'][2]['status'], 'invariant_action')

    def test_relabeling_keeps_pca_covariance_and_orbit_energy(self):
        _, h, _ = self.fixture()
        null = relabel_orbits(h, 914)
        old, new = orbit_residuals(h).reshape(-1, 8), orbit_residuals(null).reshape(-1, 8)
        np.testing.assert_allclose(old.T @ old, new.T @ new, atol=1e-10)
        np.testing.assert_allclose(h.mean(1), null.mean(1), atol=1e-12)


if __name__ == '__main__': unittest.main()
