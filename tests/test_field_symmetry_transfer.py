import json
from pathlib import Path
import unittest

import numpy as np

from experiments.algebra import composition_size, projective
from experiments.field_symmetry import group_sources
from experiments.field_symmetry_transfer import splits


class FieldSymmetryTransferTests(unittest.TestCase):
    def test_common_targets_exclude_all_source_directions_and_span_multiple_orders(self):
        config = json.loads(Path('configs/field_symmetry_transfer.json').read_text())
        groups = ('P', 'M1', 'M4', 'M2', 'M3')
        excluded = {projective(row, 5) for g in groups for row in group_sources(g, 5)}
        self.assertTrue(all(projective(t, 5) not in excluded for t in config['targets']))
        orders = {composition_size(group_sources(g, 5), target, 5) for g in groups for target in config['targets']}
        self.assertEqual(orders, {2, 3, 4})

    def test_target_support_is_nested_balanced_and_disjoint_from_test(self):
        labels = np.tile(np.arange(5), 125)
        support, test = splits(labels, 5, 17, [25, 50], 25)
        self.assertTrue(set(support[25]) <= set(support[50]))
        self.assertFalse(set(support[50]) & set(test))
        np.testing.assert_array_equal(np.bincount(labels[test]), np.repeat(25, 5))
        np.testing.assert_array_equal(np.bincount(labels[support[25]]), np.repeat(5, 5))


if __name__ == '__main__':
    unittest.main()
