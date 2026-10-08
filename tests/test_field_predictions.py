import unittest
import numpy as np

from experiments.field_predictions import heldout_combination_predictions, lookup_expectation


class FieldPredictionTests(unittest.TestCase):
    def test_full_tuple_lookup_has_no_seen_keys_on_disjoint_inputs(self):
        for b in (25, 50):
            expected = lookup_expectation(5, 4, 4, b)
            self.assertEqual(expected['seen_probability'], 0.)
            self.assertEqual(expected['accuracy'], .2)
        self.assertGreater(lookup_expectation(5, 4, 2, 50)['accuracy'], lookup_expectation(5, 4, 3, 50)['accuracy'])
        self.assertGreater(lookup_expectation(5, 4, 2, 50)['accuracy'], lookup_expectation(5, 4, 2, 25)['accuracy'])

    def test_combination_holdout_removes_all_rows_of_each_group(self):
        labels = np.repeat(['a', 'b', 'c'], 6)
        design = np.arange(18).reshape(-1, 1).astype(float)
        truth = design[:, 0]*.01
        result = heldout_combination_predictions(design, truth, labels)
        self.assertEqual(len(result['folds']), 3)
        self.assertTrue(all(r['train_count'] == 12 and r['test_count'] == 6 for r in result['folds']))
        self.assertTrue(np.all(np.isfinite(result['predictions'])))


if __name__ == '__main__': unittest.main()
