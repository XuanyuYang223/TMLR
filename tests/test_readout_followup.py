import unittest
import numpy as np
import torch
from experiments.readout_followup import ReadoutBatch, fit_field, lookup_predict, code_diagnostics
from experiments.run import balanced_splits


class ReadoutFollowupTests(unittest.TestCase):
    def test_field_fit_recovers_unknown_affine_coefficients(self):
        x = np.column_stack([np.eye(3, dtype=np.int64), np.ones(3, dtype=np.int64)])
        x = np.vstack([x, [0, 0, 0, 1]])
        true_coefficients = np.array([2, 3, 1, 4])
        y = x @ true_coefficients % 5
        original = x.copy()
        solution, rank = fit_field(x, y, 5)
        np.testing.assert_array_equal(solution, true_coefficients)
        np.testing.assert_array_equal(x, original)
        self.assertEqual(rank, 4)

    def test_dependent_source_outputs_still_allow_a_valid_fit(self):
        x = np.array([[1, 1, 1], [2, 2, 1], [3, 3, 1]])
        y = x @ np.array([1, 2, 4]) % 5
        solution, rank = fit_field(x, y, 5)
        self.assertEqual(rank, 2)
        self.assertIsNotNone(solution)
        np.testing.assert_array_equal(x @ solution % 5, y)

    def test_inconsistent_target_labels_cannot_be_silently_fitted(self):
        solution, rank = fit_field([[1, 1], [1, 1]], [0, 1], 5)
        self.assertIsNone(solution)
        self.assertEqual(rank, 1)

    def test_lookup_only_reuses_seen_source_tuples(self):
        predictions, matched = lookup_predict(np.array([[1, 2], [3, 4]]), np.array([1, 2]), np.array([[3, 4], [0, 0]]), 5)
        np.testing.assert_array_equal(matched, [True, False])
        np.testing.assert_array_equal(predictions, [2, 1])

    def test_mlp_target_heads_do_not_share_parameters(self):
        features = torch.randn(2, 4, 8)
        head = ReadoutBatch(2, 8, 5, 16, 'mlp', 17, 'cpu')
        head(features)[0].square().sum().backward()
        self.assertGreater(head.w1.grad[0].abs().sum().item(), 0)
        self.assertEqual(head.w1.grad[1].abs().sum().item(), 0)
        self.assertFalse(features.requires_grad)

    def test_field_diagnostic_learns_from_support_labels_without_target_formula(self):
        from itertools import product
        latent = np.array(list(product(range(5), repeat=3)))
        codes = np.column_stack([latent, latent.sum(1) % 5])
        # The coefficients are used here to generate data, and never passed to
        # code_diagnostics, which only receives observed support labels.
        labels = (latent @ np.array([[1, 2, 3], [2, 1, 1]]).T) % 5
        test, supports = balanced_splits(labels, [10, 25], 5, 5, 17)
        rows = code_diagnostics(codes, labels, supports, test, 5, 42)
        field = [r for r in rows if r['method'] == 'field_fit']
        self.assertTrue(all(r['test_accuracy'] == 1 for r in field))
        self.assertTrue(all(r['identifies_full_source_span'] for r in field))
        shuffled = [r for r in rows if r['method'] == 'shuffled_field_fit']
        self.assertTrue(all(not r['fit_consistent'] for r in shuffled))


if __name__ == '__main__':
    unittest.main()
