import unittest
import numpy as np
import torch
from experiments.algebra import audit, composition_size, projective, rank_mod, scenarios, target_pool, world
from experiments.analysis import grouped_predictions, linear_cka
from experiments.models import Encoder, BatchedTargets
from experiments.run import balanced_splits
from experiments.permutation_audit import transform
from experiments.diagnostics import symbolic_composition


class ProtocolTests(unittest.TestCase):
    def test_finite_field_rank_is_not_real_rank(self):
        matrix = np.array([[1, 2], [3, 1]])
        self.assertEqual(np.linalg.matrix_rank(matrix), 2)
        self.assertEqual(rank_mod(matrix, 5), 1)

    def test_matched_statistics_but_different_minimal_dependencies(self):
        groups = scenarios()
        targets = target_pool(groups, 5, 10)
        result = audit(groups, targets, 5)
        for row in result['scenarios']:
            self.assertEqual(row['rank'], 3)
            self.assertEqual(row['joint_support'], 125)
            self.assertTrue(row['uniform_marginals'])
            self.assertAlmostEqual(row['max_pairwise_mi_nats'], 0)
            self.assertEqual(row['minimum_circuit_size'], 3 if row['family'] == 'A' else 4)
        keys = {tuple(sorted(projective(v, 5) for v in g['sources'])) for g in groups}
        self.assertEqual(len(keys), len(groups), 'scalar-equivalent combinations would leak between CV folds')

    def test_target_pool_excludes_equivalent_pretraining_tasks(self):
        groups = scenarios()
        excluded = {projective(v, 5) for g in groups for v in g['sources']}
        targets = target_pool(groups, 5, 10)
        self.assertTrue(all(projective(v, 5) not in excluded for v in targets))
        for g in groups:
            self.assertTrue(all(composition_size(g['sources'], v, 5) in (2, 3) for v in targets))

    def test_input_scrambling_preserves_uniform_latent_world(self):
        _, latent, encoded, basis = world(5, 4, 17)
        self.assertEqual(rank_mod(basis, 5), 4)
        unique, counts = np.unique(latent, axis=0, return_counts=True)
        self.assertEqual(len(unique), 125)
        self.assertTrue(np.all(counts == 5))
        self.assertTrue(np.all(encoded.sum(axis=1) == 4))

    def test_nested_support_is_balanced_and_disjoint_from_test(self):
        _, latent, _, _ = world(5, 4, 17)
        labels = latent[:, :2]
        test, support = balanced_splits(labels, [10, 25, 50], 25, 5, 42)
        for t in range(2):
            previous = set()
            for budget in [10, 25, 50]:
                current = set(support[budget][t])
                self.assertTrue(previous <= current)
                self.assertFalse(current & set(test[t]))
                self.assertTrue(np.all(np.bincount(labels[support[budget][t], t], minlength=5) == budget // 5))
                previous = current

    def test_batched_adaptations_copy_encoder_and_are_independent(self):
        torch.manual_seed(17)
        encoder = Encoder(20, 16, 8)
        model = BatchedTargets(encoder, 2, 5, seed=42)
        x = torch.randn(4, 20)
        torch.testing.assert_close(model.encode(x)[0], encoder(x))
        torch.testing.assert_close(model.encode(x)[1], encoder(x))
        model(x)[0].square().sum().backward()
        self.assertEqual(model.backbone['w1'].grad[1].abs().sum().item(), 0)
        self.assertGreater(model.backbone['w1'].grad[0].abs().sum().item(), 0)

    def test_cka_translation_and_orthogonal_invariance(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(30, 8))
        q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        self.assertAlmostEqual(linear_cka(x, x @ q + 10), 1, places=12)

    def test_entire_combination_holdout_removes_all_seed_rows(self):
        rows = []
        for scenario in range(3):
            for world_seed in [17, 42]:
                for target_id in range(2):
                    rows.append({'scenario': str(scenario), 'world_seed': world_seed, 'target_id': target_id,
                                 'source_auc': .9, 'source_accuracy': 1., 'mean_single_auc': .9,
                                 'minimum_circuit_size': 3 + scenario % 2,
                                 'target_composition_size': 2 + target_id,
                                 'transfer_gain': .1 * scenario + .01 * target_id})
        result = grouped_predictions(rows, {'prediction_alpha': 10})
        self.assertEqual([fold['count'] for fold in result['baseline']['folds']], [4, 4, 4])
        self.assertEqual(len(result['baseline']['predictions']), 12)

    def test_permutation_transform_composition(self):
        value = (3, 1, 4, 2)
        for name in ['inverse', 'complement', 'reverse']:
            self.assertEqual(transform(transform(value, name), name), value)
        self.assertEqual(transform(value, 'inverse_after_complement'), transform(transform(value, 'complement'), 'inverse'))

    def test_symbolic_diagnostic_undoes_randomized_head_order(self):
        native = np.array([[1, 2, 3, 3], [4, 0, 2, 4]])
        order = np.array([3, 1, 0, 2])
        targets = np.array([[1, 2, 1], [0, 1, 3]])
        predicted = symbolic_composition(native[:, order], order, targets, 5)
        np.testing.assert_array_equal(predicted, native[:, :3] @ targets.T % 5)


if __name__ == '__main__':
    unittest.main()
