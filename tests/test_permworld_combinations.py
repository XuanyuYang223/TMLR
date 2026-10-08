import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

from experiments.permworld_combinations import generate_data, new_model, prompts, select_groups

sys.path.insert(0, str(Path('external/neurips/src').resolve()))
from neurips_permutations.math_ops import PROPERTY32_TASK_NAMES, PROPERTY_FUNCTIONS
from neurips_permutations.passage import TOKEN_TO_ID, passage_tokens, one_line_tokens


class PermWorldCombinationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('configs/permworld_combinations.json').read_text())

    def test_groups_preserve_task_count_exclude_targets_and_match_affine_edges(self):
        groups = select_groups(self.config)
        self.assertEqual(len(groups), 8)
        self.assertEqual(len({frozenset(g['tasks']) for g in groups}), 8)
        for group in groups:
            self.assertEqual(len(group['tasks']), 4)
            self.assertFalse(set(group['tasks']) & set(self.config['target_tasks']))
            self.assertEqual(int(group['selection_statistics']['minimum_known_constraint_size']), group['minimum_constraint_size'])
        for family in ('position', 'cycle', 'interior'):
            self.assertEqual(len({g['selection_statistics']['expanded_affine_pair_count'] for g in groups if g['family'] == family}), 1)

    def test_prompts_match_upstream_passage_and_contain_no_answers(self):
        permutations = [tuple(range(1, 11)), tuple(range(12, 0, -1))]
        width = 2 * 12 + 4
        prefixes = []
        for permutation in permutations:
            tokens = ['<BOS>', '<SIZE>', f'{len(permutation):02d}', *one_line_tokens(permutation)]
            encoded = [TOKEN_TO_ID[t] for t in tokens]
            prefixes.append(encoded + [TOKEN_TO_ID['<PAD>']] * (width - len(encoded)))
        tasks = ['fixed_points', 'exceedances']
        ids, mask, positions = prompts(torch.tensor(prefixes), torch.tensor([10, 12]), tasks, TOKEN_TO_ID)
        for k, task in enumerate(tasks):
            for j, permutation in enumerate(permutations):
                expected = passage_tokens(task, permutation, PROPERTY_FUNCTIONS[task](permutation))[:-2]
                actual = ids[k * 2 + j][mask[k * 2 + j]].tolist()
                self.assertEqual(actual, [TOKEN_TO_ID[t] for t in expected])
                self.assertEqual(ids[k * 2 + j, positions[k * 2 + j]], TOKEN_TO_ID['='])

    def test_generation_has_globally_disjoint_inputs_and_exact_labels(self):
        config = {**self.config, 'lengths': [10, 11], 'examples_per_length': {k: 2 for k in self.config['examples_per_length']}}
        with tempfile.TemporaryDirectory() as temp:
            data = generate_data(config, Path(temp), list(PROPERTY32_TASK_NAMES), PROPERTY_FUNCTIONS, TOKEN_TO_ID, one_line_tokens)
            inputs = [tuple(row) for split in config['examples_per_length'] for row in data[f'{split}_input']]
            self.assertEqual(len(inputs), len(set(inputs)))
            for split in config['examples_per_length']:
                for row, n, labels in zip(data[f'{split}_input'], data[f'{split}_lengths'], data[f'{split}_labels']):
                    permutation = tuple(map(int, row[4:2 * n + 3:2]))
                    self.assertEqual(sorted(permutation), list(range(1, n + 1)))
                    self.assertEqual(labels.tolist(), [PROPERTY_FUNCTIONS[name](permutation) for name in PROPERTY32_TASK_NAMES])

    def test_one_end_hidden_state_cannot_see_later_task_token(self):
        config = {**self.config, 'd_model': 16, 'layers': 1, 'heads': 2}
        model = new_model(config, 17, 'cpu').eval()
        permutation = tuple(range(1, 11))
        tokens = ['<BOS>', '<SIZE>', '10', *one_line_tokens(permutation)]
        prefix = torch.tensor([[TOKEN_TO_ID[t] for t in tokens]])
        ids, mask, _ = prompts(prefix, torch.tensor([10]), ['fixed_points', 'exceedances'], TOKEN_TO_ID)
        with torch.no_grad():
            h, valid = model._embed_inputs(ids, mask)
            for block in model.blocks:
                h = block(h, valid)
            np.testing.assert_allclose(h[0, len(tokens) - 1], h[1, len(tokens) - 1], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
