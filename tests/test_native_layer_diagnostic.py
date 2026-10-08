from pathlib import Path
import sys
import unittest

import torch

from experiments.longrun_attention import accelerate
from experiments.native_layer_diagnostic import activation_gradients

sys.path.insert(0, str(Path('external/neurips/src').resolve()))
from neurips_permutations.models import CausalTransformer


class NativeLayerDiagnosticTests(unittest.TestCase):
    def test_answer_loss_uses_earlier_prefix_but_not_final_prefix_activations(self):
        torch.manual_seed(124)
        model = accelerate(CausalTransformer(30, 16, d_model=16, layers=2, n_heads=2, dropout=0.))
        inputs = torch.randint(0, 30, (3, 12))
        gradients, loss, parameter_norm = activation_gradients(
            model, inputs, torch.ones_like(inputs, dtype=torch.bool), torch.full((3,), 11), torch.tensor([1, 2, 3]))
        for name in ('block_2', 'final_norm'):
            self.assertEqual(gradients[name][:, :11].count_nonzero().item(), 0)
            self.assertGreater(gradients[name][:, 11].abs().sum().item(), 0)
        self.assertGreater(gradients['block_1'][:, :11].abs().sum().item(), 0)
        self.assertGreater(parameter_norm, 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == '__main__': unittest.main()
