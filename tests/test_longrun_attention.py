import copy
from pathlib import Path
import sys
import unittest
import torch
from experiments.longrun_attention import accelerate

sys.path.insert(0, str(Path('external/neurips/src').resolve()))
from neurips_permutations.models import CausalTransformer


class LongRunAttentionTests(unittest.TestCase):
    def test_logits_and_gradients_match_original_with_padding(self):
        torch.manual_seed(71)
        original = CausalTransformer(30, 16, d_model=16, layers=2, n_heads=2, dropout=0.)
        faster = accelerate(copy.deepcopy(original))
        inputs = torch.randint(0, 30, (3, 12))
        mask = torch.ones_like(inputs, dtype=torch.bool)
        mask[1, 8:] = False
        mask[2] = False
        a, b = original(inputs, mask), faster(inputs, mask)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        a.square().sum().backward()
        b.square().sum().backward()
        for (name_a, parameter_a), (name_b, parameter_b) in zip(original.named_parameters(), faster.named_parameters()):
            self.assertEqual(name_a, name_b)
            torch.testing.assert_close(parameter_a.grad, parameter_b.grad, rtol=1e-4, atol=1e-5)

    def test_future_tokens_cannot_change_prefix_logits(self):
        torch.manual_seed(7)
        model = accelerate(CausalTransformer(30, 16, d_model=16, layers=2, n_heads=2, dropout=0.)).eval()
        x = torch.randint(0, 30, (1, 12))
        y = x.clone()
        y[:, 7:] = (y[:, 7:] + 1) % 30
        with torch.no_grad():
            torch.testing.assert_close(model(x)[:, :7], model(y)[:, :7], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
