import unittest
import numpy as np
import torch

from experiments.longrun_attention import accelerate
from experiments.longrun_geometry import point_kernel_cka, pooled_prefix_features
from experiments.analysis import linear_cka
from neurips_permutations.models import CausalTransformer


class LongRunGeometryTests(unittest.TestCase):
    def test_point_kernel_cka_equals_feature_kernel_cka_for_wide_features(self):
        rng = np.random.default_rng(17)
        left, right = rng.normal(size=(12, 30)), rng.normal(size=(12, 25))
        self.assertAlmostEqual(point_kernel_cka(left, right), linear_cka(left, right), places=12)

    def test_masked_pooling_ignores_additional_padding_without_labels(self):
        torch.manual_seed(17)
        model = accelerate(CausalTransformer(30, 32, d_model=16, layers=2, n_heads=2))
        prefix = torch.randint(1, 30, (2, 6))
        short = {'x_input': torch.cat([prefix, torch.zeros(2, 2, dtype=torch.long)], dim=1)}
        long = {'x_input': torch.cat([prefix, torch.zeros(2, 8, dtype=torch.long)], dim=1)}
        a = pooled_prefix_features(model, short, 'x', {'<PAD>': 0})
        b = pooled_prefix_features(model, long, 'x', {'<PAD>': 0})
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)


if __name__ == '__main__': unittest.main()
