import numpy as np
import torch

from experiments.operator_capacity_confirmation import MatchedResidual, losses, crossed_interval


def test_parameter_matched_identity_and_gelu_start_at_same_frozen_operator():
    a = np.arange(16, dtype=np.float32).reshape(4, 4)/10
    b = np.arange(4, dtype=np.float32)
    models = []
    for nonlinear in [False, True]:
        torch.manual_seed(73)
        models.append(MatchedResidual(a, b, 8, nonlinear))
    counts = [sum(p.numel() for p in m.parameters()) for m in models]
    assert counts == [76, 76]
    for (key, p), (other_key, q) in zip(models[0].state_dict().items(), models[1].state_dict().items()):
        assert key == other_key
        torch.testing.assert_close(p, q, atol=0, rtol=0)
    h = torch.randn(5, 4)
    for m in models:
        torch.testing.assert_close(m(h), h@torch.from_numpy(a)+torch.from_numpy(b), atol=0, rtol=0)
    assert not models[0].a.requires_grad and not models[0].bias.requires_grad


def test_identity_two_layer_residual_is_affine_even_after_training():
    model = MatchedResidual(np.eye(4), np.zeros(4), 8, False)
    with torch.no_grad():
        model.layers[-1].weight.normal_(); model.layers[-1].bias.normal_()
    h = torch.randn(11, 4)
    w1, w2 = model.layers[0].weight, model.layers[2].weight
    b1, b2 = model.layers[0].bias, model.layers[2].bias
    a = model.a+w1.T@w2.T
    b = model.bias+b1@w2.T+b2
    torch.testing.assert_close(model(h), h@a+b, atol=2e-6, rtol=2e-6)


def test_wrong_geometry_keeps_output_distillation_target_fixed():
    torch.manual_seed(12)
    pred, truth, wrong = torch.randn(7, 4), torch.randn(7, 4), torch.randn(7, 4)
    w, b, output = torch.randn(3, 4), torch.randn(3), torch.randn(7, 3)
    correct = losses(pred, output, truth, w, b, 1.3, 2.)
    mismatch = losses(pred, output, wrong, w, b, 1.3, 2.)
    torch.testing.assert_close(correct[0], mismatch[0], atol=0, rtol=0)
    assert correct[1] != mismatch[1]


def test_crossed_bootstrap_respects_paired_source_and_example_effects():
    constant = np.full((3, 7), 4.5)
    assert crossed_interval(constant, 100, 23) == [4.5, 4.5]
    lo, hi = crossed_interval(np.array([[-2.]*7, [0.]*7, [2.]*7]), 1000, 23)
    assert lo <= 0 <= hi
