import torch
from torch import nn

from experiments.hidden_relation_train import AffineOperators
from experiments import two_step_relation_joint as joint


def toy_batch(monkeypatch):
    torch.manual_seed(17)
    model = nn.Module()
    model.encoder = nn.Linear(4, 4)
    model.lm_head = nn.Linear(4, 31, bias=False)
    model.lm_head.weight.requires_grad_(False)
    def query(model, inputs, lengths, task, tokens):
        return model.encoder(inputs[:, :4].float())
    monkeypatch.setattr(joint, 'query_hidden', query)
    x = torch.randint(1, 9, (6, 3, 8)); n = torch.full((6,), 12)
    y = torch.randint(0, 31, (6, 3)); teacher = torch.randn(6, 3, 31)
    return model, AffineOperators(4), x, n, y, teacher, torch.arange(6)


def test_joint_pairing_changes_only_geometry_at_shared_parameters(monkeypatch):
    model, ops, x, n, y, teacher, ids = toy_batch(monkeypatch)
    correct = joint.joint_losses(model, ops, x, n, y, teacher, ids, ids, ids, 'task', {}, 2., 1.)
    wrong = joint.joint_losses(model, ops, x, n, y, teacher, ids, ids.roll(1), ids.roll(2), 'task', {}, 2., 1.)
    for a, b in zip(correct[:3], wrong[:3]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert not torch.isclose(correct[3], wrong[3])


def test_joint_loss_trains_encoder_and_operators_with_frozen_readout(monkeypatch):
    model, ops, x, n, y, teacher, ids = toy_batch(monkeypatch)
    losses = joint.joint_losses(model, ops, x, n, y, teacher, ids, ids, ids, 'task', {}, 2., 1.)
    sum(losses).backward()
    assert model.encoder.weight.grad is not None and model.encoder.weight.grad.norm() > 0
    assert ops.offset.grad is not None and ops.offset.grad.norm() > 0
    assert model.lm_head.weight.grad is None
