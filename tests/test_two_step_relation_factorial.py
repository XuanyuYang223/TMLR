from itertools import permutations

import numpy as np
import torch

from experiments.hidden_relation_train import AffineOperators
from experiments.two_step_relation_factorial import (
    apply_numpy, epoch_schedule, factorial_pairings, independent_orbit, supervision_losses,
)


def test_wrong_pairings_preserve_visible_answers_and_cannot_cancel():
    n = np.repeat([12, 18, 24, 30], [3, 4, 2, 1])
    labels = np.repeat([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]], [3, 4, 2, 1], axis=0)
    pc, pi, keep = factorial_pairings(n, labels, 17)
    ids = np.flatnonzero(keep)
    assert len(ids) == 7  # size-two strata would force double-wrong cancellation
    for partner in [pc, pi]:
        assert np.all(partner[keep] != ids)
        np.testing.assert_array_equal(labels[partner[keep]], labels[keep])
        np.testing.assert_array_equal(n[partner[keep]], n[keep])
        np.testing.assert_array_equal(np.sort(partner[keep]), ids)
    assert np.all(pc[pi[keep]] != ids)
    assert np.all(pi[pc[keep]] != ids)


def test_complete_epochs_match_every_input_exposure_under_pairings():
    n = np.repeat([12, 18], [7, 9]); y = np.ones((len(n), 3), dtype=int)
    pc, pi, keep = factorial_pairings(n, y, 42)
    counts = np.zeros(len(n), dtype=int)
    for epoch, ids in epoch_schedule(n, keep, 5, 4, 101):
        assert len(set(n[ids])) == 1
        counts[ids] += 1
    np.testing.assert_array_equal(counts, np.full(len(n), 5))
    np.testing.assert_array_equal(counts[pc], counts)
    np.testing.assert_array_equal(counts[pi], counts)


def test_only_geometric_loss_changes_when_pairings_change():
    torch.manual_seed(42)
    h = torch.randn(6, 3, 4); weights = torch.randn(7, 4); bias = torch.randn(7)
    centers = torch.randn(6, 4); teacher = (h + centers[:, None]) @ weights.T + bias
    ops = AffineOperators(4); ids = torch.arange(6); pc = ids.roll(1); pi = ids.roll(2)
    correct_geometry, correct_output = supervision_losses(ops, h, ids, ids, ids, centers, teacher, weights, bias, 2., 1.)
    wrong_geometry, wrong_output = supervision_losses(ops, h, ids, pc, pi, centers, teacher, weights, bias, 2., 1.)
    torch.testing.assert_close(correct_output, wrong_output, rtol=0, atol=0)
    assert not torch.isclose(correct_geometry, wrong_geometry)


def test_prediction_api_composes_in_input_order_without_target_access():
    x = np.array([[2., 3.]])
    maps = {'rho_c': np.array([[0., 1.], [1., 0.]]), 'bias_c': np.array([1., 0.]),
            'rho_i': np.diag([2., 3.]), 'bias_i': np.array([0., -1.])}
    expected = ((x @ maps['rho_c'] + maps['bias_c']) @ maps['rho_i'] + maps['bias_i'])
    np.testing.assert_array_equal(apply_numpy(x, 'ci', maps), expected)
    assert not np.array_equal(apply_numpy(x, 'ci', maps), apply_numpy(x, 'ic', maps))


def test_ic_answer_is_visible_but_ci_can_require_missing_information():
    def record_maximum(p):
        best = 0; count = 0
        for value in p:
            if value > best:
                count += 1; best = value
        return count
    answers = {}
    found_collision = False
    for p in permutations(range(1, 7)):
        orbit = independent_orbit(p)
        assert record_maximum(orbit[6]) == record_maximum(orbit[1])
        key = tuple(record_maximum(orbit[j]) for j in [0, 1, 4])
        answer = record_maximum(orbit[5])
        if key in answers and answers[key] != answer:
            found_collision = True
        answers[key] = answer
    assert found_collision


def test_independent_audit_normalizes_saved_numpy_integer_entries():
    orbit = independent_orbit(np.array([2, 4, 1, 3], dtype=np.int64))
    assert all(type(value) is int for p in orbit for value in p)
