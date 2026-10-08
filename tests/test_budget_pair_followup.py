import numpy as np
import pytest

from experiments.budget_matched_geometry import (
    decoder_fit, residual_update_error, encoder_matrix, reconstruction_inputs,
)
from experiments.collision_pair_followup import pair_statistics


def test_fixed_input_encoder_and_normalization_ignore_heldout_values():
    rng = np.random.default_rng(73)
    h = rng.normal(size=(12, 8, 14))
    split = np.array([0] * 6 + [1] * 3 + [2] * 3)
    lengths = np.full(12, 15)
    changed = h.copy()
    changed[split != 0] = rng.normal(size=(6, 8, 14)) * 1000
    q = encoder_matrix(h, split, lengths, 'fixed_random', 5, 712)
    q_changed = encoder_matrix(changed, split, lengths, 'fixed_random', 5, 712)
    np.testing.assert_array_equal(q, q_changed)
    _, fit = reconstruction_inputs(h, split, q)
    _, changed_fit = reconstruction_inputs(changed, split, q_changed)
    for key in ['mean', 'scale']:
        np.testing.assert_array_equal(fit[key], changed_fit[key])


def test_low_rank_update_error_matches_explicit_full_prediction():
    rng = np.random.default_rng(17)
    reconstruction, matrix = rng.normal(size=(23, 19)), rng.normal(size=(7, 7))
    q, _ = np.linalg.qr(rng.normal(size=(19, 7)), mode='reduced')
    z = reconstruction @ q
    target = rng.normal(size=(23, 19))
    error = residual_update_error(reconstruction, z, matrix, target, target @ q,
        float(np.square(reconstruction).sum()), float(np.square(target).sum()))
    prediction = reconstruction + (z @ matrix - z) @ q.T
    np.testing.assert_allclose(error, np.square(prediction - target).sum(), rtol=1e-13)


def test_decoder_has_unpenalized_intercept_for_constant_target():
    rng = np.random.default_rng(5)
    x, xv = rng.normal(size=(33, 7)), rng.normal(size=(11, 7))
    truth = np.array([2.3, -7.1, 14.9])
    coefficient, info = decoder_fit(x, np.tile(truth, (33, 1)), xv, np.tile(truth, (11, 1)), [1e-6, 1.])
    np.testing.assert_allclose(coefficient[:-1], 0, atol=1e-12)
    np.testing.assert_allclose(coefficient[-1], truth, atol=1e-12)
    assert info['effective_ridge_df'] <= 8


def test_pair_double_correctness_does_not_imply_above_half_accuracy():
    truth = np.array([3, 7, 3, 7])
    prediction = np.array([3, 7, 7, 3])
    stats, _ = pair_statistics(prediction, truth, np.array([0, 0, 1, 1]))
    assert stats['both_correct_fraction'] == .5
    assert stats['neither_correct_fraction'] == .5
    assert stats['answer_accuracy'] == .5
    assert stats['orientation_excess_all_pairs'] == 0
    assert stats['correct_orientation_given_covered_candidates'] == .5


def test_equal_known_code_predictions_cannot_answer_both_pair_members():
    stats, arrays = pair_statistics(np.array([3, 7, 3, 7]), np.array([3, 3, 7, 7]), np.array([4, 5, 4, 5]))
    assert stats['both_correct_count'] == 0
    assert stats['exactly_one_correct_fraction'] == 1
    assert stats['answer_accuracy'] == .5
    assert stats['correct_orientation_given_covered_candidates'] is None
    np.testing.assert_array_equal(arrays['pair_ids'], [4, 5])


def test_collision_pair_score_rejects_noncollision_targets():
    with pytest.raises(ValueError, match='must differ'):
        pair_statistics(np.array([3, 3]), np.array([3, 3]), np.array([7, 7]))
