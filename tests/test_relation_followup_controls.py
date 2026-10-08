import numpy as np

from experiments.answer_matched_geometry import pair_geometry
from experiments.paired_relation_geometry import exact_answer_code_ceiling, paired_metrics
from experiments.relation_observed_start import observed_start_score
from experiments.relation_operator_stability import gain_candidates, global_affine_generator, reflection_generator


def test_shared_affine_center_recovers_one_action_across_input_distributions():
    rng = np.random.default_rng(12)
    rho = np.eye(4)[[1, 0, 3, 2]]
    x = np.concatenate([rng.normal(size=(40, 4)), rng.normal(size=(40, 4)) + 3])
    y = x @ rho
    xv = rng.normal(size=(30, 4)) - 2
    fitted, _ = global_affine_generator(x, y, xv, xv @ rho, [1e-10, 1e-6])
    np.testing.assert_allclose(xv @ fitted[0] + fitted[1], xv @ rho, atol=1e-7)


def test_reflection_imposes_orthogonality_and_affine_involution():
    rng = np.random.default_rng(23)
    x, y = rng.normal(size=(90, 7)), rng.normal(size=(90, 7))
    rho, bias = reflection_generator(np.concatenate([x, y]), np.concatenate([y, x]))
    np.testing.assert_allclose(rho.T @ rho, np.eye(7), atol=1e-12)
    np.testing.assert_allclose(rho @ rho, np.eye(7), atol=1e-12)
    np.testing.assert_allclose(bias @ rho + bias, np.zeros(7), atol=1e-12)


def test_gain_selection_does_not_keep_unidentified_high_gain():
    x = np.column_stack([np.zeros(50), np.arange(50)])
    rho = np.diag([7., 1.])
    y = x @ rho
    limited, uncapped, metadata = gain_candidates(rho, x, y, x, y, [1, 2, 4, 8], .05)
    assert metadata['selected_cap'] == 1
    assert np.linalg.svd(limited[0], compute_uv=False)[0] <= 1 + 1e-12
    np.testing.assert_allclose(x @ limited[0] + limited[1], y, atol=1e-12)


def test_observed_start_predictions_never_use_withheld_state_features():
    rng = np.random.default_rng(34)
    h = rng.normal(size=(2, 8, 4))
    labels = np.ones((2, 8), dtype=np.int64)
    labels[:, 2] = labels[:, 5] = [2, 3]
    data = {'split': np.array([3, 3]), 'lengths': np.array([10, 10]), 'pair_ids': np.array([0, 0]), 'labels': labels}
    archive = {'rho_c': rng.normal(size=(4, 4)), 'rho_i': rng.normal(size=(4, 4)),
        'bias_c': rng.normal(size=4), 'bias_i': rng.normal(size=4),
        'readout_weight': rng.normal(size=(31, 4)), 'readout_bias': rng.normal(size=31)}
    # The scorer also loops through iid; provide two noncollision iid rows.
    h = np.concatenate([h, rng.normal(size=(2, 8, 4))])
    data = {key: np.concatenate([value, value]) for key, value in data.items()}
    data['split'][2:] = 2
    _, original = observed_start_score(h, data, archive)
    altered = h.copy()
    altered[:, [2, 3, 5, 6, 7]] = 100 * rng.normal(size=(4, 5, 4))
    _, changed = observed_start_score(altered, data, archive)
    for key in original:
        np.testing.assert_array_equal(original[key], changed[key])


def test_pair_logit_assignment_cancels_any_shared_class_prior():
    rng = np.random.default_rng(45)
    p, source, target = [rng.normal(size=(8, 5)) for _ in range(3)]
    labels = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    pair_ids = np.repeat(np.arange(4), 2)
    weights = rng.normal(size=(31, 5))
    a = paired_metrics(p, source, target, labels, pair_ids, weights, np.zeros(31))
    b = paired_metrics(p, source, target, labels, pair_ids, weights, rng.normal(size=31) * 100)
    assert a['paired_two_candidate_logit_assignment_accuracy'] == b['paired_two_candidate_logit_assignment_accuracy']
    np.testing.assert_allclose(a['median_signed_two_candidate_logit_contrast'], b['median_signed_two_candidate_logit_contrast'], atol=1e-12)


def test_answer_code_only_pair_predictions_tie_even_with_confident_logits():
    rng = np.random.default_rng(56)
    prediction = np.repeat(rng.normal(size=(4, 5)), 2, axis=0)
    source, target = [rng.normal(size=(8, 5)) for _ in range(2)]
    labels = np.arange(1, 9)
    result = paired_metrics(prediction, source, target, labels, np.repeat(np.arange(4), 2), rng.normal(size=(31, 5)), rng.normal(size=31))
    assert result['paired_two_candidate_logit_assignment_accuracy'] == .5
    assert result['pair_target_nmse'] == 1


def test_empirical_answer_lookup_upper_bound_can_be_tighter_than_half():
    labels = np.ones((6, 8), dtype=np.int64)
    labels[:, 5] = [1, 2, 2, 3, 3, 4]
    assert exact_answer_code_ceiling(labels, np.repeat(10, 6)) == 1 / 3


def test_answer_matched_pair_geometry_cancels_arbitrary_nonlinear_answer_codes():
    rng = np.random.default_rng(67)
    hidden = rng.normal(size=(6, 8, 4))
    pair_ids = np.repeat(np.arange(3), 2)
    archive = {'basis': np.eye(4), 'map_c': rng.normal(size=(4, 4)), 'map_r': rng.normal(size=(4, 4)), 'map_i': rng.normal(size=(4, 4))}
    # Any shared per-pair, per-state vector can be a nonlinear answer code.
    answer_code = np.repeat(rng.normal(size=(3, 8, 4)) * 100, 2, axis=0)
    original, arrays = pair_geometry(hidden, pair_ids, archive, ['c', 'ci'])
    changed, changed_arrays = pair_geometry(hidden + answer_code, pair_ids, archive, ['c', 'ci'])
    for a, b in zip(original, changed):
        for key in ['pair_action_displacement_nmse', 'pair_target_nmse', 'pair_assignment_accuracy']:
            np.testing.assert_allclose(a[key], b[key], atol=1e-10)
    for key in arrays:
        np.testing.assert_allclose(arrays[key], changed_arrays[key], atol=1e-6)
