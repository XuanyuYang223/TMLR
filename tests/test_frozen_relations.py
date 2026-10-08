import numpy as np

from experiments.frozen_relation_followup import fit_maps
from experiments.relation_composition_diagnostics import decompose_word, measures, numeric_basis
from experiments.representation_algebra import permutation_action_table


def toy_known_states(seed=42):
    rng = np.random.default_rng(seed)
    c = np.eye(4)[[1, 0, 3, 2]]
    i = np.eye(4)[[3, 1, 2, 0]]
    def sample(count):
        x = rng.normal(size=(count, 4))
        return np.stack([x, x @ c, x @ i], axis=1)
    return sample(80), np.repeat([10, 11], 40), sample(40), np.repeat([10, 11], 20), c, i


def test_known_only_frozen_fit_recovers_noncommuting_maps():
    train, ns, val, nval, c, i = toy_known_states()
    # A single common centering vector preserves a shared affine action.
    # Different length-specific centers can induce different action biases.
    ns.fill(10)
    nval.fill(10)
    maps, means, alphas, errors, ids = fit_maps(train, ns, val, nval, 30, [1e-9, 1e-6], 123, False)
    rng = np.random.default_rng(456)
    x = rng.normal(size=(20, 4))
    center = means[10]
    first = (x - center) @ maps['c'][0] + maps['c'][1]
    composed = first @ maps['i'][0] + maps['i'][1] + center
    np.testing.assert_allclose(composed, x @ c @ i, atol=1e-6)
    assert np.linalg.norm(composed - x @ i @ c) > 1


def test_frozen_correct_shuffled_share_source_anchors_and_do_not_mutate_features():
    train, ns, val, nval, *_ = toy_known_states()
    before_train, before_val = train.copy(), val.copy()
    correct = fit_maps(train, ns, val, nval, 16, [1e-6, 1e-2], 123, False)
    shuffled = fit_maps(train, ns, val, nval, 16, [1e-6, 1e-2], 123, True)
    np.testing.assert_array_equal(correct[-1], shuffled[-1])
    for n in correct[1]:
        np.testing.assert_array_equal(correct[1][n], shuffled[1][n])
    np.testing.assert_array_equal(train, before_train)
    np.testing.assert_array_equal(val, before_val)
    assert np.linalg.norm(correct[0]['c'][0] - shuffled[0]['c'][0]) > .1


def test_step_error_decomposition_preserves_cross_terms():
    rng = np.random.default_rng(77)
    hidden = rng.normal(size=(17, 8, 5))
    centers = rng.normal(size=(17, 5))
    maps = {letter: (rng.normal(size=(5, 5)), rng.normal(size=5)) for letter in ['c', 'i']}
    table = permutation_action_table()
    prediction, components, stages = decompose_word(hidden, centers, maps, 'ici', table)
    final_action = stages[-1]['output_action']
    np.testing.assert_allclose(sum(components), prediction - hidden[:, final_action], atol=1e-12)
    assert not np.isclose(np.square(sum(components)).sum(), sum(np.square(c).sum() for c in components))


def test_readout_null_error_does_not_change_numeric_answers():
    rng = np.random.default_rng(99)
    weights = np.zeros((31, 4))
    weights[:, :2] = rng.normal(size=(31, 2))
    target = rng.normal(size=(25, 4))
    labels = (target @ weights.T).argmax(-1)
    prediction = target.copy()
    prediction[:, 2:] += 100 * rng.normal(size=(25, 2))
    result = measures(prediction, np.zeros_like(target), target, labels, weights, np.zeros(31), numeric_basis(weights))
    assert result['hidden_displacement_nmse'] > 1
    assert result['numeric_contrast_displacement_nmse'] < 1e-20
    assert result['answer_accuracy'] == 1
