"""Held-out linear action probes without imposing algebraic identities.

All feature vectors are ROW vectors. A word (a,b) acts first by a, then b,
and is predicted by z @ W_a @ W_b. This avoids silently reversing the usual
column-vector convention for a group representation.
"""
from itertools import permutations
import numpy as np

from .permutation_audit import TRANSFORMS, transform


ACTION_NAMES = tuple(TRANSFORMS)
GENERATOR_WORDS = {'c': 'complement', 'r': 'reverse', 'i': 'inverse'}


def permutation_action_table():
    values = list(permutations(range(1, 6)))
    lookup = {p: i for i, p in enumerate(values)}
    maps = np.array([[lookup[transform(p, name)] for p in values] for name in ACTION_NAMES])
    signatures = {tuple(m): i for i, m in enumerate(maps)}
    assert len(signatures) == 8
    # table[a,b] is b after a.
    table = np.array([[signatures[tuple(maps[b, maps[a]])] for b in range(8)] for a in range(8)])
    assert np.array_equal(table[0], np.arange(8)) and np.array_equal(table[:, 0], np.arange(8))
    for a in range(8):
        for b in range(8):
            assert np.array_equal(maps[table[a, b]], maps[b, maps[a]])
    c, r, i = [ACTION_NAMES.index(GENERATOR_WORDS[g]) for g in ('c', 'r', 'i')]
    assert table[c, c] == table[r, r] == table[i, i] == 0
    assert table[c, r] == table[r, c]
    assert table[table[i, c], i] == r
    assert table[c, i] != table[i, c]
    return table


def orbit_residuals(features):
    """[orbit, action, feature]; the mean is local to each whole orbit."""
    features = np.asarray(features, dtype=np.float64)
    return features - features.mean(axis=1, keepdims=True)


def pca_basis(fit, dimension):
    fit = np.asarray(fit, dtype=np.float64)
    _, singular, vectors = np.linalg.svd(fit.T @ fit, hermitian=True)
    if not len(singular) or singular[0] <= 1e-20:
        return np.empty((fit.shape[1], 0))
    rank = int(np.sum(singular > singular[0]*1e-10))
    return vectors[:min(dimension, rank)].T


def ridge_map(x, y, alpha):
    gram = x.T @ x
    scale = np.trace(gram) / max(1, len(gram))
    return np.linalg.solve(gram + alpha*max(scale, 1e-20)*np.eye(len(gram)), x.T @ y)


def nmse(predicted, truth):
    denominator = float(np.square(truth).sum())
    return float(np.square(predicted-truth).sum()/denominator) if denominator > 1e-20 else None


def apply_word(x, word, maps):
    prediction = x
    for generator in word:
        prediction = prediction @ maps[generator]
    return prediction


def word_action(word, table, letters):
    result = 0
    for generator in word:
        result = int(table[result, letters[generator]])
    return result


def shuffled_rows(lengths, rng):
    order = np.arange(len(lengths))
    for length in np.unique(lengths):
        ids = np.flatnonzero(lengths == length)
        order[ids] = rng.permutation(ids)
    return order


def within_strata(features, strata):
    result = np.asarray(features, dtype=np.float64).copy()
    for value in np.unique(strata):
        ids = strata == value
        result[ids] -= result[ids].mean(0)
    return result


def cka(left, right):
    # Computing in feature space avoids an unnecessarily large point kernel.
    left, right = left-left.mean(0), right-right.mean(0)
    numerator = np.square(left.T @ right).sum()
    denominator = np.sqrt(np.square(left.T @ left).sum()*np.square(right.T @ right).sum())
    return float(numerator/denominator) if denominator > 1e-20 else None


def group_probe(features, split, strata, table, generator_names, words,
                dimension=64, ridge_grid=(1e-6, 1e-4, .01, 1.), seed=0):
    """Learn only generators on fit orbits, select ridge on validation orbits.

    features includes a complete action orbit for every input anchor. All
    states of every orbit are in one split. Derived word actions are NEVER
    regression targets. The random-pair control uses the same held-out truth
    and a separately validation-selected ridge parameter.
    """
    h = np.asarray(features, dtype=np.float64)
    residual = orbit_residuals(h)
    split, strata = np.asarray(split), np.asarray(strata)
    assert len(split) == len(h) and len(strata) == len(h)
    fit = residual[split == 0].reshape(-1, h.shape[-1])
    basis = pca_basis(fit, dimension)
    centered = within_strata(h.reshape(-1, h.shape[-1]), np.repeat(strata, h.shape[1]))
    energy = float(np.square(residual).sum()/max(np.square(centered).sum(), 1e-20))
    result = {'status': 'complete' if basis.shape[1] else 'zero_action_variance',
              'probe_dimension': basis.shape[1], 'orbit_sensitive_energy_fraction': energy,
              'generators': [], 'composites': [], 'laws': [], 'wrong_order': [], 'cka': []}
    if not basis.shape[1]: return result, {}
    z = residual @ basis
    train, validation, test = [z[split == k].reshape(-1, basis.shape[1]) for k in (0, 1, 2)]
    test_full = residual[split == 2].reshape(-1, h.shape[-1])
    result['test_pca_energy_fraction'] = float(np.square(test).sum()/np.square(test_full).sum())
    maps, wrong_maps = {}, {}
    rng = np.random.default_rng(seed)
    train_strata = np.repeat(strata[split == 0], h.shape[1])
    val_strata = np.repeat(strata[split == 1], h.shape[1])
    predictions = {}
    for letter, index in generator_names.items():
        changed = z[:, table[:, index], :]
        ytrain, yval, ytest = [changed[split == k].reshape(-1, basis.shape[1]) for k in (0, 1, 2)]
        candidates = [ridge_map(train, ytrain, a) for a in ridge_grid]
        errors = [nmse(validation @ w, yval) for w in candidates]
        chosen = int(np.argmin(errors)); maps[letter] = candidates[chosen]
        yshuffled = ytrain[shuffled_rows(train_strata, rng)]
        vshuffled = yval[shuffled_rows(val_strata, rng)]
        wrong_candidates = [ridge_map(train, yshuffled, a) for a in ridge_grid]
        chosen_wrong = int(np.argmin([nmse(validation @ w, vshuffled) for w in wrong_candidates]))
        wrong_maps[letter] = wrong_candidates[chosen_wrong]
        predicted = test @ maps[letter]
        result['generators'].append({'generator': letter, 'action': int(index),
            'alpha': ridge_grid[chosen], 'validation_nmse': errors[chosen],
            'test_nmse': nmse(predicted, ytest), 'identity_nmse': nmse(test, ytest),
            'shuffled_fit_nmse': nmse(test @ wrong_maps[letter], ytest),
            'full_space_nmse': nmse(predicted @ basis.T,
                                    residual[split == 2][:, table[:, index]].reshape(test_full.shape)),
            'shuffled_alpha': ridge_grid[chosen_wrong]})
        predictions[f'generator_{letter}'] = predicted.astype(np.float32)
    for word in words:
        action = word_action(word, table, generator_names)
        truth = z[split == 2][:, table[:, action]].reshape(test.shape)
        predicted = apply_word(test, word, maps)
        result['composites'].append({'word': ''.join(word), 'action': action,
            'test_nmse': nmse(predicted, truth), 'identity_nmse': nmse(test, truth),
            'shuffled_fit_nmse': nmse(apply_word(test, word, wrong_maps), truth),
            'full_space_nmse': nmse(predicted @ basis.T,
                residual[split == 2][:, table[:, action]].reshape(test_full.shape))})
        predictions[f'composite_{"".join(word)}'] = predicted.astype(np.float32)
    # Law residuals also retain actual input/output prediction error. A small
    # discrepancy between two zero maps is not evidence of equivariance.
    laws = [('cc', (), 'c involution'), ('rr', (), 'r involution'),
            ('ii', (), 'i involution'), ('cr', 'rc', 'commuting c and r'),
            ('ici', 'r', 'conjugating c by i')]
    if set(generator_names) == {'a', 's'}:
        laws = [('aaaa', (), 'rotation order four'), ('ss', (), 'reflection involution'),
                ('sas', 'aaa', 'reflection conjugates rotation to inverse')]
    for left, right, name in laws:
        if not set(left).union(right).issubset(maps): continue
        pleft, pright = apply_word(test, left, maps), apply_word(test, right, maps)
        action = word_action(left, table, generator_names)
        assert action == word_action(right, table, generator_names)
        truth = z[split == 2][:, table[:, action]].reshape(test.shape)
        result['laws'].append({'name': name, 'left': ''.join(left), 'right': ''.join(right),
            'consistency_nmse': float(np.square(pleft-pright).sum()/np.square(test).sum()),
            'left_prediction_nmse': nmse(pleft, truth), 'right_prediction_nmse': nmse(pright, truth)})
    different = [('ci', 'ic')] if 'i' in maps else [('as', 'sa')]
    for correct, wrong in different:
        if not set(correct+wrong).issubset(maps): continue
        ca, wa = [word_action(w, table, generator_names) for w in (correct, wrong)]
        assert ca != wa
        prediction = apply_word(test, correct, maps)
        ctruth = z[split == 2][:, table[:, ca]].reshape(test.shape)
        wtruth = z[split == 2][:, table[:, wa]].reshape(test.shape)
        result['wrong_order'].append({'correct_word': correct, 'wrong_word': wrong,
            'correct_nmse': nmse(prediction, ctruth), 'wrong_nmse': nmse(prediction, wtruth),
            'gap_wrong_minus_correct': nmse(prediction, wtruth)-nmse(prediction, ctruth),
            'target_separation_nmse': nmse(wtruth, ctruth)})
    test_raw = h[split == 2]
    length_test = np.repeat(strata[split == 2], h.shape[1])
    base = within_strata(test_raw.reshape(-1, h.shape[-1]), length_test)
    for action in range(1, h.shape[1]):
        changed = within_strata(test_raw[:, table[:, action]].reshape(base.shape), length_test)
        result['cka'].append({'action': action, 'within_stratum_cka': cka(base, changed)})
    arrays = {'basis': basis.astype(np.float32), 'test_latent': test.astype(np.float32),
              **{f'map_{g}': w.astype(np.float32) for g, w in maps.items()}, **predictions}
    return result, arrays
