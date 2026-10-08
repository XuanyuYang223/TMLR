"""The supplementary audit must reject plausible but incorrect reports."""
import copy

import numpy as np
import pytest

from experiments.overnight_inverse_addendum_verify import audit_bootstrap, counts


def grade():
    return {
        'accuracy': .6, 'modal_fraction': .4,
        'modal': {'count': 2, 'accuracy': .5},
        'nonmodal': {'count': 3, 'accuracy': 2 / 3}, 'per_length': {},
    }


def inputs():
    return (np.array([1, 0, 1, 1, 0]), np.ones(5, dtype=int),
            np.array([True, True, False, False, False]), np.array([], dtype=int))


def test_subgroup_audit_rejects_correct_total_with_wrong_subgroups():
    correct = grade()
    assert counts(correct, *inputs()) == 6
    wrong = copy.deepcopy(correct)
    wrong['modal']['accuracy'] = 1.0
    wrong['nonmodal']['accuracy'] = 1 / 3
    # Both summaries still reconstruct60% overall; the subgroup claim is false.
    with pytest.raises(AssertionError):
        counts(wrong, *inputs())


def test_subgroup_audit_rejects_wrong_denominator():
    wrong = grade()
    wrong['modal']['count'] = 3
    with pytest.raises(AssertionError):
        counts(wrong, *inputs())


def test_length_audit_rejects_changed_length_mask():
    row = grade()
    row['per_length'] = {
        '10': {'accuracy': 1.0, 'modal_fraction': 0.0,
               'modal': {'count': 0, 'accuracy': None},
               'nonmodal': {'count': 3, 'accuracy': 1.0}, 'per_length': {}},
        '15': {'accuracy': 0.0, 'modal_fraction': 1.0,
               'modal': {'count': 2, 'accuracy': 0.0},
               'nonmodal': {'count': 0, 'accuracy': None}, 'per_length': {}},
    }
    prediction = np.array([1, 1, 1, 0, 0])
    truth = np.ones(5, dtype=int)
    modal = np.array([False, False, False, True, True])
    row['modal']['accuracy'] = 0.0
    row['nonmodal']['accuracy'] = 1.0
    assert counts(row, prediction, truth, modal, np.array([10, 10, 10, 15, 15])) == 18
    with pytest.raises(AssertionError):
        counts(row, prediction, truth, modal, np.array([15, 10, 10, 15, 10]))


def test_bootstrap_audit_rejects_forged_interval():
    records = [{'source_seed': seed} for seed in [17, 42, 101, 17, 42, 101]]
    summary = {'mean': 2.0, 'paired_values': [2.0] * 6, 'positive_repeats': 6,
               'three_pair_means': [2.0] * 3, 'paired_bootstrap_95': [2.0, 2.0],
               'three_pair_bootstrap_95': [2.0, 2.0]}
    audit_bootstrap(summary, [2.0] * 6, records)
    summary['three_pair_bootstrap_95'] = [1.0, 3.0]
    with pytest.raises(AssertionError):
        audit_bootstrap(summary, [2.0] * 6, records)
