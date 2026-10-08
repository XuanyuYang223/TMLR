"""Independently recount saved baseline predictions and paired contrasts."""
import itertools
import json
import math
from pathlib import Path

import numpy as np

from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now

ROOT = Path('results/two_step_relation_output_only20')
PRIMARY = Path('results/two_step_relation_dose_confirmation')


def interval(values):
    samples = sorted(math.fsum(values[i] for i in draw) / 3
                     for draw in itertools.product(range(3), repeat=3))
    def percentile(q):
        position = q * (len(samples) - 1)
        lo = math.floor(position)
        hi = math.ceil(position)
        return samples[lo] + (position - lo) * (samples[hi] - samples[lo])
    return [percentile(.025), percentile(.975)]


def run():
    result = json.loads((ROOT / 'results.json').read_text())
    protocol = json.loads((ROOT / 'output_only_protocol.json').read_text())
    assert result['status'] == 'complete'
    assert protocol['signature']['prior_main20_outcomes_already_observed']
    assert protocol['signature']['code_sha256'] == sha('experiments/two_step_relation_output_only20.py')
    data = dict(np.load(PRIMARY / 'dataset/test/dataset.npz'))
    opened = json.loads((ROOT / 'test_opened.json').read_text())
    assert opened['primary_test_dataset_sha256'] == sha(PRIMARY / 'dataset/test/dataset.npz')
    records = result['records']
    assert len(records) == 72
    actions = {'c': 1, 'i': 4, 'ci': 5, 'ic': 6, 'cc': 0, 'ii': 0}
    checked_counts = 0
    for i in range(6):
        name = f'n{i}_hidden_both_correct'
        fit = json.loads((ROOT / 'fits' / (name + '.json')).read_text())
        main_fit = json.loads((PRIMARY / 'fits' / (name + '.json')).read_text())
        assert fit['completed_utc'] < opened['opened_utc'] and fit['epochs'] == 20
        for key in ['updates', 'anchor_exposures', 'schedule_sha256',
                    'source_initialization_sha256', 'readout_parameter_sha256']:
            assert fit[key] == main_fit[key]
        assert fit['map_sha256'] == sha(ROOT / 'maps' / (name + '.npz'))
        assert fit['checkpoint_sha256'] == sha(ROOT / 'maps' / (name + '_e20.pt'))
        answers = dict(np.load(ROOT / 'evaluations' / (name + '.npz')))
        for word, action in actions.items():
            for split_id, split in [(0, 'iid'), (1, 'collisions')]:
                use = data['split'] == split_id
                hits = answers[word + '_answers'][use] == data['labels'][use, action]
                row = next(r for r in records if
                           (r['replicate'], r['word'], r['split']) == (f'n{i}', word, split))
                assert abs(int(hits.sum()) / len(hits) - row['accuracy']) < 1e-12
                checked_counts += 1
                if split_id == 1:
                    pairs = data['pair_ids'][use]
                    both = sum(bool(hits[pairs == p].all()) for p in set(map(int, pairs)))
                    assert abs(both / len(set(map(int, pairs))) - row['pair_both_correct']) < 1e-12
                    checked_counts += 1
    main = json.loads((PRIMARY / 'evaluation_records.json').read_text())['records']
    for contrast in result['contrasts']:
        values = []
        for i in range(6):
            score = next(r['accuracy'] for r in main if
                         (r['replicate'], r['condition'], r['word'], r['split']) ==
                         (f'n{i}', contrast['condition'], 'ci', contrast['split']))
            baseline = next(r['accuracy'] for r in records if
                            (r['replicate'], r['word'], r['split']) == (f'n{i}', 'ci', contrast['split']))
            values.append(100 * (score - baseline))
        groups = [math.fsum([values[i], values[i + 3]]) / 2 for i in range(3)]
        np.testing.assert_allclose(values, contrast['paired_values'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(groups, contrast['three_source_means'], atol=1e-12, rtol=0)
        assert abs(math.fsum(values) / 6 - contrast['mean_accuracy_pp']) < 1e-12
        np.testing.assert_allclose(interval(groups), contrast['three_source_bootstrap_95_pp'], atol=1e-12, rtol=0)
    files = [ROOT / 'results.json', ROOT / 'output_only_protocol.json',
             PRIMARY / 'dataset/test/dataset.npz', PRIMARY / 'evaluation_records.json', Path(__file__)]
    files += sorted((ROOT / 'fits').glob('*.json')) + sorted((ROOT / 'evaluations').glob('*.npz'))
    atomic_json(ROOT / 'statistics_verification.json', {
        'status': 'complete', 'completed_utc': now(), 'independent_counts': checked_counts,
        'independent_contrasts_and_bootstrap_intervals': len(result['contrasts']),
        'all6_budgets_and_source_initializations_match_primary': True,
        'all6_fits_complete_before_baseline_test_opened': True,
        'registered_after_main20_compound_test': True,
        'artifact_sha256': {str(p): sha(p) for p in files},
        'limitation': 'Three-source bootstrap resamples sources only. Baseline matches20 epochs.'})
    print(json.dumps({'status': 'complete', 'counts': checked_counts, 'contrasts': len(result['contrasts'])}))


if __name__ == '__main__':
    run()
