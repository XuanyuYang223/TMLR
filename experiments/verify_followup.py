"""Validate completed follow-up artifacts without running or selecting models."""
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path

import numpy as np

from .algebra import world
from .run import balanced_splits


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


class ResourceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attributes):
        self.links.extend(value for key, value in attributes if key in ('href', 'src'))


def verify():
    output = Path('results/readout_followup')
    metadata = json.loads((output / 'metadata.json').read_text())
    config = metadata['source_config']
    plan = metadata['plan']
    parents = [Path(p) for p in plan['source_runs']]
    assert hashlib.sha256(Path('experiments/readout_followup.py').read_bytes()).hexdigest() == metadata['code_sha256']
    signature = {key: metadata[key] for key in ('plan', 'max_runs', 'source_fingerprints', 'code_sha256')}
    assert hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest() == metadata['fingerprint']
    source_code = ''.join(Path('experiments', name).read_text() for name in ('run.py', 'models.py', 'algebra.py'))
    for parent, expected in zip(parents, metadata['source_fingerprints']):
        original = json.loads((parent / 'metadata.json').read_text())
        actual = hashlib.sha256((json.dumps(original['config'], sort_keys=True) + source_code + str(original['pretrain_only'])).encode()).hexdigest()
        assert actual == original['fingerprint'] == expected

    neural = read_csv(output / 'neural_endpoints.csv')
    codes = read_csv(output / 'code_diagnostic_endpoints.csv')
    common = ('scenario', 'world_seed', 'model_seed', 'target_id', 'budget')
    key = lambda row, extra=(): tuple(row[k] for k in common + extra)
    assert len(neural) == len({key(row, ('kind', 'learning_rate')) for row in neural}) == 8640
    assert len(codes) == len({key(row, ('method',)) for row in codes}) == 6480
    runs = {(r['scenario'], r['world_seed'], r['model_seed']) for r in neural}
    assert len(runs) == 72
    for scenario, basis, initialization in runs:
        cached = json.loads((output / f'{scenario}_w{basis}_m{initialization}.json').read_text())
        assert cached['fingerprint'] == metadata['fingerprint']
        assert len(cached['neural']) == 120 and len(cached['code_diagnostics']) == 90
    controls = {}
    for row in neural:
        control_key = tuple(row[k] for k in ('world_seed', 'model_seed', 'target_id', 'budget', 'kind', 'learning_rate'))
        controls.setdefault(control_key, set()).add(row['random_accuracy'])
        assert abs(float(row['test_accuracy']) - float(row['random_accuracy']) - float(row['transfer_gain'])) < 1e-12
    assert all(len(values) == 1 for values in controls.values())
    old = {key(row): row for row in read_csv('results/combined/metrics.csv') if row['mode'] == 'probe'}
    repeated = [r for r in neural if r['kind'] == 'linear' and r['learning_rate'] == '0.03']
    assert len(repeated) == 2160
    for row in repeated:
        for column in ('test_accuracy', 'support_accuracy', 'random_accuracy', 'transfer_gain'):
            assert float(row[column]) == float(old[key(row)][column])

    for basis_seed in config['world_seeds']:
        _, latent, _, basis = world(config['p'], config['dimension'], basis_seed)
        with np.load(parents[0] / f'world_{basis_seed}.npz') as first:
            labels = latent @ first['targets'].T % config['p']
            test, supports = balanced_splits(labels, config['budgets'], config['test_per_class'], config['p'], basis_seed + 10000)
            np.testing.assert_array_equal(first['basis'], basis)
            np.testing.assert_array_equal(first['test'], test)
            for budget, support in supports.items():
                np.testing.assert_array_equal(first[f'support_{budget}'], support)
                for target in range(config['target_count']):
                    assert not set(support[target]) & set(test[target])
            for parent in parents[1:]:
                with np.load(parent / f'world_{basis_seed}.npz') as other:
                    assert set(first.files) == set(other.files)
                    for column in first.files:
                        np.testing.assert_array_equal(first[column], other[column])
    field = [r for r in codes if r['method'] == 'field_fit']
    shuffled = [r for r in codes if r['method'] == 'shuffled_field_fit']
    lookup = [r for r in codes if r['method'] == 'code_lookup']
    assert len(field) == len(shuffled) == len(lookup) == 2160
    assert all(float(r['test_accuracy']) == float(r['support_accuracy']) == 1 and r['fit_consistent'] == r['identifies_full_source_span'] == 'True' for r in field)
    assert all(float(r['test_accuracy']) == .2 and r['fit_consistent'] == r['identifies_full_source_span'] == 'False' for r in shuffled)
    assert all(0 <= float(r['seen_tuple_fraction']) < 1 for r in lookup)

    cka = read_csv('results/cka_review/symmetry_contrasts.csv')
    final = [r for r in cka if r['layer'] == 'final_norm']
    assert len(final) == 24
    assert all(float(r['correct_minus_identity']) > 0 and float(r['correct_minus_wrong']) > 0 for r in final)
    originals = read_csv('external/neurips/results/property-task-geometry/cka/symmetry_cka.csv')
    original_values = {(r['pair_id'], r['model_seed'], r['layer'], r['condition']): float(r['linear_cka']) for r in originals}
    for row in cka:
        for condition in ('identity', 'correct', 'wrong'):
            assert float(row[condition]) == original_values[row['pair_id'], row['model_seed'], row['layer'], condition]
        assert float(row['correct_minus_identity']) == float(row['correct']) - float(row['identity'])
        assert float(row['correct_minus_wrong']) == float(row['correct']) - float(row['wrong'])
    resource_count = 0
    for report in (output / 'report.html', Path('results/cka_review/report.html')):
        parser = ResourceParser()
        parser.feed(report.read_text())
        for link in parser.links:
            assert (report.parent / link).is_file(), link
            resource_count += 1
    result = {'status': 'passed', 'source_models': len(runs),
              'unique_neural_endpoints': len(neural), 'unique_code_endpoints': len(codes),
              'matched_random_controls': True, 'unchanged_source_fingerprints': True,
              'identical_original_linear_endpoints': len(repeated),
              'identical_source_and_target_splits': True,
              'field_fit_and_shuffle_controls': True, 'original_cka_values_match': True,
              'positive_final_layer_cka_contrasts': len(final),
              'report_resources_checked': resource_count}
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    verify()
