"""Verify target-data separation, all frozen forecasts and completed repeats."""
import argparse
from datetime import datetime
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

from .longrun_engine import atomic_json
from .native_target_repeat import paths, feature_matrix, fit_weights, apply_weights
from .permworld_combinations import sha


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify(prepared_only=False):
    plan, config, source_root, root = paths()
    signature = json.loads((root/'protocol.json').read_text())['signature']
    for key, path in (('code_sha256', 'experiments/native_target_repeat.py'),
                      ('source_data_sha256', source_root/'dataset/data.npz'),
                      ('source_feature_sha256', source_root/'transfer_prediction/features.json'),
                      ('target_tuning_sha256', source_root/'target_tuning.json'),
                      ('transfer_code_sha256', 'experiments/longrun_transfer.py')):
        assert signature[key] == sha(path)
    with np.load(source_root/'dataset/data.npz') as archive:
        original = {tuple(row) for key in archive.files if key.endswith('_input') for row in archive[key]}
    with np.load(root/'dataset/data.npz') as archive:
        data = {k: archive[k] for k in archive.files}
    metadata = json.loads((root/'dataset/metadata.json').read_text())
    assert metadata['data_sha256'] == sha(root/'dataset/data.npz')
    new = [tuple(row) for key in data if key.endswith('_input') for row in data[key]]
    assert len(new) == len(set(new)) == 5250 and len(original) == 52290
    assert not original & set(new)
    sys.path.insert(0, str(Path(config['repository'])/'src'))
    from neurips_permutations.math_ops import PROPERTY_FUNCTIONS
    from neurips_permutations.passage import ID_TO_TOKEN
    checked_labels = 0
    for split, count in signature['split_counts_per_length'].items():
        assert len(data[split+'_input']) == count*21
        for row, n, labels in zip(data[split+'_input'], data[split+'_lengths'], data[split+'_labels']):
            permutation = tuple(int(ID_TO_TOKEN[int(v)]) for v in row[4:4+2*n:2])
            assert sorted(permutation) == list(range(1, int(n)+1))
            np.testing.assert_array_equal([PROPERTY_FUNCTIONS[name](permutation) for name in metadata['names']], labels)
            checked_labels += len(labels)
    support = json.loads((root/'dataset/support_indices.json').read_text())
    order = np.random.default_rng(signature['seed']+50000).permutation(2100)
    for budget in config['target_budgets']: assert support[str(budget)] == order[:budget].tolist()
    features = json.loads((source_root/'transfer_prediction/features.json').read_text())
    first = json.loads((root/'forecasts.json').read_text())
    frozen_unix = datetime.fromisoformat(first['frozen_utc']).timestamp()
    assert frozen_unix < (root/'dataset/data.npz').stat().st_mtime
    assert first['new_input_labels_not_generated']
    for name, expected in first['old_behavior_sha256'].items(): assert sha(source_root/'transfer'/name) == expected
    old_records = [json.loads(p.read_text()) for p in (source_root/'transfer').glob('*.json')]
    old_rows = [{**row, 'group': record['group'], 'seed': record['seed']} for record in old_records for row in record['rows']]
    random = {(r['seed'], r['target'], r['budget'], r['mode']): r['test_accuracy'] for r in old_rows if r['group'] == 'random'}
    lookup = {(r['group'], r['seed'], r['target'], r['budget'], r['mode']): r for r in old_rows}
    forecasts = first['forecasts']; assert len(forecasts) == 5760
    forecast_lookup = {(r['group'], r['seed'], r['target'], r['mode'], r['budget'], r['variant']): r['predicted_paired_gain'] for r in forecasts}
    assert len(forecast_lookup) == len(forecasts)
    checked_weights = 0
    for record in json.loads((root/'weights.json').read_text()):
        mode, budget, version = record['mode'], record['budget'], record['variant']
        corrected = version.endswith('_corrected'); variant = version.removesuffix('_corrected')
        x = feature_matrix(features, variant, corrected)
        y = np.array([lookup[r['group'], r['seed'], r['target'], budget, mode]['test_accuracy']-random[r['seed'], r['target'], budget, mode] for r in features])
        expected = fit_weights(x, y)
        for key in expected: np.testing.assert_allclose(record['weights'][key], expected[key], rtol=0, atol=1e-12)
        predicted = apply_weights(x, expected)
        for row, value in zip(features, predicted):
            assert abs(forecast_lookup[row['group'], row['seed'], row['target'], mode, budget, version]-value) < 1e-12
        checked_weights += 1
    assert checked_weights == 60
    transfer_count, endpoints = 0, 0
    tuning = json.loads((source_root/'target_tuning.json').read_text())
    if not prepared_only:
        for path in (root/'transfer').glob('*.json'):
            record = json.loads(path.read_text())
            if record['status'] != 'complete': continue
            transfer_signature = {k: record[k] for k in ('run_id', 'source_fingerprint', 'tuning_fingerprint', 'code_sha256', 'data_sha256', 'policy', 'seed')}
            assert record['fingerprint'] == fingerprint(transfer_signature)
            assert record['tuning_fingerprint'] == tuning['fingerprint'] and record['policy'] == tuning['policy']
            assert record['data_sha256'] == metadata['data_sha256']
            assert record['code_sha256'] == sha('experiments/longrun_transfer.py')
            expected = set(product(config['target_tasks'], config['target_budgets'], ('linear_task_free', 'linear_query', 'mlp_task_free', 'finetune', 'finetune_shared')))
            assert len(record['rows']) == 40 and {(r['target'], r['budget'], r['mode']) for r in record['rows']} == expected
            if record['group'] != 'random':
                parent = json.loads((source_root/'multi'/f'{record["run_id"]}.json').read_text())
                assert record['source_fingerprint'] == parent['fingerprint']
            assert frozen_unix < path.stat().st_mtime
            transfer_count += 1; endpoints += len(record['rows'])
    result = {'status': 'prepared_inputs_and_forecasts_passed' if prepared_only else 'passed_for_completed_artifacts',
              'original_inputs': len(original), 'new_globally_disjoint_inputs': len(new),
              'property_labels_recomputed_from_input_tokens': checked_labels,
              'frozen_weight_sets_recomputed': checked_weights, 'forecasts_recomputed': len(forecasts),
              'forecasts_saved_before_new_input_labels': True, 'same_source_models_and_task_sets': True,
              'completed_transfer_conditions': transfer_count, 'completed_transfer_endpoints': endpoints,
              'code_sha256': sha(__file__)}
    atomic_json(root/'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepared-only', action='store_true')
    verify(parser.parse_args().prepared_only)
