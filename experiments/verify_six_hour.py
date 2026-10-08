"""Audit frozen inputs, provenance and validation selection without retraining."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np

from .longrun_engine import atomic_json
from .permutation_audit import IDENTITIES
from .permworld_combinations import select_groups, sha


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify(plan_path='configs/six_hour_session.json'):
    plan = json.loads(Path(plan_path).read_text())
    config = json.loads(Path(plan['base_config']).read_text())
    groups = select_groups(config)
    root = Path(plan['output'])
    metadata = json.loads((root/'dataset/metadata.json').read_text())
    assert metadata['data_sha256'] == sha(root/'dataset/data.npz')
    assert metadata['signature']['generator_sha256'] == sha('experiments/permworld_combinations.py')
    names = metadata['names']
    with np.load(root/'dataset/data.npz') as data:
        all_inputs = []
        for split, count in plan['examples_per_length'].items():
            inputs, labels, lengths = [data[f'{split}_{k}'] for k in ('input', 'labels', 'lengths')]
            assert inputs.shape[0] == count*(config['lengths'][1]-config['lengths'][0]+1)
            assert labels.shape == (len(inputs), len(names))
            assert all(np.sum(lengths == n) == count for n in range(config['lengths'][0], config['lengths'][1]+1))
            all_inputs.extend(tuple(row) for row in inputs)
            for identity in IDENTITIES:
                left = sum(coefficient*labels[:, names.index(task)] for task, coefficient in identity['terms'].items())
                assert np.array_equal(left, identity['n']*lengths+identity['constant'])
        assert len(all_inputs) == len(set(all_inputs))
        support = json.loads((root/'dataset/support_indices.json').read_text())
        order = np.random.default_rng(plan['data_seed']+50000).permutation(len(data['support_pool_input']))
        for budget in config['target_budgets']:
            assert support[str(budget)] == order[:budget].tolist()
    assert all(not set(g['tasks'])&set(config['target_tasks']) for g in groups)
    source_counts = {}
    source_records = {}
    for phase in ('calibration', 'multi', 'single'):
        checked = 0
        for path in (root/phase).glob('*.json'):
            record = json.loads(path.read_text())
            if record['status'] != 'complete': continue
            signature = {key: record[key] for key in ('job', 'architecture', 'training_policy', 'data_sha256', 'source_hashes', 'upstream_hashes')}
            assert record['fingerprint'] == fingerprint(signature)
            assert record['data_sha256'] == metadata['data_sha256']
            for code, expected in record['source_hashes'].items(): assert sha(code) == expected
            for code, expected in record['upstream_hashes'].items(): assert sha(Path(config['repository'])/'src/neurips_permutations'/code) == expected
            assert record['step'] == record['job']['steps']
            assert record['exposures_per_task'] == record['step']*plan['examples_per_task_per_step']
            assert sha(root/phase/'checkpoints'/f'{record["job_id"]}.pt') == record['checkpoint_sha256']
            assert {row['task'] for row in record['source_validation']} == set(record['job']['tasks'])
            if phase != 'calibration':
                for milestone in (5000, 10000, 20000):
                    feature = np.load(root/phase/f'{record["job_id"]}_step{milestone}_features.npy')
                    assert feature.shape == (plan['examples_per_length']['representation']*21, record['architecture']['d_model'])
                    assert np.all(np.isfinite(feature))
            source_records[record['job_id']] = record
            checked += 1
        source_counts[phase] = checked
    selection = json.loads((root/'architecture_selection.json').read_text())
    recomputed = []
    for arch in plan['architectures']:
        cells = []
        for g in plan['calibration_groups']:
            key = f"{arch['id']}_{g}_s{plan['calibration_seed']}"
            cells.extend(row['accuracy'] for row in source_records[key]['source_validation'])
        recomputed.append({'architecture': arch, 'source_validation_macro': float(np.mean(cells))})
    assert selection['scores'] == recomputed
    expected = max(recomputed, key=lambda r: (r['source_validation_macro'], -r['architecture']['d_model']))['architecture']
    assert selection['selected'] == expected
    tuning_complete = False
    checked_tuning = 0
    path = root/'target_tuning.json'
    if path.exists():
        tuning = json.loads(path.read_text())
        assert tuning['code_sha256'] == sha('experiments/longrun_transfer.py')
        signature = {key: tuning[key] for key in ('plan', 'architecture', 'code_sha256', 'data_sha256', 'seed', 'probe_preprocessing')}
        assert tuning['fingerprint'] == fingerprint(signature)
        assert tuning['seed'] == 17 and tuning['data_sha256'] == metadata['data_sha256']
        if tuning['status'] == 'complete':
            tuning_complete = True
            keys = [(r['condition'], r['mode'], r['target'], r['budget'], r['learning_rate'], r['steps']) for r in tuning['rows']]
            assert len(keys) == len(set(keys)) == (len(groups)+1)*len(config['target_tasks'])*len(config['target_budgets'])*27
            for condition in ['random']+[g['id'] for g in groups]:
                for mode in ('finetune', 'linear', 'linear_query', 'mlp'):
                    cells = [r for r in tuning['rows'] if r['condition'] == condition and r['mode'] == mode]
                    pairs = sorted({(r['learning_rate'], r['steps']) for r in cells})
                    scores = [{'learning_rate': lr, 'steps': steps,
                               'validation_macro': float(np.mean([r['validation_accuracy'] for r in cells if (r['learning_rate'], r['steps']) == (lr, steps)]))} for lr, steps in pairs]
                    chosen = max(scores, key=lambda r: (r['validation_macro'], -r['steps'], -r['learning_rate']))
                    assert tuning['policies'][condition][mode] == chosen
                    checked_tuning += 1
            assert tuning['policy'] == tuning['policies']['random']
    endpoint_keys = []
    completed_transfer = 0
    for path in (root/'transfer').glob('*.json'):
        record = json.loads(path.read_text())
        if record['status'] != 'complete': continue
        assert tuning_complete
        signature = {key: record[key] for key in ('run_id', 'source_fingerprint', 'tuning_fingerprint', 'code_sha256', 'data_sha256', 'policy', 'seed')}
        assert record['fingerprint'] == fingerprint(signature)
        assert record['tuning_fingerprint'] == tuning['fingerprint'] and record['policy'] == tuning['policy']
        assert record['code_sha256'] == sha('experiments/longrun_transfer.py')
        if record['group'] != 'random': assert record['source_fingerprint'] == source_records[record['run_id']]['fingerprint']
        keys = {(r['target'], r['budget'], r['mode']) for r in record['rows']}
        expected = set(product(config['target_tasks'], config['target_budgets'], ('linear_task_free', 'linear_query', 'mlp_task_free', 'finetune', 'finetune_shared')))
        assert keys == expected and len(record['rows']) == len(expected)
        for row in record['rows']:
            assert 0 <= row['test_accuracy'] <= 1
            endpoint_keys.append((record['run_id'], row['target'], row['budget'], row['mode']))
        completed_transfer += 1
    assert len(endpoint_keys) == len(set(endpoint_keys))
    result = {'status': 'passed_for_completed_artifacts', 'source_models': source_counts,
              'globally_disjoint_inputs': len(all_inputs), 'nested_shared_supports': True,
              'heldout_targets_absent_from_sources': True, 'six_exact_label_identities': True,
              'architecture_choice_recomputed_from_source_validation': True,
              'source_code_and_checkpoint_hashes': True, 'target_tuning_complete': tuning_complete,
              'validation_policy_choices_recomputed': checked_tuning,
              'completed_transfer_models': completed_transfer, 'unique_transfer_endpoints': len(endpoint_keys),
              'training_code_sha256': sha('experiments/longrun_engine.py'), 'verifier_sha256': sha(__file__)}
    atomic_json(root/'verification.json', result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', default='configs/six_hour_session.json')
    verify(parser.parse_args().plan)
