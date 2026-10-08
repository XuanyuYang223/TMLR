"""Independent algebra, frozen-state, budget and prediction replay."""
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import torch

from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now

ROOT = Path('results/operator_capacity_confirmation')


def word(x, letters):
    a, b, c, d = map(int, x)
    for letter in letters:
        a, b, c, d = (c, d, a, b) if letter == 'a' else (-c, -d, a-c, b-d)
        a, b, c, d = [v % 13 for v in [a, b, c, d]]
    return a, b, c, d


def orbit_key(x):
    return min(word(x, w) for w in ['', 'a', 'b', 'bb', 'ab', 'abb'])


def replay(h, state, nonlinear):
    def array(name):
        return state[name].cpu().numpy().astype(np.float64)
    middle = h@array('layers.0.weight').T+array('layers.0.bias')
    if nonlinear:
        erf = np.fromiter((math.erf(float(v)/math.sqrt(2)) for v in middle.ravel()),
                          dtype=float, count=middle.size).reshape(middle.shape)
        middle = .5*middle*(1+erf)
    residual = middle@array('layers.2.weight').T+array('layers.2.bias')
    return h@array('a')+array('bias')+residual


def encoder_replay(x, state):
    def array(name):
        return state[name].numpy().astype(float)
    # Reconstruct the numerical input convention and MLP independently.
    z = ((x.astype(np.float32)-6)*np.float32(2*np.pi/13)).astype(float)
    z = z@array('encoder.0.weight').T+array('encoder.0.bias')
    erf = np.fromiter((math.erf(float(v)/math.sqrt(2)) for v in z.ravel()),
                      dtype=float, count=z.size).reshape(z.shape)
    z = .5*z*(1+erf)
    z = z@array('encoder.2.weight').T+array('encoder.2.bias')
    z = (z-z.mean(-1, keepdims=True))/np.sqrt(z.var(-1, keepdims=True)+1e-5)
    return z*array('encoder.3.weight')+array('encoder.3.bias')


def run():
    torch.set_num_threads(1)
    config = json.loads(Path('configs/operator_capacity_confirmation.json').read_text())
    protocol = json.loads((ROOT/'protocol.json').read_text())
    checks = 0
    for p, expected in protocol['signature']['source_code_sha256'].items():
        assert sha(p) == expected; checks += 1
    for p, expected in json.loads((ROOT/'data_audit.json').read_text())['dataset_sha256'].items():
        assert sha(p) == expected; checks += 1
    partitions = {}
    for file in (ROOT/'dataset').glob('*.npz'):
        data = dict(np.load(file))
        words = ['', 'a', 'b', 'ab', 'ba', 'aa', 'bb', 'bbb', 'aba', 'bbbb'] if file.stem == 'test' else ['', 'a', 'b']
        assert data['x'].shape[1] == len(words)
        for path, labels in zip(data['x'], data['labels']):
            for x, y, w in zip(path, labels, words):
                expected = word(path[0], w)
                assert tuple(x) == expected and (expected[0]+expected[3]) % 13 == y
                checks += 1
        partitions[file.stem] = {orbit_key(x) for x in data['x'][:, 0]}
        if file.stem.startswith('support'):
            assert len(partitions[file.stem]) == config['fit_anchors']
            identity = np.arange(len(data['labels']))
            for k in ['pc', 'pi']:
                assert np.array_equal(np.sort(data[k]), identity)
                assert np.all(data[k] != identity)
                assert np.array_equal(data['labels'][data[k]], data['labels'])
                checks += 3
    train = set.union(partitions['source'], *[partitions[f'support{i}'] for i in range(3)])
    val = partitions['validation'] | partitions['pilot_validation']
    test = partitions['test']
    assert not train & val and not train & test and not val & test
    excluded = {orbit_key(x) for p in config['excluded_test_datasets'] for x in np.load(p)['x'][:, 0]}
    assert len(excluded) == 1024 and not set.union(*partitions.values()) & excluded
    test_data = dict(np.load(ROOT/'dataset/test.npz'))
    for pid in range(config['collision_pairs']):
        ids = np.flatnonzero(test_data['pair_ids'] == pid)
        assert len(ids) == 2
        assert np.array_equal(test_data['labels'][ids[0], :3], test_data['labels'][ids[1], :3])
        assert test_data['labels'][ids[0], 3] != test_data['labels'][ids[1], 3]
        checks += 3
    # These are independent sources, not twelve independently initialized encoders.
    historical = {131, 271, 389, 8123, 9133, 10151}
    assert len(set(config['source_seeds'])) == 3 and not historical & set(config['source_seeds'])
    max_diff = 0.; max_encoder_diff = 0.; reconstructed = {}; records = []
    evaluation = json.loads((ROOT/'evaluation.json').read_text())
    opened = json.loads((ROOT/'test_opened.json').read_text())
    source_checks = []
    for i in range(3):
        source_file = ROOT/'models'/f'n{i}_both_correct.pt'
        backbone = torch.load(source_file, map_location='cpu', weights_only=True)
        ordinary = torch.load(ROOT/'sources'/f's{i}.pt', map_location='cpu', weights_only=True)
        assert ordinary['source_seed'] == config['source_seeds'][i]
        for k in ['readout.weight', 'readout.bias']:
            assert torch.equal(backbone['model'][k], ordinary['model'][k]); checks += 1
        z = dict(np.load(ROOT/'states'/f's{i}_known.npz'))
        h = np.load(ROOT/'states'/f's{i}_test.npz')['hidden'].astype(float)
        for name, cached in [('test', h), (f'support{i}', z['train']), ('validation', z['validation'])]:
            inputs = np.load(ROOT/'dataset'/f'{name}.npz')['x']
            reconstructed_h = encoder_replay(inputs, backbone['model'])
            np.testing.assert_allclose(reconstructed_h, cached, atol=5e-5, rtol=5e-5)
            max_encoder_diff = max(max_encoder_diff, float(np.max(np.abs(reconstructed_h-cached))))
            checks += 1
        w = backbone['model']['readout.weight'].numpy().astype(float)
        bw = backbone['model']['readout.bias'].numpy().astype(float)
        b = backbone['operators']['maps.1.weight'].numpy().T.astype(float)
        bb = backbone['operators']['maps.1.bias'].numpy().astype(float)
        a = backbone['operators']['maps.0.weight'].numpy().T.astype(float)
        ba = backbone['operators']['maps.0.bias'].numpy().astype(float)
        np.testing.assert_array_equal(z['b'], b); np.testing.assert_array_equal(z['bias_b'], bb)
        support = dict(np.load(ROOT/'dataset'/f'support{i}.npz'))
        rng = np.random.default_rng(config['schedule_seed']+i)
        digest = sha256(); count = np.zeros(config['fit_anchors'], int)
        for _ in range(config['steps']):
            ids = rng.integers(0, config['fit_anchors'], config['batch_size'], dtype=np.int64)
            digest.update(ids.tobytes()); np.add.at(count, ids, 1)
        schedule_sha = digest.hexdigest()
        for condition in ['original']+config['conditions']:
            if condition == 'original':
                pred = h[:, 0]@a+ba
            else:
                file = ROOT/'predictors'/f's{i}_{condition}.pt'
                state = torch.load(file, map_location='cpu', weights_only=True)
                record_file = ROOT/'predictor_fits'/f's{i}_{condition}.json'
                rec = json.loads(record_file.read_text())
                assert sha(record_file) == opened['fit_record_sha256'][str(record_file)]
                assert rec['completed_utc'] < opened['opened_utc']
                assert rec['steps'] == 2000 and rec['batch_size'] == 128 and rec['anchor_exposures'] == 256000
                assert rec['schedule_sha256'] == schedule_sha
                assert rec['exposure_count_sha256'] == sha256(count.tobytes()).hexdigest()
                assert rec['checkpoint_sha256'] == sha(file) and rec['backbone_sha256'] == sha(source_file)
                params = {k: v for k, v in state['state_dict'].items() if k.startswith('layers.')}
                assert sum(v.numel() for v in params.values()) == 65920
                np.testing.assert_array_equal(state['state_dict']['a'].numpy(), a)
                np.testing.assert_array_equal(state['state_dict']['bias'].numpy(), ba)
                pair = support['pc'] if condition.endswith('_wrong') else np.arange(len(support['x']))
                assert rec['pairing_sha256'] == sha256(pair.tobytes()).hexdigest()
                np.testing.assert_allclose(rec['geometry_scale'], np.var(z['train'][:, 1], axis=0).mean(), rtol=1e-6)
                pred = replay(h[:, 0], state['state_dict'], condition.startswith('nonlinear_'))
                checks += 13
            compound = pred@b+bb
            answers = (compound@w.T+bw).argmax(1)
            first_answers = (pred@w.T+bw).argmax(1)
            cache = dict(np.load(ROOT/'evaluations'/f's{i}_{condition}.npz'))
            np.testing.assert_allclose(pred, cache['first_pred'], atol=1e-4, rtol=2e-4)
            np.testing.assert_allclose(compound, cache['compound_pred'], atol=2e-4, rtol=2e-4)
            np.testing.assert_array_equal(answers, cache['compound_answers'])
            np.testing.assert_array_equal(first_answers, cache['first_answers'])
            max_diff = max(max_diff, float(np.max(np.abs(pred-cache['first_pred']))))
            checks += 4
            hit = answers == test_data['labels'][:, 3]
            for sid, split in [(0, 'iid'), (1, 'collisions')]:
                use = test_data['split'] == sid
                row = next(r for r in evaluation['records'] if (r['source'], r['condition'], r['split']) == (i, condition, split))
                assert row['accuracy'] == float(hit[use].mean())
                assert row['first_accuracy'] == float((first_answers[use]==test_data['labels'][use, 1]).mean())
                checks += 2
                if sid == 1:
                    phit = np.asarray([hit[test_data['pair_ids']==pid] for pid in range(128)])
                    assert row['pair_both_correct'] == float(phit.all(1).mean()); checks += 1
                    reconstructed[(i, condition)] = phit
                records.append({'source': i, 'condition': condition, 'split': split, 'accuracy': row['accuracy']})
        source_checks.append({'source': i, 'frozen_encoder_B_readout_checkpoint_sha256': sha(source_file),
                              'all_four_schedules_identical': True, 'parameters_each': 65920})
    for endpoint in ['accuracy', 'pair_both_correct']:
        vals = {c: np.array([reconstructed[(i, c)].mean(1) if endpoint=='accuracy' else reconstructed[(i, c)].all(1)
                            for i in range(3)], dtype=float) for c in config['conditions']}
        interaction = 100*(vals['nonlinear_correct']-vals['nonlinear_wrong']-vals['linear_correct']+vals['linear_wrong'])
        row = next(c for c in evaluation['contrasts'] if c['endpoint']==endpoint and c['contrast']=='interaction')
        np.testing.assert_allclose(interaction.mean(1), row['source_effects_pp'], atol=1e-12)
        np.testing.assert_allclose(interaction.mean(), row['mean_pp'], atol=1e-12); checks += 2
    assert all(rec['no_compound_fit_targets'] for rec in [json.loads(p.read_text()) for p in (ROOT/'predictor_fits').glob('*.json')])
    assert opened['all12_fixed_budget_predictors_complete'] and len(opened['fit_record_sha256']) == 12
    atomic_json(ROOT/'independent_verification.json', {'status': 'passed', 'verified_utc': now(),
        'checks': checks, 'max_numpy_double_first_state_replay_difference': max_diff,
        'max_numpy_double_encoder_replay_difference': max_encoder_diff,
        'source_checks': source_checks, 'records': records,
        'independent_implementation': 'Scalar finite-field row formulas; NumPy double matrix products and math.erf GELU, outside the fitted PyTorch predictor class.',
        'limits': 'Frozen state verified from shared immutable backbone and frozen buffers; no claim to certify historical unseen inputs beyond excluded test orbits.'})
    print(json.dumps({'status': 'passed', 'checks': checks, 'max_replay_diff': max_diff}), flush=True)


if __name__ == '__main__':
    run()
