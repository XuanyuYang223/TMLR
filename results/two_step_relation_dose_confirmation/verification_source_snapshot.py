"""Independent joint-model replay and affine composition audit."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import two_step_relation_factorial as core
from . import two_step_relation_joint as joint
from .longrun_engine import atomic_json
from .native_confirmation import setup
from .permworld_combinations import sha


def verify(plan, parent, config, root, sig):
    data = dict(np.load(root / 'dataset' / 'test' / 'dataset.npz'))
    rows = json.loads((root / 'evaluation_records.json').read_text())['records']
    counts, grams = 0, 0
    for index in range(6):
        for condition in plan['conditions']:
            name = f'n{index}_hidden_{condition}'
            maps = dict(np.load(root / 'maps' / f'{name}.npz'))
            cache = dict(np.load(root / 'features' / f'{name}_test.npz'))
            h = cache['hidden'].astype(float)
            archive = np.load(root / 'evaluations' / f'{name}.npz')
            selected = [r for r in rows if r['replicate'] == f'n{index}' and r['condition'] == condition]
            assert len(selected) == 12
            for row in selected:
                word = row['word']; width = h.shape[-1]; matrix = np.eye(width + 1)
                for letter in word:
                    m = np.eye(width + 1); m[:width, :width] = maps['rho_' + letter]; m[-1, :width] = maps['bias_' + letter]
                    matrix = matrix @ m
                prediction = (np.column_stack([h[:, 0], np.ones(len(h))]) @ matrix)[:, :width]
                answer = (prediction @ maps['readout_weight'].T + maps['readout_bias']).argmax(-1)
                np.testing.assert_allclose(prediction, archive[word + '_hidden'], atol=2e-4, rtol=2e-5)
                np.testing.assert_array_equal(answer, archive[word + '_answers'])
                action = {'c': 1, 'i': 4, 'ci': 5, 'ic': 6, 'cc': 0, 'ii': 0}[word]; assert action == row['action']
                use = data['split'] == (0 if row['split'] == 'iid' else 1)
                correct = [int(a) == int(b) for a, b in zip(answer[use], data['labels'][use, action])]
                assert sum(correct) / len(correct) == row['accuracy']; counts += 1
                direct = (h[use, action] @ maps['readout_weight'].T + maps['readout_bias']).argmax(-1)
                assert float((direct == data['labels'][use, action]).mean()) == row['direct_input_accuracy']; counts += 1
                if row['split'] == 'collisions':
                    hits = np.asarray(correct); pairs = data['pair_ids'][use]
                    assert sum(bool(hits[pairs == p].all()) for p in set(pairs)) / len(set(pairs)) == row['pair_both_correct']; counts += 1
                if len(word) == 2:
                    start = 1 if word[0] == 'c' else 4; letter = word[1]
                    forced = h[:, start] @ maps['rho_' + letter] + maps['bias_' + letter]
                    forced_answer = (forced @ maps['readout_weight'].T + maps['readout_bias']).argmax(-1)
                    np.testing.assert_array_equal(forced_answer, archive[word + '_forced_answers'])
                    assert float((forced_answer[use] == data['labels'][use, action]).mean()) == row['teacher_forced_accuracy']; counts += 1
                denominator = np.square(h[use, action] - h[use, 0]).sum()
                if row['displacement_nmse'] is not None:
                    np.testing.assert_allclose(np.square(prediction[use] - h[use, action]).sum() / denominator, row['displacement_nmse'], atol=1e-10)
                ck = []
                for n in plan['lengths']:
                    ids = np.flatnonzero(use & (data['lengths'] == n))[:128]
                    x, y = prediction[ids], h[ids, action]; x -= x.mean(0); y = y - y.mean(0)
                    ck.append(float(np.square(x.T @ y).sum() / np.sqrt(np.square(x.T @ x).sum() * np.square(y.T @ y).sum()))); grams += 1
                np.testing.assert_allclose(np.mean(ck), row['cka'], atol=2e-12, rtol=2e-12)
    _, _, tokens, _ = setup(config); device = 'cuda' if torch.cuda.is_available() else 'cpu'; errors = []
    val_checks = 0
    for index in range(6):
        raw = dict(np.load(root / 'dataset' / f'n{index}' / 'support' / 'dataset.npz'))
        val = {k: raw[k][raw['split'] == 1] for k in ['input', 'lengths', 'labels']}
        for condition in plan['conditions']:
            name = f'n{index}_hidden_{condition}'; record = json.loads((root / 'fits' / f'{name}.json').read_text())
            file = root / 'maps' / f'{name}_e{plan["epochs"]}.pt'; assert sha(file) == record['checkpoint_sha256']
            model = core.source_model(sig['sources'][index % 3], parent, config, device, accelerated=False)
            readout = model.lm_head.weight.detach().cpu().clone()
            state = torch.load(file, map_location='cpu', weights_only=True); model.load_state_dict(state['model'])
            torch.testing.assert_close(model.lm_head.weight.detach().cpu(), readout, atol=0, rtol=0)
            ids = np.concatenate([np.flatnonzero(data['lengths'] == n)[:4] for n in plan['lengths']])
            replay = core.hidden_features(model, {'input': data['input'][ids], 'lengths': data['lengths'][ids]}, parent['source_task'], tokens)
            saved = np.load(root / 'features' / f'{name}_test.npz')['hidden'][ids]
            np.testing.assert_allclose(replay, saved, atol=3e-4, rtol=3e-4); errors.append(float(np.abs(replay - saved).max()))
            ops = joint.AffineOperators(parent['architecture']['d_model']).to(device); ops.load_state_dict(state['operators'])
            grade = joint.validate(model, ops, val, parent, tokens)
            for key in grade:
                np.testing.assert_allclose(grade[key], record['curve'][-1][key], atol=1e-8, rtol=1e-8)
            val_checks += 1; del model, ops, state
    for source in sig['sources']:
        assert sha(source['checkpoint']) == source['sha256']
    atomic_json(root / 'verification.json', {'status': 'complete', 'completed_utc': core.now(),
                'independent_accuracy_and_pair_counts': counts, 'independent_gram_scores': grams,
                'original_unaccelerated_joint_models_replayed': 24, 'validation_models_replayed': val_checks,
                'maximum_feature_replay_error': max(errors), 'all24_readouts_preserved': True,
                'all3_source_initializations_preserved': True})


def run():
    plan, parent, config, root, sig = joint.initialize()
    file = root / 'verification_protocol.json'
    if not file.exists():
        atomic_json(file, {'registered_utc': core.now(), 'code_sha256': sha(__file__), 'new_test_observed': (root / 'test_opened.json').exists()})
        (root / 'verification_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    while not (root / 'summary.json').exists():
        state = json.loads((root / 'state.json').read_text()) if (root / 'state.json').exists() else {}
        if state.get('status') == 'failed':
            raise RuntimeError('Joint pipeline failed')
        time.sleep(10)
    verify(plan, parent, config, root, sig)
    atomic_json(root / 'state.json', {'status': 'complete', 'completed_utc': core.now()})


if __name__ == '__main__':
    run()
