"""Independent replay, data-separation and result checks for the inverse assay."""
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.nn import functional as F

from .inverse_functional_alignment import initialize, configure, now
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .specialist_cka_controls import api, old_inputs


def key(p): return b'[' + b','.join(str(int(x)).encode() for x in p) + b']'


def inverse(p):
    result = np.empty(len(p), dtype=int)
    for i, value in enumerate(p, 1): result[value - 1] = i
    return result


def verify_data(plan, root, sig):
    audit = json.loads((root / 'dataset/audit.json').read_text()); seen = set()
    _, tokens, _, _, _ = api(plan); examples = 0
    for name in ['audit', 'test'] + [r['id'] for r in plan['replicates']]:
        path = root / 'dataset' / name / 'dataset.npz'; data = dict(np.load(path))
        if name in ['audit', 'test']: assert sha(path) == audit[name + '_sha256']
        else:
            record = next(r for r in audit['support_records'] if r['replicate'] == name)
            assert sha(path) == record['dataset_sha256']
            assert int((data['split'] == 0).sum()) == 192 and int((data['split'] == 1).sum()) == 64
            pairing = dict(np.load(path.parent / 'mismatch.npz')); assert sha(path.parent / 'mismatch.npz') == record['mismatch_sha256']
            eligible = pairing['eligible']; partner = pairing['partner']; rows = np.flatnonzero(eligible)
            assert np.all(data['split'][rows] == 0) and np.all(data['split'][partner[rows]] == 0)
            assert np.all(partner[rows] != rows)
            np.testing.assert_array_equal(data['lengths'][rows], data['lengths'][partner[rows]])
            np.testing.assert_array_equal(data['labels'][rows], data['labels'][partner[rows]])
            expected = np.array([data['split'][i] == 0 and np.sum((data['split'] == 0) & (data['lengths'] == n) & (data['labels'] == y)) >= 2 for i, (n, y) in enumerate(zip(data['lengths'], data['labels']))])
            np.testing.assert_array_equal(eligible, expected)
        for i, n in enumerate(data['lengths']):
            p = data['permutations'][i, 0, :n]; assert sorted(p) == list(range(1, n + 1))
            c = n + 1 - p
            orbit = [p, c, p[::-1], c[::-1], inverse(p), inverse(c), inverse(p[::-1]), inverse(c[::-1])]
            for j, q in enumerate(orbit):
                np.testing.assert_array_equal(q, data['permutations'][i, j, :n]); assert np.all(data['permutations'][i, j, n:] == 0)
                kk = key(q); assert kk not in seen, (name, i, j); seen.add(kk)
            q = orbit[4]; label = int(np.sum(q[:-1] > q[1:])); assert label == int(data['labels'][i])
            for action, q in enumerate([p, orbit[4]]):
                row = data['input'][i, action]; assert row[0] == tokens['<BOS>'] and row[1] == tokens['<SIZE>'] and row[2] == tokens[f'{n:02d}']
                assert row[3] == tokens['<ONE_START>'] and row[2*n+3] == tokens['<ONE_END>']
                np.testing.assert_array_equal(row[4:4+2*n:2], [tokens[f'{v:02d}'] for v in q])
                assert np.all(row[2*n+4:] == tokens['<PAD>'])
            examples += 1
    for archive in sig['excluded_archives']: assert sha(archive['path']) == archive['sha256']
    prior = old_inputs(sig['excluded_archives']); assert not seen.intersection(prior)
    manifest = Path(plan['repository']) / 'data/permutation-properties-16m-v1/manifest.json'
    parent = json.loads(manifest.read_text()); assert sha(manifest) == audit['original_parent_manifest_sha256']
    pattern = re.compile(rb'"primary"\s*:\s*(\[[^]]+\])'); scanned = 0
    for j, shard in enumerate(parent['shards']):
        path = manifest.parent / shard['filename']; assert sha(path) == shard['sha256']
        with gzip.open(path, 'rb') as handle:
            for line in handle:
                match = pattern.search(line); assert match is not None
                kk = match[1].replace(b' ', b''); assert kk not in seen, ('parent_overlap', kk)
                scanned += 1
        if j % 50 == 0: print({'independent_parent_inputs_checked': scanned}, flush=True)
    assert scanned == 16000000
    return {'examples': examples, 'eight_state_orbit_inputs': len(seen), 'original_inputs_rescanned': scanned,
            'prior_local_input_keys_rechecked': len(prior), 'all_source_inverse_labels_verified': True,
            'all_support_mismatches_train_only_length_and_answer_preserving': True}


@torch.inference_mode()
def direct(model, data, task, tokens, action, indices):
    """Original full LM forward and a norm hook; no producer prompt/forward helper."""
    device = next(model.parameters()).device; model.eval(); all_logits = []; all_hidden = []
    for start in range(0, len(indices), 128):
        ix = indices[start:start+128]; n = data['lengths'][ix]; batch = len(ix); stop = int(2 * n.max() + 6)
        ids = torch.full((batch, stop), tokens['<PAD>'], device=device, dtype=torch.long)
        for j, idx in enumerate(ix):
            end = int(2 * n[j] + 4); ids[j, :end] = torch.tensor(data['input'][idx, action, :end], device=device)
            ids[j, end] = tokens[f'<{task.upper()}>']; ids[j, end+1] = tokens['=']
        hidden = []; hook = model.final_norm.register_forward_hook(lambda module, args, out: hidden.append(out))
        logits = model(ids, ids != tokens['<PAD>']); hook.remove()
        rows = torch.arange(batch, device=device); lengths = torch.tensor(n, device=device)
        all_logits.append(logits[rows, 2*lengths+5].float().cpu().numpy())
        all_hidden.append(hidden[0][rows, 2*lengths+3].float().cpu().numpy())
    return {'logits': np.concatenate(all_logits), 'hidden': np.concatenate(all_hidden)}


def gram(x, y):
    x = x.astype(np.float64); y = y.astype(np.float64); x -= x.mean(0); y -= y.mean(0)
    k = x @ x.T; ell = y @ y.T; denom = np.sqrt(np.square(k).sum() * np.square(ell).sum())
    return None if denom < 1e-20 else float((k * ell).sum() / denom)


def check_geometry(root, plan, data):
    records = json.loads((root / 'auxiliary.json').read_text())['records']; errors = []
    for record in records:
        rep = next(r for r in plan['replicates'] if r['id'] == record['replicate'])
        teacher = dict(np.load(root / 'teacher' / f"test_s{rep['source_seed']}.npz"))
        if record['endpoint'] == 'initialization': x = np.load(root / 'evaluations' / f"{rep['id']}_initialization.npz")['hidden']
        else: x = np.load(root / 'evaluations' / f"{rep['id']}_{record['condition']}.npz")[record['endpoint'] + '_hidden']
        for row in record['scores']:
            ix = np.flatnonzero(data['lengths'] == row['length'])[:128]; a = x[ix]; b = teacher['hidden'][ix]; labels = data['labels'][ix]
            keep = np.array([(labels == v).sum() >= 3 for v in labels]); a0 = a[keep]; b0 = b[keep]; labels = labels[keep]
            ar = a0.astype(float).copy(); br = b0.astype(float).copy(); shuffled = np.arange(len(ar))
            for v in set(labels):
                j = np.flatnonzero(labels == v); ar[j] -= ar[j].mean(0); br[j] -= br[j].mean(0); shuffled[j] = np.r_[j[-1], j[:-1]]
            expected = {'correct_cka': gram(a, b), 'matched_correct_cka': gram(a0, b0), 'matched_wrong_cka': gram(a0, b0[shuffled]),
                        'answer_residual_correct_cka': gram(ar, br), 'answer_residual_wrong_cka': gram(ar, br[shuffled])}
            assert row['matched_rows'] == int(keep.sum())
            for name, value in expected.items():
                np.testing.assert_allclose(value, row[name], atol=2e-12, rtol=2e-12); errors.append(abs(value - row[name]))
    return {'gram_scores_independently_checked': len(errors), 'maximum_gram_covariance_error': max(errors)}


def run():
    plan, root, sig = initialize(); device = configure(); _, tokens, _, TrainConfig, factory = api(plan)
    assert json.loads((root / 'state.json').read_text())['status'] == 'functional_evaluation_complete'
    marker = json.loads((root / 'test_opened.json').read_text()); opened = marker['opened_utc']
    assert len(marker['selected_checkpoints']) == 30
    assert json.loads((root / 'training/state.json').read_text())['completed_utc'] < opened
    assert json.loads((root / 'auxiliary_protocol.json').read_text())['registered_utc'] < opened
    data_checks = verify_data(plan, root, sig); test = dict(np.load(root / 'dataset/test/dataset.npz'))
    sample = np.concatenate([np.flatnonzero(test['lengths'] == n)[::64] for n in plan['lengths']])
    gate = json.loads((root / 'teacher/gate.json').read_text()); teacher_errors = []; replay_errors = []; endpoints = 0; validations = 0
    for source in sig['sources']:
        assert sha(source['checkpoint']) == source['checkpoint_sha256']
        cp = torch.load(source['checkpoint'], weights_only=True, map_location='cpu'); cfg = TrainConfig.from_value(cp['config'])
        model = factory(cfg); model.load_state_dict(cp['model']); del cp; model.to(device).eval()
        audit = dict(np.load(root / 'dataset/audit/dataset.npz')); cached = dict(np.load(root / 'teacher' / f"audit_s{source['seed']}.npz"))
        hit = cached['logits'].argmax(-1) == audit['labels']; grade = next(g for g in gate['grades'] if g['source_seed'] == source['seed'])
        assert float(hit.mean()) == grade['accuracy']
        for n in plan['lengths']: assert float(hit[audit['lengths'] == n].mean()) == grade['per_length'][str(n)]
        assert grade['passes'] and hit.mean() >= plan['source_accuracy_gate']['overall']
        assert min(grade['per_length'].values()) >= plan['source_accuracy_gate']['every_length']
        out = direct(model, test, plan['source_task'], tokens, 1, sample)
        tc = dict(np.load(root / 'teacher' / f"test_s{source['seed']}.npz"))
        for branch in ['logits', 'hidden']:
            err = float(np.max(np.abs(out[branch] - tc[branch][sample]))); teacher_errors.append(err)
            np.testing.assert_allclose(out[branch], tc[branch][sample], atol=3e-4, rtol=3e-4)
        del model
    for rep in plan['replicates']:
        support = dict(np.load(root / 'dataset' / rep['id'] / 'dataset.npz')); initial = root / 'initializations' / f"{rep['id']}.pt"
        rng = np.random.default_rng(rep['data_seed'] + 81001); schedule = []
        for _ in range(plan['updates']):
            n = rng.choice(plan['lengths']); candidates = np.flatnonzero((support['split'] == 0) & (support['lengths'] == n))
            schedule.append(rng.choice(candidates, plan['batch_size'], replace=False))
        digest = hashlib.sha256(np.asarray(schedule).tobytes()).hexdigest()
        source = next(s for s in sig['sources'] if s['seed'] == rep['source_seed'])
        cp = torch.load(source['checkpoint'], weights_only=True, map_location='cpu'); cfg = TrainConfig.from_value(cp['config']); del cp
        torch.manual_seed(rep['target_seed']); initial_model = factory(cfg); state = torch.load(initial, weights_only=True, map_location='cpu')
        for name, value in initial_model.state_dict().items(): assert torch.equal(value, state[name]), ('initialization', rep['id'], name)
        del initial_model, state
        for condition in plan['conditions']:
            name = rep['id'] + '_' + condition; record = json.loads((root / 'training' / f'{name}.json').read_text())
            assert record['status'] == 'complete' and record['updated_utc'] < opened
            assert record['initialization_sha256'] == sha(initial) and record['schedule_sha256'] == digest
            assert record['teacher_archive_sha256'] == sha(root / 'teacher' / f"{rep['id']}.npz")
            assert [g['step'] for g in record['curve']] == plan['validation_updates']
            best = min(record['curve'], key=lambda g: (-g['accuracy'], g['cross_entropy'], g['step'])); assert best == record['selected']
            evaluation = json.loads((root / 'evaluations' / f'{name}.json').read_text()); file = root / 'evaluations' / f'{name}.npz'
            assert sha(file) == evaluation['archive_sha256']; cached = dict(np.load(file))
            for endpoint in ['selected', 'latest']:
                file = root / 'checkpoints' / f'{name}_{endpoint}.pt'; assert sha(file) == record[endpoint + '_checkpoint_sha256']
                if endpoint == 'selected': assert marker['selected_checkpoints'][name + '_selected'] == sha(file)
                saved = torch.load(file, weights_only=True, map_location='cpu'); assert saved['step'] == (best['step'] if endpoint == 'selected' else 1200)
                model = factory(cfg); model.load_state_dict(saved['model']); del saved; model.to(device).eval()
                answers = cached[endpoint + '_answers']; logits = cached[endpoint + '_logits']; hit = answers == test['labels']
                np.testing.assert_array_equal(answers, logits.argmax(-1)); assert float(hit.mean()) == evaluation['results'][endpoint]['answer_accuracy']
                for n in plan['lengths']: assert float(hit[test['lengths'] == n].mean()) == evaluation['results'][endpoint]['per_length'][str(n)]
                out = direct(model, test, plan['target_task'], tokens, 0, sample)
                for branch in ['logits', 'hidden']:
                    expected = cached[endpoint + '_' + branch][sample]; err = float(np.max(np.abs(out[branch] - expected))); replay_errors.append(err)
                    np.testing.assert_allclose(out[branch], expected, atol=3e-4, rtol=3e-4)
                np.testing.assert_array_equal(out['logits'].argmax(-1), answers[sample])
                if endpoint == 'selected':
                    vx = np.flatnonzero(support['split'] == 1); v = direct(model, support, plan['target_task'], tokens, 0, vx)['logits']
                    assert float((v.argmax(-1) == support['labels'][vx]).mean()) == best['accuracy']
                    ce = float(F.cross_entropy(torch.tensor(v), torch.tensor(support['labels'][vx])))
                    np.testing.assert_allclose(ce, best['cross_entropy'], atol=5e-5, rtol=5e-5); validations += 1
                endpoints += 1; del model
        print({'replicate_independently_verified': rep['id'], 'endpoints': endpoints}, flush=True)
    geometry_checks = check_geometry(root, plan, test)
    atomic_json(root / 'verification.json', {'status': 'passed', 'completed_utc': now(), 'data': data_checks,
        'accuracy_endpoints_checked': endpoints, 'validation_selections_replayed': validations,
        'original_attention_test_replays': endpoints, 'test_samples_per_replay': len(sample),
        'maximum_original_attention_logit_or_hidden_error': max(replay_errors), 'maximum_teacher_replay_error': max(teacher_errors),
        'source_checkpoint_hashes_verified': len(sig['sources']), 'target_initializations_recreated': 6,
        'paired_batch_schedules_recreated': 6, 'all_30_fits_complete_before_test_opening': True,
        'auxiliary_registered_before_test_opening': True, **geometry_checks})


if __name__ == '__main__': run()
