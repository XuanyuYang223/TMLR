"""Independent audits for the fixed readouts and extra-length measurements.

This module does not fit heads, update networks or select endpoints. It uses
integer counts, original model forwards, sample Gram matrices and independently
reconstructed ridge normal equations to check the supplementary artifacts.
"""
import argparse
from collections import Counter
from datetime import datetime
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import torch

from .inverse_functional_alignment import configure, now
from .inverse_functional_verify import direct, gram
from .longrun_engine import atomic_json
from .overnight_inverse_functional import initialize
from .permworld_combinations import sha
from .specialist_cka_controls import api


def register(root):
    folder = root / 'addendum_verification'
    folder.mkdir(exist_ok=True)
    path = folder / 'protocol.json'
    signature = {
        'code_path': __file__, 'code_sha256': sha(__file__),
        'base_protocol_sha256': sha(root / 'protocol.json'),
        'checks': [
            'Reconstruct all60 ridge heads U-only means, standard deviations, '
            'intercepts, normal equations and pseudo-answer training accuracy.',
            'Replay original unaccelerated frozen teacher and native target '
            'forwards on the first8 U examples at each length in all6 pools.',
            'Recount overall, modal/nonmodal and per-length accuracy for both '
            'readout probes with identical main-test masks.',
            'Reconstruct extra-length modes from calibration predictions using '
            'Counter; check mappings were frozen before that test was opened.',
            'Replay all42 extra-length target endpoints and3 teachers on '
            'the first8 examples per length; recompute1680 Gram CKA values.',
            'Reconstruct extra-length paired differences and empirical '
            'bootstrap intervals without calling producer summary helpers.',
        ],
        'scope': 'Verification only; no new model fit, checkpoint selection, '
                 'additional labels, or alteration of registered experiments.',
    }
    if path.exists():
        assert json.loads(path.read_text())['signature'] == signature
    else:
        assert not (root / 'test_opened.json').exists()
        atomic_json(path, {'registered_utc': now(),
                          'main_test_opened': False, 'signature': signature})
    return folder


def close(value, expected, tolerance=2e-12):
    if expected is None:
        assert value is None
    else:
        np.testing.assert_allclose(value, expected, atol=tolerance, rtol=tolerance)


def counts(row, prediction, truth, modal, lengths):
    """Count with Python integers, independently of the NumPy mean producer."""
    hit = [int(p) == int(y) for p, y in zip(prediction, truth)]
    assert len(hit) == len(truth) and len(hit) > 0
    close(row['accuracy'], sum(hit) / len(hit))
    close(row['modal_fraction'], sum(map(bool, modal)) / len(hit))
    checks = 2
    weighted_hits = 0
    for name, flag in [('modal', True), ('nonmodal', False)]:
        subset = [h for h, m in zip(hit, modal) if bool(m) == flag]
        assert row[name]['count'] == len(subset)
        close(row[name]['accuracy'], sum(subset) / len(subset) if subset else None)
        weighted_hits += sum(subset)
        checks += 2
    assert weighted_hits == sum(hit)
    for length in sorted(set(map(int, lengths))):
        use = lengths == length
        checks += counts(row['per_length'][str(length)], prediction[use],
                         truth[use], modal[use], np.array([], dtype=int))
    return checks


def audit_heads(plan, root):
    checked = 0
    maximum_residual = 0.0
    for name in ['readout_probe', 'frozen_target_probe']:
        folder = root / name
        info = json.loads((folder / 'fit.json').read_text())
        assert info['linear_heads'] == 30 and info['true_U_answers_computed'] is False
        assert json.loads((folder / 'protocol.json').read_text())['registered_utc'] < info['completed_utc']
        for rep in plan['replicates']:
            pool = np.load(root / 'dataset' / rep['id'] / 'unlabeled/dataset.npz')
            assert 'labels' not in pool.files
            teacher = np.load(root / 'teacher' / f"{rep['id']}_unlabeled.npz")
            labels = teacher['logits'].argmax(-1)
            np.testing.assert_array_equal(labels, teacher['predicted_answer'])
            features = teacher['hidden'] if name == 'readout_probe' else np.load(
                folder / f"{rep['id']}_native_U.npz")['hidden']
            file = folder / f"{rep['id']}_heads.npz"
            assert sha(file) == info['head_sha256'][file.name]
            head = np.load(file)
            for length in plan['lengths']:
                use = pool['lengths'] == length
                x = features[use].astype(np.float64)
                mean = x.mean(axis=0)
                std = np.maximum(np.sqrt(np.mean((x - mean) ** 2, axis=0)), 1e-6)
                close(head[f'n{length}_mean'], mean)
                close(head[f'n{length}_std'], std)
                classes = labels[use]
                intercept = np.array([Counter(map(int, classes))[j] / len(classes)
                                      for j in range(188)])
                close(head[f'n{length}_intercept'], intercept)
                z = (x - mean) / std
                centered = np.eye(188)[classes] - intercept
                weights = head[f'n{length}_weights']
                assert weights.shape == (x.shape[1], 188)
                residual = (z.T @ z / len(z) + .01 * np.eye(z.shape[1])) @ weights
                residual -= z.T @ centered / len(z)
                error = float(np.max(np.abs(residual)))
                assert error < 1e-9, (name, rep['id'], length, error)
                maximum_residual = max(maximum_residual, error)
                predicted = (z @ weights + intercept).argmax(-1)
                accuracy = sum(int(a) == int(b) for a, b in zip(predicted, classes)) / len(classes)
                record = next(r for r in info['records'] if r['replicate'] == rep['id'] and r['length'] == length)
                field = 'pseudo_answer_training_accuracy' if name == 'readout_probe' else 'pseudo_answer_fit_accuracy'
                close(record[field], accuracy)
                checked += 1
    return {'ridge_heads_independently_checked': checked,
            'maximum_reconstructed_normal_equation_residual': maximum_residual,
            'oracle_U_answers_computed': False}


def native_model(plan, entry, device, state=None):
    _, _, _, config, factory = api(plan)
    cp = torch.load(entry['checkpoint'], map_location='cpu', weights_only=True)
    model = factory(config.from_value(cp['config']))
    model.load_state_dict(cp['model'] if state is None else state)
    del cp
    return model.to(device).eval()


def replay_cache(model, data, task, tokens, action, saved, lengths):
    indices = np.concatenate([np.flatnonzero(data['lengths'] == n)[:8] for n in lengths])
    result = direct(model, data, task, tokens, action, indices)
    errors = []
    for branch in ['hidden', 'logits']:
        np.testing.assert_allclose(result[branch], saved[branch][indices], atol=3e-4, rtol=3e-4)
        errors.append(float(np.abs(result[branch] - saved[branch][indices]).max()))
    return max(errors)


def replay_pools(plan, root, signature, device, tokens):
    errors = []
    for kind, action, task, entries in [
        ('teacher', 1, plan['source_task'], signature['sources']),
        ('target', 0, plan['target_task'], signature['targets']),
    ]:
        for entry in entries:
            model = native_model(plan, entry, device)
            for rep in plan['replicates']:
                seed = rep['source_seed'] if kind == 'teacher' else rep['target_pretrain_seed']
                if seed != entry['seed']:
                    continue
                data = dict(np.load(root / 'dataset' / rep['id'] / 'unlabeled/dataset.npz'))
                assert 'labels' not in data
                file = root / 'teacher' / f"{rep['id']}_unlabeled.npz" if kind == 'teacher' else root / 'frozen_target_probe' / f"{rep['id']}_native_U.npz"
                saved = np.load(file)
                errors.append(replay_cache(model, data, task, tokens, action, saved, plan['lengths']))
            del model
    return {'original_forward_U_caches_replayed': len(errors),
            'maximum_U_cache_replay_error': max(errors)}


def audit_probe_results(plan, root):
    opened = json.loads((root / 'test_opened.json').read_text())['opened_utc']
    data = np.load(root / 'dataset/test/dataset.npz')
    truth, lengths = data['labels'], data['lengths']
    modes = json.loads((root / 'length_modes.json').read_text())['records']
    checked = 0
    for name in ['readout_probe', 'frozen_target_probe']:
        folder = root / name
        summary = json.loads((folder / 'summary.json').read_text())
        info = json.loads((folder / 'fit.json').read_text())
        assert info['completed_utc'] < opened
        for rep in plan['replicates']:
            teacher = np.load(root / 'teacher' / f"test_s{rep['source_seed']}.npz")
            native = np.load(root / 'evaluations' / f"{rep['id']}_initial.npz")
            features = teacher['hidden'] if name == 'readout_probe' else native['hidden']
            head = np.load(folder / f"{rep['id']}_heads.npz")
            predicted = np.empty(len(truth), dtype=int)
            for length in plan['lengths']:
                ix = np.flatnonzero(lengths == length)
                z = (features[ix] - head[f'n{length}_mean']) / head[f'n{length}_std']
                predicted[ix] = (z @ head[f'n{length}_weights'] + head[f'n{length}_intercept']).argmax(-1)
            mapping = next(r['modes'] for r in modes if r['replicate'] == rep['id'] and r['prior'] == 'teacher')
            prior = np.array([mapping[str(int(n))] for n in lengths])
            answers = {'ONE_END_ridge': predicted, 'native_ONE_END_ridge': predicted,
                       'teacher_query_output': teacher['logits'].argmax(-1),
                       'inverse_teacher_query': teacher['logits'].argmax(-1),
                       'native_query_output': native['logits'].argmax(-1),
                       'teacher_length_mode': prior}
            for row in summary['records']:
                if row['replicate'] == rep['id']:
                    checked += counts(row, answers[row['condition']], truth, prior == truth, lengths)
        for row in summary['means']:
            records = [r for r in summary['records'] if r['condition'] == row['condition']]
            assert len(records) == len(plan['replicates'])
            close(row['accuracy'], sum(r['accuracy'] for r in records) / len(records))
            close(row['nonmodal_accuracy'], sum(r['nonmodal']['accuracy'] for r in records) / len(records))
            if 'modal_accuracy' in row:
                close(row['modal_accuracy'], sum(r['modal']['accuracy'] for r in records) / len(records))
    return {'independent_probe_accuracy_and_subgroup_cells_checked': checked}


def gram_cells(out, teacher, lengths, truth, records):
    checked = 0
    for branch, cells in [('hidden', records['per_length_geometry']),
                          ('logits', records['per_length_output_geometry'])]:
        for row in cells:
            ix = np.flatnonzero(lengths == row['length'])[:128]
            a, b, labels = out[branch][ix], teacher[branch][ix], truth[ix]
            frequencies = Counter(map(int, labels))
            keep = np.array([frequencies[int(v)] >= 3 for v in labels])
            aa, bb, yy = a[keep], b[keep], labels[keep]
            assert row['rows'] == len(ix) and row['matched_rows'] == int(keep.sum())
            ar, br = aa.astype(float).copy(), bb.astype(float).copy()
            permutation = np.arange(len(aa))
            for value in sorted(set(map(int, yy))):
                j = np.flatnonzero(yy == value)
                ar[j] -= sum(ar[j]) / len(j)
                br[j] -= sum(br[j]) / len(j)
                permutation[j] = [j[-1]] + list(j[:-1])
            expected = {'correct_cka': gram(a, b),
                        'matched_correct_cka': gram(aa, bb),
                        'matched_wrong_cka': gram(aa, bb[permutation]),
                        'answer_residual_correct_cka': gram(ar, br),
                        'answer_residual_wrong_cka': gram(ar, br[permutation])}
            for key, value in expected.items():
                close(row[key], value)
                checked += 1
    for key in ['correct_cka', 'answer_residual_correct_cka', 'answer_residual_wrong_cka']:
        close(records['geometry'][key], sum(r[key] for r in records['per_length_geometry']) / len(records['per_length_geometry']))
    close(records['geometry']['answer_residual_contrast'],
          records['geometry']['answer_residual_correct_cka'] - records['geometry']['answer_residual_wrong_cka'])
    return checked


def audit_bootstrap(saved, values, reps):
    values = list(map(float, values))
    close(saved['mean'], sum(values) / len(values))
    close(saved['paired_values'], values)
    assert saved['positive_repeats'] == sum(v > 0 for v in values)
    clusters = [sum(v for v, rep in zip(values, reps) if rep['source_seed'] == seed)
                / sum(rep['source_seed'] == seed for rep in reps) for seed in [17, 42, 101]]
    close(saved['three_pair_means'], clusters)
    for sample, key in [(values, 'paired_bootstrap_95'), (clusters, 'three_pair_bootstrap_95')]:
        distribution = [sum(draw) / len(draw) for draw in product(sample, repeat=len(sample))]
        close(saved[key], np.quantile(distribution, [.025, .975]))


def audit_lengths(plan, root, signature, device, tokens):
    folder = root / 'heldout_lengths'
    protocol = json.loads((folder / 'protocol.json').read_text())
    registered = protocol['signature']
    assert sha(registered['code_path']) == registered['code_sha256']
    modes = json.loads((folder / 'length_modes.json').read_text())
    opened = json.loads((folder / 'test_opened.json').read_text())
    assert protocol['registered_utc'] < modes['frozen_utc'] < opened['opened_utc']
    assert sha(folder / 'length_modes.json') == opened['modes_sha256']
    calibration = np.load(folder / 'dataset/calibration/dataset.npz')
    assert 'labels' not in calibration.files
    data = dict(np.load(folder / 'dataset/test/dataset.npz'))
    truth, lengths = data['labels'], data['lengths']
    summary = json.loads((folder / 'summary.json').read_text())
    assert summary['lengths'] == registered['lengths'] and len(truth) == summary['test_examples'] == 4096
    checked_modes, cells, grams, errors = 0, 0, 0, []
    for source in signature['sources']:
        saved_cal = np.load(folder / 'teacher' / f"cal_s{source['seed']}.npz")
        for length in registered['lengths']:
            votes = Counter(map(int, saved_cal['logits'][calibration['lengths'] == length].argmax(-1)))
            mode = max(votes, key=lambda v: (votes[v], -v))
            assert mode == modes['mappings'][str(source['seed'])][str(length)]
            checked_modes += 1
        model = native_model(plan, source, device)
        saved = np.load(folder / 'teacher' / f"test_s{source['seed']}.npz")
        errors.append(replay_cache(model, data, plan['source_task'], tokens, 1, saved, registered['lengths']))
        errors.append(replay_cache(model, dict(calibration), plan['source_task'], tokens, 1, saved_cal, registered['lengths']))
        del model
        mapping = modes['mappings'][str(source['seed'])]
        modal = np.array([mapping[str(int(n))] == int(y) for n, y in zip(lengths, truth)])
        row = next(r for r in summary['source_accuracy'] if r['source_seed'] == source['seed'])
        cells += counts(row, saved['logits'].argmax(-1), truth, modal, lengths)
    for rep in plan['replicates']:
        target = next(t for t in signature['targets'] if t['seed'] == rep['target_pretrain_seed'])
        teacher = np.load(folder / 'teacher' / f"test_s{rep['source_seed']}.npz")
        mapping = modes['mappings'][str(rep['source_seed'])]
        modal = np.array([mapping[str(int(n))] == int(y) for n, y in zip(lengths, truth)])
        for condition in ['initial'] + plan['conditions']:
            row = next(r for r in summary['records'] if r['replicate'] == rep['id'] and r['condition'] == condition)
            file = folder / 'evaluations' / f"{rep['id']}_{condition}.npz"
            assert sha(file) == row['archive_sha256']
            out = np.load(file)
            cells += counts(row, out['logits'].argmax(-1), truth, modal, lengths)
            grams += gram_cells(out, teacher, lengths, truth, row)
            if condition == 'initial':
                model = native_model(plan, target, device)
            else:
                file = root / 'checkpoints' / f"{rep['id']}_{condition}_u40000.pt"
                record = json.loads((root / 'training' / f"{rep['id']}_{condition}.json").read_text())
                assert record['status'] == 'complete' and record['step'] == 40000
                assert sha(file) == record['checkpoint_sha256']['40000']
                state = torch.load(file, map_location='cpu', weights_only=True)
                assert state['step'] == 40000
                model = native_model(plan, target, device, state['model'])
                del state
            errors.append(replay_cache(model, data, plan['target_task'], tokens, 0, out, registered['lengths']))
            del model
        print({'verified_extra_length_repeat': rep['id']}, flush=True)
    for row in summary['means']:
        records = [r for r in summary['records'] if r['condition'] == row['condition']]
        assert len(records) == 6
        close(row['accuracy'], sum(r['accuracy'] for r in records) / len(records))
        close(row['nonmodal_accuracy'], sum(r['nonmodal']['accuracy'] for r in records) / len(records))
        close(row['cka'], sum(r['geometry']['correct_cka'] for r in records) / len(records))
    for row in summary['contrasts']:
        left, right = row['contrast'].split('-')
        values = []
        for rep in plan['replicates']:
            a = next(r for r in summary['records'] if r['replicate'] == rep['id'] and r['condition'] == left)
            b = next(r for r in summary['records'] if r['replicate'] == rep['id'] and r['condition'] == right)
            values.append(100 * (a['accuracy'] - b['accuracy']))
        audit_bootstrap(row['accuracy_pp'], values, plan['replicates'])
    return {'extra_length_modes_checked': checked_modes,
            'extra_length_accuracy_and_subgroup_cells_checked': cells,
            'extra_length_gram_scores_checked': grams,
            'extra_length_original_forward_caches_replayed': len(errors),
            'maximum_extra_length_replay_error': max(errors),
            'extra_length_paired_bootstrap_contrasts_checked': len(summary['contrasts'])}


def run(phase):
    plan, root, signature = initialize()
    folder = register(root)
    if phase == 'register':
        return
    if phase == 'heads':
        result = audit_heads(plan, root)
        atomic_json(folder / 'head_verification.json', {'completed_utc': now(), **result})
        print(result, flush=True)
        return
    deadline = datetime.fromisoformat(plan['deadline_utc']).timestamp()
    required = ['readout_probe/summary.json', 'frozen_target_probe/summary.json',
                'heldout_lengths/summary.json', 'verification.json']
    if phase == 'watch':
        while not all((root / file).exists() for file in required):
            if time.time() >= deadline:
                atomic_json(folder / 'state.json', {'status': 'incomplete', 'updated_utc': now(),
                           'missing': [file for file in required if not (root / file).exists()]})
                return
            time.sleep(10)
    assert all((root / file).exists() for file in required)
    device = configure()
    _, tokens, _, _, _ = api(plan)
    result = {**audit_heads(plan, root), **audit_probe_results(plan, root),
              **replay_pools(plan, root, signature, device, tokens),
              **audit_lengths(plan, root, signature, device, tokens)}
    atomic_json(folder / 'verification.json', {'completed_utc': now(), 'status': 'complete',
                'code_sha256': sha(__file__), **result})
    atomic_json(folder / 'state.json', {'status': 'complete', 'updated_utc': now()})
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['register', 'heads', 'verify', 'watch'])
    run(parser.parse_args().phase)
