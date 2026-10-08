"""Independent action-target and direct-block control checks on frozen probes."""
import argparse
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path
import time

import numpy as np
import torch

from .until_10_verify import ROOT, DEADLINE, read, archive, digest, save, action_order, pair_rows
from .until_10_control_verify import head


WORDS = ['', 'c', 'r', 'rc', 'i', 'ci', 'ri', 'rci']
ORDERS = [action_order(word) for word in WORDS]


def features_for(rec):
    candidates = [Path('results/joint_answer_matched_geometry'), Path('results/ordinary_relation_seed_confirmation')]
    root = next(root for root in candidates if (root / 'evaluations' / (rec['source'] + '.json')).exists())
    suffix = '_joint_test.npz' if root.name == 'ordinary_relation_seed_confirmation' else '.npz'
    path = root / 'features' / (rec['source'] + suffix)
    assert digest(path) == rec['feature_sha256']
    values = archive(path)['source_query_concat']
    model_rec = read(root / 'evaluations' / (rec['source'] + '.json'))
    weight, _ = head(model_rec, root)
    contrast = weight - weight.mean(0)
    projection = contrast.T @ np.linalg.pinv(contrast @ contrast.T, rcond=1e-12) @ contrast
    blocks = values.reshape(*values.shape[:-1], 4, -1).astype(np.float64)
    null = (blocks - blocks @ projection).reshape(values.shape)
    return root, {'source_query_concat': values, 'source_query_numeric_null': null}


def specificity(path):
    rec = read(path)
    root, features = features_for(rec)
    data = archive('results/joint_answer_matched_geometry/dataset.npz')
    pairs = pair_rows(data)
    lengths = data['lengths'][pairs[:, 0]]
    partner = np.arange(len(pairs))
    for n in np.unique(lengths):
        indices = np.flatnonzero(lengths == n)
        partner[indices] = np.roll(indices, 1)
    total = 0
    for view, hidden in features.items():
        differences = (hidden[pairs[:, 0]] - hidden[pairs[:, 1]]).astype(np.float64)
        source = differences.reshape(-1, differences.shape[-1])
        pred_path = root / 'arrays' / (rec['source'] + '_' + view + '_all_joint_answer_matched_pairs.npz')
        assert digest(pred_path) == rec['prediction_archive_sha256'][str(pred_path)]
        predictions = archive(pred_path)
        for row in [r for r in rec['results'] if r['view'] == view]:
            order = action_order(row['word'])
            action = int(order[0])
            assert action == row['correct_action']
            prediction = predictions[row['word'] + '_pair_prediction'].astype(np.float64)
            target = differences[:, order].reshape(source.shape)
            denominator = np.square(target).sum()
            errors = [float(np.square(prediction - differences[:, other].reshape(source.shape)).sum() / denominator)
                for other in ORDERS]
            np.testing.assert_allclose(errors, row['wrong_action_target_nmse'], rtol=1e-9, atol=1e-10)
            correct = errors[action]
            wrong = [v for j, v in enumerate(errors) if j != action]
            np.testing.assert_allclose(correct, row['correct_pair_target_nmse'], rtol=1e-9, atol=1e-10)
            assert 1 + sum(v < correct - 1e-12 for v in wrong) == row['correct_action_rank_among_8']
            np.testing.assert_allclose(min(wrong) - correct, row['best_wrong_minus_correct_nmse'], atol=1e-9)
            np.testing.assert_allclose(np.mean(wrong) - correct, row['mean_wrong_minus_correct_nmse'], atol=1e-9)
            shuffled = differences[partner][:, order].reshape(source.shape)
            mismatch = np.square(prediction - shuffled).sum() / denominator
            np.testing.assert_allclose(mismatch, row['length_matched_pair_derangement_nmse'], rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(mismatch - correct, row['pair_derangement_minus_correct_nmse'], atol=1e-9)
            if row['word'] == 'ci':
                np.testing.assert_allclose(errors[int(action_order('ic')[0])] - correct,
                    row['IC_target_minus_CI_target_nmse'], atol=1e-9)
            total += 1
    return {'source': rec['source'], 'status': 'passed', 'independently_replayed_endpoints': total,
        'scope': 'All eight true action targets, correct-action ranks, wrong-order and length-matched pair controls. No operator or prediction is refitted.'}


def task_blocks(path):
    rec = read(path)
    _, features = features_for(rec)
    data = archive('results/joint_answer_matched_geometry/dataset.npz')
    pairs = pair_rows(data)
    # Orders obtained by counting actual permutation records for each action,
    # rather than importing the producer's task permutation constants.
    from .until_10_verify import counts, orbit
    tasks = ['left_to_right_maxima', 'right_to_left_maxima', 'left_to_right_minima', 'right_to_left_minima']
    reference = np.random.default_rng(2026100601)
    raw = [reference.permutation(13) + 1 for _ in range(200)]
    truth = np.array([[counts(p, tasks) for p in orbit(x)] for x in raw])
    task_orders = {}
    for g in ['c', 'r', 'i']:
        action = int(action_order(g)[0])
        matches = [order for order in permutations(range(4)) if np.array_equal(truth[:, 0, list(order)], truth[:, action])]
        assert len(matches) == 1
        task_orders[g] = np.array(matches[0])
    total = 0
    for view, hidden in features.items():
        # Producer promotes block features to double before subtraction.
        values = hidden.astype(np.float64)
        delta = (values[pairs[:, 0]] - values[pairs[:, 1]]).reshape(len(pairs), 8, 4, -1)
        source = delta.reshape(-1, 4, delta.shape[-1])
        for row in rec['results'][view]:
            index = np.arange(4)
            for g in row['word']:
                index = index[task_orders[g]]
            np.testing.assert_array_equal(index, row['true_task_block_order'])
            target = delta[:, action_order(row['word'])].reshape(source.shape)
            error = np.square(source[:, index] - target).sum()
            np.testing.assert_allclose(error / np.square(target - source).sum(),
                row['block_pair_action_displacement_nmse'], rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(error / np.square(target).sum(),
                row['block_pair_target_nmse'], rtol=1e-9, atol=1e-10)
            total += 1
    return {'source': rec['source'], 'status': 'passed', 'independently_replayed_endpoints': total,
        'scope': 'All supplied correct task-block permutations independently derived from permutation record counts, and both normalized errors. The 23-wrong-block rank/mean diagnostics are not independently replayed here.'}


def run():
    cp = ROOT / 'specificity_verification_cache.json'
    cache = read(cp) if cp.exists() else {}
    code = digest(__file__) + digest('experiments/until_10_control_verify.py') + digest('experiments/until_10_verify.py')
    checks, failures = [], []
    for location, operation in [('conditional_relation_specificity', specificity), ('task_block_geometry_control', task_blocks)]:
        for path in sorted((Path('results') / location / 'evaluations').glob('*.json')):
            key = str(path)
            fingerprint = code + digest(path)
            if key in cache and cache[key]['fingerprint'] == fingerprint:
                checks.append(cache[key]['result'])
                continue
            try:
                result = {**operation(path), 'study': location}
                cache[key] = {'fingerprint': fingerprint, 'result': result}
                save(cp, cache)
                checks.append(result)
            except Exception as error:
                failures.append({'check': key, 'error': repr(error)})
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'passed_available_completed_records' if not failures else 'failed', 'checks_completed': len(checks),
        'checks': checks, 'failures': failures, 'independently_replayed_endpoints': sum(c['independently_replayed_endpoints'] for c in checks),
        'verifier_code_sha256': digest(__file__)}
    save(ROOT / 'specificity_verification.json', result)
    print({'specificity_verification': result['status'], 'checks': len(checks), 'failures': failures}, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    while True:
        run()
        if not args.watch or time.time() >= DEADLINE:
            break
        time.sleep(min(45, max(0, DEADLINE-time.time())))
