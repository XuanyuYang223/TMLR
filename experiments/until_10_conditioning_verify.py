"""Replay identical-subset comparisons with independent selectors and projections."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import torch

from .until_10_verify import ROOT, DEADLINE, read, archive, digest, save, action_order, pair_rows
from .until_10_control_verify import head


def verify(path):
    rec = read(path)
    for artifact, expected in rec['artifact_sha256'].items():
        assert digest(artifact) == expected, artifact
    data = archive('results/joint_answer_matched_geometry/dataset.npz')
    pairs = pair_rows(data)
    ids = np.unique(data['pair_ids'])
    union = read('results/joint_answer_matched_geometry/dataset_audit.json')['union_tasks']
    groups = read('configs/algebra_structure_replication.json')['groups']
    tasks = next(g['tasks'] for g in groups if g['id'] == rec['group'])
    truth = data['union_labels'][:, :, [union.index(task) for task in tasks]]
    entries = {}
    for status, key in [('trained', 'trained_record_path'), ('random', 'random_record_path')]:
        original = read(rec[key])
        root = Path(rec[key]).parent.parent
        new = root.name in ['ordinary_relation_seed_confirmation', 'ordinary_initialization_control']
        fp = root / 'features' / (original['source'] + ('_joint_test.npz' if new else '.npz'))
        hidden = archive(fp)['source_query_concat']
        weight, bias = head(original, root)
        contrast = weight - weight.mean(0)
        projection = contrast.T @ np.linalg.pinv(contrast @ contrast.T, rcond=1e-12) @ contrast
        block = hidden.reshape(len(data['lengths']), 8, 4, -1).astype(np.float64)
        null = (block - block @ projection).reshape(hidden.shape)
        entries[status] = {'root': root, 'source': original['source'], 'raw': hidden, 'null': null,
            'answers': (block @ weight.T + bias).argmax(-1)}
    answers = entries['trained']['answers']
    masks = {'trained_model_answers_agree_all_states': (answers[pairs[:, 0]] == answers[pairs[:, 1]]).all(axis=(1, 2)),
        'trained_model_answers_correct_all_states': (answers[pairs] == truth[pairs]).all(axis=(1, 2, 3))}
    for selector, selected in rec['selections'].items():
        mask = masks[selector]
        assert selected['pairs'] == int(mask.sum())
        np.testing.assert_array_equal(selected['pair_ids'], ids[mask])
        assert selected['truth_used_for_selection'] == (selector == 'trained_model_answers_correct_all_states')
    total = 0
    for status, entry in entries.items():
        for view in ['source_query_concat', 'source_query_numeric_null']:
            hidden = entry['null'] if view.endswith('numeric_null') else entry['raw']
            difference = (hidden[pairs[:, 0]] - hidden[pairs[:, 1]]).astype(np.float64)
            ap = entry['root'] / 'arrays' / (entry['source'] + '_' + view + '_all_joint_answer_matched_pairs.npz')
            predictions = archive(ap)
            np.testing.assert_array_equal(predictions['pair_ids'], ids)
            rows = [r for r in rec['results'] if r['status'] == status and r['view'] == view]
            for row in rows:
                mask = masks[row['selector']]
                prediction = predictions[row['word'] + '_pair_prediction'].reshape(difference.shape)[mask].astype(np.float64)
                target = difference[mask][:, action_order(row['word'])]
                source = difference[mask]
                error = np.square(prediction - target).sum()
                np.testing.assert_allclose(error / np.square(target).sum(), row['pair_target_nmse'], rtol=1e-9, atol=1e-10)
                np.testing.assert_allclose(error / np.square(target - source).sum(), row['pair_action_displacement_nmse'], rtol=1e-9, atol=1e-10)
                assert row['pairs'] == int(mask.sum())
                total += 1
    return {'source': rec['source'], 'status': 'passed', 'independently_replayed_endpoints': total,
        'same_pair_ids_used_for_trained_and_random': True, 'selectors_independently_reconstructed': True,
        'scope': 'Both answer-selection rules and raw/numeric-null geometry scores, with the same selected inputs for each trained/random pair. Confidence remains uncontrolled; this conditional diagnostic does not replace the full test.'}


def run():
    cache_path = ROOT / 'conditioning_verification_cache.json'
    cache = read(cache_path) if cache_path.exists() else {}
    code = digest(__file__) + digest('experiments/until_10_verify.py') + digest('experiments/until_10_control_verify.py')
    checks, failures = [], []
    for path in sorted(Path('results/matched_prediction_conditioning/evaluations').glob('*.json')):
        key = str(path)
        fingerprint = code + digest(path)
        if key in cache and cache[key]['fingerprint'] == fingerprint:
            checks.append(cache[key]['result'])
            continue
        try:
            result = verify(path)
            cache[key] = {'fingerprint': fingerprint, 'result': result}
            save(cache_path, cache)
            checks.append(result)
        except Exception as error:
            failures.append({'check': key, 'error': repr(error)})
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'passed_available_completed_records' if not failures else 'failed',
        'checks_completed': len(checks), 'checks': checks, 'failures': failures,
        'independently_replayed_endpoints': sum(c['independently_replayed_endpoints'] for c in checks),
        'verifier_code_sha256': digest(__file__)}
    save(ROOT / 'conditioning_verification.json', result)
    print({'conditioning_verification': result['status'], 'sources': len(checks), 'failures': failures}, flush=True)
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
