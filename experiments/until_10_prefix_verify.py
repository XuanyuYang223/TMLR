"""Independent matrix-product replay of task-free prefix confirmation scores."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from .until_10_verify import ROOT, DEADLINE, read, archive, digest, save, action_order, pair_rows


def verify(path):
    rec = read(path)
    for p, key in [(rec['source_record_path'], 'source_record_sha256'),
            (rec['calibration_feature_path'], 'calibration_feature_sha256'),
            (rec['test_feature_path'], 'test_feature_sha256'), (rec['map_path'], 'map_sha256')]:
        assert digest(p) == rec[key], p
    data = archive('results/joint_answer_matched_geometry/dataset.npz')
    hidden = archive(rec['test_feature_path'])['ONE_END']
    maps = archive(rec['map_path'])
    saved_path = path.parent.parent / 'arrays' / (rec['source'] + '.npz')
    assert digest(saved_path) == rec['prediction_archive_sha256']
    predictions = archive(saved_path)
    q = maps['basis'].astype(np.float64)
    np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-7)
    pairs = pair_rows(data)
    np.testing.assert_array_equal(predictions['pair_ids'], np.unique(data['pair_ids']))
    delta = (hidden[pairs[:, 0]] - hidden[pairs[:, 1]]).astype(np.float64)
    source = delta.reshape(-1, hidden.shape[-1]).T
    latent = q.T @ source
    for row in rec['results']:
        product = np.eye(q.shape[1])
        for g in row['word']:
            product = maps['map_' + g].astype(np.float64).T @ product
        predicted = (source + q @ (product @ latent - latent)).T
        target = delta[:, action_order(row['word'])].reshape(predicted.shape)
        displacement = target - source.T
        error = np.square(predicted - target).sum()
        np.testing.assert_allclose(error / np.square(target).sum(), row['pair_target_nmse'], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(error / np.square(displacement).sum(), row['pair_action_displacement_nmse'], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(np.square(displacement).sum() / np.square(target).sum(), row['identity_pair_target_nmse'], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(predicted, predictions[row['word'] + '_pair_prediction'], rtol=1e-6, atol=1e-6)
        assert row['pairs'] == len(pairs)
    return {'source': rec['source'], 'status': 'passed', 'independently_replayed_endpoints': len(rec['results']),
        'scope': 'All saved generator/composite prefix predictions, both error denominators and identity baselines. Uses independent column-oriented products and manual action orders. Calibration PCA and generator ridge fitting are not refitted by this verifier.'}


def run():
    cp = ROOT / 'prefix_verification_cache.json'
    cache = read(cp) if cp.exists() else {}
    code = digest(__file__) + digest('experiments/until_10_verify.py')
    checks, failures = [], []
    for path in sorted(Path('results/ordinary_prefix_confirmation/evaluations').glob('*.json')):
        key = str(path)
        fingerprint = code + digest(path)
        if key in cache and cache[key]['fingerprint'] == fingerprint:
            checks.append(cache[key]['result'])
            continue
        try:
            result = verify(path)
            cache[key] = {'fingerprint': fingerprint, 'result': result}
            save(cp, cache)
            checks.append(result)
        except Exception as error:
            failures.append({'check': key, 'error': repr(error)})
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'passed_available_completed_records' if not failures else 'failed', 'checks_completed': len(checks),
        'checks': checks, 'failures': failures, 'independently_replayed_endpoints': sum(c['independently_replayed_endpoints'] for c in checks),
        'verifier_code_sha256': digest(__file__)}
    save(ROOT / 'prefix_verification.json', result)
    print({'prefix_verification': result['status'], 'sources': len(checks), 'failures': failures}, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        run()
        if not args.watch or time.time() >= DEADLINE:
            break
        time.sleep(min(45, max(0, DEADLINE-time.time())))
