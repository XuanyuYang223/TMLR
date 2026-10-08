"""Replay output-only surrogate predictions against original hidden targets."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from .until_10_verify import ROOT, DEADLINE, read, archive, digest, save, action_order, pair_rows
from .until_10_confidence_verify import independent_code


def verify(path):
    rec = read(path)
    parent = read(rec['parent_record_path'])
    assert digest(rec['parent_record_path']) == rec['parent_record_sha256']
    assert digest(parent['nuisance_fit_path']) == parent['nuisance_fit_sha256']
    assert digest(parent['source_feature_paths'][1]) == parent['source_feature_sha256'][1]
    assert digest(rec['map_path']) == rec['map_sha256']
    fit = archive(parent['nuisance_fit_path'])
    data = archive('results/joint_answer_matched_geometry/dataset.npz')
    hidden = archive(parent['source_feature_paths'][1])['source_query_concat']
    null, code = independent_code(hidden, data['lengths'], fit['numeric_weight'])
    proxy = ((code - fit['code_mean']) / fit['code_sigma']) @ fit['weight'] + fit['bias']
    pairs = pair_rows(data)
    surrogate_delta = proxy[pairs[:, 0]] - proxy[pairs[:, 1]]
    true_delta = null[pairs[:, 0]] - null[pairs[:, 1]]
    maps = archive(rec['map_path'])
    q = maps['basis'].astype(np.float64)
    source = surrogate_delta.reshape(-1, q.shape[0]).T
    latent = q.T @ source
    ap = path.parent.parent / 'arrays' / (rec['source'] + '.npz')
    assert digest(ap) == rec['prediction_archive_sha256']
    saved = archive(ap)
    np.testing.assert_array_equal(saved['pair_ids'], np.unique(data['pair_ids']))
    for row in rec['results']:
        product = np.eye(q.shape[1])
        for g in row['word']:
            product = maps['map_' + g].astype(np.float64).T @ product
        predicted_latent = product @ latent
        prediction = (source + q @ (predicted_latent - latent)).T
        target = true_delta[:, action_order(row['word'])].reshape(prediction.shape)
        original_source = true_delta.reshape(prediction.shape)
        error = np.square(prediction - target).sum()
        np.testing.assert_allclose(error / np.square(target).sum(), row['pair_target_nmse'], rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(error / np.square(target - original_source).sum(), row['pair_action_displacement_nmse'], rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(predicted_latent.T, saved[row['word'] + '_pair_latent_prediction'], rtol=1e-6, atol=1e-6)
        assert row['target'] == 'original_numeric_null_pair_difference'
        assert row['pairs'] == len(pairs)
    return {'source': rec['source'], 'status': 'passed', 'independently_replayed_endpoints': len(rec['results']),
        'prediction_only_current_input_outputs_and_fixed_maps': True, 'original_target_vectors_used_only_for_scoring': True,
        'scope': 'Output/confidence reconstruction and all generator/composite predictions scored against original null-space targets, using independent projections and column-oriented products. Inherited nuisance decoder is independently fitted by confidence_verification; proxy PCA/ridge fitting is not refitted here.'}


def run():
    cp = ROOT / 'proxy_verification_cache.json'
    cache = read(cp) if cp.exists() else {}
    code = digest(__file__) + digest('experiments/until_10_verify.py') + digest('experiments/until_10_confidence_verify.py')
    checks, failures = [], []
    for path in sorted(Path('results/confidence_proxy_geometry/evaluations').glob('*.json')):
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
    save(ROOT / 'proxy_verification.json', result)
    print({'proxy_verification': result['status'], 'sources': len(checks), 'failures': failures}, flush=True)
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
