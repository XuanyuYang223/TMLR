"""Independent nuisance refits and confidence-residual composition replay."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import torch

from .until_10_verify import ROOT, DEADLINE, read, archive, digest, save, action_order, pair_rows
from .until_10_control_verify import head


ALPHAS = [1e-6, 1e-4, .01, 1.]


def independent_code(h, ns, weight):
    blocks = h.reshape(*h.shape[:-1], 4, -1).astype(np.float64)
    contrast = weight - weight.mean(0)
    projector = contrast.T @ np.linalg.pinv(contrast @ contrast.T, rcond=1e-12) @ contrast
    null = (blocks - blocks @ projector).reshape(h.shape)
    logits = blocks @ weight.T
    centered = logits - logits.mean(-1, keepdims=True)
    # Log-sum-exp normalizer, separate from the producer's ratio calculation.
    maximum = centered.max(-1, keepdims=True)
    logsum = maximum + np.log(np.exp(centered - maximum).sum(-1, keepdims=True))
    prob = np.exp(centered - logsum)
    entropy = -np.sum(prob * np.maximum(centered - logsum, np.log(1e-300)), axis=-1, keepdims=True)
    feature = np.concatenate([centered, prob, entropy, prob.max(-1, keepdims=True)], axis=-1)
    length_code = np.broadcast_to(np.eye(21)[ns - 10][:, None], (*h.shape[:2], 21))
    return null, np.concatenate([feature.reshape(*h.shape[:2], -1), length_code], axis=-1)


def verify(path):
    rec = read(path)
    assert digest(rec['source_record_path']) == rec['source_record_sha256']
    for p, expected in zip(rec['source_feature_paths'], rec['source_feature_sha256']):
        assert digest(p) == expected
    assert digest(rec['nuisance_fit_path']) == rec['nuisance_fit_sha256']
    assert digest(rec['map_path']) == rec['map_sha256']
    original = read(rec['source_record_path'])
    weight, bias = head(original, Path(rec['source_record_path']).parent.parent)
    np.testing.assert_array_equal(bias, np.zeros(31))
    fit = archive(rec['nuisance_fit_path'])
    np.testing.assert_array_equal(fit['numeric_weight'], weight)
    cal = archive('results/algebra_structure_replication/probe_dataset.npz')
    data = archive('results/joint_answer_matched_geometry/dataset.npz')
    cal_h = archive(rec['source_feature_paths'][0])['source_query_concat']
    test_h = archive(rec['source_feature_paths'][1])['source_query_concat']
    cal_null, code = independent_code(cal_h, cal['lengths'], weight)
    test_null, test_code = independent_code(test_h, data['lengths'], weight)
    train = code[cal['split'] == 0].reshape(-1, code.shape[-1])
    target = cal_null[cal['split'] == 0].reshape(-1, cal_null.shape[-1])
    val = code[cal['split'] == 1].reshape(-1, code.shape[-1])
    val_target = cal_null[cal['split'] == 1].reshape(-1, cal_null.shape[-1])
    mean, sigma = train.mean(0), np.maximum(train.std(0), 1e-4)
    np.testing.assert_allclose(mean, fit['code_mean'], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sigma, fit['code_sigma'], rtol=1e-10, atol=1e-10)
    x = (train - mean) / sigma
    xv = (val - mean) / sigma
    xm, ym = x.mean(0), target.mean(0)
    centered = x - xm
    gram, rhs = centered.T @ centered, centered.T @ (target - ym)
    scale = max(np.square(x).sum() / x.shape[1], 1e-20)
    losses, solutions = [], []
    for alpha in ALPHAS:
        normal = gram + alpha * scale * np.eye(x.shape[1])
        candidate = np.linalg.solve(normal, rhs)
        intercept = ym - xm @ candidate
        solutions.append((candidate, intercept))
        losses.append(float(np.square(xv @ candidate + intercept - val_target).sum()))
        assert np.linalg.norm(normal @ candidate - rhs) / max(np.linalg.norm(rhs), 1e-20) < 1e-9
    selected = int(np.argmin(losses))
    assert ALPHAS[selected] == rec['selection']['selected_alpha']
    np.testing.assert_allclose(losses, rec['selection']['calibration_validation_losses'], rtol=1e-8, atol=1e-7)
    np.testing.assert_allclose(solutions[selected][0], fit['weight'], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(solutions[selected][1], fit['bias'], rtol=1e-5, atol=1e-7)
    residual = test_null - ((test_code - fit['code_mean']) / fit['code_sigma']) @ fit['weight'] - fit['bias']
    pairs = pair_rows(data)
    delta = residual[pairs[:, 0]] - residual[pairs[:, 1]]
    original_delta = test_null[pairs[:, 0]] - test_null[pairs[:, 1]]
    retained = np.square(delta).sum() / np.square(original_delta).sum()
    np.testing.assert_allclose(retained, rec['retained_test_pair_energy_fraction'], rtol=1e-9, atol=1e-10)
    maps = archive(rec['map_path'])
    q = maps['basis'].astype(np.float64)
    source = delta.reshape(-1, delta.shape[-1]).T
    latent = q.T @ source
    np.testing.assert_allclose(np.square(latent).sum() / np.square(delta).sum(),
        rec['projected_residual_pair_energy_fraction'], rtol=1e-9, atol=1e-10)
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
        target = delta[:, action_order(row['word'])].reshape(prediction.shape)
        error = np.square(prediction - target).sum()
        np.testing.assert_allclose(error / np.square(target).sum(), row['pair_target_nmse'], rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(error / np.square(target - source.T).sum(), row['pair_action_displacement_nmse'], rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(predicted_latent.T, saved[row['word'] + '_pair_latent_prediction'], rtol=1e-6, atol=1e-6)
    return {'source': rec['source'], 'status': 'passed', 'independently_replayed_endpoints': len(rec['results']),
        'known_only_confidence_ridge_refit_and_selection': True, 'retained_pair_energy_independently_recomputed': True,
        'scope': 'Independent confidence features, numeric contrast projection, nuisance ridge coefficients/selection, retained energy and every residual composition score. Generator PCA/ridge fitting is not refitted here; this removes only the declared confidence feature association.'}


def run():
    cp = ROOT / 'confidence_verification_cache.json'
    cache = read(cp) if cp.exists() else {}
    code = digest(__file__) + digest('experiments/until_10_verify.py') + digest('experiments/until_10_control_verify.py')
    checks, failures = [], []
    for path in sorted(Path('results/confidence_residual_geometry/evaluations').glob('*.json')):
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
    save(ROOT / 'confidence_verification.json', result)
    print({'confidence_verification': result['status'], 'sources': len(checks), 'failures': failures}, flush=True)
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
