"""Independently refit known-endpoint route readouts and replay all predictions.

The producer's route constructor, ridge solver and scoring helper are not used.
Only true e/C/I source states and labels enter either fitted readout. The same
anchors, supplied involution routes, labels and counts enter the clean control.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from .until_10_verify import (
    ROOT, DEADLINE, read, archive, digest, save, homogeneous, action_order, replay,
)


ROUTES = [(0, ''), (1, ''), (4, ''), (0, 'c'), (0, 'i'), (1, 'c'), (4, 'i'),
    (0, 'cc'), (0, 'ii'), (1, 'cc'), (1, 'ii'), (4, 'cc'), (4, 'ii')]
ALPHAS = [1e-6, 1e-4, .01, 1.]


def known_design(hidden, labels, maps):
    columns = {0: 0, 1: 1, 4: 2}
    generated, clean, targets = [], [], []
    for start, word in ROUTES:
        end = int(action_order(word)[start])
        assert end in columns
        x = hidden[:, columns[start]].astype(np.float64)
        matrix = homogeneous(maps, word, x.shape[-1])
        predicted = np.column_stack([x, np.ones(len(x))]) @ matrix
        generated.append(predicted[:, :-1])
        clean.append(hidden[:, columns[end]].astype(np.float64))
        targets.append(labels[:, columns[end]])
    return np.concatenate(generated), np.concatenate(clean), np.concatenate(targets)


def independent_ridge(x, labels, validation, val_labels):
    # Eliminate the unpenalized intercept analytically, then solve the smaller
    # centered normal equations. The producer solves a joint augmented system.
    target = np.eye(31)[labels]
    val_target = np.eye(31)[val_labels]
    mean = x.mean(0)
    target_mean = target.mean(0)
    centered = x - mean
    gram = centered.T @ centered
    rhs = centered.T @ (target - target_mean)
    scale = max(np.trace(gram) / x.shape[1], 1e-20)
    fits, losses = [], []
    for alpha in ALPHAS:
        system = gram + alpha * scale * np.eye(x.shape[1])
        weight = np.linalg.solve(system, rhs)
        bias = target_mean - mean @ weight
        residual = np.linalg.norm(system @ weight - rhs) / max(np.linalg.norm(rhs), 1e-20)
        assert residual < 1e-9
        fits.append((weight.T, bias))
        losses.append(float(np.square(validation @ weight + bias - val_target).sum()))
    best = int(np.argmin(losses))
    return fits[best], losses, ALPHAS[best]


def verify(path):
    rec = read(path)
    folder = Path(rec['source_folder'])
    maps = archive(rec['native_map_path'])
    assert maps['mean_vectors'].size == 0
    assert digest(rec['native_map_path']) == rec['native_map_sha256']
    assert digest(rec['known_feature_path']) == rec['known_feature_sha256']
    for artifact, expected in rec['artifact_sha256'].items():
        assert digest(artifact) == expected, artifact
    known = archive(rec['known_feature_path'])
    data = archive(folder / 'source_data.npz')
    train = known_design(known['train_hidden'], data['train_labels'], maps)
    validation = known_design(known['validation_hidden'], data['validation_labels'], maps)
    assert train[0].shape == train[1].shape
    assert len(train[2]) == len(data['train_lengths']) * len(ROUTES)
    assert set(np.unique(train[2])).issubset(set(np.unique(data['train_labels'])))
    probe_folder = folder if folder.name.startswith('world') else Path('results/algebra_hidden_relations')
    probe = archive(probe_folder / 'probe_dataset.npz')
    features_path = folder / 'features' / (rec['source'] + '.npz')
    assert digest(features_path) == rec['probe_feature_sha256']
    hidden = archive(features_path)['source_query_concat'][:, :, -1]
    total = 0
    for method in rec['methods']:
        index = {'known_route_generated_ridge': 0, 'known_route_clean_matched_ridge': 1}[method['method']]
        (weight, bias), losses, chosen = independent_ridge(train[index], train[2], validation[index], validation[2])
        np.testing.assert_allclose(losses, method['selection']['known_route_validation_losses'], rtol=1e-8, atol=1e-7)
        assert chosen == method['selection']['selected_alpha']
        readout_path = path.parent.parent / 'readouts' / (rec['source'] + '_' + method['method'] + '.npz')
        stored = archive(readout_path)
        np.testing.assert_allclose(weight, stored['weight'], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(bias, stored['bias'], rtol=1e-6, atol=1e-7)
        assert method['readout_fit_rows'] == len(train[index])
        assert method['known_source_anchor_count'] == len(data['train_lengths'])
        assert method['additional_true_hidden_labels'] == 0
        actual = known['validation_hidden'].reshape(-1, hidden.shape[-1]).astype(np.float64)
        actual_labels = data['validation_labels'].reshape(-1)
        # Use the stored readout after the independent coefficient check so that
        # near-tie numerical roundoff never changes the record's scoring rule.
        w, b = stored['weight'], stored['bias']
        for features, truth, key in [
            (actual, actual_labels, 'true_known_validation_accuracy'),
            (validation[0], validation[2], 'known_generated_validation_accuracy'),
            (validation[1], validation[2], 'known_clean_matched_validation_accuracy'),
        ]:
            np.testing.assert_allclose(np.mean((features @ w.T + b).argmax(-1) == truth), method[key], atol=1e-12)
        modified = {**maps, 'readout_weight': w, 'readout_bias': b}
        saved = archive(path.parent.parent / 'arrays' / (rec['source'] + '_' + method['method'] + '.npz'))
        total += replay(hidden, probe, modified, method['rows'], saved, observed=True)['independently_replayed_endpoints']
    return {'source': rec['source'], 'status': 'passed', 'independently_replayed_endpoints': total,
        'independent_centered_ridge_refits': 2, 'no_true_compound_states_or_labels_used_for_fitting': True,
        'equal_clean_control_anchor_route_label_counts': True,
        'scope': 'All readout coefficients, known-only validation selection, unchanged native map hash, saved predictions and scores. Supplied CC/II identities remain externally imposed constraints.'}


def run():
    cache_path = ROOT / 'route_verification_cache.json'
    cache = read(cache_path) if cache_path.exists() else {}
    code = digest(__file__) + digest('experiments/until_10_verify.py')
    checks, failures = [], []
    for path in Path('results/known_route_readout_control/evaluations').glob('*.json'):
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
    save(ROOT / 'route_verification.json', result)
    print({'route_verification': result['status'], 'sources': len(checks), 'failures': failures}, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        run()
        if not args.watch or time.time() >= DEADLINE:
            break
        time.sleep(min(45, max(0, DEADLINE - time.time())))
