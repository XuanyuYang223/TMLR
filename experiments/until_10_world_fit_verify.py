"""Refit all fresh-world frozen controls using independently saved known states."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .until_10_verify import ROOT, read, archive, digest, save, replay


def verify(path):
    rec = read(path)
    folder = path.parent.parent
    known_path = Path('results/relation_readout_diagnostics/features') / (rec['source'] + '.npz')
    known = archive(known_path)
    source = archive(folder / 'source_data.npz')
    probe = archive(folder / 'probe_dataset.npz')
    hidden = archive(folder / 'features' / (rec['source'] + '.npz'))['source_query_concat'][:, :, -1]
    checkpoint = torch.load(folder / 'checkpoints' / (rec['source'] + '.pt'), map_location='cpu', weights_only=True)
    grid = read('configs/algebra_hidden_relations.json')['ridge_grid']
    outcomes = []
    endpoints = 0
    for pairing in ['correct', 'shuffled']:
        map_path = folder / 'arrays' / (rec['source'] + '_frozen_full_' + pairing + '.npz')
        maps = archive(map_path)
        np.testing.assert_array_equal(maps['readout_weight'], checkpoint['model']['lm_head.weight'][:31].numpy())
        np.testing.assert_array_equal(maps['readout_bias'], np.zeros(31))
        rng = np.random.default_rng(rec['world']['source_seed'] + 2026100671)
        ids = np.concatenate([rng.permutation(np.flatnonzero(source['train_lengths'] == n))[:256]
            for n in np.unique(source['train_lengths'])])
        np.testing.assert_array_equal(ids, maps['source_anchor_ids'])
        train = known['train_hidden'][ids].astype(np.float64)
        val = known['validation_hidden'].astype(np.float64)
        lengths = source['train_lengths'][ids]
        val_lengths = source['validation_lengths']
        for n, center in zip(maps['mean_lengths'], maps['mean_vectors']):
            np.testing.assert_allclose(train[lengths == n].mean((0, 1)), center, atol=1e-12)
            train[lengths == n] -= center
            val[val_lengths == n] -= center

        def edges(h, ns, column):
            endpoint = h[:, column]
            if pairing == 'shuffled':
                order = np.arange(len(h))
                for n in np.unique(ns):
                    rows = np.flatnonzero(ns == n)
                    order[rows] = np.roll(rows, int(rng.integers(1, len(rows))))
                endpoint = endpoint[order]
            return np.concatenate([h[:, 0], endpoint]), np.concatenate([endpoint, h[:, 0]])

        alphas = {}
        for g, column in [('c', 1), ('i', 2)]:
            x, y = edges(train, lengths, column)
            xv, yv = edges(val, val_lengths, column)
            xmean = x.mean(0)
            difference = y - x
            dmean = difference.mean(0)
            centered = x - xmean
            gram = centered.T @ centered
            rhs = centered.T @ (difference - dmean)
            # Removing the intercept changes the solve, not the registered
            # regularization magnitude: its scale uses the uncentered x norm.
            scale = max(np.square(x).sum() / x.shape[1], 1e-20)
            fits, losses = [], []
            for alpha in grid:
                system = gram + alpha * scale * np.eye(x.shape[1])
                w = np.linalg.solve(system, rhs)
                b = dmean - xmean @ w
                fits.append((w, b))
                losses.append(np.square(xv + xv @ w + b - yv).sum())
            selected = int(np.argmin(losses))
            weight, bias = fits[selected]
            np.testing.assert_allclose(maps['rho_' + g], np.eye(x.shape[1]) + weight, atol=1e-8, rtol=1e-7)
            np.testing.assert_allclose(maps['bias_' + g], bias, atol=1e-8, rtol=1e-7)
            stored = maps['rho_' + g] - np.eye(x.shape[1])
            system = gram + grid[selected] * scale * np.eye(x.shape[1])
            residual = np.linalg.norm(system @ stored - rhs) / max(np.linalg.norm(rhs), 1e-20)
            assert residual < 1e-10, residual
            alphas[g] = grid[selected]
        method = next(m for m in rec['methods'] if m['method'] == 'frozen_full_' + pairing)
        pred_path = folder / 'observed_start' / (rec['source'] + '_' + method['method'] + '.npz')
        assert digest(pred_path) == method['prediction_archive_sha256']
        endpoints += replay(hidden, probe, maps, method['metrics'], archive(pred_path), observed=True)['independently_replayed_endpoints']
        outcomes.append({'pairing': pairing, 'independently_selected_alphas': alphas, 'native_readout_unchanged': True,
            'map_path': str(map_path), 'map_sha256': digest(map_path)})
    return {'source': rec['source'], 'world': rec['world']['id'], 'status': 'passed', 'fit_checks': outcomes,
        'known_feature_sha256': digest(known_path), 'independently_replayed_endpoints': endpoints,
        'known_only_centered_ridge_refits': 4, 'equal_initial_backbone_and_anchor_inputs_within_pairing': True}


def run():
    checks, failures = [], []
    for path in sorted(Path('results/relation_world_confirmation').glob('world*/observed_start/*.json')):
        try:
            checks.append(verify(path))
        except Exception as error:
            failures.append({'check': str(path), 'error': repr(error)})
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed' if not failures else 'failed',
        'checks_completed': len(checks), 'checks': checks, 'failures': failures,
        'independently_replayed_endpoints': sum(c['independently_replayed_endpoints'] for c in checks),
        'verifier_code_sha256': digest(__file__),
        'scope': 'Independently selected known-only ridge regularization, all four generator fits and 16 frozen-control endpoints per completed fresh-world source. Native weights are fixed and shared between true and shuffled pairings.'}
    save(ROOT / 'world_fit_verification.json', result)
    print({'world_fit_verification': result['status'], 'sources': len(checks), 'failures': failures}, flush=True)
    return result


if __name__ == '__main__':
    run()
