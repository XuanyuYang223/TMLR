"""Independent refits and numerical replay for output-budget/pair diagnostics."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .until_10_verify import digest, read, save, action_order
from .until_10_confidence_verify import independent_code
from .until_10_control_verify import head


def check_pca(q, x):
    np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-9)
    covariance = x.T @ x
    values, vectors = np.linalg.eigh(covariance)
    expected = vectors[:, -q.shape[1]:]
    assert np.linalg.norm(expected - q @ (q.T @ expected)) < 1e-5
    np.testing.assert_allclose(np.trace(q.T @ covariance @ q), values[-q.shape[1]:].sum(), rtol=1e-9)


def decoder_refit(x, y, xv, yv, grid):
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    gram, rhs = xc.T @ xc, xc.T @ yc
    scale = max(float(np.square(x).sum() / x.shape[1]), 1e-20)
    solutions, losses = [], []
    for a in grid:
        normal = gram + a * scale * np.eye(x.shape[1])
        weight = np.linalg.solve(normal, rhs)
        intercept = ym - xm @ weight
        assert np.linalg.norm(normal @ weight - rhs) / max(np.linalg.norm(rhs), 1e-20) < 1e-9
        solutions.append(np.vstack([weight, intercept]))
        losses.append(float(np.square(xv @ weight + intercept - yv).sum()))
    selected = int(np.argmin(losses))
    return solutions[selected], losses, grid[selected]


def map_refit(x, y, xv, yv, grid):
    gram, rhs = x.T @ x, x.T @ y
    scale = max(float(np.square(x).sum() / x.shape[1]), 1e-20)
    losses, solutions = [], []
    identity = np.eye(x.shape[1])
    for a in grid:
        # Direct normal equation for an identity-centered ridge map.
        coefficient = np.linalg.solve(gram + a * scale * identity, rhs + a * scale * identity)
        solutions.append(coefficient)
        losses.append(float(np.square(xv @ coefficient - yv).sum()))
    selected = int(np.argmin(losses))
    return solutions[selected], losses, grid[selected]


def verify_budget(path, plan):
    rec = read(path)
    parent = read(rec['parent_path'])
    assert digest(rec['parent_path']) == rec['parent_sha256']
    assert digest(rec['map_path']) == rec['map_sha256']
    for p, expected in zip(parent['source_feature_paths'], parent['source_feature_sha256']):
        assert digest(p) == expected
    original = read(parent['source_record_path'])
    weight, bias = head(original, Path(parent['source_record_path']).parent.parent)
    assert np.all(bias == 0)
    with np.load(rec['map_path']) as a:
        maps = {k: a[k] for k in a.files}
    np.testing.assert_array_equal(maps['numeric_weight'], weight)
    cal = dict(np.load('results/algebra_structure_replication/probe_dataset.npz'))
    test = dict(np.load('results/joint_answer_matched_geometry/dataset.npz'))
    with np.load(parent['source_feature_paths'][0]) as a:
        cal_h, cal_code = independent_code(a['source_query_concat'], cal['lengths'], weight)
    with np.load(parent['source_feature_paths'][1]) as a:
        test_h, test_code = independent_code(a['source_query_concat'], test['lengths'], weight)
    pairs = np.array([np.flatnonzero(test['pair_ids'] == p) for p in np.unique(test['pair_ids'])])
    np.testing.assert_array_equal(maps['pair_ids'], np.unique(test['pair_ids']))
    np.testing.assert_array_equal(test['union_labels'][pairs[:, 0]], test['union_labels'][pairs[:, 1]])
    tf, tv = [cal_h[cal['split'] == i].reshape(-1, cal_h.shape[-1]) for i in [0, 1]]
    sizes = np.unique(cal['lengths'])
    means = np.array([cal_h[(cal['lengths'] == n) & (cal['split'] == 0)].reshape(-1, cal_h.shape[-1]).mean(0) for n in sizes])
    np.testing.assert_array_equal(maps['target_lengths'], sizes)
    np.testing.assert_allclose(maps['target_centers'], means, atol=1e-9)
    target_centered = cal_h - means[np.searchsorted(sizes, cal['lengths']), None]
    q0 = maps['target_basis']
    check_pca(q0, target_centered[cal['split'] == 0].reshape(-1, cal_h.shape[-1]))
    difference = test_h[pairs[:, 0]] - test_h[pairs[:, 1]]
    target_energy = float(np.square(difference).sum())
    difference_q0 = difference @ q0
    np.testing.assert_allclose(np.square(difference_q0).sum() / target_energy, rec['shared_pca64_pair_energy_fraction'], atol=1e-9)
    rng = np.random.default_rng(rec['seed'] + 2026100692)
    shuffles = {}
    for g in ['c', 'r', 'i']:
        indices = []
        for split in [0, 1]:
            lengths = np.repeat(cal['lengths'][cal['split'] == split], 8)
            order = np.arange(len(lengths))
            for n in np.unique(lengths):
                ids = np.where(lengths == n)[0]
                order[ids] = rng.permutation(ids)
            indices.append(order)
        shuffles[g] = indices
        for tag, ids in zip(['fit', 'validation'], indices):
            np.testing.assert_array_equal(ids, maps['shuffle_' + g + '_' + tag])
    comparable = []
    endpoints = 0
    for fitted in rec['fits']:
        prefix = fitted['pipeline']
        h, ht = (cal_code, test_code) if fitted['branch'] == 'output' else (cal_h, test_h)
        encoder = maps[prefix + '_encoder_matrix']
        assert encoder.shape == (h.shape[-1], plan['reconstruction_input_dimension'])
        if prefix == 'output_identity':
            np.testing.assert_array_equal(encoder, np.eye(h.shape[-1]))
        elif prefix == 'hidden_fixed_random':
            fixed, _ = np.linalg.qr(np.random.default_rng(plan['projection_seed']).normal(size=encoder.shape), mode='reduced')
            np.testing.assert_array_equal(encoder, fixed)
        else:
            check_pca(encoder, target_centered[cal['split'] == 0].reshape(-1, cal_h.shape[-1]))
        encoded = h @ encoder
        train = encoded[cal['split'] == 0].reshape(-1, encoder.shape[1])
        mean, scale = train.mean(0), np.maximum(train.std(0), 1e-4)
        np.testing.assert_allclose(mean, maps[prefix + '_encoder_mean'], atol=1e-9)
        np.testing.assert_allclose(scale, maps[prefix + '_encoder_scale'], atol=1e-9)
        encoded = (encoded - mean) / scale
        xf, xv = [encoded[cal['split'] == i].reshape(-1, encoder.shape[1]) for i in [0, 1]]
        decoder, losses, a = decoder_refit(xf, tf, xv, tv, plan['ridge_grid'])
        original_decoder = maps[prefix + '_decoder']
        np.testing.assert_allclose(decoder, original_decoder, atol=1e-7, rtol=1e-5)
        np.testing.assert_allclose(losses, fitted['decoder']['validation_losses'], atol=1e-6, rtol=1e-8)
        assert a == fitted['decoder']['alpha']
        gram = xf.T @ xf
        eig = np.maximum(np.linalg.eigvalsh(gram), 0.)
        ridge_df = 1 + float(np.sum(eig / (eig + a * np.trace(gram) / encoder.shape[1])))
        np.testing.assert_allclose(ridge_df, fitted['decoder']['effective_ridge_df'], atol=1e-7)
        reconstruct_cal = encoded @ original_decoder[:-1] + original_decoder[-1]
        probe_means = np.array([reconstruct_cal[(cal['lengths'] == n) & (cal['split'] == 0)].reshape(-1, cal_h.shape[-1]).mean(0) for n in sizes])
        np.testing.assert_allclose(probe_means, maps[prefix + '_probe_means'], atol=1e-9)
        probe_centered = reconstruct_cal - probe_means[np.searchsorted(sizes, cal['lengths']), None]
        q = maps[prefix + '_probe_basis']
        check_pca(q, probe_centered[cal['split'] == 0].reshape(-1, cal_h.shape[-1]))
        latent = probe_centered @ q
        zf, zv = [latent[cal['split'] == i].reshape(-1, plan['dimension']) for i in [0, 1]]
        for generator in fitted['generators']:
            g, pairing = generator['generator'], generator['pairing']
            changed = latent[:, action_order(g)]
            yf, yv = [changed[cal['split'] == i].reshape(-1, plan['dimension']) for i in [0, 1]]
            if pairing == 'shuffled':
                yf, yv = yf[shuffles[g][0]], yv[shuffles[g][1]]
            coefficient, losses, alpha = map_refit(zf, yf, zv, yv, plan['ridge_grid'])
            np.testing.assert_allclose(coefficient, maps[prefix + '_' + pairing + '_' + g], atol=1e-8, rtol=1e-6)
            np.testing.assert_allclose(losses, generator['validation_losses'], atol=1e-7, rtol=1e-8)
            assert alpha == generator['alpha']
        encoded_test = (ht @ encoder - mean) / scale
        reconstruction = encoded_test @ original_decoder[:-1] + original_decoder[-1]
        source = (reconstruction[pairs[:, 0]] - reconstruction[pairs[:, 1]]).reshape(-1, cal_h.shape[-1])
        z = source @ q
        np.testing.assert_allclose(z, maps[prefix + '_pair_latent_input'], atol=1e-8, rtol=1e-7)
        for row in rec['results']:
            if row['pipeline'] != prefix:
                continue
            word = '' if row['word'] == 'identity' else ('ic' if row['word'] == 'ic_against_ci' else row['word'])
            target_word = 'ci' if row['word'] == 'ic_against_ci' else word
            # Column orientation and explicit full reconstruction, independent
            # of the producer's thin squared-error identity.
            predicted_latent = z.T.copy()
            for g in word:
                predicted_latent = maps[prefix + '_' + row['pairing'] + '_' + g].T @ predicted_latent
            prediction = source + (q @ (predicted_latent - z.T)).T
            target = difference[:, action_order(target_word)].reshape(source.shape)
            error = float(np.square(prediction - target).sum())
            np.testing.assert_allclose(error / target_energy, row['full_null_pair_target_nmse'], atol=1e-9, rtol=1e-8)
            projected = (prediction - target) @ q0
            np.testing.assert_allclose(np.square(projected).sum() / np.square(difference_q0).sum(), row['shared_pca64_pair_target_nmse'], atol=1e-9, rtol=1e-8)
            if row['full_null_action_displacement_nmse'] is not None:
                delta = target - difference.reshape(source.shape)
                np.testing.assert_allclose(error / np.square(delta).sum(), row['full_null_action_displacement_nmse'], atol=1e-9, rtol=1e-8)
            endpoints += 1
        if prefix in plan['primary_comparison']:
            comparable.append(tuple(fitted[k] for k in ['reconstruction_input_dimension', 'latent_dimension', 'decoder_parameters',
                'generator_parameters_per_pairing', 'input_center_statistics', 'input_scale_statistics', 'fitted_input_encoder_loadings',
                'fitted_probe_loadings', 'probe_center_statistics', 'decoder_fit_rows', 'decoder_validation_rows',
                'generator_fit_rows_each', 'generator_validation_rows_each', 'ridge_candidates_per_fit']))
    assert len(comparable) == 2 and comparable[0] == comparable[1]
    # Check the output branch reproduces the previous unrestricted output code
    # without dropping features to manufacture a favorable hidden comparison.
    previous = read(Path('results/confidence_proxy_geometry/evaluations') / (rec['source'] + '.json'))
    for old in previous['results']:
        new = next(r for r in rec['results'] if r['pipeline'] == 'output_identity' and r['pairing'] == 'correct' and r['word'] == old['word'])
        np.testing.assert_allclose(new['full_null_pair_target_nmse'], old['pair_target_nmse'], atol=5e-7, rtol=5e-7)
    return {'source': rec['source'], 'status': 'passed', 'replayed_scores': endpoints,
        'all_decoders_and_generator_coefficients_independently_refitted': True,
        'all_PCA_bases_checked_against_FIT_covariance': True,
        'primary_parameter_statistics_rows_and_search_counts_match': True,
        'output_proxy_retains_all_previously_used_features': True,
        'scope': 'Independent numeric-null projection/output features, FIT-only statistics, fixed projection replay, PCA subspaces, affine and identity-centered ridge refits/selection, full-space and shared-target scores. Different input feature families and effective ranks are not matched causal interventions.'}


def verify_pair(path):
    r = read(path)
    assert digest(r['record_path']) == r['input_record_sha256']
    assert digest(r['archive_path']) == r['archive_expected_sha256']
    assert digest(r['array_path']) == r['array_sha256']
    with np.load(r['data_path']) as data:
        mask = data['split'] == 3
        ids = data['pair_ids'][mask]
        labels = data['labels'][mask, r['action']]
    with np.load(r['archive_path']) as data:
        predictions = data[r['answer_key']]
    pairs = {int(p): np.flatnonzero(ids == p) for p in np.unique(ids)}
    categories, both, swapped, hits = Counter(), [], [], []
    for p, indices in pairs.items():
        assert len(indices) == 2
        a, b = map(int, indices)
        assert labels[a] != labels[b]
        first, second = int(predictions[a] == labels[a]), int(predictions[b] == labels[b])
        categories[first + second] += 1
        both.append(first * second)
        swapped.append(int(predictions[a] == labels[b] and predictions[b] == labels[a]))
        hits.append([first, second])
    n = len(pairs)
    expected = {'both_correct_count': categories[2], 'exactly_one_correct_count': categories[1], 'neither_correct_count': categories[0],
        'both_correct_fraction': categories[2] / n, 'exactly_one_correct_fraction': categories[1] / n,
        'neither_correct_fraction': categories[0] / n, 'answer_accuracy': sum(i * categories[i] for i in range(3)) / (2*n),
        'both_correct_after_swap_count': sum(swapped), 'both_correct_after_swap_fraction': sum(swapped) / n,
        'orientation_excess_all_pairs': (sum(both) - sum(swapped)) / n,
        'random_pair_orientation_expected_both_fraction': (sum(both) + sum(swapped)) / (2*n),
        'covered_candidate_pairs': sum(both) + sum(swapped), 'covered_candidate_pair_fraction': (sum(both) + sum(swapped)) / n}
    covered = sum(both) + sum(swapped)
    if covered:
        expected['correct_orientation_given_covered_candidates'] = sum(both) / covered
    else:
        assert r['metrics']['correct_orientation_given_covered_candidates'] is None
    for key, value in expected.items():
        np.testing.assert_allclose(value, r['metrics'][key], atol=1e-15)
    with np.load(r['array_path']) as data:
        np.testing.assert_array_equal(data['both_correct'], both)
        np.testing.assert_array_equal(data['swapped_both_correct'], swapped)
        np.testing.assert_array_equal(data['correct'], hits)
        np.testing.assert_array_equal(data['pair_ids'], list(pairs))
        np.testing.assert_array_equal(data['predicted_answers'], [predictions[v] for v in pairs.values()])
        np.testing.assert_array_equal(data['true_answers'], [labels[v] for v in pairs.values()])
    return {'endpoint': path.name, 'status': 'passed', 'pairs': n,
        'scope': 'Recomputed both/one/neither and swap orientation directly from original saved answer archives and immutable labels; this does not refit original neural models or validate a causal mechanism.'}


def run(stage):
    root = Path('results/budget_matched_geometry' if stage == 'budget' else 'results/collision_pair_followup')
    protocol = read(root / 'protocol.json')
    code = digest(__file__)
    assert digest('experiments/' + ('budget_matched_geometry' if stage == 'budget' else 'collision_pair_followup') + '.py') == protocol['signature']['code_sha256']
    plan = protocol['signature'].get('plan')
    checks, failures = [], []
    for path in sorted((root / 'evaluations').glob('*.json')):
        try:
            checked = verify_budget(path, plan) if stage == 'budget' else verify_pair(path)
            checks.append(checked)
            print({'verified': stage, 'record': path.stem}, flush=True)
        except Exception as e:
            failures.append({'path': str(path), 'error': repr(e)})
            print({'verification_failed': str(path), 'error': repr(e)}, flush=True)
        save(root / 'verification.json', {'status': 'failed' if failures else 'passed_completed_records',
            'checks_completed': len(checks), 'checks': checks, 'failures': failures,
            'verifier_sha256': code, 'updated_utc': datetime.now(timezone.utc).isoformat()})
    expected = 18 if stage == 'budget' else len(protocol['signature']['jobs'])
    assert len(checks) == expected and not failures, (len(checks), expected, failures)
    print({'verification': stage, 'status': 'passed', 'records': len(checks)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['budget', 'pairs'])
    run(parser.parse_args().stage)
