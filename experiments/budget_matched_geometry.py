"""Matched decoder/generator budgets for hidden and output input branches."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .confidence_residual_geometry import null_and_code
from .longrun_engine import atomic_json
from .representation_algebra import pca_basis, permutation_action_table, word_action
from .until_10_verify import digest

CONFIG = Path('configs/budget_matched_geometry.json')
CAL = Path('results/algebra_structure_replication/probe_dataset.npz')
TEST = Path('results/joint_answer_matched_geometry/dataset.npz')
LETTERS = {'c': 1, 'r': 2, 'i': 4}


def now():
    return datetime.now(timezone.utc).isoformat()


def center_by_length(values, lengths, fit):
    sizes = np.unique(lengths)
    centers = np.array([values[(lengths == n) & fit].reshape(-1, values.shape[-1]).mean(0) for n in sizes])
    return sizes, centers


def subtract_means(values, lengths, sizes, centers):
    ids = np.searchsorted(sizes, lengths)
    assert np.array_equal(sizes[ids], lengths)
    return values - centers[ids, None]


def encoder_matrix(values, split, lengths, method, dimension, seed):
    if method == 'fixed_random':
        generator = np.random.default_rng(seed)
        matrix, _ = np.linalg.qr(generator.normal(size=(values.shape[-1], dimension)), mode='reduced')
        return matrix
    if method != 'fit_pca':
        raise ValueError(method)
    sizes, centers = center_by_length(values, lengths, split == 0)
    centered = subtract_means(values, lengths, sizes, centers)
    matrix = pca_basis(centered[split == 0].reshape(-1, values.shape[-1]), dimension)
    assert matrix.shape[1] == dimension, (method, matrix.shape)
    return matrix


def reconstruction_inputs(values, split, matrix):
    encoded = values @ matrix
    fit = encoded[split == 0].reshape(-1, matrix.shape[1])
    mean, scale = fit.mean(0), np.maximum(fit.std(0), 1e-4)
    return (encoded - mean) / scale, {'matrix': matrix, 'mean': mean, 'scale': scale}


def decoder_fit(x, y, xv, yv, grid):
    design = np.column_stack([x, np.ones(len(x))])
    validation = np.column_stack([xv, np.ones(len(xv))])
    gram, rhs = design.T @ design, design.T @ y
    scale = max(float(np.trace(gram[:-1, :-1]) / x.shape[1]), 1e-20)
    candidates, losses = [], []
    for alpha in grid:
        coefficient = np.linalg.solve(gram + np.diag([alpha * scale] * x.shape[1] + [0.]), rhs)
        candidates.append(coefficient)
        losses.append(float(np.square(validation @ coefficient - yv).sum()))
    selected = int(np.argmin(losses))
    eigenvalues = np.maximum(np.linalg.eigvalsh(gram[:-1, :-1]), 0.)
    effective_df = 1 + float(np.sum(eigenvalues / (eigenvalues + grid[selected] * scale)))
    return candidates[selected], {'alpha': grid[selected], 'validation_losses': losses, 'effective_ridge_df': effective_df,
        'validation_target_nmse': losses[selected] / float(np.square(yv).sum())}


def generator_fit(x, y, xv, yv, grid):
    gram = x.T @ x
    rhs = x.T @ (y - x)
    scale = max(float(np.trace(gram) / x.shape[1]), 1e-20)
    candidates, losses = [], []
    for alpha in grid:
        coefficient = np.eye(x.shape[1]) + np.linalg.solve(gram + alpha * scale * np.eye(x.shape[1]), rhs)
        candidates.append(coefficient)
        losses.append(float(np.square(xv @ coefficient - yv).sum()))
    selected = int(np.argmin(losses))
    return candidates[selected], {'alpha': grid[selected], 'validation_losses': losses}


def length_permutation(lengths, rng):
    order = np.arange(len(lengths))
    for n in np.unique(lengths):
        indices = np.flatnonzero(lengths == n)
        order[indices] = rng.permutation(indices)
    return order


def residual_update_error(reconstruction, latent, product, target, target_latent, reconstruction_energy, target_energy):
    """Exact full error for R + (z product-z) Q.T with orthonormal Q."""
    change = latent @ product - latent
    error = float(reconstruction_energy + target_energy - 2 * np.sum(reconstruction * target)
        + np.square(change).sum() + 2 * np.sum(change * (latent - target_latent)))
    assert error >= -max(1., target_energy) * 1e-10
    return max(error, 0.)


def initialize():
    plan = json.loads(CONFIG.read_text())
    root = Path(plan['output'])
    for folder in [root, root / 'evaluations', root / 'maps']:
        folder.mkdir(parents=True, exist_ok=True)
    parents = sorted((Path(plan['parent']) / 'evaluations').glob('*.json'))
    assert len(parents) == 18
    jobs = [json.loads(p.read_text()) for p in parents]
    assert {(r['group'], r['seed'], r['status']) for r in jobs} == {
        (g, seed, state) for g in plan['groups'] for seed in plan['source_seeds'] for state in plan['statuses']}
    signature = {'config_sha256': digest(CONFIG), 'code_sha256': digest(__file__),
        'core_sha256': {p: digest(p) for p in ['experiments/confidence_residual_geometry.py',
            'experiments/native_source_subspace.py', 'experiments/representation_algebra.py']},
        'calibration_data_sha256': digest(CAL), 'test_data_sha256': digest(TEST),
        'parents': {str(p): digest(p) for p in parents}, 'prior_completed_study_sha256': digest('results/until_10_followup/completion.json'),
        'plan': plan, 'sources': 18, 'source_retraining': False,
        'scope': plan['interpretation']}
    protocol = root / 'protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text())['signature'] == signature
    else:
        atomic_json(protocol, {'registered_utc': now(), 'new_evaluations': 0, 'signature': signature})
    return plan, root, parents


def evaluate(plan, root, path):
    parent = json.loads(path.read_text())
    destination = root / 'evaluations' / (parent['source'] + '.json')
    if destination.exists():
        return
    cal, test = [dict(np.load(p)) for p in [CAL, TEST]]
    with np.load(parent['nuisance_fit_path']) as fitted:
        weight = fitted['numeric_weight']
    feature_paths = parent['source_feature_paths']
    for p, expected in zip(feature_paths, parent['source_feature_sha256']):
        assert digest(p) == expected
    with np.load(feature_paths[0]) as a:
        cal_h, cal_code = null_and_code(a['source_query_concat'], cal['lengths'], weight)
    with np.load(feature_paths[1]) as a:
        test_h, test_code = null_and_code(a['source_query_concat'], test['lengths'], weight)
    sizes, centers = center_by_length(cal_h, cal['lengths'], cal['split'] == 0)
    targets = subtract_means(cal_h, cal['lengths'], sizes, centers)
    yf, yv = [cal_h[cal['split'] == k].reshape(-1, cal_h.shape[-1]) for k in [0, 1]]
    shared_q = pca_basis(targets[cal['split'] == 0].reshape(-1, targets.shape[-1]), plan['dimension'])
    pairs = np.array([np.flatnonzero(test['pair_ids'] == p) for p in np.unique(test['pair_ids'])])
    assert pairs.shape == (1344, 2)
    np.testing.assert_array_equal(test['union_labels'][pairs[:, 0]], test['union_labels'][pairs[:, 1]])
    true_delta = test_h[pairs[:, 0]] - test_h[pairs[:, 1]]
    projected_delta = true_delta @ shared_q
    target_energy = float(np.square(true_delta).sum())
    projected_energy = float(np.square(projected_delta).sum())
    table = permutation_action_table()
    fit_lengths = np.repeat(cal['lengths'][cal['split'] == 0], 8)
    val_lengths = np.repeat(cal['lengths'][cal['split'] == 1], 8)
    random = np.random.default_rng(parent['seed'] + 2026100692)
    shuffled = {g: (length_permutation(fit_lengths, random), length_permutation(val_lengths, random)) for g in LETTERS}
    arrays = {'target_basis': shared_q, 'target_lengths': sizes, 'target_centers': centers,
        'numeric_weight': weight, 'pair_ids': np.unique(test['pair_ids'])}
    rows, fits = [], []
    for prefix in plan['branches']:
        branch = 'output' if prefix == 'output_identity' else 'hidden'
        method = prefix.removeprefix(branch + '_')
        input_cal, input_test = (cal_code, test_code) if branch == 'output' else (cal_h, test_h)
        matrix = np.eye(input_cal.shape[-1]) if branch == 'output' else encoder_matrix(
            input_cal, cal['split'], cal['lengths'], method, plan['reconstruction_input_dimension'], plan['projection_seed'])
        encoded, encoder = reconstruction_inputs(input_cal, cal['split'], matrix)
        xf, xv = [encoded[cal['split'] == k].reshape(-1, matrix.shape[1]) for k in [0, 1]]
        decoder, decoder_info = decoder_fit(xf, yf, xv, yv, plan['ridge_grid'])
        decoded_cal = encoded @ decoder[:-1] + decoder[-1]
        latent_lengths, latent_means = center_by_length(decoded_cal, cal['lengths'], cal['split'] == 0)
        probe_centered = subtract_means(decoded_cal, cal['lengths'], latent_lengths, latent_means)
        probe_basis = pca_basis(probe_centered[cal['split'] == 0].reshape(-1, targets.shape[-1]), plan['dimension'])
        assert probe_basis.shape[1] == plan['dimension']
        latent = probe_centered @ probe_basis
        zf, zv = [latent[cal['split'] == k].reshape(-1, plan['dimension']) for k in [0, 1]]
        test_encoded = (input_test @ matrix - encoder['mean']) / encoder['scale']
        decoded_test = test_encoded @ decoder[:-1] + decoder[-1]
        pair_reconstruction = (decoded_test[pairs[:, 0]] - decoded_test[pairs[:, 1]]).reshape(-1, targets.shape[-1])
        z = pair_reconstruction @ probe_basis
        arrays.update({prefix + '_encoder_' + k: v for k, v in encoder.items()})
        arrays[prefix + '_decoder'] = decoder
        arrays[prefix + '_probe_basis'] = probe_basis
        arrays[prefix + '_probe_lengths'] = latent_lengths
        arrays[prefix + '_probe_means'] = latent_means
        arrays[prefix + '_pair_latent_input'] = z
        for g in LETTERS:
            arrays['shuffle_' + g + '_fit'] = shuffled[g][0]
            arrays['shuffle_' + g + '_validation'] = shuffled[g][1]
        maps = {'correct': {}, 'shuffled': {}}
        selections = []
        for g, action in LETTERS.items():
            changed = latent[:, table[:, action]]
            af, av = [changed[cal['split'] == k].reshape(-1, plan['dimension']) for k in [0, 1]]
            for pairing in maps:
                tf, tv = (af, av) if pairing == 'correct' else (af[shuffled[g][0]], av[shuffled[g][1]])
                fitted, selected = generator_fit(zf, tf, zv, tv, plan['ridge_grid'])
                maps[pairing][g] = fitted
                arrays[prefix + '_' + pairing + '_' + g] = fitted
                selections.append({'generator': g, 'pairing': pairing, **selected})
        fits.append({'pipeline': prefix, 'encoder': method, 'branch': branch, 'input_dimension': input_cal.shape[-1],
            'reconstruction_input_dimension': matrix.shape[1],
            'latent_dimension': plan['dimension'], 'decoder_parameters': decoder.size,
            'generator_parameters_per_pairing': 3 * plan['dimension'] ** 2,
            'input_center_statistics': encoder['mean'].size, 'input_scale_statistics': encoder['scale'].size,
            'fitted_input_encoder_loadings': matrix.size if method == 'fit_pca' else 0,
            'fitted_probe_loadings': probe_basis.size,
            'probe_center_statistics': latent_means.size,
            'input_fit_rank': int(np.linalg.matrix_rank(xf)),
            'decoder_fit_rows': len(zf), 'decoder_validation_rows': len(zv),
            'generator_fit_rows_each': len(zf), 'generator_validation_rows_each': len(zv),
            'ridge_candidates_per_fit': len(plan['ridge_grid']), 'decoder': decoder_info, 'generators': selections})
        original_targets = true_delta @ probe_basis
        reconstruction_energy = float(np.square(pair_reconstruction).sum())
        shared_source = pair_reconstruction @ shared_q
        between_bases = probe_basis.T @ shared_q
        for pairing, operators in maps.items():
            for word in ['', *plan['words'], 'ic_against_ci']:
                acting_word = 'ic' if word == 'ic_against_ci' else word
                target_word = 'ci' if word == 'ic_against_ci' else word
                product = np.eye(plan['dimension'])
                for g in acting_word:
                    product = product @ operators[g]
                action = word_action(target_word, table, LETTERS)
                order = table[:, action]
                change = z @ product - z
                target = true_delta[:, order].reshape(pair_reconstruction.shape)
                target_latent = original_targets[:, order].reshape(z.shape)
                error = residual_update_error(pair_reconstruction, z, product, target, target_latent, reconstruction_energy, target_energy)
                projected_prediction = shared_source + change @ between_bases
                projected_target = projected_delta[:, order].reshape(projected_prediction.shape)
                projected_error = float(np.square(projected_prediction - projected_target).sum())
                assert error >= -target_energy * 1e-10 and projected_error >= -projected_energy * 1e-10
                displacement_energy = float(np.square(true_delta[:, order] - true_delta).sum())
                rows.append({'pipeline': prefix, 'encoder': method, 'branch': branch, 'pairing': pairing, 'word': word or 'identity',
                    'pairs': len(pairs), 'starts_per_pair': 8, 'full_null_pair_target_nmse': max(error, 0.) / target_energy,
                    'shared_pca64_pair_target_nmse': max(projected_error, 0.) / projected_energy,
                    'full_null_action_displacement_nmse': max(error, 0.) / displacement_energy if displacement_energy > 1e-20 else None})
    map_path = root / 'maps' / (parent['source'] + '.npz')
    np.savez_compressed(map_path, **arrays)
    atomic_json(destination, {'source': parent['source'], 'group': parent['group'], 'seed': parent['seed'], 'status': parent['status'],
        'parent_path': str(path), 'parent_sha256': digest(path), 'map_path': str(map_path), 'map_sha256': digest(map_path),
        'calibration_target_dim': targets.shape[-1], 'shared_pca64_pair_energy_fraction': projected_energy / target_energy,
        'fits': fits, 'results': rows, 'completed_utc': now()})
    print({'budget_geometry_completed': parent['source'], 'primary_composites': {
        pipeline: float(np.mean([r['full_null_pair_target_nmse'] for r in rows if r['pipeline'] == pipeline
            and r['pairing'] == 'correct' and r['word'] in plan['composites']]))
        for pipeline in plan['branches']}}, flush=True)


def run():
    plan, root, parents = initialize()
    for i, path in enumerate(parents):
        evaluate(plan, root, path)
        atomic_json(root / 'state.json', {'status': 'complete' if i == len(parents) - 1 else 'evaluating',
            'completed_sources': len(list((root / 'evaluations').glob('*.json'))), 'updated_utc': now()})


if __name__ == '__main__':
    run()
