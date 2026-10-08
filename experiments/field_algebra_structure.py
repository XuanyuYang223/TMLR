"""Independent C4/D4 and noninvertible projection probes of existing models."""
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path

import numpy as np

from .field_symmetry import group_sources
from .matched_field import controlled_world
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .representation_algebra import group_probe, pca_basis, apply_word


def dihedral_actions(value):
    return np.array([np.roll(value, k) for k in range(4)]+
                    [np.roll(value, k)[::-1] for k in range(4)])


def dihedral_table():
    base = np.arange(4)
    signatures = {tuple(v): i for i, v in enumerate(dihedral_actions(base))}
    return np.array([[signatures[tuple(dihedral_actions(a)[j])] for j in range(8)]
                     for a in dihedral_actions(base)])


def field_orbits(p, seed, fit_fraction=.6, validation_fraction=.2):
    remaining = set(product(range(p), repeat=4)); orbits = []
    while remaining:
        base = min(remaining)
        orbit = dihedral_actions(np.array(base))
        remaining.difference_update(map(tuple, orbit)); orbits.append(orbit)
    order = np.random.default_rng(seed).permutation(len(orbits))
    nf, nv = int(len(orbits)*fit_fraction), int(len(orbits)*validation_fraction)
    split = np.full(len(orbits), 2, dtype=np.int64)
    split[order[:nf]], split[order[nf:nf+nv]] = 0, 1
    return np.array(orbits), split


def projection_split(latent, seed, fit_fraction=.6, validation_fraction=.2):
    """P erases latent coordinate 3. Hold out entire retained-coordinate fibers.

    This keeps every fit/validation/test input AND every image P(x) disjoint
    between probe splits, despite P being many-to-one.
    """
    keys = [tuple(z[[0, 1, 3]]) for z in latent]
    unique = sorted(set(keys)); order = np.random.default_rng(seed).permutation(len(unique))
    nf, nv = int(len(unique)*fit_fraction), int(len(unique)*validation_fraction)
    assignments = {unique[int(k)]: 0 if i < nf else 1 if i < nf+nv else 2 for i, k in enumerate(order)}
    return np.array([assignments[k] for k in keys])


def affine_fit(x, y, alpha):
    augmented = np.column_stack([x, np.ones(len(x))])
    gram = augmented.T @ augmented
    scale = np.trace(x.T @ x)/max(1, x.shape[1])
    penalty = np.diag([alpha*max(scale, 1e-20)]*x.shape[1]+[0.])
    return np.linalg.solve(gram+penalty, augmented.T @ y)


def affine_apply(x, w): return x @ w[:-1]+w[-1]


def projection_probe(hidden, images, split, dimension, ridge_grid, seed):
    hidden = np.asarray(hidden, dtype=np.float64)
    fit = split == 0; validation = split == 1; test = split == 2
    mean = hidden[fit].mean(0)
    basis = pca_basis(hidden[fit]-mean, dimension)
    x = (hidden-mean) @ basis; y = (hidden[images]-mean) @ basis
    baseline = y[fit].mean(0)
    denominator = np.square(y[test]-baseline).sum()
    if denominator <= 1e-20 or not basis.shape[1]: return {'status': 'zero_variance'}, {}
    error = lambda p: float(np.square(p-y[test]).sum()/denominator)
    candidates = [affine_fit(x[fit], y[fit], a) for a in ridge_grid]
    chosen = int(np.argmin([np.square(affine_apply(x[validation], w)-y[validation]).sum() for w in candidates]))
    w = candidates[chosen]; predicted = affine_apply(x[test], w)
    rng = np.random.default_rng(seed)
    shuffled = y[fit][rng.permutation(int(fit.sum()))]
    shuffled_val = y[validation][rng.permutation(int(validation.sum()))]
    wrong = [affine_fit(x[fit], shuffled, a) for a in ridge_grid]
    wrong_index = int(np.argmin([np.square(affine_apply(x[validation], v)-shuffled_val).sum() for v in wrong]))
    repeated = affine_apply(predicted, w)
    centered_full = hidden[images[test]]-mean
    full_baseline = hidden[images[fit]].mean(0)
    result = {'status': 'complete', 'probe_dimension': basis.shape[1], 'alpha': ridge_grid[chosen],
        'test_nmse': error(predicted), 'identity_nmse': error(x[test]),
        'shuffled_fit_nmse': error(affine_apply(x[test], wrong[wrong_index])),
        'double_prediction_nmse': error(repeated),
        'idempotence_consistency_nmse': float(np.square(repeated-predicted).sum()/denominator),
        'correct_versus_identity_gap': float((np.square(predicted-x[test]).sum()-np.square(predicted-y[test]).sum())/denominator),
        'full_space_nmse': float(np.square(predicted @ basis.T + mean-hidden[images[test]]).sum()/
                                np.square(hidden[images[test]]-full_baseline).sum()),
        'test_target_pca_energy_fraction': float(np.square(y[test]).sum()/np.square(centered_full).sum()),
        'fit_rows': int(fit.sum()), 'validation_rows': int(validation.sum()), 'test_rows': int(test.sum())}
    return result, {'basis': basis.astype(np.float32), 'map': w.astype(np.float32),
                    'test_latent': x[test].astype(np.float32), 'test_target': y[test].astype(np.float32),
                    'predicted': predicted.astype(np.float32), 'double_predicted': repeated.astype(np.float32)}


def run(config_path='configs/algebra_structure.json'):
    plan = json.loads(Path(config_path).read_text()); config = json.loads(Path(plan['field_config']).read_text())
    source = Path(plan['field_source']); root = Path(plan['output'])/'field'
    root.mkdir(parents=True, exist_ok=True); (root/'probes').mkdir(exist_ok=True); (root/'arrays').mkdir(exist_ok=True)
    source_paths = [source/f'{g}_w{w}_m{m}_step{s}_features.npy' for g, w, m, s in
                    product(config['groups'], config['world_seeds'], config['model_seeds'], (0, config['steps']))]
    signature = {'plan': plan, 'code': {p: sha(p) for p in ('experiments/representation_algebra.py',
        'experiments/field_algebra_structure.py', 'experiments/native_algebra_structure.py')},
        'sources': {str(p): sha(p) for p in source_paths}, 'scope': plan['field_scope'],
        'projection': 'erase third latent coordinate; affine probe; all retained-coordinate fibers held out together'}
    protocol = root/'protocol.json'
    if protocol.exists(): assert json.loads(protocol.read_text())['signature'] == signature
    else: atomic_json(protocol, {'registered_utc': datetime.now(timezone.utc).isoformat(), 'signature': signature,
                                'new_probe_results_at_registration': 0})
    orbits, orbit_split = field_orbits(config['p'], plan['field_split_seed'], plan['field_fit_fraction'], plan['field_validation_fraction'])
    table = dihedral_table()
    audit = {'dihedral_table': table.tolist(), 'unique_orbits': len(orbits),
             'orbit_splits': {str(k): int(np.sum(orbit_split == k)) for k in range(3)},
             'source_models_used_all_625_inputs': True, 'worlds': []}
    for world_seed in config['world_seeds']:
        _, _, _, basis = controlled_world(config['p'], config['dimension'], world_seed)
        physical = np.array(list(product(range(config['p']), repeat=4)), dtype=np.int64)
        latent = physical @ basis.T % config['p']
        lookup = {tuple(z): i for i, z in enumerate(latent)}
        orbit_ids = np.array([[lookup[tuple(z)] for z in orbit] for orbit in orbits])
        pvalues = latent.copy(); pvalues[:, 2] = 0
        images = np.array([lookup[tuple(z)] for z in pvalues])
        assert np.array_equal(images[images], images)
        split = projection_split(latent, plan['field_split_seed']+1, plan['field_fit_fraction'], plan['field_validation_fraction'])
        for a in range(3):
            for b in range(a):
                assert not set(images[split == a]).intersection(images[split == b])
                assert not set(np.flatnonzero(split == a)).intersection(images[split == b])
        audit['worlds'].append({'seed': world_seed, 'projection_split_rows': {str(k): int(np.sum(split == k)) for k in range(3)},
                                'projection_images_disjoint': True})
        for group in config['groups']:
            labels = latent @ group_sources(group, config['p']).T % config['p']
            for model_seed in config['model_seeds']:
                for status, step in [('random', 0), ('trained', config['steps'])]:
                    name = f'{group}_w{world_seed}_s{model_seed}_{status}'
                    path = root/'probes'/f'{name}.json'
                    if path.exists(): continue
                    hidden = np.load(source/f'{group}_w{world_seed}_m{model_seed}_step{step}_features.npy')
                    results, array_hashes = {}, {}
                    values = {'hidden': hidden}
                    if status == 'trained': values.update(exact_numeric=labels.astype(np.float64),
                        exact_onehot=np.eye(config['p'])[labels].reshape(len(labels), -1))
                    for view, features in values.items():
                        r, arrays = group_probe(features[orbit_ids], orbit_split, np.zeros(len(orbits)), table,
                            {'a': 1, 's': 4}, [('a', 'a'), ('a', 'a', 'a'), ('a', 's'), ('a', 'a', 's'), ('a', 'a', 'a', 's')],
                            plan['field_probe_dimension'], plan['ridge_grid'], model_seed+8000)
                        # Rotation-only C4 conclusions use generator a and words aa/aaa;
                        # D4 uses both a and s. Rotation/reflection generators are independent.
                        results[f'{view}_dihedral'] = r
                        arraypath = root/'arrays'/f'{name}_{view}_dihedral.npz'
                        np.savez_compressed(arraypath, **arrays); array_hashes[arraypath.name] = sha(arraypath)
                        r, arrays = projection_probe(features, images, split, plan['field_probe_dimension'], plan['ridge_grid'], model_seed+8100)
                        results[f'{view}_projection'] = r
                        arraypath = root/'arrays'/f'{name}_{view}_projection.npz'
                        np.savez_compressed(arraypath, **arrays); array_hashes[arraypath.name] = sha(arraypath)
                    atomic_json(path, {'status': 'complete', 'group': group, 'world_seed': world_seed,
                        'seed': model_seed, 'model_status': status, 'results': results, 'array_sha256': array_hashes})
                    print(json.dumps({'completed': name, 'rotation_nmse': results['hidden_dihedral']['generators'][0]['test_nmse'],
                        'projection_nmse': results['hidden_projection']['test_nmse']}), flush=True)
    atomic_json(root/'dataset_audit.json', audit)
    atomic_json(root/'state.json', {'status': 'complete', 'conditions': 54, 'completed_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__': run()
