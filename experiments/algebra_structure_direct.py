"""Direct h(x)->h(Tx) probes, with no orbit-mean input preprocessing.

Added transparently after inspecting early orbit-residual outputs. Subtract
only a FIT-only mean for each length (or one mean in the field system).
Fit the displacement independently for each generator: rho_g = I + W_g.
This regularizes toward identity, and measures error relative to the actual
action displacement. Full-space prediction keeps unprojected components
of h(x), so invariant or discarded energy cannot masquerade as success.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .field_algebra_structure import field_orbits, dihedral_table, controlled_world, group_sources
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .representation_algebra import (permutation_action_table, pca_basis, ridge_map,
    shuffled_rows, word_action, apply_word)


def direct_probe(features, split, strata, table, letters, words, dimension, ridge_grid, seed):
    h = np.asarray(features, dtype=np.float64)
    split, strata = np.asarray(split), np.asarray(strata)
    centered = h.copy()
    for value in np.unique(strata):
        mean = h[(strata == value) & (split == 0)].reshape(-1, h.shape[-1]).mean(0)
        centered[strata == value] -= mean
    fit = centered[split == 0].reshape(-1, h.shape[-1])
    basis = pca_basis(fit, dimension)
    z = centered @ basis
    xfit, xval, xtest = [z[split == k].reshape(-1, basis.shape[1]) for k in range(3)]
    htest = centered[split == 2].reshape(-1, h.shape[-1])
    fit_strata = np.repeat(strata[split == 0], h.shape[1])
    val_strata = np.repeat(strata[split == 1], h.shape[1])
    rng = np.random.default_rng(seed); maps, wrong_maps = {}, {}
    result = {'status': 'complete', 'probe_dimension': basis.shape[1],
        'input_preprocessing': 'FIT-only within-length mean; no orbit mean; raw h(x) only',
        'test_pca_energy_fraction': float(np.square(xtest).sum()/np.square(htest).sum()),
        'generators': [], 'composites': [], 'laws': [], 'wrong_order': []}
    arrays = {'basis': basis.astype(np.float32), 'test_latent': xtest.astype(np.float32)}

    def evaluate(predicted, action, wrong_predicted=None):
        target = centered[split == 2][:, table[:, action]].reshape(htest.shape)
        target_z = z[split == 2][:, table[:, action]].reshape(xtest.shape)
        displacement = target-htest; denom = np.square(displacement).sum()
        norm_z = np.square(target_z-xtest).sum()
        total = np.square(htest).sum()
        row = {'action': int(action), 'action_displacement_energy': float(denom/total),
               'latent_displacement_energy': float(norm_z/np.square(xtest).sum()),
               'full_space_displacement_nmse': None, 'latent_displacement_nmse': None,
               'raw_target_nmse': float(np.square(predicted-target_z).sum()/np.square(target_z).sum()),
               'status': 'invariant_action' if denom < total*1e-12 else 'complete',
               'shuffled_fit_displacement_nmse': None}
        if row['status'] == 'complete':
            # pred_full = h(x) + (pred_z-z(x)) Q^T; all discarded features
            # remain the actual h(x), rather than using target activations.
            error = (predicted-xtest) @ basis.T-displacement
            row['full_space_displacement_nmse'] = float(np.square(error).sum()/denom)
            if norm_z > 1e-20: row['latent_displacement_nmse'] = float(np.square(predicted-target_z).sum()/norm_z)
            if wrong_predicted is not None:
                row['shuffled_fit_displacement_nmse'] = float(np.square((wrong_predicted-xtest) @ basis.T-displacement).sum()/denom)
        return row

    for letter, action in letters.items():
        target = z[:, table[:, action]]
        yfit, yval = [target[split == k].reshape((-1, basis.shape[1])) for k in (0, 1)]
        candidates = [np.eye(basis.shape[1])+ridge_map(xfit, yfit-xfit, a) for a in ridge_grid]
        chosen = int(np.argmin([np.square(xval @ w-yval).sum() for w in candidates]))
        maps[letter] = candidates[chosen]
        sy = yfit[shuffled_rows(fit_strata, rng)]; sv = yval[shuffled_rows(val_strata, rng)]
        wrong_candidates = [np.eye(basis.shape[1])+ridge_map(xfit, sy-xfit, a) for a in ridge_grid]
        wrong_chosen = int(np.argmin([np.square(xval @ w-sv).sum() for w in wrong_candidates]))
        wrong_maps[letter] = wrong_candidates[wrong_chosen]
        row = evaluate(xtest @ maps[letter], action, xtest @ wrong_maps[letter])
        row.update(generator=letter, alpha=ridge_grid[chosen], shuffled_alpha=ridge_grid[wrong_chosen])
        result['generators'].append(row)
        arrays[f'map_{letter}'] = maps[letter].astype(np.float32)
    for word in words:
        action = word_action(word, table, letters)
        row = evaluate(apply_word(xtest, word, maps), action, apply_word(xtest, word, wrong_maps))
        row['word'] = ''.join(word); result['composites'].append(row)
    laws = [('cc', '', 'c involution'), ('rr', '', 'r involution'), ('ii', '', 'i involution'),
            ('cr', 'rc', 'commuting c and r'), ('ici', 'r', 'conjugating c by i')]
    if set(letters) == {'a', 's'}: laws = [('aaaa', '', 'rotation order four'), ('ss', '', 'reflection involution'), ('sas', 'aaa', 'reflection conjugates rotation to inverse')]
    for left, right, name in laws:
        if not set(left+right).issubset(letters): continue
        a = word_action(left, table, letters)
        assert a == word_action(right, table, letters)
        pleft, pright = apply_word(xtest, left, maps), apply_word(xtest, right, maps)
        result['laws'].append({'name': name, 'consistency_nmse': float(np.square(pleft-pright).sum()/np.square(xtest).sum()),
            'left_raw_target_nmse': evaluate(pleft, a)['raw_target_nmse'],
            'right_raw_target_nmse': evaluate(pright, a)['raw_target_nmse']})
    left, wrong = ('ci', 'ic') if 'i' in maps else ('as', 'sa')
    ca, wa = [word_action(w, table, letters) for w in (left, wrong)]
    target_c = centered[split == 2][:, table[:, ca]].reshape(htest.shape)
    target_w = centered[split == 2][:, table[:, wa]].reshape(htest.shape)
    predicted = htest+(apply_word(xtest, left, maps)-xtest) @ basis.T
    denom = np.square(target_c-htest).sum()
    separation = np.square(target_c-target_w).sum()
    result['wrong_order'].append({'correct_word': left, 'wrong_word': wrong,
        'status': 'indistinguishable_action_on_this_representation' if separation < np.square(htest).sum()*1e-12 else 'complete',
        'target_separation_energy': float(separation/np.square(htest).sum()),
        'gap_wrong_minus_correct': float((np.square(predicted-target_w).sum()-np.square(predicted-target_c).sum())/denom) if denom > 1e-20 else None})
    return result, arrays


def run():
    plan = json.loads(Path('configs/algebra_structure.json').read_text()); root = Path(plan['output'])
    output = root/'direct'; output.mkdir(exist_ok=True); (output/'native').mkdir(exist_ok=True); (output/'field').mkdir(exist_ok=True)
    native = root/'native'; data = dict(np.load(native/'dataset.npz'))
    sources = sorted((native/'features').glob('*.npz')); assert len(sources) == 48
    protocol = output/'protocol.json'
    signature = {'code_sha256': sha(__file__), 'config': plan, 'data_sha256': sha(native/'dataset.npz'),
        'field_protocol_sha256': sha(root/'field/protocol.json'),
        'core_code_sha256': sha('experiments/representation_algebra.py'),
        'scope': 'follow-up registered after orbit-centered endpoints; direct endpoints not inspected before registration',
        'reason': 'orbit centering can construct sign changes from raw invariant task codes; direct displacement is required for h(Tx)~rho h(x)',
        'native_sources': {p.name: sha(p) for p in sources},
        'evaluation': 'full hidden-space action-displacement NMSE; identity baseline exactly one for noninvariant actions; no test centering or target inputs in predictor'}
    if protocol.exists(): assert json.loads(protocol.read_text())['signature'] == signature
    else: atomic_json(protocol, {'registered_utc': datetime.now(timezone.utc).isoformat(), 'signature': signature})
    table = permutation_action_table()
    from .native_confirmation import setup
    from .native_source_subspace import contrast_basis
    from .longrun_transfer import make_model
    import torch
    config = json.loads(Path(plan['native_base_config']).read_text()); setup(config)
    initial_bases = {}
    for path in sources:
        dest = output/'native'/f'{path.stem}.json'
        if dest.exists(): continue
        r = json.loads((native/'probes'/f'{path.stem}.json').read_text()); hidden = dict(np.load(path))
        if r['model_status'] == 'trained':
            state = torch.load(Path(plan['native_source'])/'multi/checkpoints'/f"d256_l4_{r['group']}_s{r['seed']}.pt", map_location='cpu', weights_only=True, mmap=True)['model']
            numeric_basis, _ = contrast_basis(state['lm_head.weight'][:31].numpy())
        else:
            if r['seed'] not in initial_bases:
                model = make_model(config, {'d_model':256, 'layers':4, 'heads':8}, r['seed'], 'cpu')
                initial_bases[r['seed']] = contrast_basis(model.lm_head.weight[:31].detach().numpy())[0]
                del model
            numeric_basis = initial_bases[r['seed']]
        result = {}
        for landmark in plan['native_landmarks']:
            full = hidden[landmark][:, :, -1].astype(np.float64)
            blocks = full.reshape(*full.shape[:-1], -1, 256)
            numeric = ((blocks @ numeric_basis) @ numeric_basis.T).reshape(full.shape)
            for view, values in [(landmark, full), (f'{landmark}_numeric_null', full-numeric)]:
                result[view], arrays = direct_probe(values, data['split'], data['lengths'], table,
                    {'c': 1, 'r': 2, 'i': 4}, [('r', 'c'), ('c', 'i'), ('r', 'i'), ('r', 'c', 'i')],
                    plan['pca_dimension'], plan['ridge_grid'], r['seed']+9200)
                np.savez_compressed(output/'native'/f'{path.stem}_{view}.npz', **arrays)
        atomic_json(dest, {'group': r['group'], 'seed': r['seed'], 'model_status': r['model_status'], 'results': result, 'status': 'complete'})
    # Source-label codes use exactly the same direct probe, to expose the
    # difference between label algebra and orbit-centered sign construction.
    _, functions, _, _ = setup(config)
    teachers = sorted((native/'probes').glob('teacher_*.json'))
    for path in teachers:
        dest = output/'native'/path.name
        if dest.exists(): continue
        r = json.loads(path.read_text()); labels = np.array([[[functions[t](tuple(map(int, p[:n]))) for t in r['tasks']] for p in orbit] for orbit, n in zip(data['permutations'], data['lengths'])])
        results = {}
        for view, h in [('numeric', labels.astype(np.float64)), ('onehot', np.eye(31)[labels].reshape(len(labels), 8, -1))]:
            results[view], _ = direct_probe(h, data['split'], data['lengths'], table, {'c': 1, 'r': 2, 'i': 4},
                [('r', 'c'), ('c', 'i'), ('r', 'i'), ('r', 'c', 'i')], plan['pca_dimension'], plan['ridge_grid'], 9200)
        atomic_json(dest, {'group': r['group'], 'results': results})
    from itertools import product
    fconfig = json.loads(Path(plan['field_config']).read_text()); field_source = Path(plan['field_source'])
    orbits, split = field_orbits(fconfig['p'], plan['field_split_seed'], plan['field_fit_fraction'], plan['field_validation_fraction'])
    table = dihedral_table()
    for w in fconfig['world_seeds']:
        physical, _, _, basis = controlled_world(fconfig['p'], fconfig['dimension'], w)
        latent = physical @ basis.T % fconfig['p']; lookup = {tuple(v): i for i, v in enumerate(latent)}
        ids = np.array([[lookup[tuple(v)] for v in orbit] for orbit in orbits])
        for g, m, status in product(fconfig['groups'], fconfig['model_seeds'], ('random', 'trained')):
            name = f'{g}_w{w}_s{m}_{status}'; dest = output/'field'/f'{name}.json'
            if dest.exists(): continue
            step = 0 if status == 'random' else fconfig['steps']
            h = np.load(field_source/f'{g}_w{w}_m{m}_step{step}_features.npy')
            values = {'hidden': h}
            if status == 'trained':
                labels = latent @ group_sources(g, fconfig['p']).T % fconfig['p']
                values['onehot'] = np.eye(5)[labels].reshape(len(labels), -1)
            result = {}
            for view, features in values.items():
                result[view], arrays = direct_probe(features[ids], split, np.zeros(len(orbits)), table, {'a': 1, 's': 4},
                    [('a', 'a'), ('a', 'a', 'a'), ('a', 's'), ('a', 'a', 's'), ('a', 'a', 'a', 's')],
                    plan['field_probe_dimension'], plan['ridge_grid'], m+9200)
                np.savez_compressed(output/'field'/f'{name}_{view}.npz', **arrays)
            atomic_json(dest, {'group': g, 'world_seed': w, 'seed': m, 'model_status': status, 'results': result, 'status': 'complete'})
    atomic_json(output/'state.json', {'status': 'complete', 'native_conditions': 48, 'field_conditions': 54})


if __name__ == '__main__': run()
