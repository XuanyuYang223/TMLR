"""Recompute primary direct errors, verify orbit splits and frozen sources."""
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch

from .field_algebra_structure import controlled_world, field_orbits, dihedral_table, projection_split
from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .native_confirmation import setup, permutation_row
from .native_source_subspace import contrast_basis
from .permutation_audit import transform
from .permworld_combinations import sha
from .representation_algebra import ACTION_NAMES, permutation_action_table, word_action, apply_word


def check_direct(h, split, strata, table, letters, result, arraypath):
    h = np.asarray(h, dtype=np.float64).copy()
    for n in np.unique(strata):
        mean = h[(strata == n) & (split == 0)].reshape(-1, h.shape[-1]).mean(0)
        h[strata == n] -= mean
    with np.load(arraypath) as a:
        basis = a['basis'].astype(np.float64)
        latent = h @ basis; test = latent[split == 2].reshape(-1, basis.shape[1])
        actual = h[split == 2].reshape(-1, h.shape[-1])
        np.testing.assert_allclose(test, a['test_latent'], rtol=2e-5, atol=3e-5)
        maps = {g: a[f'map_{g}'].astype(np.float64) for g in letters}
        count = 0
        for kind in ('generators', 'composites'):
            for r in result[kind]:
                word = r.get('generator', r.get('word')); action = word_action(word, table, letters)
                assert action == r['action']
                predicted = apply_word(test, word, maps)
                target = h[split == 2][:, table[:, action]].reshape(actual.shape)
                delta = target-actual; denominator = np.square(delta).sum()
                if r['status'] == 'invariant_action':
                    assert denominator < np.square(actual).sum()*1e-12; continue
                error = float(np.square((predicted-test) @ basis.T-delta).sum()/denominator)
                np.testing.assert_allclose(error, r['full_space_displacement_nmse'], atol=3e-5, rtol=3e-5)
                count += 1
        for r in result['laws']:
            # Recover words by matching the named law.
            words = {'c involution': ('cc',''), 'r involution': ('rr',''), 'i involution': ('ii',''),
                'commuting c and r': ('cr','rc'), 'conjugating c by i': ('ici','r'),
                'rotation order four': ('aaaa',''), 'reflection involution': ('ss',''),
                'reflection conjugates rotation to inverse': ('sas','aaa')}
            left, right = words[r['name']]
            error = float(np.square(apply_word(test, left, maps)-apply_word(test, right, maps)).sum()/np.square(test).sum())
            np.testing.assert_allclose(error, r['consistency_nmse'], atol=3e-5, rtol=3e-5)
            count += 1
        return count


def run():
    plan = json.loads(Path('configs/algebra_structure.json').read_text()); root = Path(plan['output'])
    native = root/'native'; config = json.loads(Path(plan['native_base_config']).read_text())
    _, _, tokens, one_line = setup(config)
    for subdir in ('native', 'field'):
        signature = json.loads((root/subdir/'protocol.json').read_text())['signature']
        for path, digest in signature['code'].items(): assert sha(path) == digest
    old_signature = json.loads((Path(plan['native_source'])/'protocol.json').read_text())['signature']
    for path, digest in old_signature['core_sha256'].items(): assert sha(path) == digest
    data = dict(np.load(native/'dataset.npz')); table = permutation_action_table()
    previous = set()
    for path in plan['excluded_datasets']:
        with np.load(path) as archive:
            for key in archive.files:
                if key.endswith('_input'):
                    previous.update(permutation_row(row, int(n)) for row, n in zip(archive[key], archive[key[:-6]+'_lengths']))
    probe_sets = [set() for _ in range(3)]; rows_checked = 0
    for orbit, raw, n, split in zip(data['input'], data['permutations'], data['lengths'], data['split']):
        base = tuple(map(int, raw[0,:n]))
        for a, name in enumerate(ACTION_NAMES):
            p = transform(base, name)
            assert p == tuple(map(int, raw[a,:n])) and p == permutation_row(orbit[a], int(n))
            assert p not in previous and all(p not in s for s in probe_sets)
            probe_sets[int(split)].add(p)
            prefix = [tokens['<BOS>'], tokens['<SIZE>'], int(n)]+[tokens[t] for t in one_line(p)]
            prefix += [tokens['<PAD>']]*(orbit.shape[-1]-len(prefix))
            np.testing.assert_array_equal(prefix, orbit[a]); rows_checked += 1
    protocol = json.loads((native/'protocol.json').read_text())['signature']
    for name, digest in protocol['source_models'].items():
        assert sha(Path(plan['native_source'])/'multi/checkpoints'/f'{name}.pt') == digest
    direct_proto = json.loads((root/'direct/protocol.json').read_text())['signature']
    assert sha('experiments/algebra_structure_direct.py') == direct_proto['code_sha256']
    native_checks, file_hash_checks = 0, 0; initial_bases = {}
    for path in sorted((native/'probes').glob('*.json')):
        if path.name.startswith('teacher_'): continue
        r = json.loads(path.read_text()); feature_path = native/'features'/f'{path.stem}.npz'
        assert sha(feature_path) == r['feature_sha256'] == direct_proto['native_sources'][feature_path.name]
        for name, digest in r['array_sha256'].items(): assert sha(native/'arrays'/name) == digest; file_hash_checks += 1
        direct = json.loads((root/'direct/native'/path.name).read_text())
        hidden = dict(np.load(feature_path))
        if r['model_status'] == 'trained':
            state = torch.load(Path(plan['native_source'])/'multi/checkpoints'/f"d256_l4_{r['group']}_s{r['seed']}.pt", map_location='cpu', weights_only=True, mmap=True)['model']
            basis, _ = contrast_basis(state['lm_head.weight'][:31].numpy())
        else:
            if r['seed'] not in initial_bases:
                model = make_model(config, {'d_model':256, 'layers':4, 'heads':8}, r['seed'], 'cpu')
                initial_bases[r['seed']] = contrast_basis(model.lm_head.weight[:31].detach().numpy())[0]
            basis = initial_bases[r['seed']]
        for landmark in plan['native_landmarks']:
            full = hidden[landmark][:,:,-1].astype(np.float64)
            blocks = full.reshape(*full.shape[:-1], -1, 256)
            numeric = ((blocks @ basis) @ basis.T).reshape(full.shape)
            for view, h in [(landmark, full), (f'{landmark}_numeric_null', full-numeric)]:
                array = root/'direct/native'/f'{path.stem}_{view}.npz'
                native_checks += check_direct(h, data['split'], data['lengths'], table, {'c':1, 'r':2, 'i':4}, direct['results'][view], array)
    field_proto = json.loads((root/'field/protocol.json').read_text())['signature']
    for path, digest in field_proto['sources'].items(): assert sha(path) == digest; file_hash_checks += 1
    fconfig = json.loads(Path(plan['field_config']).read_text()); source = Path(plan['field_source'])
    orbits, split = field_orbits(5, plan['field_split_seed'], plan['field_fit_fraction'], plan['field_validation_fraction'])
    table = dihedral_table(); field_checks, projection_checks = 0, 0
    from .field_symmetry import group_sources
    for w in fconfig['world_seeds']:
        inputs, _, _, basis = controlled_world(5,4,w)
        latent = inputs @ basis.T % 5; lookup = {tuple(z): i for i,z in enumerate(latent)}
        orbit_ids = np.array([[lookup[tuple(z)] for z in orbit] for orbit in orbits])
        psplit = projection_split(latent, plan['field_split_seed']+1)
        values = latent.copy(); values[:,2] = 0; images = np.array([lookup[tuple(z)] for z in values])
        assert np.array_equal(images[images], images)
        for a,b in ((0,1),(0,2),(1,2)): assert not set(images[psplit == a]) & set(images[psplit == b])
        for g,m,status in product(fconfig['groups'], fconfig['model_seeds'], ('random','trained')):
            name = f'{g}_w{w}_s{m}_{status}'
            step = 0 if status == 'random' else fconfig['steps']
            hidden = np.load(source/f'{g}_w{w}_m{m}_step{step}_features.npy')
            d = json.loads((root/'direct/field'/f'{name}.json').read_text())
            views = {'hidden':hidden}
            if status == 'trained':
                labels = latent @ group_sources(g,5).T % 5; views['onehot'] = np.eye(5)[labels].reshape(len(labels),-1)
            for view,h in views.items():
                field_checks += check_direct(h[orbit_ids], split, np.zeros(len(orbits)), table, {'a':1,'s':4}, d['results'][view], root/'direct/field'/f'{name}_{view}.npz')
            r = json.loads((root/'field/probes'/f'{name}.json').read_text())
            for filename,digest in r['array_sha256'].items(): assert sha(root/'field/arrays'/filename) == digest; file_hash_checks += 1
            p = r['results']['hidden_projection']; a = dict(np.load(root/'field/arrays'/f'{name}_hidden_projection.npz'))
            fit = psplit == 0; test = psplit == 2
            mean = hidden[fit].mean(0); q = a['basis'].astype(np.float64)
            x = (hidden-mean) @ q; target = (hidden[images]-mean) @ q
            prediction = x[test] @ a['map'][:-1]+a['map'][-1]
            baseline = target[fit].mean(0); denom = np.square(target[test]-baseline).sum()
            error = float(np.square(prediction-target[test]).sum()/denom)
            np.testing.assert_allclose(error,p['test_nmse'],atol=3e-5,rtol=3e-5)
            projection_checks += 1
    completion = {'status':'passed', 'verified_utc':datetime.now(timezone.utc).isoformat(),
        'native_orbit_states_exactly_verified': rows_checked,
        'native_split_states': [len(s) for s in probe_sets], 'excluded_distinct_inputs':len(previous),
        'native_direct_error_and_law_values_recomputed':native_checks,
        'field_direct_error_and_law_values_recomputed':field_checks,
        'field_projection_endpoint_errors_recomputed':projection_checks,
        'stored_auxiliary_arrays_and_field_features_hash_checked':file_hash_checks,
        'native_source_checkpoint_hashes_verified':len(protocol['source_models']),
        'previous_frozen_confirmation_code_unchanged':True,
        'scope':'Primary direct full-space errors, laws, exact orbit generation, input exclusions, many-to-one projection split disjointness; Float32 stored probe matrices checked with 3e-5 tolerance.'}
    atomic_json(root/'verification.json',completion); print(json.dumps(completion,indent=2))


if __name__ == '__main__': run()
