"""Prospective geometry assay of frozen ordinary PermWorld checkpoints."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .native_confirmation import setup, permutation_row
from .native_source_subspace import contrast_basis
from .permworld_combinations import prompts, select_groups, sha
from .permutation_audit import transform
from .representation_algebra import ACTION_NAMES, permutation_action_table, group_probe


def now(): return datetime.now(timezone.utc).isoformat()


def prepare(plan, config, root, tokens, one_line):
    path = root/'dataset.npz'
    if path.exists():
        return dict(np.load(path))
    seen = set()
    provenance = {}
    for filename in plan['excluded_datasets']:
        p = Path(filename)
        if not p.exists(): raise FileNotFoundError(p)
        provenance[str(p)] = sha(p)
        with np.load(p) as archive:
            for key in archive.files:
                if key.endswith('_input'):
                    lengths = archive[key[:-6]+'_lengths']
                    seen.update(permutation_row(x, int(n)) for x, n in zip(archive[key], lengths))
    prior_inputs = len(seen)
    rng = np.random.default_rng(plan['data_seed'])
    inputs, permutations, splits, lengths = [], [], [], []
    rejected_symmetric, rejected_seen = 0, 0
    for split, (name, count) in enumerate(plan['orbits_per_length'].items()):
        for n in range(plan['lengths'][0], plan['lengths'][1]+1):
            for _ in range(count):
                while True:
                    base = tuple(map(int, rng.permutation(n)+1))
                    orbit = [transform(base, a) for a in ACTION_NAMES]
                    if len(set(orbit)) != len(ACTION_NAMES):
                        rejected_symmetric += 1; continue
                    if any(x in seen for x in orbit):
                        rejected_seen += 1; continue
                    seen.update(orbit); break
                rows, raw = [], []
                for p in orbit:
                    prefix = [tokens['<BOS>'], tokens['<SIZE>'], n]+[tokens[t] for t in one_line(p)]
                    rows.append(prefix+[tokens['<PAD>']]*(2*config['lengths'][1]+4-len(prefix)))
                    raw.append(list(p)+[0]*(config['lengths'][1]-n))
                inputs.append(rows); permutations.append(raw); splits.append(split); lengths.append(n)
    data = {'input': np.array(inputs, dtype=np.int64), 'permutations': np.array(permutations, dtype=np.int64),
            'split': np.array(splits, dtype=np.int64), 'lengths': np.array(lengths, dtype=np.int64)}
    np.savez_compressed(path, **data)
    atomic_json(root/'dataset_audit.json', {'created_utc': now(), 'excluded_provenance': provenance,
        'previous_distinct_inputs': prior_inputs, 'orbits': len(inputs), 'states': len(inputs)*8,
        'full_size_orbits_only': True, 'rejected_symmetric': rejected_symmetric,
        'rejected_prior_or_probe_overlap': rejected_seen, 'sha256': sha(path),
        'split_orbits': {name: int(np.sum(data['split'] == i)) for i, name in enumerate(plan['orbits_per_length'])}})
    return data


@torch.no_grad()
def extract(model, data, tasks, tokens, batch_size):
    """Gather prefix and four source queries in one causally masked forward.

    Early layers are block outputs. Layer four includes final_norm, as used
    by the numeric readout. Labels/answer tokens never enter the forward pass.
    """
    model.eval(); device = next(model.parameters()).device
    inputs = data['input'].reshape(-1, data['input'].shape[-1])
    lengths = np.repeat(data['lengths'], 8)
    prefix, query = [], []
    for start in range(0, len(inputs), batch_size):
        x = torch.tensor(inputs[start:start+batch_size], device=device)
        n = torch.tensor(lengths[start:start+batch_size], device=device)
        ids, mask, position = prompts(x, n, tasks, tokens)
        hidden, valid = model._embed_inputs(ids, mask)
        layer_prefix, layer_query = [], []
        for k, block in enumerate(model.blocks):
            hidden = block(hidden, valid)
            chosen = model.final_norm(hidden) if k == len(model.blocks)-1 else hidden
            batch = len(x)
            # Future query tokens cannot affect the earlier ONE_END position.
            layer_prefix.append(chosen[torch.arange(batch, device=device), 2*n+3].cpu().numpy())
            q = chosen[torch.arange(len(ids), device=device), position].reshape(len(tasks), batch, -1)
            layer_query.append(q.permute(1, 0, 2).reshape(batch, -1).cpu().numpy())
        prefix.append(np.stack(layer_prefix, axis=1)); query.append(np.stack(layer_query, axis=1))
    shape = (len(data['lengths']), 8, len(model.blocks))
    return {'ONE_END': np.concatenate(prefix).reshape(*shape, -1),
            'source_query_concat': np.concatenate(query).reshape(*shape, -1)}


def probe_view(hidden, data, plan, seed):
    table = permutation_action_table()
    return group_probe(hidden, data['split'], data['lengths'], table,
                       {'c': 1, 'r': 2, 'i': 4}, [('r', 'c'), ('c', 'i'), ('r', 'i'), ('r', 'c', 'i')],
                       plan['pca_dimension'], plan['ridge_grid'], seed)


def run(config_path='configs/algebra_structure.json'):
    plan = json.loads(Path(config_path).read_text())
    config = json.loads(Path(plan['native_base_config']).read_text())
    source = Path(plan['native_source']); root = Path(plan['output'])/'native'
    root.mkdir(parents=True, exist_ok=True)
    for subdir in ('features', 'probes', 'arrays'): (root/subdir).mkdir(exist_ok=True)
    names, functions, tokens, one_line = setup(config)
    groups = select_groups(config)
    arch = json.loads((source/'multi/d256_l4_interior_none_s1009.json').read_text())['architecture']
    signature = {'config': plan, 'code': {p: sha(p) for p in ('experiments/native_algebra_structure.py',
        'experiments/representation_algebra.py', 'experiments/field_algebra_structure.py')},
        'source_models': {}, 'algebra_table': permutation_action_table().tolist(), 'action_names': ACTION_NAMES,
        'convention': 'row features; table[a,b] and W_a W_b both mean first a then b',
        'scope': plan['primary_scope'], 'training_policy': 'all eight four-task groups, fixed 20k checkpoint, three new source seeds'}
    for group in groups:
        for seed in plan['native_seeds']:
            name = f"d256_l4_{group['id']}_s{seed}"
            record = json.loads((source/'multi'/f'{name}.json').read_text())
            path = source/'multi/checkpoints'/f'{name}.pt'
            assert record['status'] == 'complete' and sha(path) == record['checkpoint_sha256']
            signature['source_models'][name] = record['checkpoint_sha256']
    protocol = root/'protocol.json'
    if protocol.exists(): assert json.loads(protocol.read_text())['signature'] == signature
    else: atomic_json(protocol, {'registered_utc': now(), 'signature': signature,
                                'new_probe_results_at_registration': 0})
    data = prepare(plan, config, root, tokens, one_line)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for group in groups:
        # Exact task codes provide an interpretable output-statistics control,
        # not a claim that a neural representation has this structure.
        teacher_path = root/'probes'/f"teacher_{group['id']}.json"
        if not teacher_path.exists():
            labels = np.array([[[functions[t](tuple(map(int, p[:n]))) for t in group['tasks']]
                for p in orbit] for orbit, n in zip(data['permutations'], data['lengths'])])
            teacher_results = {}
            for name, hidden in [('numeric', labels.astype(np.float64)),
                ('onehot', np.eye(31)[labels].reshape(len(labels), 8, -1))]:
                result, _ = probe_view(hidden, data, plan, plan['data_seed'])
                teacher_results[name] = result
            atomic_json(teacher_path, {'group': group['id'], 'tasks': group['tasks'], 'results': teacher_results})
        for seed in plan['native_seeds']:
            for status in ('random', 'trained'):
                name = f"{group['id']}_s{seed}_{status}"
                result_path = root/'probes'/f'{name}.json'
                if result_path.exists(): continue
                started = time.monotonic()
                atomic_json(root/'state.json', {'status': 'running', 'group': group['id'], 'seed': seed,
                                               'model_status': status, 'updated_utc': now()})
                model = make_model(config, arch, seed, device)
                if status == 'trained':
                    checkpoint = source/'multi/checkpoints'/f"d256_l4_{group['id']}_s{seed}.pt"
                    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)['model'])
                feature_path = root/'features'/f'{name}.npz'
                if feature_path.exists(): hidden = dict(np.load(feature_path))
                else:
                    if status == 'random':
                        cache = root/'features'/f'initial_s{seed}'
                        cache.mkdir(exist_ok=True)
                        missing = [t for t in group['tasks'] if not (cache/f'{t}.npy').exists()]
                        if missing:
                            values = extract(model, data, missing, tokens, plan['batch_size'])
                            np.save(cache/'ONE_END.npy', values['ONE_END'])
                            queries = values['source_query_concat'].reshape(*values['ONE_END'].shape[:-1], len(missing), arch['d_model'])
                            for j, task in enumerate(missing): np.save(cache/f'{task}.npy', queries[..., j, :])
                        hidden = {'ONE_END': np.load(cache/'ONE_END.npy'), 'source_query_concat':
                                  np.concatenate([np.load(cache/f'{t}.npy') for t in group['tasks']], axis=-1)}
                    else: hidden = extract(model, data, group['tasks'], tokens, plan['batch_size'])
                    np.savez_compressed(feature_path, **hidden)
                basis, _ = contrast_basis(model.lm_head.weight[:31].detach().cpu().numpy())
                results, array_hashes = {}, {}
                for landmark in plan['native_landmarks']:
                    for layer in plan['native_layers']:
                        key = f'{landmark}_layer{layer}'
                        result, arrays = probe_view(hidden[landmark][:, :, layer-1], data, plan, seed+9000)
                        results[key] = result
                        path = root/'arrays'/f'{name}_{key}.npz'
                        np.savez_compressed(path, **arrays); array_hashes[path.name] = sha(path)
                    final = hidden[landmark][:, :, -1].astype(np.float64)
                    blocks = final.reshape(*final.shape[:-1], -1, arch['d_model'])
                    projected = ((blocks @ basis) @ basis.T).reshape(final.shape)
                    for component, values in [('numeric_contrast', projected), ('numeric_null', final-projected)]:
                        key = f'{landmark}_{component}'
                        result, arrays = probe_view(values, data, plan, seed+9000)
                        result['variance_fraction_of_orbit_residual'] = float(np.square(values-values.mean(1, keepdims=True)).sum()/
                            max(np.square(final-final.mean(1, keepdims=True)).sum(), 1e-20))
                        results[key] = result
                        path = root/'arrays'/f'{name}_{key}.npz'
                        np.savez_compressed(path, **arrays); array_hashes[path.name] = sha(path)
                atomic_json(result_path, {'status': 'complete', 'group': group['id'], 'tasks': group['tasks'],
                    'seed': seed, 'model_status': status, 'dataset_sha256': sha(root/'dataset.npz'),
                    'feature_sha256': sha(feature_path), 'array_sha256': array_hashes,
                    'source_checkpoint_sha256': signature['source_models'][f"d256_l4_{group['id']}_s{seed}"] if status == 'trained' else None,
                    'source_audit_accuracy': float(np.mean([x['accuracy'] for x in json.loads((source/'multi'/f"d256_l4_{group['id']}_s{seed}.json").read_text())['source_audit']])) if status == 'trained' else None,
                    'results': results, 'seconds': time.monotonic()-started, 'completed_utc': now()})
                del model
                print(json.dumps({'completed': name, 'seconds': round(time.monotonic()-started, 2),
                    'prefix_generator_nmse': float(np.mean([r['test_nmse'] for r in results['ONE_END_layer4']['generators']])),
                    'query_generator_nmse': float(np.mean([r['test_nmse'] for r in results['source_query_concat_layer4']['generators']]))}), flush=True)
    atomic_json(root/'state.json', {'status': 'complete', 'completed_utc': now(), 'conditions': 48})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='configs/algebra_structure.json')
    run(parser.parse_args().config)
