"""Explore representation landmarks after preliminary ONE_END observations.

Source-query concatenations are compared only within the same task group.
A common heldout query and masked prefix pooling permit cross-group comparison.
No source or target answer labels enter feature extraction.
"""
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import torch

from .analysis import linear_cka
from .longrun_engine import atomic_json, load_session
from .longrun_transfer import make_model, query_features
from .native_transform_audit import OPERATORS
from .permworld_combinations import features, sha
from .permworld_combinations_report import within_length_center
from .six_hour_report import write_rows


def point_kernel_cka(left, right):
    """Equivalent linear CKA using point kernels for wide concatenations."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    left, right = left-left.mean(0), right-right.mean(0)
    k, l = left@left.T, right@right.T
    denominator = np.sqrt(np.square(k).sum()*np.square(l).sum())
    return float((k*l).sum()/denominator) if denominator else None


@torch.no_grad()
def pooled_prefix_features(model, data, split, token_ids):
    model.eval()
    values = []
    for start in range(0, len(data[f'{split}_input']), 128):
        x = data[f'{split}_input'][start:start+128]
        mask = x != token_ids['<PAD>']
        hidden, valid = model._embed_inputs(x, mask)
        for block in model.blocks: hidden = block(hidden, valid)
        hidden = model.final_norm(hidden)
        values.append(((hidden*mask[..., None]).sum(1)/mask.sum(1, keepdim=True)).cpu().numpy())
    return np.concatenate(values)


def run():
    plan, config, groups, data, names, token_ids, device = load_session()
    root = Path(plan['output'])
    output = root/'landmarks'; output.mkdir(exist_ok=True)
    arch = json.loads((root/'architecture_selection.json').read_text())['selected']
    signature = {'source_architecture': arch, 'data_sha256': sha(root/'dataset/data.npz'),
                 'code_sha256': sha(__file__), 'query_extraction_sha256': sha('experiments/longrun_transfer.py'),
                 'common_heldout_query': config['target_tasks'][0],
                 'transformed_data_sha256': sha(root/'native_transform_audit/transformed_representation.npz'),
                 'status': 'exploratory after preliminary ONE_END geometry, before target adaptation results',
                 'landmarks': ['ONE_END', 'masked prefix mean', 'common heldout query at equals', 'concatenated trained-task queries within group only']}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    if (output/'metadata.json').exists(): assert json.loads((output/'metadata.json').read_text())['fingerprint'] == fingerprint
    atomic_json(output/'metadata.json', {**signature, 'fingerprint': fingerprint})
    source_tasks = sorted({t for g in groups for t in g['tasks']})
    with np.load(root/'native_transform_audit/transformed_representation.npz') as archive:
        for operator in OPERATORS:
            data[operator+'_input'] = torch.tensor(archive[operator+'_input'], device=device)
            data[operator+'_lengths'] = data['representation_lengths']
    for seed in plan['model_seeds']:
        model = make_model(config, arch, seed, device)
        for task in source_tasks:
            path = output/f'initial_s{seed}_query_{task}.npy'
            if not path.exists(): np.save(path, query_features(model, data, 'representation', task, token_ids))
        for landmark in ('heldout_query', 'prefix_mean'):
            path = output/f'initial_s{seed}_{landmark}.npy'
            if not path.exists(): np.save(path, query_features(model, data, 'representation', config['target_tasks'][0], token_ids) if landmark == 'heldout_query' else pooled_prefix_features(model, data, 'representation', token_ids))
        for operator in OPERATORS:
            for task in source_tasks:
                path = output/f'initial_s{seed}_{operator}_query_{task}.npy'
                if not path.exists(): np.save(path, query_features(model, data, operator, task, token_ids))
            for landmark in ('ONE_END', 'heldout_query', 'prefix_mean'):
                path = output/f'initial_s{seed}_{operator}_{landmark}.npy'
                if not path.exists(): np.save(path, extract(model, data, operator, landmark, config, None, token_ids))
        for group in groups:
            job_id = f"{arch['id']}_{group['id']}_s{seed}"
            record = json.loads((root/'multi'/f'{job_id}.json').read_text())
            assert record['status'] == 'complete'
            checkpoint = root/'multi'/'checkpoints'/f'{job_id}.pt'
            assert sha(checkpoint) == record['checkpoint_sha256']
            model = make_model(config, arch, seed, device)
            model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device)['model'])
            for landmark in ('heldout_query', 'prefix_mean', 'source_query_concat'):
                path = output/f'{job_id}_{landmark}.npy'
                if path.exists(): continue
                if landmark == 'heldout_query':
                    h = query_features(model, data, 'representation', config['target_tasks'][0], token_ids)
                elif landmark == 'prefix_mean':
                    h = pooled_prefix_features(model, data, 'representation', token_ids)
                else:
                    h = np.concatenate([query_features(model, data, 'representation', t, token_ids) for t in group['tasks']], axis=1)
                np.save(path, h)
            for operator in OPERATORS:
                for landmark in ('ONE_END', 'heldout_query', 'prefix_mean', 'source_query_concat'):
                    path = output/f'{job_id}_{operator}_{landmark}.npy'
                    if not path.exists(): np.save(path, extract(model, data, operator, landmark, config, group, token_ids))
            print(json.dumps({'landmarks_complete': job_id}), flush=True)
    analyze(plan, groups, arch, data['representation_lengths'].cpu().numpy())
    analyze_transforms(plan, groups, arch, data['representation_lengths'].cpu().numpy())


def extract(model, data, split, landmark, config, group, token_ids):
    if landmark == 'ONE_END': return features(model, data, split, token_ids)
    if landmark == 'heldout_query': return query_features(model, data, split, config['target_tasks'][0], token_ids)
    if landmark == 'prefix_mean': return pooled_prefix_features(model, data, split, token_ids)
    return np.concatenate([query_features(model, data, split, t, token_ids) for t in group['tasks']], axis=1)


def analyze(plan, groups, arch, lengths):
    root = Path(plan['output']); output = root/'landmarks'
    cross_seed, cross_group = [], []
    for landmark in ('ONE_END', 'heldout_query', 'prefix_mean', 'source_query_concat'):
        for group in groups:
            for a, b in combinations(plan['model_seeds'], 2):
                trained, initial = [], []
                for seed in (a, b):
                    run_id = f"{arch['id']}_{group['id']}_s{seed}"
                    trained.append(np.load(root/'multi'/f'{run_id}_features.npy') if landmark == 'ONE_END' else np.load(output/f'{run_id}_{landmark}.npy'))
                    if landmark == 'source_query_concat':
                        initial.append(np.concatenate([np.load(output/f'initial_s{seed}_query_{t}.npy') for t in group['tasks']], axis=1))
                    elif landmark == 'ONE_END':
                        initial.append(np.load(root/'initial'/f's{seed}_features.npy'))
                    else:
                        initial.append(np.load(output/f'initial_s{seed}_{landmark}.npy'))
                for centering in ('raw', 'within_length'):
                    left = [within_length_center(v, lengths) for v in trained] if centering == 'within_length' else trained
                    right = [within_length_center(v, lengths) for v in initial] if centering == 'within_length' else initial
                    measure = point_kernel_cka if landmark == 'source_query_concat' else linear_cka
                    trained_cka, initial_cka = measure(*left), measure(*right)
                    cross_seed.append({'landmark': landmark, 'group': group['id'], 'seed_a': a, 'seed_b': b,
                                       'centering': centering, 'trained_cka': trained_cka, 'initial_cka': initial_cka,
                                       'change_from_initial': trained_cka-initial_cka})
        if landmark == 'source_query_concat': continue
        for seed in plan['model_seeds']:
            for a, b in combinations(groups, 2):
                features_pair = []
                for group in (a, b):
                    run_id = f"{arch['id']}_{group['id']}_s{seed}"
                    path = root/'multi'/f'{run_id}_features.npy' if landmark == 'ONE_END' else output/f'{run_id}_{landmark}.npy'
                    features_pair.append(np.load(path))
                for centering in ('raw', 'within_length'):
                    pair = [within_length_center(v, lengths) for v in features_pair] if centering == 'within_length' else features_pair
                    cross_group.append({'landmark': landmark, 'seed': seed, 'left': a['id'], 'right': b['id'],
                                        'centering': centering, 'linear_cka': linear_cka(*pair)})
    write_rows(output/'cross_seed.csv', cross_seed)
    write_rows(output/'cross_group.csv', cross_group)
    summary = []
    for landmark in ('ONE_END', 'heldout_query', 'prefix_mean', 'source_query_concat'):
        for mode in ('raw', 'within_length'):
            cells = [r for r in cross_seed if (r['landmark'], r['centering']) == (landmark, mode)]
            summary.append({'landmark': landmark, 'centering': mode, 'cells': len(cells),
                            'trained_cka': float(np.mean([r['trained_cka'] for r in cells])),
                            'initial_cka': float(np.mean([r['initial_cka'] for r in cells])),
                            'positive_changes': sum(r['change_from_initial'] > 0 for r in cells)})
    atomic_json(output/'summary.json', {'status': 'exploratory landmark sensitivity', 'cross_seed': summary,
                                      'source_query_scope': 'same trained query set within group only; no cross-group source-query comparison',
                                      'source_answer_labels_used': False})
    print(json.dumps(summary, indent=2))


def analyze_transforms(plan, groups, arch, lengths):
    import csv
    root = Path(plan['output']); output = root/'landmarks'
    with (root/'native_transform_audit/categorical_kernel.csv').open() as handle:
        codes = {(r['group'], r['operator'], r['centering']): r for r in csv.DictReader(handle)}
    rows = []
    for seed in plan['model_seeds']:
        for group in groups:
            run_id = f"{arch['id']}_{group['id']}_s{seed}"
            for landmark in ('ONE_END', 'heldout_query', 'prefix_mean', 'source_query_concat'):
                original = np.load(root/'multi'/f'{run_id}_features.npy') if landmark == 'ONE_END' else np.load(output/f'{run_id}_{landmark}.npy')
                initial = (np.concatenate([np.load(output/f'initial_s{seed}_query_{t}.npy') for t in group['tasks']], axis=1) if landmark == 'source_query_concat'
                           else np.load(root/'initial'/f's{seed}_features.npy') if landmark == 'ONE_END' else np.load(output/f'initial_s{seed}_{landmark}.npy'))
                measure = point_kernel_cka if landmark == 'source_query_concat' else linear_cka
                for operator in OPERATORS:
                    changed = np.load(output/f'{run_id}_{operator}_{landmark}.npy')
                    initial_changed = (np.concatenate([np.load(output/f'initial_s{seed}_{operator}_query_{t}.npy') for t in group['tasks']], axis=1) if landmark == 'source_query_concat'
                                       else np.load(output/f'initial_s{seed}_{operator}_{landmark}.npy'))
                    for centering in ('raw', 'within_length'):
                        pair = [within_length_center(v, lengths) for v in (original, changed)] if centering == 'within_length' else [original, changed]
                        start_pair = [within_length_center(v, lengths) for v in (initial, initial_changed)] if centering == 'within_length' else [initial, initial_changed]
                        trained_cka, initial_cka = measure(*pair), measure(*start_pair)
                        oracle = codes[group['id'], operator, centering]
                        rows.append({'group': group['id'], 'seed': seed, 'operator': operator, 'landmark': landmark,
                                     'centering': centering, 'trained_cka': trained_cka, 'initial_cka': initial_cka,
                                     'change_from_initial': trained_cka-initial_cka,
                                     'categorical_code_cka': float(oracle['categorical_code_cka']),
                                     'closure_status': oracle['closure_status'],
                                     'transformed_source_training_overlap': int(oracle['source_training_overlap_count'])})
    write_rows(output/'transformed_cka.csv', rows)
    print(json.dumps({'transformed_cka_endpoints': len(rows)}))


if __name__ == '__main__': run()
