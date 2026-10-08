"""Shared answer-orbit matching across three ordinary task combinations."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from .algebra_structure_replication import collect_prior_inputs, load_plan
from .answer_matched_geometry import extract_final, pair_geometry
from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .native_confirmation import setup
from .native_source_subspace import contrast_basis
from .permworld_combinations import sha
from .permutation_audit import transform
from .representation_algebra import ACTION_NAMES


CONFIG = 'configs/joint_answer_matched_geometry.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize():
    plan = json.loads(Path(CONFIG).read_text())
    previous, config, parent = load_plan()
    root = Path(plan['output'])
    root.mkdir(parents=True, exist_ok=True)
    for name in ['features', 'evaluations', 'arrays']:
        (root / name).mkdir(exist_ok=True)
    excludes = [*previous['excluded_datasets'], str(parent / 'probe_dataset.npz'),
        'results/algebra_hidden_relations/probe_dataset.npz', 'results/answer_matched_geometry/dataset.npz']
    world_root = Path('results/relation_world_confirmation')
    raw_excludes = ['results/algebra_hidden_relations/training_orbit_audit.npz']
    for output in sorted(world_root.glob('world*')):
        if (output / 'probe_dataset.npz').exists():
            excludes.append(str(output / 'probe_dataset.npz'))
        if (output / 'training_orbit_audit.npz').exists():
            raw_excludes.append(str(output / 'training_orbit_audit.npz'))
    signature = {'config_sha256': sha(CONFIG), 'code_sha256': sha(__file__),
        'helper_sha256': sha('experiments/answer_matched_geometry.py'), 'plan': plan,
        'parent_protocol_sha256': sha(parent / 'protocol.json'),
        'excluded_data_sha256': {path: sha(path) for path in [*excludes, *raw_excludes]},
        'frozen_map_sha256': {f'{group}_{seed}_{status}_{view}': sha(parent / 'arrays' / f'{group}_s{seed}_{status}_{view}.npz')
            for group in plan['groups'] for seed in plan['source_seeds'] for status in plan['model_statuses'] for view in plan['views']}}
    path = root / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())['signature'] == signature
    else:
        atomic_json(path, {'registered_utc': now(), 'signature': signature, 'new_joint_answer_pair_evaluations': 0})
    return plan, previous, config, root, excludes, raw_excludes


def prepare(plan, previous, config, root, excludes, raw_excludes, functions, tokens, one_line):
    path = root / 'dataset.npz'
    tasks = list(dict.fromkeys(task for group in previous['groups'] for task in group['tasks']))
    if path.exists():
        return dict(np.load(path)), tasks
    seen = collect_prior_inputs(excludes)
    for filename in raw_excludes:
        a = dict(np.load(filename))
        seen.update(tuple(map(int, p[:n])) for orbit, n in zip(a['permutations'], a['lengths']) for p in orbit)
    prior_count = len(seen)
    rng = np.random.default_rng(plan['data_seed'])
    rows, raw, labels, lengths, pair_ids = [], [], [], [], []
    pair = 0
    width = 2 * plan['lengths'][1] + 4
    for n in range(plan['lengths'][0], plan['lengths'][1] + 1):
        pending = {}
        accepted = 0
        attempts = 0
        while accepted < plan['pairs_per_length']:
            attempts += 1
            base = tuple(map(int, rng.permutation(n) + 1))
            orbit = [transform(base, name) for name in ACTION_NAMES]
            if len(set(orbit)) != 8 or any(p in seen for p in orbit):
                continue
            answers = [[functions[task](p) for task in tasks] for p in orbit]
            key = tuple(value for ys in answers for value in ys)
            old = pending.get(key)
            if old is not None and any(p in seen for p in old):
                pending.pop(key); old = None
            if old is None:
                pending[key] = orbit
                continue
            if set(old) & set(orbit):
                continue
            pending.pop(key)
            seen.update(old); seen.update(orbit)
            for member in [old, orbit]:
                encoded = []
                for p in member:
                    row = [tokens['<BOS>'], tokens['<SIZE>'], n] + [tokens[t] for t in one_line(p)]
                    encoded.append(row + [tokens['<PAD>']] * (width - len(row)))
                ys = [[functions[task](p) for task in tasks] for p in member]
                np.testing.assert_array_equal(ys, answers)
                rows.append(encoded); raw.append([list(p) + [0] * (plan['lengths'][1] - n) for p in member]); labels.append(ys)
                lengths.append(n); pair_ids.append(pair)
            pair += 1; accepted += 1
        print(json.dumps({'joint_pair_length_complete': n, 'attempts': attempts, 'pairs': accepted}), flush=True)
    data = {'input': np.asarray(rows, dtype=np.int64), 'permutations': np.asarray(raw, dtype=np.int64),
        'union_labels': np.asarray(labels, dtype=np.int64), 'lengths': np.asarray(lengths, dtype=np.int64), 'pair_ids': np.asarray(pair_ids, dtype=np.int64)}
    np.savez_compressed(path, **data)
    atomic_json(root / 'dataset_audit.json', {'dataset_sha256': sha(path), 'union_tasks': tasks, 'pairs': pair,
        'states': len(rows) * 8, 'excluded_distinct_inputs': prior_count,
        'matched_complete_output_orbit_signature_for_all_groups': True})
    return data, tasks


def run():
    plan, previous, config, root, excludes, raw_excludes = initialize()
    torch.set_num_threads(4)
    _, functions, tokens, one_line = setup(config)
    data, union_tasks = prepare(plan, previous, config, root, excludes, raw_excludes, functions, tokens, one_line)
    parent = Path(plan['parent'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pair_rows = np.stack([np.flatnonzero(data['pair_ids'] == pair) for pair in np.unique(data['pair_ids'])])
    for seed in plan['source_seeds']:
        for group in previous['groups']:
            if group['id'] not in plan['groups']:
                continue
            task_ids = [union_tasks.index(task) for task in group['tasks']]
            truth = data['union_labels'][:, :, task_ids]
            for status in plan['model_statuses']:
                name = f'{group["id"]}_s{seed}_{status}'
                dest = root / 'evaluations' / f'{name}.json'
                if dest.exists():
                    continue
                if time.time() >= datetime.fromisoformat(plan['deadline_utc']).timestamp():
                    return
                model = make_model(config, previous['architecture'], seed, device)
                if status == 'trained':
                    state = torch.load(parent / 'source/checkpoints' / f'{group["id"]}_s{seed}.pt', weights_only=True, map_location=device)
                    model.load_state_dict(state['model'])
                fp = root / 'features' / f'{name}.npz'
                if fp.exists():
                    features = dict(np.load(fp))
                else:
                    features = extract_final(model, data, group['tasks'], tokens, plan['batch_size'])
                    np.savez_compressed(fp, **features)
                weight = model.lm_head.weight[:31].detach().cpu().numpy().astype(np.float64)
                b = model.lm_head.bias
                bias = np.zeros(31) if b is None else b[:31].detach().cpu().numpy().astype(np.float64)
                block = features['source_query_concat'].reshape(len(data['lengths']), 8, 4, -1).astype(np.float64)
                predictions = (block @ weight.T + bias).argmax(-1)
                agreement = (predictions[pair_rows[:, 0]] == predictions[pair_rows[:, 1]]).all(axis=(1, 2))
                correct = (predictions[pair_rows] == truth[pair_rows]).all(axis=(1, 2, 3))
                basis, _ = contrast_basis(weight)
                null = (block - (block @ basis) @ basis.T).reshape(features['source_query_concat'].shape)
                views = {'source_query_concat': features['source_query_concat'], 'source_query_concat_full_width': features['source_query_concat'],
                    'source_query_numeric_null': null, 'ONE_END': features['ONE_END'], 'ONE_END_full_width': features['ONE_END']}
                results, hashes = [], {}
                for view, hidden in views.items():
                    archive_path = parent / 'arrays' / f'{name}_{view}.npz'
                    maps = dict(np.load(archive_path))
                    for subset_name, subset in [('all_joint_answer_matched_pairs', None), ('model_answers_agree_all_states', agreement), ('model_answers_correct_all_states', correct)]:
                        if subset is not None and subset.sum() == 0:
                            continue
                        rows, arrays = pair_geometry(hidden, data['pair_ids'], maps, plan['words'], subset)
                        selected = pair_rows if subset is None else pair_rows[subset]
                        difference = hidden[selected[:, 0]] - hidden[selected[:, 1]]
                        q = maps['basis'].astype(np.float64)
                        captured = float(np.square(difference @ q).sum() / np.square(difference).sum())
                        for row in rows:
                            row.update(view=view, subset=subset_name, paired_projection_energy_fraction=captured)
                        results.extend(rows)
                        ap = root / 'arrays' / f'{name}_{view}_{subset_name}.npz'
                        np.savez_compressed(ap, **arrays); hashes[ap.name] = sha(ap)
                atomic_json(dest, {'source': name, 'group': group['id'], 'seed': seed, 'status': status, 'feature_sha256': sha(fp),
                    'same_model_answer_pairs': int(agreement.sum()), 'all_orbit_correct_pairs': int(correct.sum()),
                    'model_query_accuracy': float(np.mean(predictions == truth)), 'results': results,
                    'array_sha256': hashes, 'completed_utc': now()})
                print(json.dumps({'joint_answer_geometry': name, 'primary': [r for r in results if r['view'] == plan['primary_view'] and r['subset'] == 'all_joint_answer_matched_pairs']}), flush=True)
                del model
                if device == 'cuda':
                    torch.cuda.empty_cache()


if __name__ == '__main__':
    run()
