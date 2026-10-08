"""Test ordinary-training geometry on exact answer-matched input differences."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from .algebra_structure_replication import collect_prior_inputs, load_plan
from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .native_confirmation import setup
from .native_source_subspace import contrast_basis
from .permworld_combinations import prompts, sha
from .permutation_audit import transform
from .representation_algebra import ACTION_NAMES, permutation_action_table, word_action


CONFIG = 'configs/answer_matched_geometry.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize():
    plan = json.loads(Path(CONFIG).read_text())
    previous_plan, config, previous_root = load_plan()
    root = Path(plan['output'])
    root.mkdir(parents=True, exist_ok=True)
    for folder in ['features', 'evaluations', 'arrays']:
        (root / folder).mkdir(exist_ok=True)
    inputs = [*previous_plan['excluded_datasets'], str(previous_root / 'probe_dataset.npz'), 'results/algebra_hidden_relations/probe_dataset.npz']
    signature = {'config_sha256': sha(CONFIG), 'code_sha256': sha(__file__),
        'parent_protocol_sha256': sha(previous_root / 'protocol.json'), 'plan': plan,
        'excluded_dataset_sha256': {path: sha(path) for path in inputs},
        'partial_training_orbit_sha256': sha('results/algebra_hidden_relations/training_orbit_audit.npz'),
        'frozen_map_sha256': {f'{seed}_{status}_{view}': sha(previous_root / 'arrays' / f'novel_records_s{seed}_{status}_{view}.npz')
            for seed in plan['source_seeds'] for status in plan['model_statuses'] for view in plan['views']},
        'trained_source_sha256': {str(seed): sha(previous_root / 'source/checkpoints' / f'novel_records_s{seed}.pt') for seed in plan['source_seeds']}}
    protocol = root / 'protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text())['signature'] == signature
    else:
        atomic_json(protocol, {'registered_utc': now(), 'signature': signature, 'new_conditional_geometry_evaluations': 0})
    return plan, previous_plan, config, root, inputs


def prepare(plan, config, root, excludes, functions, tokens, one_line):
    path = root / 'dataset.npz'
    if path.exists():
        return dict(np.load(path))
    seen = collect_prior_inputs(excludes)
    a = dict(np.load('results/algebra_hidden_relations/training_orbit_audit.npz'))
    seen.update(tuple(map(int, p[:n])) for orbit, n in zip(a['permutations'], a['lengths']) for p in orbit)
    initial_count = len(seen)
    rng = np.random.default_rng(plan['data_seed'])
    rows, raw, labels, lengths, pair_ids = [], [], [], [], []
    identity = 0
    width = 2 * plan['lengths'][1] + 4
    for n in range(plan['lengths'][0], plan['lengths'][1] + 1):
        pending = {}
        accepted = 0
        while accepted < plan['pairs_per_length']:
            base = tuple(map(int, rng.permutation(n) + 1))
            orbit = [transform(base, name) for name in ACTION_NAMES]
            if len(set(orbit)) != 8 or any(p in seen for p in orbit):
                continue
            key = tuple(functions[task](base) for task in plan['tasks'])
            old = pending.get(key)
            if old is not None and any(p in seen for p in old):
                pending.pop(key)
                old = None
            if old is None:
                pending[key] = orbit
                continue
            if set(old) & set(orbit):
                continue
            pending.pop(key)
            seen.update(old)
            seen.update(orbit)
            answer_tuples = []
            for member in [old, orbit]:
                encoded = []
                for p in member:
                    row = [tokens['<BOS>'], tokens['<SIZE>'], n] + [tokens[t] for t in one_line(p)]
                    encoded.append(row + [tokens['<PAD>']] * (width - len(row)))
                rows.append(encoded)
                raw.append([list(p) + [0] * (plan['lengths'][1] - n) for p in member])
                answer_tuples.append([[functions[task](p) for task in plan['tasks']] for p in member])
                lengths.append(n)
                pair_ids.append(identity)
            np.testing.assert_array_equal(answer_tuples[0], answer_tuples[1])
            labels.extend(answer_tuples)
            identity += 1
            accepted += 1
    data = {'input': np.asarray(rows, dtype=np.int64), 'permutations': np.asarray(raw, dtype=np.int64),
        'labels': np.asarray(labels, dtype=np.int64), 'lengths': np.asarray(lengths, dtype=np.int64), 'pair_ids': np.asarray(pair_ids, dtype=np.int64)}
    np.savez_compressed(path, **data)
    atomic_json(root / 'dataset_audit.json', {'dataset_sha256': sha(path), 'excluded_distinct_inputs': initial_count,
        'pairs': identity, 'states': len(rows) * 8, 'all_transformed_true_answer_tuples_equal_within_pairs': True,
        'conditional_test_only_no_fit_or_validation_data': True})
    return data


@torch.no_grad()
def extract_final(model, data, tasks, tokens, batch_size):
    model.eval()
    device = next(model.parameters()).device
    x = data['input'].reshape(-1, data['input'].shape[-1])
    ns = np.repeat(data['lengths'], 8)
    prefixes, queries = [], []
    for start in range(0, len(x), batch_size):
        ids, mask, position = prompts(torch.as_tensor(x[start:start + batch_size], device=device), torch.as_tensor(ns[start:start + batch_size], device=device), tasks, tokens)
        hidden, valid = model._embed_inputs(ids, mask)
        for block in model.blocks:
            hidden = block(hidden, valid)
        hidden = model.final_norm(hidden)
        batch = min(batch_size, len(x) - start)
        n = torch.as_tensor(ns[start:start + batch_size], device=device)
        prefixes.append(hidden[torch.arange(batch, device=device), 2 * n + 3].float().cpu().numpy())
        q = hidden[torch.arange(len(ids), device=device), position].reshape(len(tasks), batch, -1)
        queries.append(q.permute(1, 0, 2).reshape(batch, -1).float().cpu().numpy())
    return {'ONE_END': np.concatenate(prefixes).reshape(len(ns) // 8, 8, -1),
        'source_query_concat': np.concatenate(queries).reshape(len(ns) // 8, 8, -1)}


def pair_geometry(hidden, pair_ids, archive, words, subset=None):
    ids = np.unique(pair_ids)
    if subset is not None:
        ids = ids[subset]
    assert len(ids)
    indices = np.stack([np.flatnonzero(pair_ids == pair) for pair in ids])
    assert indices.shape[1] == 2
    differences = hidden[indices[:, 0]] - hidden[indices[:, 1]]
    differences = differences.astype(np.float64)
    q = archive['basis'].astype(np.float64)
    source = differences.reshape(-1, differences.shape[-1])
    latent = source @ q
    maps = {g: archive[f'map_{g}'].astype(np.float64) for g in ['c', 'r', 'i']}
    table, letters = permutation_action_table(), {'c': 1, 'r': 2, 'i': 4}
    output, arrays = [], {}
    for word in words:
        product = np.eye(q.shape[-1])
        for g in word:
            product = product @ maps[g]
        predicted = source + (latent @ product - latent) @ q.T
        action = word_action(word, table, letters)
        target = differences[:, table[:, action]].reshape(source.shape)
        error = float(np.square(predicted - target).sum())
        denom = float(np.square(target - source).sum())
        norm = float(np.square(target).sum())
        dot = np.sum(predicted * target, axis=-1)
        output.append({'word': word, 'pairs': len(ids), 'pair_action_displacement_nmse': error / denom,
            'pair_target_nmse': error / norm, 'identity_pair_target_nmse': denom / norm,
            'pair_assignment_accuracy': float(np.mean(np.where(dot > 1e-12, 1., np.where(dot < -1e-12, 0., .5)))),
            'zero_pair_prediction_target_nmse': 1., 'identity_pair_action_displacement_nmse': 1.,
            'pair_source_rms': float(np.sqrt(np.mean(source**2))), 'pair_target_rms': float(np.sqrt(np.mean(target**2)))})
        arrays[f'{word}_pair_prediction'] = predicted.astype(np.float32)
    arrays['pair_ids'] = ids
    return output, arrays


def run():
    plan, previous, config, root, excludes = initialize()
    torch.set_num_threads(4)
    _, functions, tokens, one_line = setup(config)
    data = prepare(plan, config, root, excludes, functions, tokens, one_line)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parent = Path(plan['parent'])
    for seed in plan['source_seeds']:
        for status in plan['model_statuses']:
            name = f'novel_records_s{seed}_{status}'
            dest = root / 'evaluations' / f'{name}.json'
            if dest.exists():
                continue
            if time.time() >= datetime.fromisoformat(plan['deadline_utc']).timestamp():
                return
            model = make_model(config, previous['architecture'], seed, device)
            if status == 'trained':
                state = torch.load(parent / 'source/checkpoints' / f'novel_records_s{seed}.pt', weights_only=True, map_location=device)
                model.load_state_dict(state['model'])
            fp = root / 'features' / f'{name}.npz'
            if fp.exists():
                features = dict(np.load(fp))
            else:
                features = extract_final(model, data, plan['tasks'], tokens, plan['batch_size'])
                np.savez_compressed(fp, **features)
            weight = model.lm_head.weight[:31].detach().cpu().numpy().astype(np.float64)
            b = model.lm_head.bias
            bias = np.zeros(31) if b is None else b[:31].detach().cpu().numpy().astype(np.float64)
            block = features['source_query_concat'].reshape(len(data['lengths']), 8, 4, -1).astype(np.float64)
            predictions = (block @ weight.T + bias).argmax(-1)
            pair_rows = np.stack([np.flatnonzero(data['pair_ids'] == pair) for pair in np.unique(data['pair_ids'])])
            agreement = (predictions[pair_rows[:, 0]] == predictions[pair_rows[:, 1]]).all(axis=(1, 2))
            all_correct = (predictions[pair_rows] == data['labels'][pair_rows]).all(axis=(1, 2, 3))
            basis, _ = contrast_basis(weight)
            numeric = (block @ basis) @ basis.T
            null = (block - numeric).reshape(features['source_query_concat'].shape)
            views = {'source_query_concat': features['source_query_concat'], 'source_query_concat_full_width': features['source_query_concat'],
                'source_query_numeric_null': null, 'ONE_END': features['ONE_END'], 'ONE_END_full_width': features['ONE_END']}
            results = []
            archive_hashes = {}
            for view, hidden in views.items():
                maps = dict(np.load(parent / 'arrays' / f'{name}_{view}.npz'))
                for subset_name, subset in [('all_answer_matched_pairs', None), ('all_orbit_model_answers_agree', agreement), ('all_orbit_model_answers_correct', all_correct)]:
                    if subset is not None and subset.sum() == 0:
                        continue
                    rows, arrays = pair_geometry(hidden, data['pair_ids'], maps, plan['words'], subset)
                    for row in rows:
                        row.update(view=view, subset=subset_name)
                    results.extend(rows)
                    output = root / 'arrays' / f'{name}_{view}_{subset_name}.npz'
                    np.savez_compressed(output, **arrays)
                    archive_hashes[output.name] = sha(output)
            atomic_json(dest, {'source': name, 'seed': seed, 'status': status, 'feature_sha256': sha(fp),
                'same_model_answer_pairs': int(agreement.sum()), 'all_orbit_correct_pairs': int(all_correct.sum()),
                'model_query_accuracy': float(np.mean(predictions == data['labels'])), 'results': results,
                'array_sha256': archive_hashes, 'completed_utc': now()})
            print(json.dumps({'answer_matched_geometry': name, 'primary': [r for r in results if r['view'] == plan['primary_view'] and r['subset'] == 'all_answer_matched_pairs']}), flush=True)
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()


if __name__ == '__main__':
    run()
