"""Fresh input worlds with deadline-bounded source training and fixed probes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from .algebra_structure_replication import collect_prior_inputs
from .frozen_relation_followup import extract_known, fit_maps
from .hidden_relation_train import collision_pair, load_plan, train_one
from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .native_confirmation import setup
from .permworld_combinations import sha
from .permutation_audit import transform
from .relation_observed_start import observed_start_score
from .relation_seed_extension import evaluate_one
from .representation_algebra import ACTION_NAMES


CONFIG = 'configs/relation_world_confirmation.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def orbit_set(archive):
    return {tuple(map(int, p[:n])) for orbit, n in zip(archive['permutations'], archive['lengths']) for p in orbit}


def create_world(world, parent_plan, config, seen, root, tokens, one_line, function):
    rng = np.random.default_rng(world['data_seed'])
    width = 2 * parent_plan['lengths'][1] + 4
    def encode(orbit, n):
        encoded = []
        for p in orbit:
            row = [tokens['<BOS>'], tokens['<SIZE>'], n] + [tokens[t] for t in one_line(p)]
            encoded.append(row + [tokens['<PAD>']] * (width - len(row)))
        return encoded
    def fresh(n):
        while True:
            base = tuple(map(int, rng.permutation(n) + 1))
            orbit = [transform(base, action) for action in ACTION_NAMES]
            if len(set(orbit)) == 8 and not any(p in seen for p in orbit):
                seen.update(orbit)
                return orbit
    tx, ty, tn, traw = [], [], [], []
    px, py, pn, praw, splits, pairs = [], [], [], [], [], []
    for n in range(parent_plan['lengths'][0], parent_plan['lengths'][1] + 1):
        for _ in range(parent_plan['training_orbits_per_length']):
            orbit = fresh(n)
            encoded = encode(orbit, n)
            tx.append([encoded[j] for j in [0, 1, 4]])
            ty.append([function(orbit[j]) for j in [0, 1, 4]])
            tn.append(n)
            traw.append([list(p) + [0] * (parent_plan['lengths'][1] - n) for p in orbit])
    for split, count in enumerate(parent_plan['probe_orbits_per_length'].values()):
        for n in range(parent_plan['lengths'][0], parent_plan['lengths'][1] + 1):
            for _ in range(count):
                orbit = fresh(n)
                px.append(encode(orbit, n)); py.append([function(p) for p in orbit]); pn.append(n)
                praw.append([list(p) + [0] * (parent_plan['lengths'][1] - n) for p in orbit]); splits.append(split); pairs.append(-1)
    pair_id = 0
    for n in range(parent_plan['lengths'][0], parent_plan['lengths'][1] + 1):
        pending = {}
        for _ in range(parent_plan['collision_pairs_per_length']):
            for orbit in collision_pair(rng, n, seen, pending, function):
                px.append(encode(orbit, n)); py.append([function(p) for p in orbit]); pn.append(n)
                praw.append([list(p) + [0] * (parent_plan['lengths'][1] - n) for p in orbit]); splits.append(3); pairs.append(pair_id)
            pair_id += 1
    probe = {'input': np.asarray(px, dtype=np.int64), 'labels': np.asarray(py, dtype=np.int64), 'lengths': np.asarray(pn, dtype=np.int64),
        'permutations': np.asarray(praw, dtype=np.int64), 'split': np.asarray(splits, dtype=np.int64), 'pair_ids': np.asarray(pairs, dtype=np.int64)}
    val = probe['split'] == 1
    source = {'train_input': np.asarray(tx, dtype=np.int64), 'train_labels': np.asarray(ty, dtype=np.int64), 'train_lengths': np.asarray(tn, dtype=np.int64),
        'validation_input': probe['input'][val][:, [0, 1, 4]], 'validation_labels': probe['labels'][val][:, [0, 1, 4]], 'validation_lengths': probe['lengths'][val]}
    np.savez_compressed(root / 'source_data.npz', **source)
    np.savez_compressed(root / 'training_orbit_audit.npz', permutations=np.asarray(traw, dtype=np.int64), lengths=source['train_lengths'])
    np.savez_compressed(root / 'probe_dataset.npz', **probe)
    atomic_json(root / 'data_hashes.json', {name: sha(root / name) for name in ['source_data.npz', 'training_orbit_audit.npz', 'probe_dataset.npz']})


def initialize():
    campaign = json.loads(Path(CONFIG).read_text())
    parent, config, _ = load_plan()
    root = Path(campaign['output'])
    root.mkdir(parents=True, exist_ok=True)
    original = Path(campaign['parent'])
    old_audit = json.loads((original / 'dataset_audit.json').read_text())
    excludes = [*old_audit['excluded_datasets'], str(original / 'probe_dataset.npz'), 'results/answer_matched_geometry/dataset.npz']
    signature = {'config_sha256': sha(CONFIG), 'code_sha256': sha(__file__),
        'parent_protocol_sha256': sha(original / 'protocol.json'),
        'original_train_code_sha256': sha('experiments/hidden_relation_train.py'), 'campaign': campaign,
        'excluded_data_sha256': {path: sha(path) for path in excludes},
        'parent_training_orbit_sha256': sha(original / 'training_orbit_audit.npz')}
    protocol = root / 'protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text())['signature'] == signature
    else:
        atomic_json(protocol, {'registered_utc': now(), 'signature': signature, 'new_world_models': 0})
    _, functions, tokens, one_line = setup(config)
    seen = collect_prior_inputs(excludes)
    seen.update(orbit_set(dict(np.load(original / 'training_orbit_audit.npz'))))
    for world in campaign['worlds']:
        output = root / world['id']
        output.mkdir(exist_ok=True)
        for subdir in ['source', 'checkpoints', 'features', 'arrays', 'evaluations', 'observed_start']:
            (output / subdir).mkdir(exist_ok=True)
        if not (output / 'source_data.npz').exists():
            create_world(world, parent, config, seen, output, tokens, one_line, functions[parent['source_task']])
        else:
            for name, expected in json.loads((output / 'data_hashes.json').read_text()).items():
                assert sha(output / name) == expected
            seen.update(orbit_set(dict(np.load(output / 'training_orbit_audit.npz'))))
            seen.update(orbit_set(dict(np.load(output / 'probe_dataset.npz'))))
        plan = {**parent, 'output': str(output), 'source_seeds': [world['source_seed']], 'conditions': campaign['conditions'], 'data_seed': world['data_seed']}
        atomic_json(output / 'protocol.json', {'campaign_protocol_sha256': sha(protocol), 'world': world, 'plan': plan,
            'source_data_sha256': sha(output / 'source_data.npz')}) if not (output / 'protocol.json').exists() else None
    return campaign, parent, config, root, tokens


def observed_world(campaign, parent, config, output, world, condition, tokens, device):
    seed = world['source_seed']
    name = f'{condition}_s{seed}'
    dest = output / 'observed_start' / f'{name}.json'
    if dest.exists():
        return
    data = dict(np.load(output / 'probe_dataset.npz'))
    source = dict(np.load(output / 'source_data.npz'))
    hidden = np.load(output / 'features' / f'{name}.npz')['source_query_concat'][:, :, -1].astype(np.float64)
    state = torch.load(output / 'checkpoints' / f'{name}.pt', weights_only=True, map_location=device)
    model = make_model(config, parent['architecture'], seed, device)
    model.load_state_dict(state['model'])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    known = {split: extract_known(model, source[f'{split}_input'], source[f'{split}_lengths'], parent['source_task'], tokens, 64) for split in ['train', 'validation']}
    variants = [('native_operators', dict(np.load(output / 'arrays' / f'{name}_native_operators.npz')))]
    for shuffled in [False, True]:
        maps, means, alphas, validation, ids = fit_maps(known['train'], source['train_lengths'], known['validation'], source['validation_lengths'], 256, parent['ridge_grid'], seed + 2026100671, shuffled)
        archive = {f'rho_{g}': pair[0] for g, pair in maps.items()}
        archive.update({f'bias_{g}': pair[1] for g, pair in maps.items()})
        archive.update(mean_lengths=np.array(list(means)), mean_vectors=np.array(list(means.values())),
            readout_weight=variants[0][1]['readout_weight'], readout_bias=variants[0][1]['readout_bias'], source_anchor_ids=ids)
        label = 'frozen_full_shuffled' if shuffled else 'frozen_full_correct'
        ap = output / 'arrays' / f'{name}_{label}.npz'
        np.savez_compressed(ap, **archive)
        variants.append((label, archive))
    results = []
    for method, archive in variants:
        rows, arrays = observed_start_score(hidden, data, archive)
        ap = output / 'observed_start' / f'{name}_{method}.npz'
        np.savez_compressed(ap, **arrays)
        results.append({'method': method, 'metrics': rows, 'prediction_archive_sha256': sha(ap)})
    atomic_json(dest, {'world': world, 'source': name, 'condition': condition, 'methods': results, 'completed_utc': now()})
    print(json.dumps({'new_world_observed_start': name, 'world': world['id'], 'metrics': [r for r in results[0]['metrics'] if r['split'] == 'answer_collisions']}), flush=True)
    del model, state
    if device == 'cuda':
        torch.cuda.empty_cache()


def run():
    campaign, parent, config, root, tokens = initialize()
    deadline = datetime.fromisoformat(campaign['deadline_utc']).timestamp()
    waiting = Path(campaign['wait_for_extension']) / 'state.json'
    while time.time() < deadline and json.loads(waiting.read_text())['status'] not in ['complete', 'stopped_at_deadline']:
        atomic_json(root / 'state.json', {'status': 'waiting_for_source_seed_extension', 'updated_utc': now()})
        time.sleep(min(45, deadline - time.time()))
    if time.time() >= deadline:
        atomic_json(root / 'state.json', {'status': 'stopped_at_deadline_before_training', 'updated_utc': now()})
        return
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def timeout(*_):
        raise TimeoutError('10:00 deadline reached')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
    try:
        for world in campaign['worlds']:
            output = root / world['id']
            raw = dict(np.load(output / 'source_data.npz'))
            data = {key: torch.as_tensor(value, device=device) for key, value in raw.items()}
            plan = {**parent, 'output': str(output), 'source_seeds': [world['source_seed']], 'conditions': campaign['conditions'], 'data_seed': world['data_seed']}
            for condition in campaign['conditions']:
                name = f'{condition}_s{world["source_seed"]}'
                atomic_json(root / 'state.json', {'status': 'source_training', 'world': world['id'], 'source': name, 'updated_utc': now()})
                train_one(plan, config, output, tokens, data, raw, world['source_seed'], condition, device)
                evaluate_one({'parent': str(output)}, plan, config, output, name, world['source_seed'], condition, tokens, device)
                observed_world(campaign, parent, config, output, world, condition, tokens, device)
        atomic_json(root / 'state.json', {'status': 'complete', 'worlds': 3, 'source_models': 9, 'updated_utc': now()})
    except TimeoutError:
        atomic_json(root / 'state.json', {'status': 'stopped_at_deadline', 'updated_utc': now()})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    run()
