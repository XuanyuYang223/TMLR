"""Prospective 2x2 generator correctness assay with frozen source/readout.

Only the correspondence in geometric supervision changes. Visible output
supervision is always correct and identical. Predictors receive the base
vector and learned operators, never transformed test inputs or answers.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from itertools import product
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .hidden_relation_train import AffineOperators, load_plan as parent_plan, query_hidden
from .longrun_engine import atomic_json, atomic_torch
from .longrun_transfer import make_model
from .native_confirmation import setup
from .permworld_combinations import sha
from .permutation_audit import transform
from .representation_algebra import ACTION_NAMES, permutation_action_table, word_action
from .specialist_cka_controls import input_key, old_inputs

CONFIG = Path('configs/two_step_relation_factorial.json')
FLAGS = {'both_correct': (True, True), 'c_correct_i_wrong': (True, False),
         'c_wrong_i_correct': (False, True), 'both_wrong': (False, False)}


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize():
    plan = json.loads(CONFIG.read_text())
    parent, config, _ = parent_plan()
    root = Path(plan['output'])
    for name in ['', 'dataset', 'features', 'fits', 'maps', 'evaluations']:
        (root / name).mkdir(parents=True, exist_ok=True)
    protocol = root / 'protocol.json'
    dependencies = ['experiments/hidden_relation_train.py', 'experiments/longrun_transfer.py',
                    'experiments/longrun_attention.py', 'experiments/permworld_combinations.py',
                    'experiments/permutation_audit.py', 'experiments/representation_algebra.py',
                    'experiments/specialist_cka_controls.py']
    if protocol.exists():
        sig = json.loads(protocol.read_text())['signature']
        effective_sha = sig['code_sha256']
        amendments = root / 'implementation_amendments.json'
        if amendments.exists():
            for amendment in json.loads(amendments.read_text()):
                assert amendment['old_code_sha256'] == effective_sha
                effective_sha = amendment['new_code_sha256']
                assert sha(root / amendment['source_snapshot']) == effective_sha
        assert sig['plan'] == plan and effective_sha == sha(__file__) and sig['config_sha256'] == sha(CONFIG)
        for file, digest in sig['dependencies_sha256'].items():
            assert sha(file) == digest, file
        for source in sig['sources']:
            assert sha(source['checkpoint']) == source['sha256']
        return plan, parent, config, root, sig
    sources = [{'seed': seed, 'checkpoint': str(Path(plan['parent']) / 'checkpoints' /
                 f"{plan['source_condition']}_s{seed}.pt")} for seed in plan['source_seeds']]
    for source in sources:
        source['sha256'] = sha(source['checkpoint'])
    names = {'data.npz', 'probe_dataset.npz', 'source_data.npz', 'training_orbit_audit.npz', 'dataset.npz'}
    archives = []
    digests = set()
    for file in sorted(Path('results').rglob('*.npz')):
        if file.name not in names or root in file.parents:
            continue
        digest = sha(file)
        if digest not in digests:
            archives.append({'path': str(file), 'sha256': digest})
            digests.add(digest)
    sig = {'plan': plan, 'code_sha256': sha(__file__), 'config_sha256': sha(CONFIG),
           'sources': sources, 'excluded_archives': archives,
           'dependencies_sha256': {file: sha(file) for file in dependencies},
           'parent_protocol_sha256': sha(Path(plan['parent']) / 'protocol.json'),
           'limits': 'Source models were trained from scratch on the local e/C/I source dataset. Local archive orbits are excluded. This study does not claim exclusion against the unrelated original16M upstream corpus, on which these backbones were not pretrained.'}
    atomic_json(protocol, {'registered_utc': now(), 'new_outcomes_observed': False, 'signature': sig})
    (root / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return plan, parent, config, root, sig


def factorial_pairings(lengths, labels, seed, minimum=3):
    """Preserve all three visible answers and exclude cancelling pair maps."""
    keys = np.column_stack([lengths, labels])
    _, groups, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    eligible = counts[groups] >= minimum
    pc = np.arange(len(keys))
    pi = pc.copy()
    rng = np.random.default_rng(seed)
    for group in np.unique(groups[eligible]):
        ids = np.flatnonzero(groups == group)
        assert len(ids) >= 3
        for _ in range(10000):
            a, b = rng.permutation(len(ids)), rng.permutation(len(ids))
            identity = np.arange(len(ids))
            if (np.all(a != identity) and np.all(b != identity)
                    and np.all(a[b] != identity) and np.all(b[a] != identity)):
                pc[ids], pi[ids] = ids[a], ids[b]
                break
        else:
            raise RuntimeError('Could not find noncancelling paired derangements')
    for partner in (pc, pi):
        assert np.array_equal(np.sort(partner[eligible]), np.flatnonzero(eligible))
        assert np.all(partner[eligible] != np.flatnonzero(eligible))
        assert np.array_equal(keys[partner[eligible]], keys[eligible])
    assert np.all(pc[pi[eligible]] != np.flatnonzero(eligible))
    assert np.all(pi[pc[eligible]] != np.flatnonzero(eligible))
    return pc, pi, eligible


def epoch_schedule(lengths, eligible, epochs, batch_size, seed):
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        for n in rng.permutation(np.unique(lengths)):
            ids = rng.permutation(np.flatnonzero(eligible & (lengths == n)))
            assert len(ids) > 0
            for start in range(0, len(ids), batch_size):
                yield epoch, ids[start:start + batch_size]


def fresh_orbit(rng, n, seen):
    while True:
        base = tuple(map(int, rng.permutation(n) + 1))
        orbit = [transform(base, name) for name in ACTION_NAMES]
        keys = [input_key(p) for p in orbit]
        if len(set(keys)) == 8 and not any(key in seen for key in keys):
            seen.update(keys)
            return orbit


def collision_orbits(rng, n, seen, function):
    pending = {}
    while True:
        orbit = fresh_orbit(rng, n, seen)
        key = tuple(function(orbit[j]) for j in (0, 1, 4))
        old = pending.get(key)
        if old is not None and function(old[5]) != function(orbit[5]):
            return old, orbit
        pending[key] = orbit


def encode(orbits, lengths, actions, function, tokens, one_line):
    rows, raw, labels = [], [], []
    for orbit, n in zip(orbits, lengths):
        encoded = []
        for j in actions:
            prefix = [tokens['<BOS>'], tokens['<SIZE>'], int(n)] + [tokens[v] for v in one_line(orbit[j])]
            encoded.append(prefix + [tokens['<PAD>']] * (64 - len(prefix)))
        rows.append(encoded)
        raw.append([list(p) + [0] * (30 - int(n)) for p in orbit])
        labels.append([function(orbit[j]) for j in actions])
    return {'input': np.asarray(rows, dtype=np.int64), 'permutations': np.asarray(raw, dtype=np.int64),
            'lengths': np.asarray(lengths), 'labels': np.asarray(labels, dtype=np.int64)}


def prepare_data(plan, parent, config, root, sig):
    if (root / 'data_verification.json').exists():
        return
    if (root / 'data_manifest.json').exists():
        verify_data(plan, parent, config, root, sig)
        return
    _, functions, tokens, one_line = setup(config)
    function = functions[parent['source_task']]
    seen = old_inputs(sig['excluded_archives'])
    prior = len(seen)
    cohorts = []
    for index, seed in enumerate(plan['support_seeds']):
        rng = np.random.default_rng(seed)
        orbits, lengths, splits = [], [], []
        for n in plan['lengths']:
            for split, count in [(0, plan['fit_per_length']), (1, plan['validation_per_length'])]:
                for _ in range(count):
                    orbits.append(fresh_orbit(rng, n, seen)); lengths.append(n); splits.append(split)
        data = encode(orbits, lengths, (0, 1, 4), function, tokens, one_line)
        data['split'] = np.asarray(splits)
        file = root / 'dataset' / f'n{index}' / 'support' / 'dataset.npz'
        file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(file, **data)
        fit = data['split'] == 0
        pc, pi, eligible = factorial_pairings(data['lengths'][fit], data['labels'][fit], seed + 6001,
                                             plan['minimum_pairing_stratum'])
        np.savez_compressed(file.parent / 'pairings.npz', pc=pc, pi=pi, eligible=eligible)
        cohorts.append({'id': f'n{index}', 'path': str(file), 'sha256': sha(file),
                        'pairings_sha256': sha(file.parent / 'pairings.npz'),
                        'fit_anchors': int(fit.sum()), 'eligible_anchors': int(eligible.sum()),
                        'eligible_by_length': {str(n): int((eligible & (data['lengths'][fit] == n)).sum()) for n in plan['lengths']}})
        print(json.dumps(cohorts[-1]), flush=True)
    rng = np.random.default_rng(plan['test_seed'])
    orbits, lengths, splits, pair_ids = [], [], [], []
    pair = 0
    for n in plan['lengths']:
        for _ in range(plan['iid_per_length']):
            orbits.append(fresh_orbit(rng, n, seen)); lengths.append(n); splits.append(0); pair_ids.append(-1)
        for _ in range(plan['collision_pairs_per_length']):
            for orbit in collision_orbits(rng, n, seen, function):
                orbits.append(orbit); lengths.append(n); splits.append(1); pair_ids.append(pair)
            pair += 1
    data = encode(orbits, lengths, tuple(range(8)), function, tokens, one_line)
    data.update(split=np.asarray(splits), pair_ids=np.asarray(pair_ids))
    file = root / 'dataset' / 'test' / 'dataset.npz'
    file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(file, **data)
    cohorts.append({'id': 'test', 'path': str(file), 'sha256': sha(file)})
    atomic_json(root / 'data_manifest.json', {'created_utc': now(), 'cohorts': cohorts,
                'prior_local_distinct_inputs': prior, 'collision_pairs': pair})
    verify_data(plan, parent, config, root, sig)


def independent_orbit(p):
    def inv(q):
        out = [0] * len(q)
        for index, value in enumerate(q, 1):
            out[value - 1] = index
        return tuple(out)
    p = tuple(map(int, p)); c = tuple(len(p) + 1 - v for v in p); r = p[::-1]
    rc = c[::-1]
    return [p, c, r, rc, inv(p), inv(c), inv(r), inv(rc)]


def verify_data(plan, parent, config, root, sig):
    _, functions, _, _ = setup(config)
    f = functions[parent['source_task']]
    seen = old_inputs(sig['excluded_archives'])
    manifest = json.loads((root / 'data_manifest.json').read_text())
    examples = 0
    for cohort in manifest['cohorts']:
        file = Path(cohort['path']); assert sha(file) == cohort['sha256']
        a = dict(np.load(file))
        actions = (0, 1, 4) if cohort['id'] != 'test' else tuple(range(8))
        for raw, n, rows, labels in zip(a['permutations'], a['lengths'], a['input'], a['labels']):
            orbit = independent_orbit(raw[0, :n])
            assert len(set(orbit)) == 8
            for saved, p in zip(raw, orbit):
                assert tuple(map(int, saved[:n])) == p
                key = input_key(p); assert key not in seen; seen.add(key)
            for row, action, answer in zip(rows, actions, labels):
                assert tuple(map(int, row[4:4 + 2 * n:2])) == orbit[action]
                assert int(answer) == f(orbit[action])
            # ic is an output-degenerate secondary test for LR-max.
            assert f(orbit[6]) == f(orbit[1])
            examples += 1
        if cohort['id'] != 'test':
            fit = a['split'] == 0; pairings = dict(np.load(file.parent / 'pairings.npz'))
            labels, ns = a['labels'][fit], a['lengths'][fit]
            keep = pairings['eligible']; ids = np.flatnonzero(keep)
            for name in ('pc', 'pi'):
                partner = pairings[name]
                assert Counter(map(int, partner[keep])) == Counter(map(int, ids))
                assert np.all(partner[keep] != ids)
                assert np.array_equal(labels[partner[keep]], labels[keep])
                assert np.array_equal(ns[partner[keep]], ns[keep])
            assert np.all(pairings['pc'][pairings['pi'][keep]] != ids)
            assert np.all(pairings['pi'][pairings['pc'][keep]] != ids)
        else:
            for pair in np.unique(a['pair_ids'][a['split'] == 1]):
                ids = np.flatnonzero(a['pair_ids'] == pair); assert len(ids) == 2
                assert a['lengths'][ids[0]] == a['lengths'][ids[1]]
                assert np.array_equal(a['labels'][ids[0], [0, 1, 4]], a['labels'][ids[1], [0, 1, 4]])
                assert a['labels'][ids[0], 5] != a['labels'][ids[1], 5]
    atomic_json(root / 'data_verification.json', {'status': 'complete', 'completed_utc': now(),
                'fresh_examples_checked': examples, 'full_orbit_states_checked': examples * 8,
                'collision_pairs_checked': manifest['collision_pairs'],
                'answer_only_ci_accuracy_ceiling': 0.5, 'answer_only_pair_both_correct_ceiling': 0.0,
                'pairing_compositions_have_no_fixed_points': True})


def source_model(source, parent, config, device, accelerated=True):
    if accelerated:
        model = make_model(config, parent['architecture'], source['seed'], device)
    else:
        from .permworld_combinations import new_model
        model = new_model({**config, **parent['architecture']}, source['seed'], device)
    state = torch.load(source['checkpoint'], map_location='cpu', weights_only=True)
    model.load_state_dict(state['model']); del state
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def hidden_features(model, data, task, tokens):
    x = data['input']; n = data['lengths']; count = x.shape[1]
    flat = x.reshape(-1, x.shape[-1]); lengths = np.repeat(n, count)
    device = next(model.parameters()).device; features = []
    for start in range(0, len(flat), 128):
        h = query_hidden(model, torch.as_tensor(flat[start:start + 128], device=device),
                         torch.as_tensor(lengths[start:start + 128], device=device), task, tokens)
        features.append(h.float().cpu().numpy())
    return np.concatenate(features).reshape(len(n), count, -1)


def cache_supports(plan, parent, config, root, sig):
    _, _, tokens, _ = setup(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for index, _ in enumerate(plan['support_seeds']):
        source = sig['sources'][index % 3]; dest = root / 'features' / f'n{index}_support.npz'
        if dest.exists():
            continue
        data = dict(np.load(root / 'dataset' / f'n{index}' / 'support' / 'dataset.npz'))
        assert data['input'].shape[1] == 3
        model = source_model(source, parent, config, device)
        h = hidden_features(model, data, parent['source_task'], tokens)
        weights = model.lm_head.weight[:31].detach().cpu().numpy()
        bias = np.zeros(31, dtype=np.float32) if model.lm_head.bias is None else model.lm_head.bias[:31].detach().cpu().numpy()
        logits = h @ weights.T + bias
        fit = data['split'] == 0
        np.savez_compressed(dest, train_hidden=h[fit], validation_hidden=h[~fit],
                            train_output=logits[fit], validation_output=logits[~fit],
                            readout_weight=weights, readout_bias=bias)
        atomic_json(dest.with_suffix('.json'), {'source_seed': source['seed'], 'feature_sha256': sha(dest),
                    'source_checkpoint_sha256': source['sha256'], 'completed_utc': now(),
                    'visible_validation_accuracy_e_C_I': (logits[~fit].argmax(-1) == data['labels'][~fit]).mean(0).tolist()})
        del model
        print(json.dumps({'support_features': f'n{index}', 'visible_accuracy': json.loads(dest.with_suffix('.json').read_text())['visible_validation_accuracy_e_C_I']}), flush=True)


def apply_numpy(x, word, maps):
    """Pure prediction API: base vectors and learned parameters only."""
    out = np.asarray(x, dtype=np.float64)
    for letter in word:
        out = out @ maps[f'rho_{letter}'] + maps[f'bias_{letter}']
    return out


def known_grade(ops, h, centers, weights, bias, labels):
    with torch.no_grad():
        out = []
        for index, action in [(0, 1), (1, 2)]:
            prediction = ops(h[:, 0], index) + centers
            out.append({'action': 'c' if index == 0 else 'i',
                        'accuracy': float((((prediction @ weights.T + bias).argmax(-1)) == labels[:, action]).float().mean().cpu()),
                        'mse': float((prediction - (h[:, action] + centers)).square().mean().cpu())})
    return out


def supervision_losses(ops, h, ids, pc, pi, centers, teacher, weights, bias, temperature, scale):
    """The KD path sees original visible pairs under every condition."""
    base = h[ids, 0]; center = centers[ids]
    geometry, kd = [], []
    for j, action, partner in [(0, 1, pc), (1, 2, pi)]:
        forward = ops(base, j)
        backward = ops(h[ids, action], j)
        reverse_geometric = ops(h[partner, action], j)
        geometry.extend([(forward - h[partner, action]).square().mean(),
                         (reverse_geometric - base).square().mean()])
        for prediction, target in [(forward + center, teacher[ids, action]),
                                   (backward + center, teacher[ids, 0])]:
            logits = prediction @ weights.T + bias
            kd.append(F.kl_div(F.log_softmax(logits / temperature, -1),
                               F.softmax(target / temperature, -1), reduction='batchmean') * temperature ** 2)
    return torch.stack(geometry).mean() / max(scale, 1e-6), torch.stack(kd).mean()


def train_fit(plan, root, index, view, condition, device):
    name = f'n{index}_{view}_{condition}'; record = root / 'fits' / f'{name}.json'
    if record.exists() and json.loads(record.read_text())['status'] == 'complete':
        return
    data = dict(np.load(root / 'dataset' / f'n{index}' / 'support' / 'dataset.npz'))
    cached = dict(np.load(root / 'features' / f'n{index}_support.npz'))
    pairings = dict(np.load(root / 'dataset' / f'n{index}' / 'support' / 'pairings.npz'))
    fit = data['split'] == 0; ns = data['lengths'][fit]; eligible = pairings['eligible']
    h = cached[f'train_{view}'].astype(np.float32); hv = cached[f'validation_{view}'].astype(np.float32)
    center_by_length = {int(n): h[(ns == n) & eligible].reshape(-1, h.shape[-1]).mean(0) for n in plan['lengths']}
    centers = np.asarray([center_by_length[int(n)] for n in ns])
    val_centers = np.asarray([center_by_length[int(n)] for n in data['lengths'][~fit]])
    scale = float(np.var((h - centers[:, None])[eligible], axis=(0, 1)).mean())
    h = torch.as_tensor(h - centers[:, None], device=device)
    hv = torch.as_tensor(hv - val_centers[:, None], device=device)
    centers = torch.as_tensor(centers, device=device); val_centers = torch.as_tensor(val_centers, device=device)
    if view == 'hidden':
        weights, bias = cached['readout_weight'], cached['readout_bias']
    else:
        weights, bias = np.eye(31, dtype=np.float32), np.zeros(31, dtype=np.float32)
    weights, bias = torch.as_tensor(weights, device=device), torch.as_tensor(bias, device=device)
    teacher = torch.as_tensor(cached['train_output'], device=device)
    val_labels = torch.as_tensor(data['labels'][~fit], device=device)
    ops = AffineOperators(h.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(ops.parameters(), lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
    schedule = list(epoch_schedule(ns, eligible, plan['epochs'], plan['batch_size'], plan['support_seeds'][index] + 8001))
    digest = sha256(); exposure = np.zeros(len(ns), dtype=np.int64)
    correct_c, correct_i = FLAGS[condition]
    pc = np.arange(len(ns)) if correct_c else pairings['pc']
    pi = np.arange(len(ns)) if correct_i else pairings['pi']
    cp = root / 'maps' / f'{name}_resume.pt'; start_epoch = 0; curve = []
    if cp.exists():
        state = torch.load(cp, map_location=device, weights_only=True)
        ops.load_state_dict(state['operators']); optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']; curve = state['curve']
    temperature = plan['distillation_temperature']; last_step = 0
    for step, (epoch, ids) in enumerate(schedule, 1):
        digest.update(np.asarray([epoch], dtype=np.int64).tobytes() + ids.tobytes()); exposure[ids] += 1
        last_step = step
        if epoch <= start_epoch:
            continue
        if time.time() >= datetime.fromisoformat(plan['deadline_utc']).timestamp():
            raise TimeoutError('Registered factorial study reached10AM; final test remains closed if fits incomplete')
        ids_t = torch.as_tensor(ids, device=device); pc_t = torch.as_tensor(pc[ids], device=device); pi_t = torch.as_tensor(pi[ids], device=device)
        optimizer.zero_grad(set_to_none=True)
        geometry_loss, output_loss = supervision_losses(ops, h, ids_t, pc_t, pi_t, centers,
                                                       teacher, weights, bias, temperature, scale)
        loss = output_loss + plan['geometry_weight'] * geometry_loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(ops.parameters(), 1.)
        ratio = min(1., step / plan['warmup_updates'])
        cosine = (1 + math.cos(math.pi * step / len(schedule))) / 2
        lr = plan['learning_rate'] * ratio * (plan['minimum_learning_rate_ratio'] + (1 - plan['minimum_learning_rate_ratio']) * cosine)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        boundary = step == len(schedule) or schedule[step][0] != epoch
        if boundary and (epoch % 25 == 0 or epoch == plan['epochs']):
            grade = known_grade(ops, hv, val_centers, weights, bias, val_labels)
            curve.append({'epoch': epoch, 'step': step, 'visible_generators': grade,
                          'geometry_loss': float(geometry_loss.detach().cpu()), 'output_loss': float(output_loss.detach().cpu())})
            atomic_torch(cp, {'operators': ops.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'curve': curve})
            if epoch in plan['checkpoint_epochs']:
                atomic_torch(root / 'maps' / f'{name}_e{epoch}.pt', {'operators': ops.state_dict(), 'epoch': epoch})
            atomic_json(root / 'current_job.json', {'fit': name, 'epoch': epoch, 'epochs': plan['epochs'], 'updated_utc': now()})
    assert np.all(exposure[eligible] == plan['epochs']) and not exposure[~eligible].any()
    for partner in (pc, pi):
        assert np.array_equal(exposure[partner[eligible]], exposure[eligible])
    offsets = ops.offset.detach().cpu().numpy().astype(np.float64)
    biases = ops.bias.detach().cpu().numpy().astype(np.float64)
    dest = root / 'maps' / f'{name}.npz'
    np.savez_compressed(dest, rho_c=np.eye(h.shape[-1]) + offsets[0], rho_i=np.eye(h.shape[-1]) + offsets[1],
                        bias_c=biases[0], bias_i=biases[1], mean_lengths=np.asarray(plan['lengths']),
                        mean_vectors=np.asarray([center_by_length[n] for n in plan['lengths']]),
                        readout_weight=weights.cpu().numpy(), readout_bias=bias.cpu().numpy())
    atomic_json(record, {'status': 'complete', 'replicate': f'n{index}', 'source_seed': plan['source_seeds'][index % 3],
                'view': view, 'condition': condition, 'completed_utc': now(), 'epochs': plan['epochs'],
                'updates': last_step, 'schedule_sha256': digest.hexdigest(), 'map_sha256': sha(dest),
                'support_features_sha256': sha(root / 'features' / f'n{index}_support.npz'),
                'eligible_anchors': int(eligible.sum()), 'anchor_exposures': int(exposure.sum()),
                'exact_same_marginal_exposures': True, 'curve': curve})
    print(json.dumps({'completed_operator_fit': name, 'updates': last_step, 'visible_generators': curve[-1]['visible_generators']}), flush=True)


def train_all(plan, root):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    done = 0
    for view in plan['views']:
        for index in range(len(plan['support_seeds'])):
            for condition in plan['conditions']:
                train_fit(plan, root, index, view, condition, device); done += 1
                atomic_json(root / 'training_state.json', {'status': 'running', 'completed': done,
                            'planned': 48, 'updated_utc': now()})
    atomic_json(root / 'training_state.json', {'status': 'complete', 'completed': done, 'updated_utc': now()})


def evaluate(plan, parent, config, root, sig):
    state = json.loads((root / 'training_state.json').read_text()); assert state['status'] == 'complete' and state['completed'] == 48
    atomic_json(root / 'test_opened.json', {'opened_utc': now(), 'all48_fits_complete': True,
                'fit_record_sha256': {p.name: sha(p) for p in (root / 'fits').glob('*.json')}})
    data = dict(np.load(root / 'dataset' / 'test' / 'dataset.npz'))
    _, _, tokens, _ = setup(config); device = 'cuda' if torch.cuda.is_available() else 'cpu'
    table = permutation_action_table()
    rows = []
    for source in sig['sources']:
        model = source_model(source, parent, config, device)
        h = hidden_features(model, data, parent['source_task'], tokens)
        weights = model.lm_head.weight[:31].detach().cpu().numpy()
        bias = np.zeros(31, dtype=np.float32) if model.lm_head.bias is None else model.lm_head.bias[:31].detach().cpu().numpy()
        output = h @ weights.T + bias
        np.savez_compressed(root / 'features' / f"test_s{source['seed']}.npz", hidden=h, output=output, readout_weight=weights, readout_bias=bias)
        del model
        for index in range(len(plan['support_seeds'])):
            if plan['source_seeds'][index % 3] != source['seed']:
                continue
            for view in plan['views']:
                vectors = (h if view == 'hidden' else output).astype(np.float64)
                for condition in plan['conditions']:
                    name = f'n{index}_{view}_{condition}'; maps = dict(np.load(root / 'maps' / f'{name}.npz'))
                    centers = np.asarray([maps['mean_vectors'][list(maps['mean_lengths']).index(n)] for n in data['lengths']])
                    arrays = {}
                    for word in plan['words']:
                        action = word_action(word, table, {'c': 1, 'i': 4})
                        prediction = apply_numpy(vectors[:, 0] - centers, word, maps) + centers
                        answers = (prediction @ maps['readout_weight'].T + maps['readout_bias']).argmax(-1)
                        arrays[word + '_hidden'] = prediction.astype(np.float32); arrays[word + '_answers'] = answers
                        forced_answers = None
                        if len(word) == 2:
                            intermediate_action = 1 if word[0] == 'c' else 4
                            forced = apply_numpy(vectors[:, intermediate_action] - centers, word[1], maps) + centers
                            forced_answers = (forced @ maps['readout_weight'].T + maps['readout_bias']).argmax(-1)
                            arrays[word + '_forced_answers'] = forced_answers
                        target = vectors[:, action].astype(np.float64); base = vectors[:, 0].astype(np.float64)
                        for split, tag in [(0, 'iid'), (1, 'collisions')]:
                            use = data['split'] == split; hit = answers[use] == data['labels'][use, action]
                            delta = target[use] - base[use]; denominator = float(np.square(delta).sum())
                            per_length_cka = []
                            for n in plan['lengths']:
                                ids = np.flatnonzero(use & (data['lengths'] == n))[:128]
                                per_length_cka.append(gram_cka(prediction[ids], target[ids]))
                            row = {'replicate': f'n{index}', 'source_seed': source['seed'], 'view': view,
                                   'condition': condition, 'word': word, 'action': action, 'split': tag,
                                   'accuracy': float(hit.mean()), 'examples': int(use.sum()),
                                   'teacher_forced_accuracy': float((forced_answers[use] == data['labels'][use, action]).mean()) if forced_answers is not None else None,
                                   'direct_input_accuracy': float((output[use, action].argmax(-1) == data['labels'][use, action]).mean()),
                                   'displacement_nmse': float(np.square(prediction[use] - target[use]).sum() / denominator) if denominator > 1e-20 else None,
                                   'cka': float(np.mean(per_length_cka)), 'pair_both_correct': None,
                                   'composed_displacement_energy': float(np.square(prediction[use] - base[use]).sum() / max(np.square(base[use] - base[use].mean(0)).sum(), 1e-20))}
                            if split == 1:
                                pair_ids = data['pair_ids'][use]
                                row['pair_both_correct'] = float(np.mean([hit[pair_ids == pair].all() for pair in np.unique(pair_ids)]))
                            rows.append(row)
                    np.savez_compressed(root / 'evaluations' / f'{name}.npz', **arrays)
    atomic_json(root / 'evaluation_records.json', {'completed_utc': now(), 'records': rows})


def gram_cka(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    x, y = x - x.mean(0), y - y.mean(0)
    a, b = x @ x.T, y @ y.T
    denominator = np.sqrt(np.square(a).sum() * np.square(b).sum())
    return float((a * b).sum() / denominator) if denominator > 1e-20 else 0.0


def statistics(values):
    values = np.asarray(values, dtype=float)
    def interval(v):
        ids = np.asarray(list(product(range(len(v)), repeat=len(v))))
        return np.quantile(v[ids].mean(1), [.025, .975]).tolist()
    clusters = np.asarray([values[[0, 3]].mean(), values[[1, 4]].mean(), values[[2, 5]].mean()])
    return {'mean': float(values.mean()), 'paired_values': values.tolist(), 'positive_repeats': int((values > 0).sum()),
            'six_fit_bootstrap_95': interval(values), 'three_source_means': clusters.tolist(), 'three_source_bootstrap_95': interval(clusters)}


def summarize(plan, root):
    rows = json.loads((root / 'evaluation_records.json').read_text())['records']
    means, contrasts = [], []
    for view in plan['views']:
        for split in ['iid', 'collisions']:
            for word in plan['words']:
                for condition in plan['conditions']:
                    selected = [r for r in rows if (r['view'], r['split'], r['word'], r['condition']) == (view, split, word, condition)]
                    means.append({'view': view, 'split': split, 'word': word, 'condition': condition,
                                  **{key: float(np.mean([r[key] for r in selected])) if selected[0][key] is not None else None
                                     for key in ['accuracy', 'teacher_forced_accuracy', 'direct_input_accuracy', 'pair_both_correct', 'displacement_nmse', 'cka', 'composed_displacement_energy']}})
                for right in plan['conditions'][1:] + ['interaction']:
                    values = []
                    for index in range(6):
                        selected = {r['condition']: r for r in rows if (r['view'], r['split'], r['word'], r['replicate']) == (view, split, word, f'n{index}')}
                        if right == 'interaction':
                            delta = selected['both_correct']['accuracy'] - selected['c_correct_i_wrong']['accuracy'] - selected['c_wrong_i_correct']['accuracy'] + selected['both_wrong']['accuracy']
                        else:
                            delta = selected['both_correct']['accuracy'] - selected[right]['accuracy']
                        values.append(delta * 100)
                    contrasts.append({'view': view, 'split': split, 'word': word,
                                      'contrast': 'interaction' if right == 'interaction' else 'both_correct-' + right,
                                      'primary': view == 'hidden' and word == plan['primary_word'],
                                      'accuracy_pp': statistics(values)})
    primary = [r for r in means if r['view'] == 'hidden' and r['word'] == plan['primary_word']]
    correct_rows = [r for r in rows if r['view'] == 'hidden' and r['word'] == plan['primary_word']
                    and r['split'] == 'collisions' and r['condition'] == 'both_correct']
    source_accuracy = [float(np.mean([r['accuracy'] for r in correct_rows if r['source_seed'] == seed])) for seed in plan['source_seeds']]
    source_pairs = [float(np.mean([r['pair_both_correct'] for r in correct_rows if r['source_seed'] == seed])) for seed in plan['source_seeds']]
    gates = {'both_correct_collision_above_half_in_all3_sources': all(v > .5 for v in source_accuracy),
             'both_correct_pair_both_correct_positive_in_all3_sources': all(v > 0 for v in source_pairs)}
    for split in ['iid', 'collisions']:
        selected = [r for r in contrasts if r['primary'] and r['split'] == split and r['contrast'] != 'interaction']
        gates[f'both_correct_beats_all3_controls_in_each_source_{split}'] = all(
            all(v > 0 for v in r['accuracy_pp']['three_source_means']) for r in selected)
    summary = {'completed_utc': now(), 'means': means, 'contrasts': contrasts, 'primary_means': primary,
               'operator_fits': 48, 'frozen_backbones': 3, 'fit_repeats': 6,
               'primary_word': plan['primary_word'], 'primary_epoch': plan['primary_epoch'],
               'fixed_prediction_gates': gates, 'both_correct_collision_three_source_accuracy': source_accuracy,
               'both_correct_collision_three_source_pair_both_correct': source_pairs,
               'scope': plan['scope'], 'answer_only_collision_accuracy_ceiling': 0.5,
               'answer_only_collision_pair_both_correct_ceiling': 0.0,
               'spontaneous_relation_discovery_tested': False, 'output_independent_mechanism_established': False}
    atomic_json(root / 'summary.json', summary)
    display = []
    names = {'both_correct': '双正确', 'c_correct_i_wrong': '仅补排列正确',
             'c_wrong_i_correct': '仅取逆正确', 'both_wrong': '双错配'}
    for view in plan['views']:
        for condition in plan['conditions']:
            iid = next(r for r in means if (r['view'], r['word'], r['condition'], r['split']) == (view, 'ci', condition, 'iid'))
            collision = next(r for r in means if (r['view'], r['word'], r['condition'], r['split']) == (view, 'ci', condition, 'collisions'))
            display.append([view, names[condition], f"{100*iid['accuracy']:.2f}%", f"{100*collision['accuracy']:.2f}%",
                            f"{100*collision['pair_both_correct']:.2f}%", f"{collision['displacement_nmse']:.4f}", f"{collision['cka']:.4f}"])
    table = '<table><tr>' + ''.join('<th>' + h + '</th>' for h in ['空间', '条件', 'IID准确率', '碰撞准确率', '配对双正确', '位移NMSE', 'CKA']) + '</tr>'
    table += ''.join('<tr>' + ''.join('<td>' + escape(str(cell)) + '</td>' for cell in row) + '</tr>' for row in display) + '</table>'
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>两步关系正确性：2×2组合检验</title><style>body{font-family:system-ui;max-width:1150px;margin:40px auto;line-height:1.7}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px}</style><h1>两步关系正确性与未监督组合</h1><p>主操作为先补排列、后取逆（ci）。先取逆再补排列（ic）的LR-max答案恒等于可见C答案，因此保留为次要诊断。所有预测只输入原排列的一个冻结表征，再调用两个学到的算子与固定数字读出器。直接读取变换输入的成绩仅作诊断。</p>'
    html += table + '<p>四组共享单步输出监督、初始化、每个可用输入的精确曝光次数及训练日程；几何错配保持长度和三个可见答案。小于3样本的分组在所有条件下都不用于拟合。离散错配的复合不存在固定点，这不保证数值算子不会产生抵消；全部往返与恒等位移结果见原始记录。</p><p>隐藏空间24次拟合为主实验，输出空间24次拟合为诊断，参数量不同。6份拟合数据复用3个已有源模型，区间仅作描述。碰撞样本的已知答案向量预测上限为50%、配对双正确上限为0；这不是完整概率输出的信息上限。所有关系与调用顺序由实验提供，不能证明自发发现代数规律。</p><p><a href="summary.json">全部分支与对比</a> · <a href="evaluation_records.json">逐次结果</a> · <a href="protocol.json">实验登记</a> · <a href="verification.json">独立复核</a></p></html>'
    (root / 'report.html').write_text(html)


def verify_results(plan, parent, config, root, sig):
    data = dict(np.load(root / 'dataset' / 'test' / 'dataset.npz'))
    rows = json.loads((root / 'evaluation_records.json').read_text())['records']; checks = 0; grams = 0
    for row in rows:
        assert row['action'] == {'c': 1, 'i': 4, 'ci': 5, 'ic': 6, 'cc': 0, 'ii': 0}[row['word']]
        name = f"{row['replicate']}_{row['view']}_{row['condition']}"
        maps = dict(np.load(root / 'maps' / f'{name}.npz'))
        cache = dict(np.load(root / 'features' / f"test_s{row['source_seed']}.npz"))
        vectors = cache[row['view']].astype(np.float64)
        means = dict(zip(map(int, maps['mean_lengths']), maps['mean_vectors']))
        base = vectors[:, 0]; center = np.asarray([means[int(n)] for n in data['lengths']])
        # Independently multiply homogeneous affine matrices in row convention.
        width = base.shape[1]; matrix = np.eye(width + 1)
        for letter in row['word']:
            transform_matrix = np.eye(width + 1)
            transform_matrix[:width, :width] = maps['rho_' + letter]
            transform_matrix[-1, :width] = maps['bias_' + letter]
            matrix = matrix @ transform_matrix
        prediction = np.column_stack([base - center, np.ones(len(base))]) @ matrix
        prediction = prediction[:, :width] + center
        answers = (prediction @ maps['readout_weight'].T + maps['readout_bias']).argmax(-1)
        archive = np.load(root / 'evaluations' / f'{name}.npz')
        np.testing.assert_allclose(prediction, archive[row['word'] + '_hidden'], atol=2e-4, rtol=2e-5)
        np.testing.assert_array_equal(answers, archive[row['word'] + '_answers'])
        use = data['split'] == (0 if row['split'] == 'iid' else 1)
        hits = [int(a) == int(b) for a, b in zip(answers[use], data['labels'][use, row['action']])]
        assert sum(hits) / len(hits) == row['accuracy']; checks += 1
        if row['split'] == 'collisions':
            hit = np.asarray(hits); pairs = data['pair_ids'][use]
            assert sum(bool(hit[pairs == pair].all()) for pair in set(pairs)) / len(set(pairs)) == row['pair_both_correct']; checks += 1
        target = vectors[:, row['action']]
        if row['teacher_forced_accuracy'] is not None:
            action = 1 if row['word'][0] == 'c' else 4
            letter = row['word'][1]
            forced = (vectors[:, action] - center) @ maps['rho_' + letter] + maps['bias_' + letter] + center
            forced_answer = (forced @ maps['readout_weight'].T + maps['readout_bias']).argmax(-1)
            np.testing.assert_array_equal(forced_answer, archive[row['word'] + '_forced_answers'])
            assert float((forced_answer[use] == data['labels'][use, row['action']]).mean()) == row['teacher_forced_accuracy']; checks += 1
        direct_hits = cache['output'][use, row['action']].argmax(-1) == data['labels'][use, row['action']]
        assert float(direct_hits.mean()) == row['direct_input_accuracy']; checks += 1
        energy = np.square(prediction[use] - base[use]).sum() / max(np.square(base[use] - base[use].mean(0)).sum(), 1e-20)
        np.testing.assert_allclose(energy, row['composed_displacement_energy'], atol=1e-10)
        denominator = np.square(target[use] - base[use]).sum()
        if row['displacement_nmse'] is not None:
            np.testing.assert_allclose(np.square(prediction[use] - target[use]).sum() / denominator, row['displacement_nmse'], atol=1e-10)
        values = []
        for n in plan['lengths']:
            ids = np.flatnonzero(use & (data['lengths'] == n))[:128]
            x, y = prediction[ids], target[ids]; x = x - x.mean(0); y = y - y.mean(0)
            values.append(float(np.square(x.T @ y).sum() / np.sqrt(np.square(x.T @ x).sum() * np.square(y.T @ y).sum())))
            grams += 1
        np.testing.assert_allclose(np.mean(values), row['cka'], atol=2e-12, rtol=2e-12)
    _, _, tokens, _ = setup(config); device = 'cuda' if torch.cuda.is_available() else 'cpu'; replay_error = []
    for source in sig['sources']:
        model = source_model(source, parent, config, device, accelerated=False)
        ids = np.concatenate([np.flatnonzero(data['lengths'] == n)[:4] for n in plan['lengths']])
        replay = hidden_features(model, {'input': data['input'][ids], 'lengths': data['lengths'][ids]}, parent['source_task'], tokens)
        saved = np.load(root / 'features' / f"test_s{source['seed']}.npz")['hidden'][ids]
        np.testing.assert_allclose(replay, saved, atol=3e-4, rtol=3e-4)
        replay_error.append(float(np.abs(replay - saved).max())); del model
        assert sha(source['checkpoint']) == source['sha256']
    for index in range(6):
        source = sig['sources'][index % 3]
        model = source_model(source, parent, config, device, accelerated=False)
        support = dict(np.load(root / 'dataset' / f'n{index}' / 'support' / 'dataset.npz'))
        ids = np.concatenate([np.flatnonzero((support['lengths'] == n) & (support['split'] == 0))[:2]
                              for n in plan['lengths']])
        replay = hidden_features(model, {'input': support['input'][ids], 'lengths': support['lengths'][ids]}, parent['source_task'], tokens)
        saved = np.load(root / 'features' / f'n{index}_support.npz')['train_hidden'][np.searchsorted(np.flatnonzero(support['split'] == 0), ids)]
        np.testing.assert_allclose(replay, saved, atol=3e-4, rtol=3e-4)
        replay_error.append(float(np.abs(replay - saved).max())); del model
    for index in range(6):
        for view in plan['views']:
            fits = [json.loads((root / 'fits' / f'n{index}_{view}_{condition}.json').read_text()) for condition in plan['conditions']]
            assert len({row['schedule_sha256'] for row in fits}) == 1
            assert len({row['anchor_exposures'] for row in fits}) == 1
            for row in fits:
                assert row['epochs'] == plan['epochs'] and row['status'] == 'complete'
                assert sha(root / 'maps' / f"{row['replicate']}_{view}_{row['condition']}.npz") == row['map_sha256']
    atomic_json(root / 'verification.json', {'status': 'complete', 'completed_utc': now(),
                'independent_accuracy_and_pair_counts': checks, 'independent_gram_scores': grams,
                'independent_affine_composition_rows': len(rows), 'original_backbones_replayed': 3,
                'maximum_source_feature_replay_error': max(replay_error), 'frozen_source_checkpoints_preserved': 3,
                'support_feature_caches_replayed': 6,
                'all48_fit_budgets_and_hashes_checked': True})


def run():
    plan, parent, config, root, sig = initialize()
    _, _, _, _ = setup(config)
    try:
        for stage in ['prepare', 'features', 'train', 'evaluate', 'summarize', 'verify']:
            atomic_json(root / 'state.json', {'status': 'running', 'stage': stage, 'updated_utc': now()})
            if stage == 'prepare': prepare_data(plan, parent, config, root, sig)
            elif stage == 'features': cache_supports(plan, parent, config, root, sig)
            elif stage == 'train': train_all(plan, root)
            elif stage == 'evaluate': evaluate(plan, parent, config, root, sig)
            elif stage == 'summarize': summarize(plan, root)
            elif stage == 'verify': verify_results(plan, parent, config, root, sig)
        atomic_json(root / 'state.json', {'status': 'complete', 'completed_utc': now()})
    except BaseException as error:
        atomic_json(root / 'state.json', {'status': 'failed', 'error': repr(error), 'updated_utc': now()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['run', 'register', 'verify'])
    args = parser.parse_args()
    if args.phase == 'run': run()
    elif args.phase == 'register': initialize()
    else:
        plan, parent, config, root, sig = initialize(); verify_results(plan, parent, config, root, sig)
