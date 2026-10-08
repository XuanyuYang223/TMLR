"""Known-label-only readout calibration for fixed composite predictions."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from .field_algebra_structure import affine_fit
from .frozen_relation_followup import extract_known
from .hidden_relation_train import load_plan
from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .native_confirmation import setup
from .permworld_combinations import sha
from .relation_observed_start import observed_start_score


CONFIG = 'configs/relation_readout_diagnostics.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def fit_readouts(train, labels, val, val_labels, grid):
    x = train.reshape(-1, train.shape[-1]).astype(np.float64)
    y = labels.reshape(-1)
    xv = val.reshape(-1, val.shape[-1]).astype(np.float64)
    yv = val_labels.reshape(-1)
    center = x.mean(0)
    xc, xvc = x - center, xv - center
    target, validation_target = np.eye(31)[y], np.eye(31)[yv]
    fits = [affine_fit(xc, target, alpha) for alpha in grid]
    losses = [float(np.square(xvc @ w[:-1] + w[-1] - validation_target).sum()) for w in fits]
    chosen = int(np.argmin(losses))
    fitted = fits[chosen]
    ridge = (fitted[:-1].T, fitted[-1] - center @ fitted[:-1])
    observed = np.unique(y)
    classes = np.zeros((31, x.shape[-1]))
    for label in observed:
        classes[label] = x[y == label].mean(0)
    euclidean = (2 * classes, -np.square(classes).sum(-1))
    cosine = (classes / np.maximum(np.linalg.norm(classes, axis=-1, keepdims=True), 1e-12), np.zeros(31))
    for _, bias in [euclidean, cosine]:
        bias[np.setdiff1d(np.arange(31), observed)] = -1e10
    variants = [euclidean, cosine]
    accuracies = [float(np.mean((xv @ weight.T + bias).argmax(-1) == yv)) for weight, bias in variants]
    picked = int(np.argmax(accuracies))
    return {'known_onehot_ridge': ridge, 'known_prototype_validation_selected': variants[picked]}, {
        'ridge_alpha': grid[chosen], 'ridge_validation_squared_error': losses[chosen],
        'prototype_selected': ['euclidean', 'cosine'][picked], 'prototype_validation_accuracies': accuracies,
        'known_training_label_classes': observed.tolist(),
    }


def initialize():
    plan = json.loads(Path(CONFIG).read_text())
    root = Path(plan['output'])
    root.mkdir(parents=True, exist_ok=True)
    for folder in ['features', 'readouts', 'evaluations', 'arrays']:
        (root / folder).mkdir(exist_ok=True)
    signature = {'code_sha256': sha(__file__), 'config_sha256': sha(CONFIG), 'plan': plan,
        'parent_protocol_sha256': sha(Path(plan['parent']) / 'protocol.json'),
        'extension_protocol_sha256': sha(Path(plan['extension']) / 'protocol.json'),
        'world_protocol_sha256': sha(Path(plan['world_confirmation']) / 'protocol.json')}
    path = root / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())['signature'] == signature
    else:
        atomic_json(path, {'registered_utc': now(), 'signature': signature, 'new_readout_calibrations': 0})
    return plan, root


def known_features(location, name, seed, parent, config, root):
    for path in [Path('results/frozen_relation_followup/features') / f'{name}.npz', Path('results/relation_operator_stability/features') / f'{name}.npz', root / 'features' / f'{name}.npz']:
        if path.exists():
            return dict(np.load(path)), sha(path)
    _, _, tokens, _ = setup(config)
    raw = dict(np.load(location / 'source_data.npz'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = make_model(config, parent['architecture'], seed, device)
    model.load_state_dict(torch.load(location / 'checkpoints' / f'{name}.pt', weights_only=True, map_location=device)['model'])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    output = {f'{tag}_hidden': extract_known(model, raw[f'{tag}_input'], raw[f'{tag}_lengths'], parent['source_task'], tokens, 64) for tag in ['train', 'validation']}
    path = root / 'features' / f'{name}.npz'
    np.savez_compressed(path, **output)
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return output, sha(path)


def evaluate_available():
    plan, root = initialize()
    parent, config, _ = load_plan()
    torch.set_num_threads(4)
    locations = [(Path(plan['parent']), plan['source_seeds'][:3], 'original_exploratory'),
        (Path(plan['extension']), plan['source_seeds'][3:], 'additional_seeds')]
    world_plan = json.loads(Path('configs/relation_world_confirmation.json').read_text())
    locations.extend((Path(plan['world_confirmation']) / world['id'], [world['source_seed']], world['id']) for world in world_plan['worlds'])
    completed = 0
    for location, seeds, scope in locations:
        for seed in seeds:
            for condition in plan['conditions']:
                name = f'{condition}_s{seed}'
                dest = root / 'evaluations' / f'{name}.json'
                if dest.exists():
                    completed += 1; continue
                cp = location / 'checkpoints' / f'{name}.pt'
                ep, fp = location / 'evaluations' / f'{name}.json', location / 'features' / f'{name}.npz'
                if not ep.exists() or not fp.exists():
                    continue
                if time.time() >= datetime.fromisoformat(plan['deadline_utc']).timestamp():
                    return completed
                raw = dict(np.load(location / 'source_data.npz'))
                probe_location = location if scope.startswith('world') else Path(plan['parent'])
                probe = dict(np.load(probe_location / 'probe_dataset.npz'))
                known, known_sha = known_features(location, name, seed, parent, config, root)
                readouts, selection = fit_readouts(known['train_hidden'], raw['train_labels'], known['validation_hidden'], raw['validation_labels'], plan['ridge_grid'])
                original_map = location / 'arrays' / (f'{name}_query_native_operators.npz' if scope == 'original_exploratory' else f'{name}_native_operators.npz')
                base = dict(np.load(original_map))
                readouts['original_lm_head'] = (base['readout_weight'].astype(np.float64), base['readout_bias'].astype(np.float64))
                hidden = np.load(fp)['source_query_concat'][:, :, -1].astype(np.float64)
                calibrations, hashes = [], {}
                for label in plan['readouts']:
                    weight, bias = readouts[label]
                    archive = {**base, 'readout_weight': weight, 'readout_bias': bias}
                    rows, arrays = observed_start_score(hidden, probe, archive)
                    xval = known['validation_hidden'].reshape(-1, hidden.shape[-1]).astype(np.float64)
                    yval = raw['validation_labels'].reshape(-1)
                    val_accuracy = float(np.mean((xval @ weight.T + bias).argmax(-1) == yval))
                    ap = root / 'arrays' / f'{name}_{label}.npz'
                    np.savez_compressed(ap, **arrays)
                    rp = root / 'readouts' / f'{name}_{label}.npz'
                    np.savez_compressed(rp, weight=weight, bias=bias)
                    hashes[ap.name] = sha(ap); hashes[rp.name] = sha(rp)
                    calibrations.append({'readout': label, 'known_validation_accuracy': val_accuracy, 'rows': rows})
                atomic_json(dest, {'source': name, 'source_condition': condition, 'seed': seed, 'scope': scope,
                    'source_checkpoint_sha256': sha(cp), 'native_map_sha256': sha(original_map), 'known_feature_sha256': known_sha,
                    'readout_selection': selection, 'calibrations': calibrations, 'archive_sha256': hashes, 'completed_utc': now()})
                print(json.dumps({'readout_evaluated': name, 'scope': scope, 'primary': [{
                    'readout': c['readout'], 'known_accuracy': c['known_validation_accuracy'],
                    'base_CI_accuracy': next(r['answer_accuracy'] for r in c['rows'] if r['split'] == 'answer_collisions' and r['case'] == 'base_CI')
                } for c in calibrations]}), flush=True)
                completed += 1
    return completed


if __name__ == '__main__':
    plan, _ = initialize()
    deadline = datetime.fromisoformat(plan['deadline_utc']).timestamp()
    while time.time() < deadline:
        count = evaluate_available()
        if count == 36:
            break
        time.sleep(min(45, max(0, deadline - time.time())))
