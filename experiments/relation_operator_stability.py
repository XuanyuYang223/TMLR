"""Test predefined gain controls using known-state-only validation selection."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from .field_algebra_structure import affine_fit
from .frozen_relation_followup import extract_known
from .hidden_relation_train import load_plan
from .hidden_relation_evaluate import score_method
from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .native_confirmation import setup
from .permworld_combinations import sha


CONFIG = 'configs/relation_operator_stability.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def paired_edges(hidden, index):
    return np.concatenate([hidden[:, 0], hidden[:, index]]).astype(np.float64), np.concatenate([hidden[:, index], hidden[:, 0]]).astype(np.float64)


def global_affine_generator(x, y, xv, yv, grid):
    center = x.mean(0)
    xc, yc, xvc, yvc = x - center, y - center, xv - center, yv - center
    candidates = [affine_fit(xc, yc - xc, alpha) for alpha in grid]
    errors = [float(np.square(xvc + xvc @ w[:-1] + w[-1] - yvc).sum()) for w in candidates]
    best = int(np.argmin(errors))
    weight = candidates[best]
    rho = np.eye(x.shape[-1]) + weight[:-1]
    bias = weight[-1] + center - center @ rho
    return (rho, bias), {'ridge_alpha': grid[best], 'validation_squared_error': errors[best]}


def reflection_generator(x, y):
    center = (x.mean(0) + y.mean(0)) / 2
    gram = (x - center).T @ (y - center)
    gram = (gram + gram.T) / 2
    eigenvalues, basis = np.linalg.eigh(gram)
    rho = (basis * np.where(eigenvalues >= 0, 1., -1.)) @ basis.T
    bias = center - center @ rho
    return rho, bias


def gain_candidates(rho, x, y, xv, yv, caps, tolerance):
    u, singular, vt = np.linalg.svd(rho, full_matrices=False)
    candidates = []
    center_x, center_y = x.mean(0), y.mean(0)
    for cap in [*caps, None]:
        limited = rho if cap is None else (u * np.minimum(singular, cap)) @ vt
        bias = center_y - center_x @ limited
        error = float(np.square(xv @ limited + bias - yv).sum())
        candidates.append({'cap': cap, 'validation_squared_error': error, 'map': (limited, bias)})
    best = min(item['validation_squared_error'] for item in candidates)
    eligible = [item for item in candidates if item['validation_squared_error'] <= best * (1 + tolerance) + 1e-12]
    chosen = min(eligible, key=lambda item: float('inf') if item['cap'] is None else item['cap'])
    metadata = {'original_spectral_norm': float(singular[0]), 'selected_cap': chosen['cap'],
        'candidates': [{key: item[key] for key in ['cap', 'validation_squared_error']} for item in candidates]}
    return chosen['map'], candidates[-1]['map'], metadata


def initialize():
    plan = json.loads(Path(CONFIG).read_text())
    parent, config, _ = load_plan()
    root = Path(plan['output'])
    root.mkdir(parents=True, exist_ok=True)
    for folder in ['features', 'maps', 'evaluations']:
        (root / folder).mkdir(exist_ok=True)
    signature = {'config_sha256': sha(CONFIG), 'code_sha256': sha(__file__),
        'parent_protocol_sha256': sha(Path(plan['parent']) / 'protocol.json'),
        'extension_protocol_sha256': sha(Path(plan['extension']) / 'protocol.json'),
        'known_source_data_sha256': sha(Path(plan['parent']) / 'source_data.npz'),
        'scope': plan['interpretation_scope'], 'hyperparameter_selection': plan['selection_scope']}
    dest = root / 'protocol.json'
    if dest.exists():
        assert json.loads(dest.read_text())['signature'] == signature
    else:
        atomic_json(dest, {'registered_utc': now(), 'signature': signature, 'new_stability_evaluations': 0})
    return plan, parent, config, root


def known_features(plan, parent, config, root, location, name, seed, state):
    existing = Path('results/frozen_relation_followup/features') / f'{name}.npz'
    if location == Path(plan['parent']) and existing.exists():
        return dict(np.load(existing)), sha(existing)
    path = root / 'features' / f'{name}.npz'
    if path.exists():
        return dict(np.load(path)), sha(path)
    raw = dict(np.load(Path(plan['parent']) / 'source_data.npz'))
    _, _, tokens, _ = setup(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = make_model(config, parent['architecture'], seed, device)
    model.load_state_dict(state['model'])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    output = {f'{split}_hidden': extract_known(model, raw[f'{split}_input'], raw[f'{split}_lengths'], parent['source_task'], tokens, plan['batch_size']) for split in ['train', 'validation']}
    output['readout_weight'] = model.lm_head.weight[:31].float().cpu().numpy()
    rb = model.lm_head.bias
    output['readout_bias'] = np.zeros(31, dtype=np.float32) if rb is None else rb[:31].float().cpu().numpy()
    np.savez_compressed(path, **output)
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return output, sha(path)


def evaluate_available():
    plan, parent, config, root = initialize()
    torch.set_num_threads(4)
    probe = dict(np.load(Path(plan['parent']) / 'probe_dataset.npz'))
    completed = 0
    for location, seeds in [(Path(plan['parent']), plan['source_seeds']), (Path(plan['extension']), plan['extension_seeds'])]:
        for seed in seeds:
            for condition in plan['source_conditions']:
                name = f'{condition}_s{seed}'
                cp, rec = location / 'checkpoints' / f'{name}.pt', location / 'source' / f'{name}.json'
                fp = location / 'features' / f'{name}.npz'
                if not rec.exists() or not fp.exists():
                    continue
                if all((root / 'evaluations' / f'{name}_{method}.json').exists() for method in plan['methods']):
                    completed += 1
                    continue
                if time.time() >= datetime.fromisoformat(plan['deadline_utc']).timestamp():
                    return completed
                record = json.loads(rec.read_text())
                assert record['status'] == 'complete' and sha(cp) == record['checkpoint_sha256']
                state = torch.load(cp, weights_only=True, map_location='cpu')
                known, known_sha = known_features(plan, parent, config, root, location, name, seed, state)
                hidden = np.load(fp)['source_query_concat'][:, :, -1].astype(np.float64)
                variants = {method: ({}, {}) for method in plan['methods']}
                for j, (letter, index) in enumerate([('c', 1), ('i', 2)]):
                    x, y = paired_edges(known['train_hidden'], index)
                    xv, yv = paired_edges(known['validation_hidden'], index)
                    affine, affine_meta = global_affine_generator(x, y, xv, yv, plan['ridge_grid'])
                    variants['shared_center_affine'][0][letter] = affine
                    variants['shared_center_affine'][1][letter] = affine_meta
                    reflection = reflection_generator(x, y)
                    variants['orthogonal_reflection'][0][letter] = reflection
                    variants['orthogonal_reflection'][1][letter] = {'validation_squared_error': float(np.square(xv @ reflection[0] + reflection[1] - yv).sum()), 'imposed_laws': 'A.T A = I and A²=I, with involutive affine bias'}
                    native_rho = np.eye(hidden.shape[-1]) + state['operators']['offset'][j].numpy().astype(np.float64)
                    limited, refit, metadata = gain_candidates(native_rho, x, y, xv, yv, plan['spectral_norm_caps'], plan['cap_validation_error_relative_tolerance'])
                    variants['native_validation_gain_cap'][0][letter] = limited
                    variants['native_validation_gain_cap'][1][letter] = metadata
                    variants['native_bias_refit'][0][letter] = refit
                    variants['native_bias_refit'][1][letter] = {'validation_squared_error': metadata['candidates'][-1]['validation_squared_error']}
                for method, (maps, metadata) in variants.items():
                    dest = root / 'evaluations' / f'{name}_{method}.json'
                    if dest.exists():
                        continue
                    metrics, arrays = score_method(hidden, probe, maps, {}, known['readout_weight'].astype(np.float64), known['readout_bias'].astype(np.float64), 'query')
                    arrays.update({f'rho_{letter}': value[0] for letter, value in maps.items()})
                    arrays.update({f'bias_{letter}': value[1] for letter, value in maps.items()})
                    arrays.update(readout_weight=known['readout_weight'], readout_bias=known['readout_bias'])
                    ap = root / 'maps' / f'{name}_{method}.npz'
                    np.savez_compressed(ap, **arrays)
                    atomic_json(dest, {'source': name, 'seed': seed, 'source_condition': condition, 'method': method,
                        'known_feature_sha256': known_sha, 'source_checkpoint_sha256': sha(cp), 'map_archive_sha256': sha(ap),
                        'source_validation_accuracy': record['observed_validation_accuracy'], 'generator_selection': metadata,
                        'metrics': metrics, 'completed_utc': now()})
                    print(json.dumps({'stability_evaluated': name, 'method': method, 'primary': [{k: r[k] for k in ['word', 'answer_accuracy', 'displacement_nmse']} for r in metrics if r['split'] == 'answer_collisions' and r['word'] in ['ci', 'ici']]}), flush=True)
                completed += 1
    atomic_json(root / 'state.json', {'status': 'available_evaluations_complete', 'source_models_evaluated': completed, 'updated_utc': now()})
    return completed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if not args.watch:
        evaluate_available()
    else:
        plan, _, _, _ = initialize()
        deadline = datetime.fromisoformat(plan['deadline_utc']).timestamp()
        while time.time() < deadline:
            count = evaluate_available()
            if count == 27:
                break
            time.sleep(min(45, max(0, deadline - time.time())))
