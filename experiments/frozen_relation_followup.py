"""Fixed-backbone relation controls and stepwise composition diagnostics."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from .field_algebra_structure import affine_fit
from .hidden_relation_train import load_plan as parent_plan, query_hidden
from .hidden_relation_evaluate import transform_features, score_method
from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .native_confirmation import setup
from .permworld_combinations import sha
from .representation_algebra import permutation_action_table, word_action


CONFIG = 'configs/frozen_relation_followup.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def load_plan():
    plan = json.loads(Path(CONFIG).read_text())
    parent, config, _ = parent_plan()
    return plan, parent, config, Path(plan['output'])


def initialize():
    plan, parent, config, root = load_plan()
    root.mkdir(parents=True, exist_ok=True)
    for name in ['features', 'maps', 'evaluations', 'diagnostics']:
        (root / name).mkdir(exist_ok=True)
    previous = Path(plan['parent'])
    signature = {
        'config_sha256': sha(CONFIG),
        'code_sha256': sha(__file__),
        'parent_protocol_sha256': sha(previous / 'protocol.json'),
        'parent_data_sha256': {name: sha(previous / name) for name in ['source_data.npz', 'probe_dataset.npz']},
        'source_checkpoint_sha256': {
            f'{condition}_s{seed}': sha(previous / 'checkpoints' / f'{condition}_s{seed}.pt')
            for seed in plan['source_seeds'] for condition in plan['source_conditions']
        },
        'core_sha256': {name: sha(name) for name in [
            'experiments/field_algebra_structure.py', 'experiments/hidden_relation_evaluate.py',
            'experiments/hidden_relation_train.py', 'experiments/representation_algebra.py']},
        'scope': 'Exploratory follow-up of already inspected parent outcomes; full frozen comparison registered before its new full-source fits and intermediate-state diagnostics.'
    }
    protocol = root / 'protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text())['signature'] == signature
    else:
        atomic_json(protocol, {'registered_utc': now(), 'signature': signature, 'new_frozen_evaluations': 0})
    return plan, parent, config, root


@torch.no_grad()
def extract_known(model, inputs, lengths, task, tokens, batch_size):
    model.eval()
    device = next(model.parameters()).device
    flat = inputs.reshape(-1, inputs.shape[-1])
    ns = np.repeat(lengths, 3)
    result = []
    for start in range(0, len(flat), batch_size):
        x = torch.as_tensor(flat[start:start + batch_size], device=device)
        n = torch.as_tensor(ns[start:start + batch_size], device=device)
        result.append(query_hidden(model, x, n, task, tokens).float().cpu().numpy())
    return np.concatenate(result).reshape(len(lengths), 3, -1)


def features():
    plan, parent, config, root = initialize()
    torch.set_num_threads(4)
    _, _, tokens, _ = setup(config)
    previous = Path(plan['parent'])
    source = dict(np.load(previous / 'source_data.npz'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for seed in plan['source_seeds']:
        for condition in plan['source_conditions']:
            name = f'{condition}_s{seed}'
            dest = root / 'features' / f'{name}.npz'
            meta = root / 'features' / f'{name}.json'
            if meta.exists():
                assert sha(dest) == json.loads(meta.read_text())['feature_sha256']
                continue
            if datetime.now(timezone.utc).timestamp() >= datetime.fromisoformat(plan['deadline_utc']).timestamp():
                return
            cp = previous / 'checkpoints' / f'{name}.pt'
            state = torch.load(cp, weights_only=True, map_location=device)
            model = make_model(config, parent['architecture'], seed, device)
            model.load_state_dict(state['model'])
            assert all(not parameter.grad for parameter in model.parameters())
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            outputs = {}
            for split in ['train', 'validation']:
                outputs[f'{split}_hidden'] = extract_known(model, source[f'{split}_input'], source[f'{split}_lengths'], parent['source_task'], tokens, plan['batch_size'])
            weights = model.lm_head.weight[:31].float().cpu().numpy()
            bias = np.zeros(31, dtype=np.float32) if model.lm_head.bias is None else model.lm_head.bias[:31].float().cpu().numpy()
            outputs.update(readout_weight=weights, readout_bias=bias)
            np.savez_compressed(dest, **outputs)
            visible_accuracy = (np.argmax(outputs['validation_hidden'] @ weights.T + bias, axis=-1) == source['validation_labels']).mean(0).tolist()
            atomic_json(meta, {'source': name, 'source_checkpoint_sha256': sha(cp), 'feature_sha256': sha(dest),
                'source_data_sha256': sha(previous / 'source_data.npz'), 'visible_accuracy_e_C_I': visible_accuracy,
                'backbone_and_readout_frozen': True, 'completed_utc': now()})
            atomic_json(root / 'state.json', {'status': 'extracting_known_features', 'source': name, 'updated_utc': now()})
            print(json.dumps({'known_features': name, 'visible_accuracy_e_C_I': visible_accuracy}), flush=True)
            del model, state
            if device == 'cuda':
                torch.cuda.empty_cache()


def fit_maps(train, train_lengths, val, val_lengths, count, grid, seed, shuffled):
    """Nested source anchors; validation and centering see only known states."""
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([rng.permutation(np.flatnonzero(train_lengths == n))[:count] for n in np.unique(train_lengths)])
    train, train_lengths = np.asarray(train[chosen], dtype=np.float64), train_lengths[chosen]
    val = np.asarray(val, dtype=np.float64).copy()
    means = {}
    for n in np.unique(train_lengths):
        means[int(n)] = train[train_lengths == n].reshape(-1, train.shape[-1]).mean(0)
        train[train_lengths == n] -= means[int(n)]
        val[val_lengths == n] -= means[int(n)]
    def edge_values(h, ns, index):
        other = h[:, index]
        if shuffled:
            order = np.arange(len(h))
            for n in np.unique(ns):
                ids = np.flatnonzero(ns == n)
                assert len(ids) > 1
                order[ids] = np.roll(ids, int(rng.integers(1, len(ids))))
            other = other[order]
        return np.concatenate([h[:, 0], other]), np.concatenate([other, h[:, 0]])
    maps, alphas, validation = {}, {}, {}
    for letter, index in [('c', 1), ('i', 2)]:
        x, y = edge_values(train, train_lengths, index)
        xv, yv = edge_values(val, val_lengths, index)
        candidates = [affine_fit(x, y - x, alpha) for alpha in grid]
        errors = [float(np.square(xv + xv @ w[:-1] + w[-1] - yv).sum()) for w in candidates]
        best = int(np.argmin(errors))
        w = candidates[best]
        maps[letter] = (np.eye(x.shape[-1]) + w[:-1], w[-1])
        alphas[letter] = grid[best]
        validation[letter] = errors[best] / max(float(np.square(yv - xv).sum()), 1e-20)
    return maps, means, alphas, validation, chosen


def fit_and_evaluate():
    plan, parent, config, root = initialize()
    previous = Path(plan['parent'])
    source = dict(np.load(previous / 'source_data.npz'))
    probe = dict(np.load(previous / 'probe_dataset.npz'))
    for seed in plan['source_seeds']:
        for condition in plan['source_conditions']:
            name = f'{condition}_s{seed}'
            known = dict(np.load(root / 'features' / f'{name}.npz'))
            hidden = np.load(previous / 'features' / f'{name}.npz')['source_query_concat'][:, :, -1].astype(np.float64)
            for count in plan['fit_anchors_per_length']:
                for pairing in plan['relation_pairings']:
                    ident = f'{name}_k{count}_{pairing}'
                    dest = root / 'evaluations' / f'{ident}.json'
                    if dest.exists():
                        continue
                    if datetime.now(timezone.utc).timestamp() >= datetime.fromisoformat(plan['deadline_utc']).timestamp():
                        return
                    maps, means, alphas, val_error, ids = fit_maps(known['train_hidden'], source['train_lengths'], known['validation_hidden'], source['validation_lengths'], count, plan['ridge_grid'], seed + 2026100671, pairing == 'shuffled')
                    metrics, arrays = score_method(hidden, probe, maps, means, known['readout_weight'].astype(np.float64), known['readout_bias'].astype(np.float64), 'query')
                    arrays.update({f'rho_{g}': m[0] for g, m in maps.items()})
                    arrays.update({f'bias_{g}': m[1] for g, m in maps.items()})
                    arrays.update(mean_lengths=np.array(list(means)), mean_vectors=np.array(list(means.values())), source_anchor_ids=ids,
                        readout_weight=known['readout_weight'], readout_bias=known['readout_bias'])
                    ap = root / 'maps' / f'{ident}.npz'
                    np.savez_compressed(ap, **arrays)
                    record = {'source': name, 'source_condition': condition, 'seed': seed, 'fit_anchors_per_length': count, 'pairing': pairing,
                        'source_checkpoint_sha256': sha(previous / 'checkpoints' / f'{name}.pt'), 'map_archive_sha256': sha(ap),
                        'known_feature_sha256': sha(root / 'features' / f'{name}.npz'), 'probe_feature_sha256': sha(previous / 'features' / f'{name}.npz'),
                        'visible_accuracy_e_C_I': json.loads((root / 'features' / f'{name}.json').read_text())['visible_accuracy_e_C_I'],
                        'selected_generator_alphas': alphas, 'known_validation_displacement_nmse': val_error,
                        'metrics': metrics, 'completed_utc': now()}
                    atomic_json(dest, record)
                    atomic_json(root / 'state.json', {'status': 'frozen_fit_evaluation', 'condition': ident, 'updated_utc': now()})
                    print(json.dumps({'frozen_evaluation': ident, 'primary': [{k: a[k] for k in ['word', 'answer_accuracy', 'displacement_nmse']} for a in metrics if a['split'] == 'answer_collisions' and a['word'] in ['ci', 'ici']]}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['features', 'evaluate', 'all'])
    args = parser.parse_args()
    if args.stage in ['features', 'all']:
        features()
    if args.stage in ['evaluate', 'all']:
        fit_and_evaluate()
