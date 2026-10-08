"""Continue the fixed source-seed replication up to the requested deadline."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from .hidden_relation_train import load_plan, train_one
from .hidden_relation_evaluate import partial_probe, score_method
from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .native_algebra_structure import extract
from .native_confirmation import setup
from .permworld_combinations import sha


CONFIG = 'configs/relation_seed_extension.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize():
    extension = json.loads(Path(CONFIG).read_text())
    parent, config, _ = load_plan()
    root = Path(extension['output'])
    root.mkdir(parents=True, exist_ok=True)
    for name in ['source', 'checkpoints', 'features', 'arrays', 'evaluations']:
        (root / name).mkdir(exist_ok=True)
    previous = Path(extension['parent'])
    source = root / 'source_data.npz'
    if not source.exists():
        shutil.copyfile(previous / 'source_data.npz', source)
    assert sha(source) == sha(previous / 'source_data.npz')
    plan = {**parent, **{key: extension[key] for key in ['output', 'source_seeds', 'conditions']}}
    signature = {
        'config_sha256': sha(CONFIG), 'code_sha256': sha(__file__),
        'original_train_code_sha256': sha('experiments/hidden_relation_train.py'),
        'parent_protocol_sha256': sha(previous / 'protocol.json'),
        'source_data_sha256': sha(source), 'probe_data_sha256': sha(previous / 'probe_dataset.npz'),
        'plan': plan, 'scope': extension['scope'],
        'deadline_policy': 'SIGALRM stops current work at requested deadline; original trainer saves every 1000 steps and verifies RNG/sampler replay on resume. Interrupted jobs are partial, never counted as 20000-step results.'
    }
    dest = root / 'protocol.json'
    if dest.exists():
        assert json.loads(dest.read_text())['signature'] == signature
    else:
        atomic_json(dest, {'registered_utc': now(), 'signature': signature, 'new_models': 0})
    return extension, plan, config, root


def evaluate_one(extension, plan, config, root, name, seed, condition, tokens, device):
    dest = root / 'evaluations' / f'{name}.json'
    if dest.exists():
        return
    previous = Path(extension['parent'])
    probe = dict(np.load(previous / 'probe_dataset.npz'))
    source = json.loads((root / 'source' / f'{name}.json').read_text())
    cp = root / 'checkpoints' / f'{name}.pt'
    assert source['status'] == 'complete' and sha(cp) == source['checkpoint_sha256']
    state = torch.load(cp, weights_only=True, map_location=device)
    model = make_model(config, plan['architecture'], seed, device)
    model.load_state_dict(state['model'])
    fp = root / 'features' / f'{name}.npz'
    if fp.exists():
        features = dict(np.load(fp))
    else:
        features = extract(model, probe, [plan['source_task']], tokens, plan['batch_size'])
        np.savez_compressed(fp, **features)
    hidden = features['source_query_concat'][:, :, -1].astype(np.float64)
    weights = model.lm_head.weight[:31].detach().cpu().numpy().astype(np.float64)
    rb = model.lm_head.bias
    bias = np.zeros(31) if rb is None else rb[:31].detach().cpu().numpy().astype(np.float64)
    offsets = state['operators']['offset'].cpu().numpy().astype(np.float64)
    biases = state['operators']['bias'].cpu().numpy().astype(np.float64)
    native = {letter: (np.eye(hidden.shape[-1]) + offsets[j], biases[j]) for j, letter in enumerate(['c', 'i'])}
    variants = [('native_operators', native, {}, {})]
    for shuffled in [False, True]:
        maps, means, alphas = partial_probe(hidden, probe, plan['ridge_grid'], seed + 9300, shuffled)
        variants.append(('posthoc_shuffled_generators' if shuffled else 'posthoc_correct_generators', maps, means, alphas))
    methods, archives = [], {}
    for method, maps, means, alphas in variants:
        metrics, arrays = score_method(hidden, probe, maps, means, weights, bias, 'query')
        arrays.update({f'rho_{letter}': value[0] for letter, value in maps.items()})
        arrays.update({f'bias_{letter}': value[1] for letter, value in maps.items()})
        arrays.update(mean_lengths=np.array(list(means)), mean_vectors=np.array(list(means.values())).reshape(-1, hidden.shape[-1]),
            readout_weight=weights, readout_bias=bias)
        ap = root / 'arrays' / f'{name}_{method}.npz'
        np.savez_compressed(ap, **arrays)
        archives[ap.name] = sha(ap)
        methods.append({'method': method, 'generator_alphas': alphas, 'metrics': metrics})
    atomic_json(dest, {'source': name, 'seed': seed, 'condition': condition, 'source_validation_accuracy': source['observed_validation_accuracy'],
        'checkpoint_sha256': sha(cp), 'feature_sha256': sha(fp), 'array_sha256': archives, 'methods': methods, 'completed_utc': now()})
    primary = [a for a in methods[0]['metrics'] if a['split'] == 'answer_collisions' and a['word'] in ['ci', 'ici']]
    print(json.dumps({'new_seed_evaluated': name, 'primary': primary}), flush=True)
    del model, state
    if device == 'cuda':
        torch.cuda.empty_cache()


def run(stage='all'):
    extension, plan, config, root = initialize()
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, _, tokens, _ = setup(config)
    raw = dict(np.load(root / 'source_data.npz'))
    data = {key: torch.as_tensor(value, device=device) for key, value in raw.items()}
    deadline = datetime.fromisoformat(extension['deadline_utc']).timestamp()
    if time.time() >= deadline:
        atomic_json(root / 'state.json', {'status': 'stopped_at_deadline', 'updated_utc': now()})
        return
    def deadline_handler(*_):
        raise TimeoutError('Requested 10:00 deadline reached')
    signal.signal(signal.SIGALRM, deadline_handler)
    signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
    name = None
    try:
        for seed in plan['source_seeds']:
            for condition in plan['conditions']:
                name = f'{condition}_s{seed}'
                if stage in ['train', 'all']:
                    atomic_json(root / 'state.json', {'status': 'source_training', 'source': name, 'deadline_utc': extension['deadline_utc'], 'updated_utc': now()})
                    train_one(plan, config, root, tokens, data, raw, seed, condition, device)
                if stage in ['evaluate', 'all']:
                    atomic_json(root / 'state.json', {'status': 'evaluating', 'source': name, 'updated_utc': now()})
                    evaluate_one(extension, plan, config, root, name, seed, condition, tokens, device)
        atomic_json(root / 'state.json', {'status': 'complete', 'source_models': len(plan['source_seeds']) * len(plan['conditions']), 'updated_utc': now()})
    except TimeoutError:
        checkpoint = root / 'checkpoints' / f'{name}.pt' if name else None
        saved = None
        if checkpoint and checkpoint.exists():
            state = torch.load(checkpoint, weights_only=True, map_location='cpu')
            saved = {'source': name, 'step': state['step'], 'checkpoint_sha256': sha(checkpoint)}
        atomic_json(root / 'state.json', {'status': 'stopped_at_deadline', 'last_saved_checkpoint': saved, 'updated_utc': now()})
        print(json.dumps({'stopped_at_deadline': True, 'last_saved_checkpoint': saved}), flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['train', 'evaluate', 'all'], nargs='?', default='all')
    run(parser.parse_args().stage)
