"""Prospective behavior endpoint paired with the exact-statistics cohort."""
import argparse
from datetime import datetime
import hashlib
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import torch

from .algebra import composition_size, projective, world
from .analysis import write_csv
from .field_mechanism import categorical_decoder
from .field_symmetry import group_sources
from .longrun_engine import atomic_json
from .longrun_transfer import fit_probe
from .models import SourceModel
from .readout_followup import lookup_predict


def splits(labels, p, seed, budgets, test_per_class):
    rng = np.random.default_rng(seed)
    test, pools = [], []
    for category in range(p):
        ids = np.flatnonzero(labels == category)
        rng.shuffle(ids)
        test.extend(ids[:test_per_class])
        pools.append(ids[test_per_class:])
    support = {b: np.concatenate([ids[:b//p] for ids in pools]) for b in budgets}
    test = np.array(test)
    assert all(not set(ids)&set(test) for ids in support.values())
    return support, test


def run(config_path='configs/field_symmetry_transfer.json'):
    config = json.loads(Path(config_path).read_text())
    source_config = json.loads(Path(config['source_config']).read_text())
    source_root = Path(source_config['output'])
    output = Path(config['output']); output.mkdir(exist_ok=True)
    parent = json.loads((source_root/'metadata.json').read_text())
    signature = {'config': config, 'source_fingerprint': parent['fingerprint'],
                 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'head_code_sha256': hashlib.sha256(Path('experiments/longrun_transfer.py').read_bytes()).hexdigest(),
                 'categorical_code_sha256': hashlib.sha256(Path('experiments/field_mechanism.py').read_bytes()).hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    if (output/'metadata.json').exists():
        assert json.loads((output/'metadata.json').read_text())['fingerprint'] == fingerprint
    atomic_json(output/'metadata.json', {**signature, 'fingerprint': fingerprint})
    targets = np.array(config['targets'])
    p = source_config['p']
    excluded = {projective(row, p) for g in source_config['groups'] for row in group_sources(g, p)}
    assert all(projective(target, p) not in excluded for target in targets)
    policy = {'steps': config['head_steps'], 'learning_rate': config['head_learning_rate']}
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stop_at = min(datetime.fromisoformat(config['deadline_utc']).timestamp()-config['analysis_reserve_seconds'], time.time()+config['maximum_runtime_seconds'])
    all_rows = []
    for w, m in product(source_config['world_seeds'], source_config['model_seeds']):
        inputs, _, encoded, basis = world(p, source_config['dimension'], w)
        latent = inputs@basis.T % p
        labels = latent@targets.T % p
        split = [splits(labels[:, t], p, w+90000+t, config['budgets'], config['test_per_class']) for t in range(len(targets))]
        for g in ['random']+source_config['groups']:
            path = output/f'{g}_w{w}_m{m}.json'
            if path.exists():
                saved = json.loads(path.read_text()); assert saved['fingerprint'] == fingerprint
                if saved['status'] == 'complete':
                    all_rows.extend(saved['rows']); continue
            if time.time() >= stop_at:
                if all_rows: write_csv(output/'endpoints.csv', all_rows)
                return
            source_gate = None
            codes = None
            if g == 'random':
                h = np.load(source_root/f'P_w{w}_m{m}_step0_features.npy').astype(np.float32)
            else:
                source_record = json.loads((source_root/f'{g}_w{w}_m{m}.json').read_text())
                if source_record['status'] != 'complete': continue
                source_gate = source_record['source_gate_passed']
                h = np.load(source_root/f'{g}_w{w}_m{m}_step{source_config["steps"]}_features.npy').astype(np.float32)
                model = SourceModel(encoded.shape[1], source_config['hidden'], source_config['features'], p)
                checkpoint = source_root/'checkpoints'/f'{g}_w{w}_m{m}.pt'
                assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == source_record['checkpoint_sha256']
                model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'))
                with torch.no_grad(): codes = model(torch.tensor(encoded)).argmax(-1).numpy()
            rows = []
            for t, target in enumerate(targets):
                support, test = split[t]
                order = composition_size(group_sources(g, p), target, p) if g != 'random' else None
                for budget in config['budgets']:
                    ids = support[budget]
                    common = {'group': g, 'world_seed': w, 'model_seed': m, 'target_id': t, 'budget': budget,
                              'target_composition_size': order, 'source_gate_passed': source_gate}
                    for kind in ('linear', 'mlp'):
                        result = fit_probe(h[ids], torch.tensor(labels[ids, t], device=device), h[test],
                                           torch.tensor(labels[test, t], device=device), h.shape[1], p, policy, m, kind, device)
                        rows.append({**common, 'mode': kind, 'accuracy': result['test_accuracy'],
                                     'support_accuracy': result['support_accuracy'], 'tuple_seen_fraction': None, 'chosen_subset_size': None})
                    if codes is not None:
                        predicted, seen, chosen = categorical_decoder(codes[ids], labels[ids, t], codes[test], p)
                        rows.append({**common, 'mode': 'categorical_subset', 'accuracy': float(np.mean(predicted == labels[test, t])),
                                     'support_accuracy': None, 'tuple_seen_fraction': float(np.mean(seen)), 'chosen_subset_size': len(chosen)})
                        predicted, seen = lookup_predict(codes[ids], labels[ids, t], codes[test], p)
                        rows.append({**common, 'mode': 'full_tuple_lookup', 'accuracy': float(np.mean(predicted == labels[test, t])),
                                     'support_accuracy': None, 'tuple_seen_fraction': float(np.mean(seen)), 'chosen_subset_size': 4})
            atomic_json(path, {'fingerprint': fingerprint, 'status': 'complete', 'group': g, 'rows': rows})
            all_rows.extend(rows)
            print(json.dumps({'run': path.stem, 'endpoints': len(rows)}), flush=True)
    write_csv(output/'endpoints.csv', all_rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='configs/field_symmetry_transfer.json')
    run(parser.parse_args().config)
