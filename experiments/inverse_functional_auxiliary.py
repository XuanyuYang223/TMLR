"""Prospectively fixed, secondary geometry measurements for the functional assay."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .inverse_functional_alignment import initialize, configure, now, extract
from .longrun_attention import accelerate
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .specialist_cka_controls import api


RULES = {
    'scope': 'Secondary measurements; no checkpoint, loss or task selection from CKA.',
    'subset': 'First 128 saved test rows of each length, identical for every condition and initialization.',
    'feature': 'Target and frozen inverse-input teacher final_norm at task-free ONE_END.',
    'aggregation': 'Compute within each length and average the five lengths equally.',
    'matched_control': 'Keep answer strata with at least three rows; cyclic shift each stratum by one row, preserving length and true target answer. Use identical kept rows in correct and wrong scores.',
    'answer_residual': 'Subtract each true-answer stratum mean from both hidden matrices on the same kept rows. This removes linear answer means, not all output information.',
    'initialization': 'Replay the exact stored target initialization for each replicate, with the same teacher and test subset.',
    'endpoints': ['selected', 'latest', 'initialization'],
}


def cka(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(0); y = y - y.mean(0)
    denom = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    if denom < 1e-20: return None
    return float(np.square(x.T @ y).sum() / denom)


def scores(x, y, lengths, answers, sizes):
    rows = []
    for n in sizes:
        ix = np.flatnonzero(lengths == n)[:128]
        a, b, label = x[ix], y[ix], answers[ix]
        kept = np.concatenate([np.flatnonzero(label == v) for v in np.unique(label) if (label == v).sum() >= 3])
        kept.sort(); a0, b0, labels = a[kept], b[kept], label[kept]
        permutation = np.arange(len(kept))
        ar = a0.astype(np.float64).copy(); br = b0.astype(np.float64).copy()
        for v in np.unique(labels):
            j = np.flatnonzero(labels == v); permutation[j] = np.roll(j, 1)
            ar[j] -= ar[j].mean(0); br[j] -= br[j].mean(0)
        assert np.all(permutation != np.arange(len(kept)))
        rows.append({'length': n, 'rows': len(ix), 'matched_rows': len(kept),
                     'correct_cka': cka(a, b), 'matched_correct_cka': cka(a0, b0),
                     'matched_wrong_cka': cka(a0, b0[permutation]),
                     'answer_residual_correct_cka': cka(ar, br),
                     'answer_residual_wrong_cka': cka(ar, br[permutation])})
    return rows


def register(root):
    path = root / 'auxiliary_protocol.json'
    if path.exists():
        old = json.loads(path.read_text()); assert old['rules'] == RULES and old['code_sha256'] == sha(__file__)
        return
    assert not (root / 'test_opened.json').exists(), 'Auxiliary choices must precede test opening.'
    atomic_json(path, {'registered_utc': now(), 'rules': RULES, 'code_sha256': sha(__file__),
                      'primary_training_has_started': bool(list((root / 'training').glob('r*.json'))),
                      'new_test_outcomes_observed': False})


def run(plan, root, signature):
    register(root)
    assert json.loads((root / 'state.json').read_text())['status'] == 'functional_evaluation_complete'
    device = configure(); _, tokens, _, TrainConfig, factory = api(plan)
    data = dict(np.load(root / 'dataset/test/dataset.npz')); records = []
    for rep in plan['replicates']:
        source = next(s for s in signature['sources'] if s['seed'] == rep['source_seed'])
        cp = torch.load(source['checkpoint'], weights_only=True, map_location='cpu')
        cfg = TrainConfig.from_value(cp['config']); del cp
        teacher = dict(np.load(root / 'teacher' / f"test_s{rep['source_seed']}.npz"))
        file = root / 'evaluations' / f"{rep['id']}_initialization.npz"
        if not file.exists():
            model = accelerate(factory(cfg)); model.load_state_dict(torch.load(root / 'initializations' / f"{rep['id']}.pt", weights_only=True, map_location='cpu')); model.to(device)
            out = extract(model, data, plan['target_task'], tokens, 0); np.savez_compressed(file, **out); del model
        init = dict(np.load(file))
        records.append({'replicate': rep['id'], 'condition': 'initialization', 'endpoint': 'initialization',
                        'scores': scores(init['hidden'], teacher['hidden'], data['lengths'], data['labels'], plan['lengths'])})
        for condition in plan['conditions']:
            arrays = dict(np.load(root / 'evaluations' / f"{rep['id']}_{condition}.npz"))
            for endpoint in ['selected', 'latest']:
                records.append({'replicate': rep['id'], 'condition': condition, 'endpoint': endpoint,
                                'scores': scores(arrays[endpoint + '_hidden'], teacher['hidden'], data['lengths'], data['labels'], plan['lengths'])})
    atomic_json(root / 'auxiliary.json', {'records': records, 'completed_utc': now(), 'secondary_only': True})
    print({'secondary_geometry_records': len(records)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['register', 'run'])
    args = parser.parse_args(); plan, root, sig = initialize()
    if args.stage == 'register': register(root)
    else: run(plan, root, sig)
