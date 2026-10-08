"""Compare fitted latent actions with exact source-task block permutations."""
from datetime import datetime, timezone
import json
from itertools import permutations
from pathlib import Path
import time

import numpy as np

from .longrun_engine import atomic_json
from .native_source_subspace import contrast_basis
from .permworld_combinations import sha
from .representation_algebra import permutation_action_table, word_action


ROOT = Path('results/task_block_geometry_control')
JOINT = Path('results/joint_answer_matched_geometry')
PARENT = Path('results/algebra_structure_replication')
NEW = Path('results/ordinary_relation_seed_confirmation')
TASK_PERMUTATIONS = {'c': np.array([2, 3, 0, 1]), 'r': np.array([1, 0, 3, 2]), 'i': np.array([3, 1, 2, 0])}


def block_controls(hidden, pair_ids):
    rows = np.stack([np.flatnonzero(pair_ids == pair) for pair in np.unique(pair_ids)])
    delta = (hidden[rows[:, 0]] - hidden[rows[:, 1]]).astype(np.float64).reshape(len(rows), 8, 4, -1)
    source = delta.reshape(-1, 4, delta.shape[-1])
    table, letters = permutation_action_table(), {'c': 1, 'r': 2, 'i': 4}
    results = []
    for word in ['c', 'r', 'i', 'rc', 'ci', 'ri', 'rci']:
        index = np.arange(4)
        for g in word:
            index = index[TASK_PERMUTATIONS[g]]
        action = word_action(word, table, letters)
        target = delta[:, table[:, action]].reshape(source.shape)
        denominator = float(np.square(target - source).sum())
        target_norm = float(np.square(target).sum())
        pred = source[:, index]
        error = float(np.square(pred - target).sum())
        other = [float(np.square(source[:, np.array(order)] - target).sum()) for order in permutations(range(4)) if tuple(order) != tuple(index)]
        results.append({'word': word, 'true_task_block_order': index.tolist(),
            'block_pair_action_displacement_nmse': error / denominator,
            'block_pair_target_nmse': error / target_norm,
            'wrong_block_permutation_mean_displacement_nmse': float(np.mean(other)) / denominator,
            'correct_block_permutation_rank_among_24': 1 + sum(value < error - 1e-9 for value in other),
            'scope': 'Task block identities are provided mathematically; no fitting. Ranking against wrong block orders is a diagnostic, not test-set selection.'})
    return results


def initialize():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'evaluations').mkdir(exist_ok=True)
    signature = {'code_sha256': sha(__file__), 'joint_test_protocol_sha256': sha(JOINT / 'protocol.json'),
        'new_ordinary_source_protocol_sha256': sha(NEW / 'protocol.json'),
        'scope': 'Exploratory control after original records-only paired outcomes. Checks whether four task-query reordering explains conditional geometry; no new training or composite-map fitting.'}
    path = ROOT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())['signature'] == signature
    else:
        atomic_json(path, {'registered_utc': datetime.now(timezone.utc).isoformat(), 'signature': signature, 'new_block_control_values': 0})


def evaluate_available():
    initialize()
    data = dict(np.load(JOINT / 'dataset.npz'))
    candidates = []
    for path in (JOINT / 'evaluations').glob('novel_records_s*.json'):
        record = json.loads(path.read_text())
        candidates.append((path.stem, JOINT / 'features' / f'{path.stem}.npz', PARENT / 'source/checkpoints' / f"novel_records_s{record['seed']}.pt", record['status'], record['seed']))
    for path in (NEW / 'evaluations').glob('novel_records_s*.json'):
        record = json.loads(path.read_text())
        candidates.append((path.stem, NEW / 'features' / f'{path.stem}_joint_test.npz', NEW / 'source/checkpoints' / f'{path.stem}.pt', 'trained', record['seed']))
    count = 0
    for name, feature, cp, status, seed in candidates:
        dest = ROOT / 'evaluations' / f'{name}.json'
        if dest.exists():
            count += 1; continue
        values = dict(np.load(feature))
        full = values['source_query_concat'].astype(np.float64)
        # Null-space projection is read from the independently saved pair
        # feature derivation: obtain the fixed readout of this same model.
        import torch
        if status == 'trained':
            weight = torch.load(cp, weights_only=True, map_location='cpu')['model']['lm_head.weight'][:31].numpy().astype(np.float64)
        else:
            from .algebra_structure_replication import load_plan
            from .longrun_transfer import make_model
            plan, config, _ = load_plan()
            model = make_model(config, plan['architecture'], seed, 'cpu')
            weight = model.lm_head.weight[:31].detach().numpy().astype(np.float64)
        basis, _ = contrast_basis(weight)
        block = full.reshape(*full.shape[:-1], 4, -1)
        null = (block - (block @ basis) @ basis.T).reshape(full.shape)
        results = {view: block_controls(hidden, data['pair_ids']) for view, hidden in [('source_query_concat', full), ('source_query_numeric_null', null)]}
        atomic_json(dest, {'source': name, 'seed': seed, 'status': status, 'feature_sha256': sha(feature), 'results': results,
            'reported_utc': datetime.now(timezone.utc).isoformat()})
        print(json.dumps({'block_control': name, 'numeric_null': results['source_query_numeric_null']}), flush=True)
        count += 1
    return count


if __name__ == '__main__':
    deadline = datetime(2026, 10, 6, 17, tzinfo=timezone.utc).timestamp()
    while time.time() < deadline:
        count = evaluate_available()
        if count == 9:
            break
        time.sleep(min(45, max(0, deadline - time.time())))
