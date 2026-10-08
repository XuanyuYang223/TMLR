"""Exploratory relation-specificity controls on frozen conditional predictions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .algebra_structure_replication import load_plan
from .native_source_subspace import contrast_basis
from .permworld_combinations import sha
from .representation_algebra import permutation_action_table, word_action


ROOT = Path('results/conditional_relation_specificity')
DEADLINE = datetime(2026, 10, 6, 17, tzinfo=timezone.utc).timestamp()
VIEWS = ['source_query_concat', 'source_query_numeric_null']
WORDS = ['c', 'r', 'i', 'rc', 'ci', 'ri', 'rci']


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize():
    ROOT.mkdir(exist_ok=True)
    (ROOT / 'evaluations').mkdir(exist_ok=True)
    signature = {'code_sha256': sha(__file__), 'data_sha256': sha('results/joint_answer_matched_geometry/dataset.npz'),
        'words': WORDS, 'views': VIEWS, 'controls': ['all_seven_wrong_group_actions', 'pair_derangement_within_length', 'ci_versus_ic'],
        'scope': 'Exploratory follow-up after the older three-source conditional means were seen. Uses only existing frozen predictions and no fitting or hyperparameter selection. Correct-action rank is diagnostic, not test-set action selection. New source endpoints are evaluated by the same fixed rules.'}
    protocol = ROOT / 'protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text())['signature'] == signature
    else:
        atomic_json(protocol, {'registered_utc': now(), 'signature': signature, 'new_specificity_evaluations': 0})


def evaluate(path):
    rec = json.loads(path.read_text())
    root = path.parent.parent
    new = root.name == 'ordinary_relation_seed_confirmation'
    name = rec['source']
    dest = ROOT / 'evaluations' / (name + '.json')
    if dest.exists():
        return
    data = dict(np.load('results/joint_answer_matched_geometry/dataset.npz'))
    feature_path = root / 'features' / (name + ('_joint_test.npz' if new else '.npz'))
    features = dict(np.load(feature_path))
    plan, config, _ = load_plan()
    if rec['status'] == 'trained':
        cp = root / 'source/checkpoints' / (name + '.pt') if new else Path('results/algebra_structure_replication/source/checkpoints') / f"{rec['group']}_s{rec['seed']}.pt"
        weight = torch.load(cp, weights_only=True, map_location='cpu')['model']['lm_head.weight'][:31].numpy().astype(np.float64)
    else:
        model = make_model(config, plan['architecture'], rec['seed'], 'cpu')
        weight = model.lm_head.weight[:31].detach().numpy().astype(np.float64)
    basis, _ = contrast_basis(weight)
    block = features['source_query_concat'].reshape(len(data['lengths']), 8, 4, -1).astype(np.float64)
    null = (block - (block @ basis) @ basis.T).reshape(features['source_query_concat'].shape)
    pairs = np.stack([np.flatnonzero(data['pair_ids'] == pair) for pair in np.unique(data['pair_ids'])])
    ns = data['lengths'][pairs[:, 0]]
    partner = np.arange(len(pairs))
    for n in np.unique(ns):
        ids = np.flatnonzero(ns == n)
        partner[ids] = np.roll(ids, 1)
    assert np.all(partner != np.arange(len(partner)))
    table, letters = permutation_action_table(), {'c': 1, 'r': 2, 'i': 4}
    rows, hashes = [], {}
    for view, hidden in [('source_query_concat', features['source_query_concat']), ('source_query_numeric_null', null)]:
        difference = (hidden[pairs[:, 0]] - hidden[pairs[:, 1]]).astype(np.float64)
        source = difference.reshape(-1, difference.shape[-1])
        ap = root / 'arrays' / (name + '_' + view + '_all_joint_answer_matched_pairs.npz')
        predictions = dict(np.load(ap)); hashes[str(ap)] = sha(ap)
        for word in WORDS:
            pred = predictions[word + '_pair_prediction'].astype(np.float64)
            action = word_action(word, table, letters)
            target = difference[:, table[:, action]].reshape(source.shape)
            norm = float(np.square(target).sum())
            errors = [float(np.square(pred - difference[:, table[:, candidate]].reshape(source.shape)).sum() / norm) for candidate in range(8)]
            correct = errors[action]
            wrong = [error for j, error in enumerate(errors) if j != action]
            shuffled = difference[partner][:, table[:, action]].reshape(source.shape)
            row = {'view': view, 'word': word, 'correct_action': action, 'pairs': len(pairs),
                'correct_pair_target_nmse': correct, 'wrong_action_target_nmse': errors,
                'correct_action_rank_among_8': 1 + sum(e < correct - 1e-12 for e in wrong),
                'best_wrong_minus_correct_nmse': min(wrong) - correct,
                'mean_wrong_minus_correct_nmse': float(np.mean(wrong) - correct),
                'length_matched_pair_derangement_nmse': float(np.square(pred - shuffled).sum() / norm),
                'pair_derangement_minus_correct_nmse': float((np.square(pred-shuffled).sum() - np.square(pred-target).sum())/norm)}
            if word == 'ci':
                wrong_order = word_action('ic', table, letters)
                row['IC_target_minus_CI_target_nmse'] = errors[wrong_order] - correct
            rows.append(row)
    atomic_json(dest, {'source': name, 'group': rec['group'], 'seed': rec['seed'], 'status': rec['status'],
        'cohort': 'additional_ordinary_seeds' if new else 'original_ordinary_seeds_new_conditional_test',
        'feature_sha256': sha(feature_path), 'prediction_archive_sha256': hashes, 'results': rows, 'completed_utc': now()})
    print(json.dumps({'specificity_evaluated': name, 'null_CI': next(r for r in rows if r['view'].endswith('numeric_null') and r['word'] == 'ci')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    initialize()
    torch.set_num_threads(2)
    while True:
        for location in ['joint_answer_matched_geometry', 'ordinary_relation_seed_confirmation']:
            for path in (Path('results') / location / 'evaluations').glob('*.json'):
                if time.time() >= DEADLINE:
                    break
                evaluate(path)
        if not args.watch or time.time() >= DEADLINE or len(list((ROOT / 'evaluations').glob('*.json'))) == 27:
            break
        time.sleep(min(45, max(0, DEADLINE - time.time())))
