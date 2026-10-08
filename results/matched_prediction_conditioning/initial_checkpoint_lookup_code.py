"""Score trained and initialization probes on identical answer-selected pairs."""
from datetime import datetime, timezone
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


ROOT = Path('results/matched_prediction_conditioning')
JOINT = Path('results/joint_answer_matched_geometry')
DEADLINE = datetime(2026, 10, 6, 17, tzinfo=timezone.utc).timestamp()
WORDS = ['c', 'r', 'i', 'rc', 'ci', 'ri', 'rci']
VIEWS = ['source_query_concat', 'source_query_numeric_null']


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize():
    ROOT.mkdir(exist_ok=True)
    (ROOT / 'evaluations').mkdir(exist_ok=True)
    signature = {'code_sha256': sha(__file__), 'test_data_sha256': sha(JOINT / 'dataset.npz'),
        'core_sha256': {p: sha(p) for p in ['experiments/longrun_transfer.py', 'experiments/algebra_structure_replication.py',
            'experiments/native_source_subspace.py', 'experiments/representation_algebra.py']},
        'words': WORDS, 'views': VIEWS, 'deadline_utc': '2026-10-06T17:00:00+00:00',
        'scope': 'Exploratory control after old ordinary outcomes and the first new records/minima sources were inspected. A trained model selects pairs with equal discrete predicted answers across all eight states, or correct answers for both complete orbits. The same selected pair IDs are scored for its trained probe and same-seed initialization probe. No source or operator refitting, and no selection by geometry error. Saved float32 pair predictions are scored as stored. Pair selection is conditional on the trained model and its behavior; this is not an unbiased primary test. Confidence and input-answer interactions remain uncontrolled. All groups, seeds and selection rules are reported, including empty subsets.'}
    path = ROOT / 'protocol.json'
    import json
    if path.exists():
        assert json.loads(path.read_text())['signature'] == signature
    else:
        atomic_json(path, {'registered_utc': now(), 'signature': signature, 'new_matched_conditioning_evaluations': 0})


def read(path):
    import json
    return json.loads(path.read_text())


def heads(rec, root):
    plan, config, _ = load_plan()
    if rec['status'] == 'trained':
        cp = root / 'source/checkpoints' / (rec['source'] + '.pt') if root.name == 'ordinary_relation_seed_confirmation' else Path('results/algebra_structure_replication/source/checkpoints') / (rec['source'] + '.pt')
        state = torch.load(cp, map_location='cpu', weights_only=True)['model']
        return state['lm_head.weight'][:31].numpy().astype(np.float64)
    model = make_model(config, plan['architecture'], rec['seed'], 'cpu')
    return model.lm_head.weight[:31].detach().numpy().astype(np.float64)


def evaluate(trained_path, random_path):
    trained, random = read(trained_path), read(random_path)
    dest = ROOT / 'evaluations' / (trained['source'] + '.json')
    if dest.exists():
        return
    assert trained['seed'] == random['seed'] and trained['group'] == random['group']
    data = dict(np.load(JOINT / 'dataset.npz'))
    pairs = np.stack([np.flatnonzero(data['pair_ids'] == pair) for pair in np.unique(data['pair_ids'])])
    plan, _, _ = load_plan()
    tasks = next(g['tasks'] for g in plan['groups'] if g['id'] == trained['group'])
    union = read(JOINT / 'dataset_audit.json')['union_tasks']
    truth = data['union_labels'][:, :, [union.index(task) for task in tasks]]
    cached, hashes = {}, {}
    for rec, path in [(trained, trained_path), (random, random_path)]:
        root = path.parent.parent
        new = root.name in ['ordinary_relation_seed_confirmation', 'ordinary_initialization_control']
        fp = root / 'features' / (rec['source'] + ('_joint_test.npz' if new else '.npz'))
        features = dict(np.load(fp))['source_query_concat']
        weight = heads(rec, root)
        blocks = features.reshape(len(data['lengths']), 8, 4, -1).astype(np.float64)
        basis, _ = contrast_basis(weight)
        null = (blocks - (blocks @ basis) @ basis.T).reshape(features.shape)
        cached[rec['status']] = {'rec': rec, 'root': root, 'features': features,
            'null': null, 'predictions': (blocks @ weight.T).argmax(-1)}
        hashes[str(fp)] = sha(fp)
        hashes[str(path)] = sha(path)
    answer = cached['trained']['predictions']
    masks = {'trained_model_answers_agree_all_states': (answer[pairs[:, 0]] == answer[pairs[:, 1]]).all(axis=(1, 2)),
        'trained_model_answers_correct_all_states': (answer[pairs] == truth[pairs]).all(axis=(1, 2, 3))}
    selections = {name: {'pairs': int(mask.sum()), 'pair_ids': np.unique(data['pair_ids'])[mask].tolist(),
        'truth_used_for_selection': name == 'trained_model_answers_correct_all_states'} for name, mask in masks.items()}
    rows = []
    table, letters = permutation_action_table(), {'c': 1, 'r': 2, 'i': 4}
    for status, entry in cached.items():
        for view in VIEWS:
            hidden = entry['null'] if view.endswith('numeric_null') else entry['features']
            difference = (hidden[pairs[:, 0]] - hidden[pairs[:, 1]]).astype(np.float64)
            ap = entry['root'] / 'arrays' / (entry['rec']['source'] + '_' + view + '_all_joint_answer_matched_pairs.npz')
            saved = dict(np.load(ap)); hashes[str(ap)] = sha(ap)
            np.testing.assert_array_equal(saved['pair_ids'], np.unique(data['pair_ids']))
            for selector, mask in masks.items():
                if not mask.any():
                    continue
                source = difference[mask].reshape(-1, difference.shape[-1])
                for word in WORDS:
                    action = word_action(word, table, letters)
                    target = difference[mask][:, table[:, action]].reshape(source.shape)
                    prediction = saved[word + '_pair_prediction'].reshape(difference.shape)[mask].reshape(source.shape).astype(np.float64)
                    error = np.square(prediction - target).sum()
                    rows.append({'selector': selector, 'status': status, 'view': view, 'word': word, 'pairs': int(mask.sum()),
                        'pair_target_nmse': float(error / np.square(target).sum()),
                        'pair_action_displacement_nmse': float(error / np.square(target - source).sum())})
    atomic_json(dest, {'source': trained['source'], 'group': trained['group'], 'seed': trained['seed'],
        'cohort': 'new_source_seeds' if trained_path.parent.parent.name == 'ordinary_relation_seed_confirmation' else 'old_source_seeds',
        'trained_record_path': str(trained_path), 'random_record_path': str(random_path),
        'selections': selections, 'results': rows, 'artifact_sha256': hashes, 'completed_utc': now()})
    print({'matched_answer_conditioning': trained['source'], 'pair_counts': {k: v['pairs'] for k, v in selections.items()}}, flush=True)


if __name__ == '__main__':
    initialize()
    torch.set_num_threads(2)
    while time.time() < DEADLINE:
        for folder, random_folder in [(JOINT, JOINT), (Path('results/ordinary_relation_seed_confirmation'), Path('results/ordinary_initialization_control'))]:
            for path in (folder / 'evaluations').glob('*.json'):
                rec = read(path)
                if rec['status'] != 'trained':
                    continue
                rp = random_folder / 'evaluations' / ((rec['source'].removesuffix('_trained') if folder == JOINT else rec['source']) + '_random.json')
                if rp.exists() and time.time() < DEADLINE:
                    evaluate(path, rp)
        count = len(list((ROOT / 'evaluations').glob('*.json')))
        atomic_json(ROOT / 'state.json', {'status': 'complete' if count == 18 else 'waiting_or_evaluating', 'paired_sources': count, 'updated_utc': now()})
        if count == 18:
            break
        time.sleep(min(45, max(0, DEADLINE - time.time())))
