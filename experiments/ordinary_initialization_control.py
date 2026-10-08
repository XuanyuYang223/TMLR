"""Seed-matched untrained networks for the additional ordinary source models."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from .algebra_structure_replication import load_plan
from .answer_matched_geometry import extract_final, pair_geometry
from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .algebra_structure_direct import direct_probe
from .native_confirmation import setup
from .native_source_subspace import contrast_basis
from .permworld_combinations import sha
from .representation_algebra import permutation_action_table


ROOT = Path('results/ordinary_initialization_control')
DEADLINE = datetime(2026, 10, 6, 17, tzinfo=timezone.utc).timestamp()


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize():
    campaign = json.loads(Path('configs/ordinary_relation_seed_confirmation.json').read_text())
    plan, config, _ = load_plan()
    ROOT.mkdir(exist_ok=True)
    for name in ['features', 'maps', 'arrays', 'evaluations']:
        (ROOT / name).mkdir(exist_ok=True)
    signature = {'code_sha256': sha(__file__), 'trained_campaign_protocol_sha256': sha(Path(campaign['output']) / 'protocol.json'),
        'test_data_sha256': sha('results/joint_answer_matched_geometry/dataset.npz'),
        'calibration_sha256': sha('results/algebra_structure_replication/probe_dataset.npz'),
        'core_sha256': {p: sha(p) for p in ['experiments/answer_matched_geometry.py', 'experiments/algebra_structure_direct.py', 'experiments/longrun_transfer.py']},
        'source_seeds': campaign['source_seeds'], 'groups': campaign['groups'], 'views': campaign['views'],
        'scope': 'Seed-matched initial networks, zero network training labels. Same complete-orbit calibration fit/validation and conditional test as the trained ordinary models. Fit generators separately on calibration and multiply them for unseen compositions; no test-set fitting. Registered before the new ordinary models start.'}
    path = ROOT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())['signature'] == signature
    else:
        atomic_json(path, {'registered_utc': now(), 'signature': signature, 'new_initialization_evaluations': 0})
    return campaign, plan, config


def evaluate(campaign, plan, config, group, seed, tokens, device):
    name = f"{group['id']}_s{seed}_random"
    dest = ROOT / 'evaluations' / (name + '.json')
    if dest.exists():
        return
    model = make_model(config, plan['architecture'], seed, device)
    calibration = dict(np.load('results/algebra_structure_replication/probe_dataset.npz'))
    test = dict(np.load('results/joint_answer_matched_geometry/dataset.npz'))
    features = []
    for tag, data in [('calibration', calibration), ('joint_test', test)]:
        path = ROOT / 'features' / (name + '_' + tag + '.npz')
        if path.exists():
            value = dict(np.load(path))
        else:
            value = extract_final(model, data, group['tasks'], tokens, 24)
            np.savez_compressed(path, **value)
        features.append(value)
    weight = model.lm_head.weight[:31].detach().cpu().numpy().astype(np.float64)
    head_bias = model.lm_head.bias
    bias = np.zeros(31) if head_bias is None else head_bias[:31].detach().cpu().numpy().astype(np.float64)
    basis, _ = contrast_basis(weight)
    def null_view(value):
        full = value['source_query_concat'].astype(np.float64)
        block = full.reshape(*full.shape[:-1], 4, -1)
        return (block - (block @ basis) @ basis.T).reshape(full.shape)
    cal_views = {'source_query_concat': features[0]['source_query_concat'], 'source_query_numeric_null': null_view(features[0])}
    test_views = {'source_query_concat': features[1]['source_query_concat'], 'source_query_numeric_null': null_view(features[1])}
    union = json.loads(Path('results/joint_answer_matched_geometry/dataset_audit.json').read_text())['union_tasks']
    truth = test['union_labels'][:, :, [union.index(task) for task in group['tasks']]]
    blocks = features[1]['source_query_concat'].reshape(len(test['lengths']), 8, 4, -1)
    answers = (blocks @ weight.T + bias).argmax(-1)
    pairs = np.stack([np.flatnonzero(test['pair_ids'] == pair) for pair in np.unique(test['pair_ids'])])
    agreement = (answers[pairs[:, 0]] == answers[pairs[:, 1]]).all((1, 2))
    correct = (answers[pairs] == truth[pairs]).all((1, 2, 3))
    rows, hashes, map_metadata = [], {}, {}
    for view in campaign['views']:
        fitting, maps = direct_probe(cal_views[view], calibration['split'], calibration['lengths'], permutation_action_table(),
            {'c': 1, 'r': 2, 'i': 4}, [('r', 'c'), ('c', 'i'), ('r', 'i'), ('r', 'c', 'i')], plan['probe_dimension'], plan['ridge_grid'], seed + 9200)
        mp = ROOT / 'maps' / (name + '_' + view + '.npz')
        np.savez_compressed(mp, **maps)
        map_metadata[view] = {'map_sha256': sha(mp), 'known_input_calibration_results': fitting}
        for subset_name, subset in [('all_joint_answer_matched_pairs', None), ('model_answers_agree_all_states', agreement), ('model_answers_correct_all_states', correct)]:
            if subset is not None and not subset.any():
                continue
            outcomes, arrays = pair_geometry(test_views[view], test['pair_ids'], maps, ['c', 'r', 'i', 'rc', 'ci', 'ri', 'rci'], subset)
            selected = pairs if subset is None else pairs[subset]
            delta = test_views[view][selected[:, 0]] - test_views[view][selected[:, 1]]
            captured = float(np.square(delta @ maps['basis'].astype(np.float64)).sum() / np.square(delta).sum())
            for row in outcomes:
                row.update(view=view, subset=subset_name, paired_projection_energy_fraction=captured)
            rows.extend(outcomes)
            ap = ROOT / 'arrays' / (name + '_' + view + '_' + subset_name + '.npz')
            np.savez_compressed(ap, **arrays); hashes[ap.name] = sha(ap)
    atomic_json(dest, {'source': name, 'group': group['id'], 'seed': seed, 'status': 'random',
        'conditional_query_accuracy': float(np.mean(answers == truth)), 'same_model_answer_pairs': int(agreement.sum()),
        'all_orbit_correct_pairs': int(correct.sum()), 'source_label_exposures': 0, 'results': rows, 'map_metadata': map_metadata,
        'array_sha256': hashes, 'test_feature_sha256': sha(ROOT / 'features' / (name + '_joint_test.npz')), 'completed_utc': now()})
    print(json.dumps({'ordinary_initialization_evaluated': name}), flush=True)
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()


if __name__ == '__main__':
    campaign, plan, config = initialize()
    torch.set_num_threads(2)
    _, _, tokens, _ = setup(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    while time.time() < DEADLINE:
        count = 0
        for seed in campaign['source_seeds']:
            for group in plan['groups']:
                if group['id'] not in campaign['groups']:
                    continue
                name = f"{group['id']}_s{seed}"
                if (Path(campaign['output']) / 'source' / (name + '.json')).exists() and time.time() < DEADLINE:
                    evaluate(campaign, plan, config, group, seed, tokens, device)
                count += (ROOT / 'evaluations' / (name + '_random.json')).exists()
        atomic_json(ROOT / 'state.json', {'status': 'complete' if count == 9 else 'waiting_for_completed_trained_source', 'completed': count, 'updated_utc': now()})
        if count == 9:
            break
        time.sleep(min(45, max(0, DEADLINE-time.time())))
