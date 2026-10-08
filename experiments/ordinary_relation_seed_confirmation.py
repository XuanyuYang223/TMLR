"""Fresh ordinary source seeds for answer-controlled group geometry."""
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from .algebra_structure_direct import direct_probe
from .algebra_structure_replication import load_plan
from .answer_matched_geometry import extract_final, pair_geometry
from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .native_ablation import train_one
from .native_confirmation import setup
from .native_source_subspace import contrast_basis
from .permworld_combinations import sha
from .representation_algebra import permutation_action_table


CONFIG = 'configs/ordinary_relation_seed_confirmation.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize():
    campaign = json.loads(Path(CONFIG).read_text())
    original, config, _ = load_plan()
    root = Path(campaign['output'])
    root.mkdir(parents=True, exist_ok=True)
    for folder in ['source/checkpoints', 'dataset', 'features', 'maps', 'evaluations', 'arrays']:
        (root / folder).mkdir(parents=True, exist_ok=True)
    source_link = root / 'dataset/data.npz'
    if not source_link.exists():
        source_link.symlink_to(Path(original['source_data']).resolve())
    signature = {'code_sha256': sha(__file__), 'config_sha256': sha(CONFIG),
        'original_train_code_sha256': sha('experiments/native_ablation.py'),
        'parent_protocol_sha256': sha(Path(campaign['parent']) / 'protocol.json'),
        'source_data_sha256': sha(source_link), 'campaign': campaign,
        'calibration_data_sha256': sha(Path(campaign['parent']) / 'probe_dataset.npz'),
        'joint_test_protocol_sha256': sha(Path(campaign['joint_test']) / 'protocol.json')}
    dest = root / 'protocol.json'
    if dest.exists():
        assert json.loads(dest.read_text())['signature'] == signature
    else:
        atomic_json(dest, {'registered_utc': now(), 'signature': signature, 'new_ordinary_sources': 0})
    return campaign, original, config, root


def evaluate_one(campaign, plan, config, root, group, seed, tokens, device):
    name = f'{group["id"]}_s{seed}'
    dest = root / 'evaluations' / f'{name}.json'
    if dest.exists():
        return
    cp = root / 'source/checkpoints' / f'{name}.pt'
    record = json.loads((root / 'source' / f'{name}.json').read_text())
    assert record['status'] == 'complete' and sha(cp) == record['checkpoint_sha256']
    model = make_model(config, plan['architecture'], seed, device)
    model.load_state_dict(torch.load(cp, weights_only=True, map_location=device)['model'])
    calibration = dict(np.load(Path(campaign['parent']) / 'probe_dataset.npz'))
    test = dict(np.load(Path(campaign['joint_test']) / 'dataset.npz'))
    fs = []
    for tag, data in [('calibration', calibration), ('joint_test', test)]:
        path = root / 'features' / f'{name}_{tag}.npz'
        if path.exists():
            values = dict(np.load(path))
        else:
            values = extract_final(model, data, group['tasks'], tokens, 24)
            np.savez_compressed(path, **values)
        fs.append(values)
    weight = model.lm_head.weight[:31].detach().cpu().numpy().astype(np.float64)
    bias_head = model.lm_head.bias
    bias = np.zeros(31) if bias_head is None else bias_head[:31].detach().cpu().numpy().astype(np.float64)
    basis, _ = contrast_basis(weight)
    def null_view(features):
        full = features['source_query_concat'].astype(np.float64)
        block = full.reshape(*full.shape[:-1], 4, -1)
        return (block - (block @ basis) @ basis.T).reshape(full.shape)
    cal_views = {'source_query_concat': fs[0]['source_query_concat'], 'source_query_numeric_null': null_view(fs[0])}
    test_views = {'source_query_concat': fs[1]['source_query_concat'], 'source_query_numeric_null': null_view(fs[1])}
    union = json.loads((Path(campaign['joint_test']) / 'dataset_audit.json').read_text())['union_tasks']
    truth = test['union_labels'][:, :, [union.index(task) for task in group['tasks']]]
    blocks = fs[1]['source_query_concat'].reshape(len(test['lengths']), 8, 4, -1)
    predictions = (blocks @ weight.T + bias).argmax(-1)
    pair_rows = np.stack([np.flatnonzero(test['pair_ids'] == pair) for pair in np.unique(test['pair_ids'])])
    agreement = (predictions[pair_rows[:, 0]] == predictions[pair_rows[:, 1]]).all(axis=(1, 2))
    correct = (predictions[pair_rows] == truth[pair_rows]).all(axis=(1, 2, 3))
    results, hashes, map_metadata = [], {}, {}
    for view in campaign['views']:
        fit_result, maps = direct_probe(cal_views[view], calibration['split'], calibration['lengths'], permutation_action_table(),
            {'c': 1, 'r': 2, 'i': 4}, [('r', 'c'), ('c', 'i'), ('r', 'i'), ('r', 'c', 'i')], plan['probe_dimension'], plan['ridge_grid'], seed + 9200)
        mp = root / 'maps' / f'{name}_{view}.npz'
        np.savez_compressed(mp, **maps)
        map_metadata[view] = {'map_sha256': sha(mp), 'known_input_calibration_results': fit_result}
        for subset_name, subset in [('all_joint_answer_matched_pairs', None), ('model_answers_agree_all_states', agreement), ('model_answers_correct_all_states', correct)]:
            if subset is not None and subset.sum() == 0:
                continue
            rows, arrays = pair_geometry(test_views[view], test['pair_ids'], maps, ['c', 'r', 'i', 'rc', 'ci', 'ri', 'rci'], subset)
            selected = pair_rows if subset is None else pair_rows[subset]
            delta = test_views[view][selected[:, 0]] - test_views[view][selected[:, 1]]
            projection_fraction = float(np.square(delta @ maps['basis'].astype(np.float64)).sum() / np.square(delta).sum())
            for row in rows:
                row.update(view=view, subset=subset_name, paired_projection_energy_fraction=projection_fraction)
            results.extend(rows)
            ap = root / 'arrays' / f'{name}_{view}_{subset_name}.npz'
            np.savez_compressed(ap, **arrays); hashes[ap.name] = sha(ap)
    atomic_json(dest, {'source': name, 'group': group['id'], 'seed': seed, 'status': 'trained',
        'source_audit_accuracy': float(np.mean([r['accuracy'] for r in record['source_audit']])),
        'conditional_query_accuracy': float(np.mean(predictions == truth)), 'same_model_answer_pairs': int(agreement.sum()),
        'all_orbit_correct_pairs': int(correct.sum()), 'results': results, 'map_metadata': map_metadata, 'array_sha256': hashes,
        'source_checkpoint_sha256': sha(cp), 'calibration_feature_sha256': sha(root / 'features' / f'{name}_calibration.npz'),
        'test_feature_sha256': sha(root / 'features' / f'{name}_joint_test.npz'), 'completed_utc': now()})
    print(json.dumps({'ordinary_seed_confirmed': name, 'primary': [r for r in results if r['view'] == campaign['primary_view'] and r['subset'] == 'all_joint_answer_matched_pairs']}), flush=True)
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()


def run():
    campaign, original, config, root = initialize()
    deadline = datetime.fromisoformat(campaign['deadline_utc']).timestamp()
    waiting = Path(campaign['wait_for_world_confirmation']) / 'state.json'
    while time.time() < deadline and json.loads(waiting.read_text())['status'] not in ['complete', 'stopped_at_deadline', 'stopped_at_deadline_before_training']:
        atomic_json(root / 'state.json', {'status': 'waiting_for_world_confirmation', 'updated_utc': now()})
        time.sleep(min(45, deadline - time.time()))
    if time.time() >= deadline:
        atomic_json(root / 'state.json', {'status': 'stopped_at_deadline_before_training', 'updated_utc': now()})
        return
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    names, _, tokens, _ = setup(config)
    raw = dict(np.load(root / 'dataset/data.npz'))
    data = {key: torch.as_tensor(value, device=device) for key, value in raw.items()}
    plan = {**original, 'output': str(root), 'source_seeds': campaign['source_seeds']}
    def timeout(*_):
        raise TimeoutError('10:00 deadline reached')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
    try:
        for seed in campaign['source_seeds']:
            for group in original['groups']:
                if group['id'] not in campaign['groups']:
                    continue
                job = {**group, 'seed': seed, 'batch': original['examples_per_task_per_step'], 'steps': campaign['source_steps'], 'kind': 'new'}
                atomic_json(root / 'state.json', {'status': 'source_training', 'source': f'{group["id"]}_s{seed}', 'updated_utc': now()})
                train_one(plan, config, root, job, raw, data, names, tokens, device)
                evaluate_one(campaign, plan, config, root, group, seed, tokens, device)
        atomic_json(root / 'state.json', {'status': 'complete', 'sources': 9, 'updated_utc': now()})
    except TimeoutError:
        atomic_json(root / 'state.json', {'status': 'stopped_at_deadline', 'updated_utc': now()})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    run()
