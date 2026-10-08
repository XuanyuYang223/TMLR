"""Secondary task-free prefix geometry on the newly trained ordinary models."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from .algebra_structure_replication import load_plan
from .algebra_structure_direct import direct_probe
from .answer_matched_geometry import pair_geometry
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .representation_algebra import permutation_action_table


ROOT = Path('results/ordinary_prefix_confirmation')
DEADLINE = datetime(2026, 10, 6, 17, tzinfo=timezone.utc).timestamp()


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize():
    ROOT.mkdir(exist_ok=True)
    for folder in ['evaluations', 'maps', 'arrays']:
        (ROOT / folder).mkdir(exist_ok=True)
    signature = {'code_sha256': sha(__file__), 'test_data_sha256': sha('results/joint_answer_matched_geometry/dataset.npz'),
        'calibration_data_sha256': sha('results/algebra_structure_replication/probe_dataset.npz'),
        'core_sha256': {p: sha(p) for p in ['experiments/algebra_structure_direct.py', 'experiments/answer_matched_geometry.py',
            'experiments/representation_algebra.py', 'experiments/algebra_structure_replication.py']},
        'scope': 'Secondary boundary check registered after old prefix failures and five additional query-geometry outcomes. New prefix outcomes have not been inspected. Uses existing trained and seed-matched initialization features at the task-free ONE_END position, before task prompts. Same complete-orbit calibration fitting and validation as query probes, fixed 64-dimensional probe and ridge grid; composites are matrix products without compound fitting. All groups and statuses are reported on all joint answer-matched pairs, no selected subset. No new network training.'}
    path = ROOT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())['signature'] == signature
    else:
        atomic_json(path, {'registered_utc': now(), 'signature': signature, 'new_prefix_evaluations': 0})


def evaluate(path):
    rec = json.loads(path.read_text())
    root = path.parent.parent
    dest = ROOT / 'evaluations' / (rec['source'] + '.json')
    if dest.exists():
        return
    plan, _, _ = load_plan()
    cal_path = root / 'features' / (rec['source'] + '_calibration.npz')
    test_path = root / 'features' / (rec['source'] + '_joint_test.npz')
    cal = np.load(cal_path)['ONE_END']
    hidden = np.load(test_path)['ONE_END']
    calibration = dict(np.load('results/algebra_structure_replication/probe_dataset.npz'))
    data = dict(np.load('results/joint_answer_matched_geometry/dataset.npz'))
    fitting, maps = direct_probe(cal, calibration['split'], calibration['lengths'], permutation_action_table(),
        {'c': 1, 'r': 2, 'i': 4}, [('r', 'c'), ('c', 'i'), ('r', 'i'), ('r', 'c', 'i')],
        plan['probe_dimension'], plan['ridge_grid'], rec['seed'] + 9200)
    mp = ROOT / 'maps' / (rec['source'] + '.npz')
    np.savez_compressed(mp, **maps)
    rows, predictions = pair_geometry(hidden, data['pair_ids'], maps, ['c', 'r', 'i', 'rc', 'ci', 'ri', 'rci'])
    ap = ROOT / 'arrays' / (rec['source'] + '.npz')
    np.savez_compressed(ap, **predictions)
    atomic_json(dest, {'source': rec['source'], 'seed': rec['seed'], 'group': rec['group'], 'status': rec['status'],
        'source_record_path': str(path), 'source_record_sha256': sha(path),
        'calibration_feature_path': str(cal_path), 'calibration_feature_sha256': sha(cal_path),
        'test_feature_path': str(test_path), 'test_feature_sha256': sha(test_path),
        'map_path': str(mp), 'map_sha256': sha(mp), 'prediction_archive_sha256': sha(ap),
        'calibration_results': fitting, 'results': rows, 'completed_utc': now()})
    print({'prefix_evaluated': rec['source'], 'composite_target_nmse': np.mean([r['pair_target_nmse'] for r in rows if len(r['word']) > 1])}, flush=True)


if __name__ == '__main__':
    initialize()
    while time.time() < DEADLINE:
        for folder in ['ordinary_relation_seed_confirmation', 'ordinary_initialization_control']:
            for path in (Path('results') / folder / 'evaluations').glob('*.json'):
                if time.time() < DEADLINE:
                    evaluate(path)
        count = len(list((ROOT / 'evaluations').glob('*.json')))
        atomic_json(ROOT / 'state.json', {'status': 'complete' if count == 18 else 'waiting_or_evaluating', 'sources': count, 'updated_utc': now()})
        if count == 18:
            break
        time.sleep(min(45, max(0, DEADLINE-time.time())))
