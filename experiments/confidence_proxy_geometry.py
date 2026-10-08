"""Output-confidence-only hidden proxy, calibrated without conditional-test targets."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from .algebra_structure_direct import direct_probe
from .algebra_structure_replication import load_plan
from .confidence_residual_geometry import null_and_code
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .representation_algebra import permutation_action_table, word_action


ROOT = Path('results/confidence_proxy_geometry')
DEADLINE = datetime(2026, 10, 6, 17, tzinfo=timezone.utc).timestamp()


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize():
    ROOT.mkdir(exist_ok=True)
    for name in ['evaluations', 'maps', 'arrays']:
        (ROOT / name).mkdir(exist_ok=True)
    signature = {'code_sha256': sha(__file__), 'test_data_sha256': sha('results/joint_answer_matched_geometry/dataset.npz'),
        'calibration_data_sha256': sha('results/algebra_structure_replication/probe_dataset.npz'),
        'parent_nuisance_protocol_sha256': sha('results/confidence_residual_geometry/protocol.json'),
        'core_sha256': {p: sha(p) for p in ['experiments/confidence_residual_geometry.py', 'experiments/algebra_structure_direct.py',
            'experiments/algebra_structure_replication.py', 'experiments/representation_algebra.py']},
        'scope': 'Exploratory direct-surrogate check after the residual-confidence outcomes were inspected. Reuse the known-only nuisance decoder, construct a hidden proxy using only four tasks numeric output/confidence features and length, and fit generator maps on calibration proxies. No conditional-test target hidden vector or true label enters decoder or generator fitting/selection. Predictions start from the output-only proxy at the current input; original numeric-null target vectors are used only for scoring. All groups and trained/initialization cases are reported. Decoder training does use calibration hidden vectors, so this is a learned output surrogate, not a representation constructed without any hidden-vector access. Word rules and generator fitting are supplied by the framework.'}
    path = ROOT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())['signature'] == signature
    else:
        atomic_json(path, {'registered_utc': now(), 'signature': signature, 'new_proxy_evaluations': 0})


def evaluate(path):
    parent = json.loads(path.read_text())
    dest = ROOT / 'evaluations' / (parent['source'] + '.json')
    if dest.exists():
        return
    plan, _, _ = load_plan()
    cal = dict(np.load('results/algebra_structure_replication/probe_dataset.npz'))
    test = dict(np.load('results/joint_answer_matched_geometry/dataset.npz'))
    fit = dict(np.load(parent['nuisance_fit_path']))
    cal_hidden, test_hidden = [np.load(p)['source_query_concat'] for p in parent['source_feature_paths']]
    _, cal_code = null_and_code(cal_hidden, cal['lengths'], fit['numeric_weight'])
    true_null, test_code = null_and_code(test_hidden, test['lengths'], fit['numeric_weight'])
    def proxy(code):
        return ((code - fit['code_mean']) / fit['code_sigma']) @ fit['weight'] + fit['bias']
    cal_proxy, test_proxy = proxy(cal_code), proxy(test_code)
    fitting, maps = direct_probe(cal_proxy, cal['split'], cal['lengths'], permutation_action_table(),
        {'c': 1, 'r': 2, 'i': 4}, [('r', 'c'), ('c', 'i'), ('r', 'i'), ('r', 'c', 'i')],
        plan['probe_dimension'], plan['ridge_grid'], parent['seed'] + 9200)
    mp = ROOT / 'maps' / (parent['source'] + '.npz')
    np.savez_compressed(mp, **maps)
    pairs = np.stack([np.flatnonzero(test['pair_ids'] == pair) for pair in np.unique(test['pair_ids'])])
    surrogate_difference = test_proxy[pairs[:, 0]] - test_proxy[pairs[:, 1]]
    true_difference = true_null[pairs[:, 0]] - true_null[pairs[:, 1]]
    q = maps['basis'].astype(np.float64)
    source = surrogate_difference.reshape(-1, q.shape[0])
    latent = source @ q
    table, letters = permutation_action_table(), {'c': 1, 'r': 2, 'i': 4}
    rows, saved = [], {'pair_ids': np.unique(test['pair_ids'])}
    for word in ['c', 'r', 'i', 'rc', 'ci', 'ri', 'rci']:
        product = np.eye(q.shape[1])
        for g in word:
            product = product @ maps['map_' + g].astype(np.float64)
        predicted_latent = latent @ product
        prediction = source + (predicted_latent - latent) @ q.T
        action = word_action(word, table, letters)
        target = true_difference[:, table[:, action]].reshape(source.shape)
        original_source = true_difference.reshape(source.shape)
        error = np.square(prediction - target).sum()
        rows.append({'word': word, 'pairs': len(pairs), 'target': 'original_numeric_null_pair_difference',
            'pair_target_nmse': float(error / np.square(target).sum()),
            'pair_action_displacement_nmse': float(error / np.square(target - original_source).sum())})
        saved[word + '_pair_latent_prediction'] = predicted_latent.astype(np.float32)
    ap = ROOT / 'arrays' / (parent['source'] + '.npz')
    np.savez_compressed(ap, **saved)
    atomic_json(dest, {'source': parent['source'], 'group': parent['group'], 'seed': parent['seed'], 'status': parent['status'],
        'parent_record_path': str(path), 'parent_record_sha256': sha(path), 'map_path': str(mp), 'map_sha256': sha(mp),
        'calibration_results': fitting, 'prediction_archive_sha256': sha(ap), 'results': rows, 'completed_utc': now()})
    print({'confidence_proxy_evaluated': parent['source'], 'composite_original_target_nmse':
        np.mean([r['pair_target_nmse'] for r in rows if len(r['word']) > 1])}, flush=True)


if __name__ == '__main__':
    initialize()
    while time.time() < DEADLINE:
        for path in Path('results/confidence_residual_geometry/evaluations').glob('*.json'):
            if time.time() < DEADLINE:
                evaluate(path)
        count = len(list((ROOT / 'evaluations').glob('*.json')))
        atomic_json(ROOT / 'state.json', {'status': 'complete' if count == 18 else 'waiting_or_evaluating', 'sources': count, 'updated_utc': now()})
        if count == 18:
            break
        time.sleep(min(45, max(0, DEADLINE-time.time())))
