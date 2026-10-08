"""Answer-conditioned pair differences and prior-free readout diagnostics."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from .longrun_engine import atomic_json
from .permworld_combinations import sha


ROOT = Path('results/paired_relation_geometry')
OBSERVED = Path('results/relation_observed_start')
PARENT = Path('results/algebra_hidden_relations')
EXTENSION = Path('results/relation_seed_extension')


def exact_answer_code_ceiling(labels, lengths, target=5):
    counts = defaultdict(Counter)
    for ys, n in zip(labels, lengths):
        key = (int(n), *map(int, ys[[0, 1, 4]]))
        counts[key][int(ys[target])] += 1
    return sum(max(group.values()) for group in counts.values()) / len(labels)


def paired_metrics(prediction, start_hidden, target_hidden, labels, pair_ids, weight, bias):
    rows = np.stack([np.flatnonzero(pair_ids == pair) for pair in np.unique(pair_ids)])
    assert rows.shape[1] == 2
    first, second = rows[:, 0], rows[:, 1]
    pd = prediction[first] - prediction[second]
    td = target_hidden[first] - target_hidden[second]
    sd = start_hidden[first] - start_hidden[second]
    assert np.all(labels[first] != labels[second])
    error, norm = float(np.square(pd - td).sum()), float(np.square(td).sum())
    delta_norm = float(np.square(td - sd).sum())
    logits = prediction @ weight.T + bias
    a, b = labels[first], labels[second]
    signed = logits[first, a] - logits[first, b] - logits[second, a] + logits[second, b]
    assignment = np.where(signed > 1e-10, 1., np.where(signed < -1e-10, 0., .5))
    geometry_dot = np.sum(pd * td, axis=-1)
    geometry_assignment = np.where(geometry_dot > 1e-10, 1., np.where(geometry_dot < -1e-10, 0., .5))
    return {'pairs': len(rows), 'pair_target_nmse': error / norm,
        'pair_action_displacement_nmse': error / delta_norm,
        'zero_pair_prediction_target_nmse': 1., 'identity_pair_action_displacement_nmse': 1.,
        'pair_geometric_assignment_accuracy': float(geometry_assignment.mean()),
        'paired_two_candidate_logit_assignment_accuracy': float(assignment.mean()),
        'answer_code_only_logit_assignment_baseline': .5,
        'logit_assignment_tie_fraction': float(np.mean(np.abs(signed) <= 1e-10)),
        'median_signed_two_candidate_logit_contrast': float(np.median(signed))}


def initialize():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'evaluations').mkdir(exist_ok=True)
    signature = {'code_sha256': sha(__file__), 'observed_start_protocol_sha256': sha(OBSERVED / 'protocol.json'),
        'probe_data_sha256': sha(PARENT / 'probe_dataset.npz'),
        'scope': 'Exploratory complementary endpoints, defined before inspecting their additional-source-seed values. Pair differences remove any deterministic function of the three matched true answers, length and task. Two-candidate assignment is a diagnostic using the evaluation answer pair, not exact-answer inference. Frozen original 0.5 exact-answer gates remain unchanged.'}
    path = ROOT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())['signature'] == signature
    else:
        atomic_json(path, {'registered_utc': datetime.now(timezone.utc).isoformat(), 'signature': signature,
            'new_pair_contrast_evaluations': 0})


def evaluate_available():
    initialize()
    data = dict(np.load(PARENT / 'probe_dataset.npz'))
    use = data['split'] == 3
    ceiling = exact_answer_code_ceiling(data['labels'][use], data['lengths'][use])
    assert ceiling <= .5
    atomic_json(ROOT / 'answer_code_ceiling.json', {'same_collision_test': True, 'rows': int(use.sum()),
        'general_pair_upper_bound': .5, 'tight_empirical_arbitrary_answer_code_ceiling': ceiling,
        'scope': 'Privileged lookup oracle over all test answers; an empirical finite-sample upper bound, not a deployable predictor or a replacement of the registered 50% gate.'})
    cases = {'base_CI': (0, 5), 'observed_C_then_I': (1, 5), 'base_ICI': (0, 2), 'observed_I_then_CI': (4, 2)}
    completed = 0
    for path in (OBSERVED / 'evaluations').glob('*.json'):
        record = json.loads(path.read_text())
        dest = ROOT / 'evaluations' / path.name
        if dest.exists():
            completed += 1
            continue
        name = record['source']
        location = PARENT if record['scope'] == 'original_exploratory' else EXTENSION
        features = np.load(location / 'features' / f'{name}.npz')['source_query_concat'][:, :, -1].astype(np.float64)
        output = []
        for method in record['methods']:
            maps = dict(np.load(method['map_archive_path']))
            saved = dict(np.load(OBSERVED / 'arrays' / f"{name}_{method['method']}.npz"))
            for case, (start, target) in cases.items():
                metrics = paired_metrics(saved[f'answer_collisions_{case}_hidden'].astype(np.float64), features[use, start], features[use, target],
                    data['labels'][use, target], data['pair_ids'][use], maps['readout_weight'].astype(np.float64), maps['readout_bias'].astype(np.float64))
                metrics.update(method=method['method'], case=case)
                output.append(metrics)
        atomic_json(dest, {'source': name, 'source_condition': record['source_condition'], 'seed': record['seed'], 'scope': record['scope'],
            'observed_start_record_sha256': sha(path), 'metrics': output, 'reported_utc': datetime.now(timezone.utc).isoformat()})
        print(json.dumps({'paired_geometry': name, 'primary': [r for r in output if r['method'] == 'native_operators']}), flush=True)
        completed += 1
    return completed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    deadline = datetime(2026, 10, 6, 17, tzinfo=timezone.utc).timestamp()
    while time.time() < deadline:
        count = evaluate_available()
        if not args.watch or count == 27:
            break
        time.sleep(min(45, max(0, deadline - time.time())))
