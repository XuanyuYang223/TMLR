"""Separate unseen-edge prediction from autonomous base-only composition."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from .hidden_relation_evaluate import pair_scores, transform_features
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .representation_algebra import permutation_action_table, word_action


CONFIG = 'configs/relation_observed_start.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize():
    plan = json.loads(Path(CONFIG).read_text())
    root = Path(plan['output'])
    root.mkdir(parents=True, exist_ok=True)
    for folder in ['evaluations', 'arrays']:
        (root / folder).mkdir(exist_ok=True)
    signature = {'config_sha256': sha(CONFIG), 'code_sha256': sha(__file__),
        'parent_protocol_sha256': sha(Path(plan['parent']) / 'protocol.json'),
        'extension_protocol_sha256': sha(Path(plan['extension']) / 'protocol.json'),
        'probe_data_sha256': sha(Path(plan['parent']) / 'probe_dataset.npz'), 'plan': plan}
    path = root / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())['signature'] == signature
    else:
        atomic_json(path, {'registered_utc': now(), 'signature': signature, 'new_additional_seed_observed_start_results': 0})
    return plan, root


def observed_start_score(hidden, data, archive):
    maps = {g: (archive[f'rho_{g}'], archive[f'bias_{g}']) for g in ['c', 'i']}
    means = {int(n): value for n, value in zip(archive.get('mean_lengths', []), archive.get('mean_vectors', []))}
    weight, bias = archive['readout_weight'].astype(np.float64), archive['readout_bias'].astype(np.float64)
    table, letters = permutation_action_table(), {'c': 1, 'i': 4}
    rows, arrays = [], {}
    for split, tag in [(2, 'iid'), (3, 'answer_collisions')]:
        use = data['split'] == split
        h = hidden[use].astype(np.float64)
        centers = np.array([means.get(int(n), np.zeros(h.shape[-1])) for n in data['lengths'][use]])
        for name, start, word in [('base_CI', 0, 'ci'), ('observed_C_then_I', 1, 'i'), ('base_ICI', 0, 'ici'), ('observed_I_then_CI', 4, 'ci')]:
            destination = start
            for g in word:
                destination = int(table[destination, letters[g]])
            pred = transform_features(h[:, start] - centers, word, maps) + centers
            target = h[:, destination]
            denominator = np.square(target - h[:, start]).sum()
            answers = (pred @ weight.T + bias).argmax(-1)
            truth = data['labels'][use, destination]
            row = {'case': name, 'split': tag, 'start_action': start, 'destination_action': destination,
                'start_type_seen_during_source_training': start in [0, 1, 4],
                'displacement_nmse': float(np.square(pred - target).sum() / denominator),
                'answer_accuracy': float(np.mean(answers == truth))}
            if split == 3:
                ids = data['pair_ids'][use]
                row.update(pair_scores(answers == truth, ids))
                # Swap a known-start hidden vector within answer-identical pairs.
                # This is diagnostic-only, with partner accuracy reported separately.
                partner = np.arange(len(ids))
                for pair in np.unique(ids):
                    indices = np.flatnonzero(ids == pair)
                    assert len(indices) == 2
                    partner[indices] = indices[::-1]
                row['swapped_start_self_answer_accuracy'] = float(np.mean(answers[partner] == truth))
                row['swapped_start_partner_answer_accuracy'] = float(np.mean(answers[partner] == truth[partner]))
                assert row['swapped_start_self_answer_accuracy'] + row['swapped_start_partner_answer_accuracy'] <= 1 + 1e-12
                arrays[f'{tag}_{name}_partner_rows'] = partner
            rows.append(row)
            arrays[f'{tag}_{name}_hidden'] = pred.astype(np.float32)
            arrays[f'{tag}_{name}_answers'] = answers
    return rows, arrays


def evaluate_available():
    plan, root = initialize()
    data = dict(np.load(Path(plan['parent']) / 'probe_dataset.npz'))
    count = 0
    for location, seeds, scope in [(Path(plan['parent']), plan['original_seeds_exploratory'], 'original_exploratory'),
            (Path(plan['extension']), plan['additional_seeds_prospective_endpoint'], 'additional_seed_secondary_forecast')]:
        for seed in seeds:
            for condition in plan['conditions']:
                name = f'{condition}_s{seed}'
                ep = location / 'evaluations' / f'{name}.json'
                if not ep.exists():
                    continue
                dest = root / 'evaluations' / f'{name}.json'
                if dest.exists():
                    count += 1
                    continue
                feature = location / 'features' / f'{name}.npz'
                hidden = np.load(feature)['source_query_concat'][:, :, -1]
                variants = [('native_operators', location / 'arrays' / (f'{name}_query_native_operators.npz' if scope == 'original_exploratory' else f'{name}_native_operators.npz'))]
                for pairing in ['correct', 'shuffled']:
                    source_map = Path('results/frozen_relation_followup/maps') / f'{name}_k256_{pairing}.npz'
                    if source_map.exists():
                        variants.append((f'frozen_full_{pairing}', source_map))
                methods = []
                for method, path in variants:
                    archive = dict(np.load(path))
                    rows, arrays = observed_start_score(hidden, data, archive)
                    output = root / 'arrays' / f'{name}_{method}.npz'
                    np.savez_compressed(output, **arrays)
                    methods.append({'method': method, 'map_archive_path': str(path), 'map_sha256': sha(path),
                        'prediction_archive_sha256': sha(output), 'rows': rows})
                source = json.loads((location / 'source' / f'{name}.json').read_text())
                atomic_json(dest, {'source': name, 'source_condition': condition, 'seed': seed, 'scope': scope,
                    'source_validation_accuracy': source['observed_validation_accuracy'], 'feature_sha256': sha(feature), 'methods': methods, 'completed_utc': now()})
                count += 1
                native = methods[0]['rows']
                print(json.dumps({'observed_start_evaluated': name, 'scope': scope,
                    'collision': [r for r in native if r['split'] == 'answer_collisions']}), flush=True)
    atomic_json(root / 'state.json', {'status': 'available_evaluations_complete', 'source_models': count, 'updated_utc': now()})
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    plan, _ = initialize()
    deadline = datetime.fromisoformat(plan['deadline_utc']).timestamp()
    while time.time() < deadline:
        count = evaluate_available()
        if not args.watch or count == 27:
            break
        time.sleep(min(45, max(0, deadline - time.time())))
