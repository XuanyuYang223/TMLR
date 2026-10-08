"""Evaluate the first-pool length rules on fresh target data, without tuning."""
from itertools import product
import json
from pathlib import Path

import numpy as np

from .longrun_engine import atomic_json
from .native_length_baseline import majority_predictions, smooth_prediction
from .native_target_repeat import paths
from .permworld_combinations import sha
from .six_hour_report import write_rows


def run():
    _, config, source_root, root = paths()
    selected = json.loads((source_root/'length_baselines/selection.json').read_text())['selected']
    names = json.loads((root/'dataset/metadata.json').read_text())['names']
    support = json.loads((root/'dataset/support_indices.json').read_text())
    with np.load(root/'dataset/data.npz') as archive:
        data = {key: archive[key] for key in ('support_pool_lengths', 'support_pool_labels', 'target_test_lengths', 'target_test_labels')}
    rows = []
    for target, budget in product(config['target_tasks'], config['target_budgets']):
        ids = support[str(budget)]; t = names.index(target)
        n, y = data['support_pool_lengths'][ids], data['support_pool_labels'][ids, t]
        test_n, truth = data['target_test_lengths'], data['target_test_labels'][:, t]
        predictions = list(majority_predictions(n, y, test_n))
        predictions.append(smooth_prediction(n, y, test_n, selected['bandwidth'], config['lengths'][1]+1))
        for mode, prediction in zip(('global_majority', 'exact_length_majority', 'smooth_length'), predictions):
            rows.append({'target': target, 'budget': budget, 'mode': mode, 'test_accuracy': float(np.mean(prediction == truth))})
    summaries = [{'budget': budget, 'mode': mode, 'test_macro': float(np.mean([r['test_accuracy'] for r in rows if (r['budget'], r['mode']) == (budget, mode)]))}
                 for budget, mode in product(config['target_budgets'], ('global_majority', 'exact_length_majority', 'smooth_length'))]
    write_rows(root/'length_baselines.csv', rows)
    atomic_json(root/'length_baselines.json', {'scope': 'same first-pool practical support-only baselines; original bandwidth reused without new tuning',
                                             'old_selected': selected, 'target_validation_read': False,
                                             'data_sha256': sha(root/'dataset/data.npz'), 'code_sha256': sha(__file__),
                                             'endpoints': rows, 'macro': summaries})
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__': run()
