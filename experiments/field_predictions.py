"""Freeze behavior-prediction analyses before the new finite-field cohort runs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .algebra import world
from .analysis import write_csv
from .field_symmetry import group_sources, spectral_power


def lookup_expectation(p, dimension, composition, budget):
    """Oracle minimal-subset lookup, balanced supports, disjoint input rows.

    Conditional on one test input, each support point in its target class is
    drawn among p**(dimension-1)-1 alternatives. p**(dimension-composition)-1
    share its partial source code. No-hit probability is hypergeometric.
    """
    class_size = p**(dimension-1)
    key_multiplicity = p**(dimension-composition)
    per_class = budget//p
    assert budget % p == 0 and 1 <= composition <= dimension
    failure = class_size-key_multiplicity
    probability_no_hit = np.prod([(failure-i)/(class_size-1-i) for i in range(per_class)])
    seen = 1-float(probability_no_hit)
    return {'seen_probability': seen, 'accuracy': 1/p+(1-1/p)*seen}


def heldout_combination_predictions(design, truth, labels, alpha=1.):
    predictions = np.full(len(truth), np.nan)
    folds = []
    for group in sorted(set(labels)):
        test = labels == group
        train = ~test
        mean, sigma = design[train].mean(0), design[train].std(0)
        sigma = np.where(sigma > 1e-8, sigma, 1.)
        x = np.column_stack([np.ones(train.sum()), (design[train]-mean)/sigma])
        z = np.column_stack([np.ones(test.sum()), (design[test]-mean)/sigma])
        penalty = np.eye(x.shape[1])*alpha; penalty[0, 0] = 0
        weights = np.linalg.solve(x.T@x+penalty, x.T@truth[train])
        predictions[test] = z@weights
        folds.append({'group': str(group), 'train_count': int(train.sum()), 'test_count': int(test.sum()),
                      'mae': float(np.mean(abs(predictions[test]-truth[test]))),
                      'rmse': float(np.sqrt(np.mean((predictions[test]-truth[test])**2)))})
    total = float(np.square(truth-truth.mean()).sum())
    return {'r2': 1-float(np.square(predictions-truth).sum())/total if total > 0 else None,
            'mae': float(np.mean(abs(predictions-truth))), 'folds': folds, 'predictions': predictions.tolist()}


def register():
    root = Path('results/six_hour_session')
    assert not Path('results/field_symmetry/metadata.json').exists()
    config = {'registered_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'registered_before_new_source_cohort_training_and_target_behavior',
              'ridge_alpha': 1., 'holdout': 'remove every world, initialization, target and budget for one entire source combination',
              'baseline': ['target fixed effects', 'input-basis fixed effects', 'initialization fixed effects',
                           'source final mean accuracy', 'source learning-curve mean AUC', 'mean physical source coefficient support size'],
              'structural_features': ['minimum target composition size'],
              'learned_feature_statistic': 'Fourier energy fraction along the four nonzero character multiples of the target coefficient',
              'analytical_reference': 'oracle-minimum-subset lookup occupancy; correct subset supplied externally, not generic-decoder performance theorem',
              'outcomes': ['paired random-adjusted linear accuracy', 'paired random-adjusted MLP accuracy', 'generic categorical-subset accuracy'],
              'budgets': [25, 50], 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limitations': ['Only five source-combination folds; folds share an abstract algebra family.',
                              'Crossed worlds, seeds and targets are correlated observations; no independent dataset p-value.',
                              'Source learning statistics are observed after source training; these are conditional predictions of downstream outcomes.',
                              'An analytical lookup reference is specific to its oracle subset and sampling assumptions.']}
    root.joinpath('field_prediction_preregistration.json').write_text(json.dumps(config, indent=2)+'\n')
    print(json.dumps({'registered_utc': config['registered_utc'], 'code_sha256': config['code_sha256']}))


def analyze():
    protocol = json.loads(Path('results/six_hour_session/field_prediction_preregistration.json').read_text())
    assert protocol['code_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    transfer_root = Path('results/field_symmetry_transfer')
    metadata = json.loads((transfer_root/'metadata.json').read_text())
    config = metadata['config']
    source_config = json.loads(Path(config['source_config']).read_text())
    source_root = Path(source_config['output'])
    p = source_config['p']
    complete = [json.loads(path.read_text()) for path in transfer_root.glob('*_w*_m*.json')]
    rows = [row for record in complete if record['status'] == 'complete' for row in record['rows']]
    baseline = {(r['world_seed'], r['model_seed'], r['target_id'], r['budget'], r['mode']): r['accuracy'] for r in rows if r['group'] == 'random'}
    augmented = []
    cache = {}
    for row in rows:
        if row['group'] == 'random' or row['mode'] == 'full_tuple_lookup': continue
        g, w, m = row['group'], row['world_seed'], row['model_seed']
        key = g, w, m
        if key not in cache:
            record = json.loads((source_root/f'{g}_w{w}_m{m}.json').read_text())
            inputs, _, _, basis = world(p, source_config['dimension'], w)
            latent = inputs@basis.T % p
            h = np.load(source_root/f'{g}_w{w}_m{m}_step{source_config["steps"]}_features.npy')
            power = spectral_power(h, latent, p)
            times = np.array([r['step'] for r in record['curve']])
            scores = np.array([r['source_accuracy'] for r in record['curve']])
            cache[key] = {'power': power/power.sum(), 'source_final_accuracy': float(scores[-1]),
                          'source_auc': float(np.trapezoid(scores, times)/times[-1]),
                          'source_physical_complexity': float(np.count_nonzero(group_sources(g, p)@basis % p, axis=1).mean())}
        statistics = cache[key]
        target = np.array(config['targets'][row['target_id']])
        energy = sum(statistics['power'][tuple(a*target % p)] for a in range(1, p))
        base_key = w, m, row['target_id'], row['budget'], row['mode']
        truth = row['accuracy']-baseline[base_key] if row['mode'] in ('linear', 'mlp') and base_key in baseline else row['accuracy']
        augmented.append({**row, 'prediction_outcome': truth,
                          **{k: v for k, v in statistics.items() if k != 'power'},
                          'target_character_energy_fraction': float(energy),
                          'oracle_subset_lookup_expected_accuracy': lookup_expectation(p, source_config['dimension'], row['target_composition_size'], row['budget'])['accuracy']})
    results = {}
    for mode in ('linear', 'mlp', 'categorical_subset'):
        for budget in config['budgets']:
            selected = [r for r in augmented if (r['mode'], r['budget']) == (mode, budget)]
            if len({r['group'] for r in selected}) != len(source_config['groups']): continue
            design = np.array([[r['source_final_accuracy'], r['source_auc'], r['source_physical_complexity']]
                + [float(r['target_id'] == t) for t in range(1, len(config['targets']))]
                + [float(r['world_seed'] == w) for w in source_config['world_seeds'][1:]]
                + [float(r['model_seed'] == m) for m in source_config['model_seeds'][1:]] for r in selected])
            structure = np.array([[r['target_composition_size']] for r in selected])
            spectrum = np.array([[r['target_character_energy_fraction']] for r in selected])
            truth = np.array([r['prediction_outcome'] for r in selected])
            labels = np.array([r['group'] for r in selected])
            scores = {}
            for name, matrix in (('baseline', design), ('plus_composition', np.column_stack([design, structure])),
                                 ('plus_target_spectrum', np.column_stack([design, spectrum])),
                                 ('plus_both', np.column_stack([design, structure, spectrum]))):
                scores[name] = heldout_combination_predictions(matrix, truth, labels, protocol['ridge_alpha'])
            results[f'{mode}_{budget}'] = scores
    output = transfer_root/'prediction'; output.mkdir(exist_ok=True)
    if augmented: write_csv(output/'features_and_outcomes.csv', augmented)
    output.joinpath('scores.json').write_text(json.dumps({'protocol': protocol, 'endpoints': len(augmented), 'results': results}, indent=2)+'\n')
    print(json.dumps({key: {name: {'r2': r['r2'], 'mae': r['mae']} for name, r in value.items()} for key, value in results.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--register', action='store_true'); args = parser.parse_args()
    (register if args.register else analyze)()
