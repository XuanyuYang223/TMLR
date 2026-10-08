"""Fit original-cohort predictors and save independent-world forecasts first."""
import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np

from .algebra import composition_size, world
from .field_predictions import heldout_combination_predictions, lookup_expectation
from .field_symmetry import group_sources, spectral_power
from .longrun_engine import atomic_json
from .matched_field import controlled_world
from .permworld_combinations import sha
from .six_hour_report import write_rows


VARIANTS = ('baseline', 'plus_composition', 'plus_oracle_occupancy', 'plus_target_spectrum', 'plus_both')


def source_statistics(source_config, group, world_seed, model_seed, provider):
    root = Path(source_config['output'])
    record = json.loads((root/f'{group}_w{world_seed}_m{model_seed}.json').read_text())
    assert record['status'] == 'complete'
    inputs, _, _, basis = provider(source_config['p'], source_config['dimension'], world_seed)
    latent = inputs@basis.T % source_config['p']
    h = np.load(root/f'{group}_w{world_seed}_m{model_seed}_step{source_config["steps"]}_features.npy')
    power = spectral_power(h, latent, source_config['p'])
    times = np.array([r['step'] for r in record['curve']])
    accuracy = np.array([r['source_accuracy'] for r in record['curve']])
    return {'source_final_accuracy': float(accuracy[-1]),
            'source_auc': float(np.trapezoid(accuracy, times)/times[-1]),
            'source_physical_complexity': float(np.count_nonzero(group_sources(group, source_config['p'])@basis % source_config['p'], axis=1).mean()),
            'basis': basis, 'power': power/power.sum(), 'source_fingerprint': record['fingerprint']}


def augment(row, config, source_config, statistics):
    p = source_config['p']; target = np.array(config['targets'][row['target_id']])
    order = composition_size(group_sources(row['group'], p), target, p)
    return {**row, **{k: statistics[k] for k in ('source_final_accuracy', 'source_auc', 'source_physical_complexity', 'source_fingerprint')},
            'target_physical_complexity': int(np.count_nonzero(target@statistics['basis'] % p)),
            'target_composition_size': order,
            'target_character_energy_fraction': float(sum(statistics['power'][tuple(a*target % p)] for a in range(1, p))),
            'oracle_occupancy': lookup_expectation(p, source_config['dimension'], order, row['budget'])['accuracy']}


def design(rows, variant, config, source_config):
    matrix = []
    for row in rows:
        values = [row[k] for k in ('source_final_accuracy', 'source_auc', 'source_physical_complexity', 'target_physical_complexity')]
        values += [float(row['target_id'] == t) for t in range(1, len(config['targets']))]
        for name in ('world_seed', 'model_seed'):
            categories = source_config['world_seeds' if name == 'world_seed' else 'model_seeds']
            values += [float(row[name] == c) if row[name] in categories else 1/len(categories) for c in categories[1:]]
        if variant in ('plus_composition', 'plus_both'): values.append(row['target_composition_size'])
        if variant == 'plus_oracle_occupancy': values.append(row['oracle_occupancy'])
        if variant in ('plus_target_spectrum', 'plus_both'): values.append(row['target_character_energy_fraction'])
        matrix.append(values)
    return np.array(matrix, dtype=np.float64)


def fit_ridge(x, truth, alpha):
    mean, sigma = x.mean(0), x.std(0)
    sigma = np.where(sigma > 1e-8, sigma, 1.)
    z = np.column_stack([np.ones(len(x)), (x-mean)/sigma])
    penalty = np.eye(z.shape[1])*alpha; penalty[0, 0] = 0
    weights = np.linalg.solve(z.T@z+penalty, z.T@truth)
    return {'mean': mean.tolist(), 'sigma': sigma.tolist(), 'weights': weights.tolist()}


def predict(fit, x):
    return np.column_stack([np.ones(len(x)), (x-np.array(fit['mean']))/np.array(fit['sigma'])])@np.array(fit['weights'])


def register():
    protocol = {'registered_utc': datetime.now(timezone.utc).isoformat(), 'code_sha256': sha(__file__),
                'status': 'registered before either field cohort is trained and before all behavior endpoints',
                'fit_cohort': 'all five original groups, three original input bases, three initialization seeds',
                'forecast_cohort': 'P, M1, M2 on three independently selected all-source-support-four bases, all three initialization seeds',
                'forecast_timing': 'save forecasts after new source features exist but before new target-readout process starts',
                'ridge_alpha': 1., 'variants': list(VARIANTS),
                'baseline': ['source final accuracy', 'source learning AUC', 'physical source support count', 'physical target support count',
                             'target, initialization and original input-basis fixed effects'],
                'unseen_world_policy': 'average training-world fixed effect; each non-reference world dummy gets 1/3',
                'outcomes': ['linear gain over paired random encoder', 'MLP gain over paired random encoder', 'categorical subset accuracy'],
                'structural_features': ['target minimum source composition', 'analytical oracle-subset key occupancy'],
                'learned_feature_diagnostic': 'target-character energy in frozen source features; privileged analytical task-definition diagnostic',
                'limitations': ['The three forecast source-group formulas also occur in the fitting cohort; only the input bases and model runs are unseen.',
                               'Generic categorical dependency order is not specific evidence of finite-field algebra learning.',
                               'Oracle lookup occupancy assumes the correct source subset; generic subset discovery need not achieve it.',
                               'The full source grid is exposed; no new-input source generalization.']}
    atomic_json('results/six_hour_session/independent_forecast_preregistration.json', protocol)
    print(json.dumps({'registered_utc': protocol['registered_utc'], 'code_sha256': protocol['code_sha256']}))


def run():
    protocol = json.loads(Path('results/six_hour_session/independent_forecast_preregistration.json').read_text())
    assert protocol['code_sha256'] == sha(__file__)
    output = Path('results/field_forecast'); output.mkdir(exist_ok=True)
    if (output/'forecasts.json').exists():
        assert json.loads((output/'forecasts.json').read_text())['protocol'] == protocol
        return
    assert not Path('results/field_matched_support_transfer/metadata.json').exists(), 'Forecast must be saved before independent target readouts'
    old_config = json.loads(Path('configs/field_symmetry_transfer.json').read_text())
    old_source = json.loads(Path(old_config['source_config']).read_text())
    old_records = [json.loads(p.read_text()) for p in Path(old_config['output']).glob('*_w*_m*.json')]
    old_records = [r for r in old_records if r['status'] == 'complete']
    if len(old_records) != 54:
        atomic_json(output/'status.json', {'status': 'unavailable', 'reason': 'Original behavior cohort incomplete; no independent forecast claim'})
        return
    old_rows = [r for record in old_records for r in record['rows']]
    baseline = {(r['world_seed'], r['model_seed'], r['target_id'], r['budget'], r['mode']): r['accuracy'] for r in old_rows if r['group'] == 'random'}
    cache, fit_rows = {}, []
    for row in old_rows:
        if row['group'] == 'random' or row['mode'] == 'full_tuple_lookup': continue
        key = row['group'], row['world_seed'], row['model_seed']
        if key not in cache: cache[key] = source_statistics(old_source, *key, world)
        value = augment(row, old_config, old_source, cache[key])
        value['prediction_outcome'] = row['accuracy'] if row['mode'] == 'categorical_subset' else row['accuracy']-baseline[row['world_seed'], row['model_seed'], row['target_id'], row['budget'], row['mode']]
        fit_rows.append(value)
    new_config = json.loads(Path('configs/field_matched_support_transfer.json').read_text())
    new_source = json.loads(Path(new_config['source_config']).read_text())
    new_rows = []
    for g, w, m in product(new_source['groups'], new_source['world_seeds'], new_source['model_seeds']):
        statistics = source_statistics(new_source, g, w, m, controlled_world)
        for t, b, mode in product(range(len(new_config['targets'])), new_config['budgets'], ('linear', 'mlp', 'categorical_subset')):
            new_rows.append(augment({'group': g, 'world_seed': w, 'model_seed': m, 'target_id': t, 'budget': b, 'mode': mode}, new_config, new_source, statistics))
    forecasts, fitted, cv = [], {}, {}
    for mode, budget in product(('linear', 'mlp', 'categorical_subset'), old_config['budgets']):
        old = [r for r in fit_rows if (r['mode'], r['budget']) == (mode, budget)]
        new = [r for r in new_rows if (r['mode'], r['budget']) == (mode, budget)]
        truth = np.array([r['prediction_outcome'] for r in old]); labels = np.array([r['group'] for r in old])
        for variant in VARIANTS:
            x = design(old, variant, old_config, old_source)
            fit = fit_ridge(x, truth, protocol['ridge_alpha'])
            name = f'{mode}_{budget}_{variant}'
            fitted[name] = fit
            cv[name] = heldout_combination_predictions(x, truth, labels, protocol['ridge_alpha'])
            estimates = predict(fit, design(new, variant, old_config, old_source))
            forecasts.extend({**r, 'variant': variant, 'predicted_outcome': float(y)} for r, y in zip(new, estimates))
    atomic_json(output/'forecasts.json', {'protocol': protocol, 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'independent_behavior_not_started': True, 'original_behavior_fingerprint': json.loads(Path(old_config['output']).joinpath('metadata.json').read_text())['fingerprint'],
        'independent_source_fingerprint': json.loads(Path(new_source['output']).joinpath('metadata.json').read_text())['fingerprint'],
        'forecasts': forecasts, 'fitted_models': fitted, 'original_whole_combination_cv': cv})
    write_rows(output/'fitting_features.csv', fit_rows)
    write_rows(output/'independent_source_features.csv', new_rows)
    write_rows(output/'forecasts.csv', forecasts)
    print(json.dumps({'frozen_forecasts': len(forecasts), 'before_independent_target_readout': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--register', action='store_true'); args = parser.parse_args()
    (register if args.register else run)()
