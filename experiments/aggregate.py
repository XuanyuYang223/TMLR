"""Aggregate compatible seed replications without treating them as new bases."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from .algebra import scenarios
from .analysis import summarize


def aggregate(inputs, output):
    inputs = [Path(p) for p in inputs]
    metadata = [json.loads((p / 'metadata.json').read_text()) for p in inputs]
    configurations = [m['config'] for m in metadata]
    reference = {k: v for k, v in configurations[0].items() if k != 'model_seeds'}
    if any({k: v for k, v in c.items() if k != 'model_seeds'} != reference for c in configurations[1:]):
        raise ValueError('configurations differ beyond initialization seeds')
    seeds = [s for c in configurations for s in c['model_seeds']]
    if len(seeds) != len(set(seeds)):
        raise ValueError('duplicate initialization seeds cannot be counted twice')
    audit = json.loads((inputs[0] / 'algebra_audit.json').read_text())
    if any(json.loads((p / 'algebra_audit.json').read_text()) != audit for p in inputs[1:]):
        raise ValueError('algebra/targets do not match')
    for w in reference['world_seeds']:
        arrays = [np.load(p / f'world_{w}.npz') for p in inputs]
        for key in arrays[0].files:
            if any(not np.array_equal(arrays[0][key], a[key]) for a in arrays[1:]):
                raise ValueError('basis or support/test indices do not match')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = {**configurations[0], 'model_seeds': sorted(seeds)}
    provenance = {'config': config, 'aggregation_of': [{'directory': str(p.resolve()), 'fingerprint': m['fingerprint']} for p, m in zip(inputs, metadata)],
                  'fingerprint': 'aggregation_only_not_a_training_run',
                  'replication': 'same three input bases crossed with three independent initialization seeds'}
    (output / 'metadata.json').write_text(json.dumps(provenance, indent=2))
    shutil.copyfile(inputs[0] / 'algebra_audit.json', output / 'algebra_audit.json')
    for w in config['world_seeds']:
        shutil.copyfile(inputs[0] / f'world_{w}.npz', output / f'world_{w}.npz')
    groups = scenarios()
    groups = [g for pair in zip(groups[:4], groups[4:]) for g in pair][:config['scenario_limit']]
    metrics = []
    for p, c in zip(inputs, configurations):
        for w in c['world_seeds']:
            for m in c['model_seeds']:
                shutil.copyfile(p / f'initial_w{w}_m{m}.npy', output / f'initial_w{w}_m{m}.npy')
                for g in groups:
                    rid = f"{g['id']}_w{w}_m{m}"
                    for suffix in ['.json', '_features.npy']:
                        shutil.copyfile(p / f'{rid}{suffix}', output / f'{rid}{suffix}')
                    metrics.extend(json.loads((p / f'{rid}.json').read_text())['metrics'])
    # The oracle has no learned target parameters; retain per-run provenance.
    oracles = [json.loads((p / 'symbolic_oracle.json').read_text()) for p in inputs]
    result = {k: v for k, v in oracles[0].items() if k != 'rows'}
    result['rows'] = [r for oracle in oracles for r in oracle['rows']]
    result['mean_oracle_accuracy'] = float(np.mean([r['oracle_test_accuracy'] for r in result['rows']]))
    result['minimum_oracle_accuracy'] = min(r['oracle_test_accuracy'] for r in result['rows'])
    (output / 'symbolic_oracle.json').write_text(json.dumps(result, indent=2))
    summarize(output, metrics, groups, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', default=['results/pilot', 'results/replication'])
    parser.add_argument('--output', default='results/combined')
    args = parser.parse_args()
    aggregate(args.inputs, args.output)
