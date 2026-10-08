"""Independent field cohort with exactly four physical inputs per source task."""
from itertools import product
import json
from pathlib import Path

import numpy as np

from .algebra import rank_mod
from . import field_symmetry
from .field_symmetry import group_sources
from .longrun_engine import atomic_json
from .permworld_combinations import sha


def controlled_world(p, dimension, seed):
    assert p == 5 and dimension == 4
    rng = np.random.default_rng(seed)
    while True:
        basis = rng.integers(1, p, size=(dimension, dimension))
        if rank_mod(basis, p) != dimension: continue
        if all(np.all(group_sources(g, p)@basis % p != 0) for g in ('P', 'M1', 'M2')): break
    inputs = np.array(list(product(range(p), repeat=dimension)), dtype=np.int64)
    encoded = np.eye(p, dtype=np.float32)[inputs].reshape(len(inputs), -1)
    return inputs, inputs@basis[:3].T % p, encoded, basis


def run():
    path = Path('configs/field_matched_support.json')
    config = json.loads(path.read_text())
    assert config['world_provider_sha256'] == sha(__file__)
    audit = []
    for seed in config['world_seeds']:
        inputs, _, _, basis = controlled_world(config['p'], config['dimension'], seed)
        groups = []
        for name in config['groups']:
            physical = group_sources(name, config['p'])@basis % config['p']
            assert np.all(physical != 0)
            labels = inputs@physical.T % config['p']
            assert len(np.unique(labels, axis=0)) == config['p']**4
            for coefficient in physical:
                changed = inputs*coefficient % config['p']
                assert len(np.unique(changed, axis=0)) == len(inputs)
                assert np.array_equal(changed.sum(1) % config['p'], inputs@coefficient % config['p'])
            groups.append({'group': name, 'physical_source_coefficients': physical.tolist(),
                           'physical_inputs_per_source': [4, 4, 4, 4],
                           'single_task_isomorphism': 'each task is modular sum of four inputs after independent categorical permutations in every input coordinate'})
        audit.append({'world_seed': seed, 'basis': basis.tolist(), 'groups': groups})
    root = Path(config['output']); root.mkdir(exist_ok=True)
    atomic_json(root/'controlled_worlds.json', {'provider_sha256': sha(__file__), 'worlds': audit,
        'complexity_scope': 'individual task functions are isomorphic under input-channel permutations; training trajectories need not coincide',
        'source_input_scope': 'same complete 625 physical inputs, independently chosen latent-coordinate bases'})
    field_symmetry.world = controlled_world
    field_symmetry.run(str(path))


if __name__ == '__main__': run()
