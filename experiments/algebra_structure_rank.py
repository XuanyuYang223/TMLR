"""One prespecified full-width sensitivity after the first direct results.

All task groups/seeds are included. No search for a favorable probe rank.
The same fit/validation/test inputs and ridge grid are retained; PCA now keeps
every numerically supported input direction. No new source training.
"""
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path

import numpy as np

from .algebra_structure_direct import direct_probe
from .field_algebra_structure import (controlled_world, field_orbits, dihedral_table,
    projection_split, projection_probe)
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .representation_algebra import permutation_action_table


def run():
    plan = json.loads(Path('configs/algebra_structure.json').read_text()); root = Path(plan['output'])
    output = root/'rank_sensitivity'; output.mkdir(exist_ok=True)
    for name in ('native', 'field'): (output/name).mkdir(exist_ok=True)
    signature = {'code_sha256': sha(__file__), 'direct_code_sha256': sha('experiments/algebra_structure_direct.py'),
        'upstream_direct_protocol_sha256': sha(root/'direct/protocol.json'),
        'scope': 'added after first low-rank direct outputs; one complete-width sensitivity for every group/seed',
        'dimensions': 'native ONE_END <=256, source-query concatenation <=1024, field <=64; retain all fit-supported PCA directions',
        'controls': 'same fit/validation/test rows, validation-selected ridge grid, and matched random models; no new training or test-based rank selection'}
    protocol = output/'protocol.json'
    if protocol.exists(): assert json.loads(protocol.read_text())['signature'] == signature
    else: atomic_json(protocol, {'registered_utc': datetime.now(timezone.utc).isoformat(), 'signature': signature})
    data = dict(np.load(root/'native/dataset.npz')); table = permutation_action_table()
    for path in sorted((root/'native/features').glob('*.npz')):
        dest = output/'native'/f'{path.stem}.json'
        if dest.exists(): continue
        record = json.loads((root/'native/probes'/f'{path.stem}.json').read_text()); hidden = dict(np.load(path))
        results = {}
        for landmark in plan['native_landmarks']:
            h = hidden[landmark][:,:,-1]
            results[landmark], arrays = direct_probe(h, data['split'], data['lengths'], table, {'c':1,'r':2,'i':4},
                [('r','c'),('c','i'),('r','i'),('r','c','i')], h.shape[-1], plan['ridge_grid'], record['seed']+9200)
            np.savez_compressed(output/'native'/f'{path.stem}_{landmark}.npz', **arrays)
        atomic_json(dest, {'group':record['group'], 'seed':record['seed'], 'model_status':record['model_status'], 'results':results, 'status':'complete'})
        print(path.stem, flush=True)
    config = json.loads(Path(plan['field_config']).read_text()); source = Path(plan['field_source'])
    orbits, split = field_orbits(5, plan['field_split_seed'], plan['field_fit_fraction'], plan['field_validation_fraction']); table = dihedral_table()
    for w in config['world_seeds']:
        inputs,_,_,basis = controlled_world(5,4,w); latent=inputs @ basis.T % 5
        lookup={tuple(z):i for i,z in enumerate(latent)}; ids=np.array([[lookup[tuple(z)] for z in orbit] for orbit in orbits])
        projected=latent.copy(); projected[:,2]=0; images=np.array([lookup[tuple(z)] for z in projected])
        psplit=projection_split(latent,plan['field_split_seed']+1)
        for g,m,status in product(config['groups'],config['model_seeds'],('random','trained')):
            name=f'{g}_w{w}_s{m}_{status}'; dest=output/'field'/f'{name}.json'
            if dest.exists(): continue
            step=0 if status=='random' else config['steps']; h=np.load(source/f'{g}_w{w}_m{m}_step{step}_features.npy')
            result,arrays=direct_probe(h[ids],split,np.zeros(len(orbits)),table,{'a':1,'s':4},
                [('a','a'),('a','a','a'),('a','s'),('a','a','s'),('a','a','a','s')],h.shape[-1],plan['ridge_grid'],m+9200)
            np.savez_compressed(output/'field'/f'{name}_hidden.npz',**arrays)
            projection,arrays=projection_probe(h,images,psplit,h.shape[-1],plan['ridge_grid'],m+8100)
            np.savez_compressed(output/'field'/f'{name}_projection.npz',**arrays)
            atomic_json(dest,{'group':g,'world_seed':w,'seed':m,'model_status':status,'results':{'hidden':result,'projection':projection},'status':'complete'})
    atomic_json(output/'state.json', {'status':'complete','native_conditions':48,'field_conditions':54})


if __name__ == '__main__': run()
