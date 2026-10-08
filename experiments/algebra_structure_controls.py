"""Supplementary orbit-relabeling null, preserving every orbit's moments.

This control was added after the first primary assay outputs, and is labeled
as a diagnostic rather than part of the original prospective hypothesis.
Independent permutations within each orbit destroy a globally meaningful
operator label while preserving the whole-orbit centering constraint, PCA
covariance, orbit energy, and the abstract multiplication table.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .longrun_engine import atomic_json
from .native_algebra_structure import probe_view
from .permworld_combinations import sha


def relabel_orbits(hidden, seed):
    rng = np.random.default_rng(seed)
    return np.stack([orbit[rng.permutation(len(orbit))] for orbit in hidden])


def run():
    plan = json.loads(Path('configs/algebra_structure.json').read_text())
    root = Path(plan['output']); output = root/'orbit_null'; output.mkdir(exist_ok=True)
    native = root/'native'; data = dict(np.load(native/'dataset.npz'))
    sources = sorted((native/'features').glob('*_trained.npz'))
    assert len(sources) == 24
    signature = {'code_sha256': sha(__file__), 'upstream_protocol_sha256': sha(native/'protocol.json'),
        'features': {p.name: sha(p) for p in sources},
        'scope': 'post hoc diagnostic added after early primary endpoints; no new source training or selection',
        'control': 'independently relabel eight states of each orbit before refitting generators; preserves centered PCA spectrum exactly'}
    path = output/'protocol.json'
    if path.exists(): assert json.loads(path.read_text())['signature'] == signature
    else: atomic_json(path, {'registered_utc': datetime.now(timezone.utc).isoformat(), 'signature': signature})
    for source in sources:
        path = output/f'{source.stem}.json'
        if path.exists(): continue
        record = json.loads((native/'probes'/f'{source.stem}.json').read_text())
        values = dict(np.load(source)); results = {}
        for landmark in plan['native_landmarks']:
            hidden = values[landmark][:, :, -1]
            control = relabel_orbits(hidden, record['seed']+9100)
            result, _ = probe_view(control, data, plan, record['seed']+9000)
            results[landmark] = result
        atomic_json(path, {'group': record['group'], 'seed': record['seed'], 'results': results,
                          'status': 'complete', 'source_feature_sha256': sha(source)})
    atomic_json(output/'state.json', {'status': 'complete', 'models': 24})


if __name__ == '__main__': run()
