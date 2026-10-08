"""Registered joint resampling of sources and heldout inputs; no fitting."""
import argparse
from math import fsum
import json
from pathlib import Path
import time

import numpy as np

from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now

ROOT = Path('results/two_step_relation_dose_confirmation')
CONFIG = Path('configs/two_step_relation_dose_confirmation.json')
DRAWS = 10000
SEED = 261074501


def register():
    signature = {'code_sha256': sha(__file__), 'config_sha256': sha(CONFIG),
                 'bootstrap_draws': DRAWS, 'bootstrap_seed': SEED,
                 'method': 'Resample three source blocks; retain both fitting repeats per selected source. Independently resample heldout examples within each fixed length, using complete two-row collision pairs for the collision split. All conditions share each resample. Report percentile intervals for three both-correct contrasts and the2x2 interaction, separately for IID and collisions.',
                 'scope': 'Secondary uncertainty audit. No model, setting, endpoint, metric or fixed prediction gate is selected or changed. Three sources remain a limitation.'}
    file = ROOT / 'uncertainty_protocol.json'
    if file.exists():
        assert json.loads(file.read_text())['signature'] == signature
    else:
        atomic_json(file, {'registered_utc': now(), 'compound_test_opened_at_registration': (ROOT / 'test_opened.json').exists(),
                          'signature': signature})
        (ROOT / 'uncertainty_source_snapshot.py').write_bytes(Path(__file__).read_bytes())


def run():
    register()
    while not (ROOT / 'completion.json').exists():
        state = json.loads((ROOT / 'state.json').read_text()) if (ROOT / 'state.json').exists() else {}
        if state.get('status') in ['failed', 'not_started_visible_feasibility_failed']:
            raise RuntimeError('Confirmation did not complete; uncertainty not evaluated')
        time.sleep(10)
    plan = json.loads(CONFIG.read_text()); data = np.load(ROOT / 'dataset/test/dataset.npz')
    summary = json.loads((ROOT / 'summary.json').read_text())
    hits = np.asarray([[np.load(ROOT / 'evaluations' / f'n{i}_hidden_{c}.npz')['ci_answers'] == data['labels'][:, 5]
                        for c in plan['conditions']] for i in range(6)], dtype=float)
    sources = np.asarray([(hits[j] + hits[j + 3]) / 2 for j in range(3)])
    names = ['both_correct-' + c for c in plan['conditions'][1:]] + ['interaction']
    differences = np.stack([sources[:, 0] - sources[:, k] for k in [1,2,3]] +
                           [sources[:,0] - sources[:,1] - sources[:,2] + sources[:,3]])
    outputs, arrays, checks = [], {}, 0
    for split, label in [(0, 'iid'), (1, 'collisions')]:
        blocks = []
        for n in plan['lengths']:
            ids = np.flatnonzero((data['split'] == split) & (data['lengths'] == n))
            if split == 1:
                pair_ids = data['pair_ids'][ids]
                pairs = [ids[pair_ids == p] for p in sorted(set(map(int,pair_ids)))]
                assert all(len(pair) == 2 for pair in pairs)
                blocks.append(np.stack([differences[:,:,pair].mean(-1) for pair in pairs], axis=-1))
            else:
                blocks.append(differences[:,:,ids])
        effects = np.stack(blocks, axis=2) # contrast, source, fixed length, input/pair
        count = effects.shape[-1]; draws = []; rng = np.random.default_rng(SEED + split)
        for start in range(0, DRAWS, 250):
            size = min(250, DRAWS-start)
            sc = rng.multinomial(3, np.ones(3)/3, size=size)
            ic = np.stack([rng.multinomial(count, np.ones(count)/count, size=size) for _ in plan['lengths']], axis=1)
            assert np.all(sc.sum(1) == 3) and np.all(ic.sum(-1) == count)
            values = 100*np.einsum('bs,blp,cslp->bc', sc/3, ic/count, effects, optimize=True)/len(plan['lengths'])
            if start == 0:
                # Scalar weighted sums independently verify axes/normalizers.
                for b in range(16):
                    for c in range(4):
                        expected = 100*fsum(float(sc[b,s]*ic[b,l,p]*effects[c,s,l,p])
                                           for s in range(3) for l in range(len(plan['lengths'])) for p in range(count))/(3*count*len(plan['lengths']))
                        np.testing.assert_allclose(values[b,c], expected, atol=1e-12, rtol=1e-12); checks += 1
            draws.append(values)
        draws = np.concatenate(draws); arrays[label + '_accuracy_pp_draws'] = draws
        for c, contrast in enumerate(names):
            row = next(r for r in summary['contrasts'] if r['primary'] and r['split'] == label and r['contrast'] == contrast)
            point = float(100*effects[c].mean()); np.testing.assert_allclose(point, row['accuracy_pp']['mean'], atol=1e-10)
            outputs.append({'split':label, 'contrast':contrast, 'mean_accuracy_pp':point,
                            'source_and_input_bootstrap_95_pp':np.quantile(draws[:,c],[.025,.975]).tolist(),
                            'input_units_per_length':count, 'input_unit':'whole collision pair' if split else 'individual IID example'})
    np.savez_compressed(ROOT / 'uncertainty_draws.npz', **arrays)
    atomic_json(ROOT / 'uncertainty.json', {'status':'complete', 'completed_utc':now(), 'draws':DRAWS,
                'sources':3, 'fitting_repeats_per_source':2, 'scalar_weighted_sum_checks':checks,
                'primary_mean_checks':len(outputs), 'results':outputs,
                'summary_sha256':sha(ROOT/'summary.json'), 'draws_sha256':sha(ROOT/'uncertainty_draws.npz'),
                'limitation':'Secondary registered percentile bootstrap with only three existing source initializations; it does not establish broad generality or independence of six fits.'})


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['register','run'])
    register() if parser.parse_args().phase == 'register' else run()
