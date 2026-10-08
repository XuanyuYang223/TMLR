"""Independent aggregate/exposure audit, using multinomial bootstrap counts."""
from datetime import datetime, timezone
import json
from math import fsum, isclose
from pathlib import Path
import time

import numpy as np

from .overnight_inverse_statistics_verify import bootstrap_interval
from .permworld_combinations import sha
from .longrun_engine import atomic_json

ROOT = Path('results/two_step_relation_joint')


def close(actual, expected):
    assert isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-10), (actual, expected)


def average(values):
    return fsum(values) / len(values)


def verify():
    summary = json.loads((ROOT / 'summary.json').read_text())
    records = json.loads((ROOT / 'evaluation_records.json').read_text())['records']
    plan = json.loads(Path('configs/two_step_relation_joint.json').read_text())
    for mean in summary['means']:
        rows = [r for r in records if all(r[k] == mean[k] for k in ['view', 'split', 'word', 'condition'])]
        assert len(rows) == 6
        for key in ['accuracy', 'teacher_forced_accuracy', 'direct_input_accuracy', 'pair_both_correct',
                    'displacement_nmse', 'cka', 'composed_displacement_energy']:
            if mean[key] is None:
                assert all(r[key] is None for r in rows)
            else:
                close(mean[key], average([r[key] for r in rows]))
    for row in summary['contrasts']:
        differences = []
        for i in range(6):
            scores = {r['condition']: r['accuracy'] for r in records if
                      (r['view'], r['split'], r['word'], r['replicate']) ==
                      (row['view'], row['split'], row['word'], f'n{i}')}
            if row['contrast'] == 'interaction':
                difference = scores['both_correct'] - scores['c_correct_i_wrong'] - scores['c_wrong_i_correct'] + scores['both_wrong']
            else:
                difference = scores['both_correct'] - scores[row['contrast'].split('-')[1]]
            differences.append(100 * difference)
        clusters = [average([differences[i] for i in (j, j + 3)]) for j in range(3)]
        report = row['accuracy_pp']
        close(report['mean'], average(differences))
        assert report['positive_repeats'] == sum(v > 0 for v in differences)
        for field, values in [('paired_values', differences), ('three_source_means', clusters),
                              ('six_fit_bootstrap_95', bootstrap_interval(differences)),
                              ('three_source_bootstrap_95', bootstrap_interval(clusters))]:
            for actual, expected in zip(report[field], values):
                close(actual, expected)
    test_opened = json.loads((ROOT / 'test_opened.json').read_text())
    opened_time = datetime.fromisoformat(test_opened['opened_utc'])
    assert len(test_opened['fit_record_sha256']) == 24
    reconstructed = 0
    for i in range(6):
        support_path = ROOT / 'dataset' / f'n{i}' / 'support'
        data = np.load(support_path / 'dataset.npz'); ns = data['lengths'][data['split'] == 0]
        pair = np.load(support_path / 'pairings.npz'); keep = pair['eligible']
        rng = np.random.default_rng(plan['support_seeds'][i] + 8001)
        import hashlib
        digest = hashlib.sha256(); counts = np.zeros(len(ns), dtype=np.int64); steps = 0
        for epoch in range(1, plan['epochs'] + 1):
            for length in rng.permutation(sorted(set(map(int, ns)))):
                ids = rng.permutation(np.flatnonzero(keep & (ns == length)))
                for start in range(0, len(ids), plan['batch_size']):
                    batch = ids[start:start + plan['batch_size']]
                    digest.update(np.asarray([epoch], dtype=np.int64).tobytes() + batch.tobytes())
                    counts[batch] += 1; steps += 1
        assert np.all(counts[keep] == plan['epochs']) and not counts[~keep].any()
        for partner in [pair['pc'], pair['pi']]:
            destinations = np.zeros(len(ns), dtype=np.int64)
            np.add.at(destinations, partner, counts)
            np.testing.assert_array_equal(destinations, counts)
        for view in plan['views']:
            for condition in plan['conditions']:
                file = ROOT / 'fits' / f'n{i}_{view}_{condition}.json'
                record = json.loads(file.read_text())
                assert record['updates'] == steps and record['schedule_sha256'] == digest.hexdigest()
                assert record['anchor_exposures'] == int(counts.sum())
                assert datetime.fromisoformat(record['completed_utc']) < opened_time
                assert sha(file) == test_opened['fit_record_sha256'][file.name]
                assert record['source_seed'] == plan['source_seeds'][i % 3]
        reconstructed += 1
    output = {'status': 'complete', 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'independent_condition_means': len(summary['means']),
              'independent_paired_contrasts': len(summary['contrasts']),
              'multinomial_bootstrap_intervals': 2 * len(summary['contrasts']),
              'independent_schedules_and_exact_marginal_exposures': reconstructed,
              'all24_fits_completed_before_test_opened': True,
              'summary_sha256': sha(ROOT / 'summary.json'), 'verifier_sha256': sha(__file__)}
    atomic_json(ROOT / 'statistics_verification.json', output)
    files = {str(p): sha(p) for p in ROOT.rglob('*') if p.is_file()
             and p.name not in ['completion.json', 'state.json', 'current_job.json'] and not p.name.endswith('.tmp')}
    atomic_json(ROOT / 'completion.json', {'status': 'complete', 'completed_utc': datetime.now(timezone.utc).isoformat(),
                'operator_fits': 24, 'source_models': 3, 'statistics_verification': output,
                'primary_verification': json.loads((ROOT / 'verification.json').read_text()), 'artifact_sha256': files})
    print(json.dumps(output), flush=True)


def run():
    protocol = ROOT / 'statistics_verification_protocol.json'
    if not protocol.exists():
        atomic_json(protocol, {'registered_utc': datetime.now(timezone.utc).isoformat(),
                    'code_sha256': sha(__file__), 'multinomial_helper_sha256': sha('experiments/overnight_inverse_statistics_verify.py'),
                    'new_test_observed': (ROOT / 'test_opened.json').exists(),
                    'scope': 'Verification only; does not choose or change any fit or metric.'})
        (ROOT / 'statistics_verifier_snapshot.py').write_bytes(Path(__file__).read_bytes())
    while not (ROOT / 'verification.json').exists():
        if json.loads((ROOT / 'state.json').read_text()).get('status') == 'failed':
            raise RuntimeError('Factorial pipeline failed before final verification')
        time.sleep(10)
    verify()


if __name__ == '__main__':
    run()
