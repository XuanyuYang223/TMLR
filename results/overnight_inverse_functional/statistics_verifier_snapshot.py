"""Audit completed main aggregates; no fitting, selection or new outcomes."""
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
from math import factorial, fsum, isclose
from pathlib import Path


ROOT = Path('results/overnight_inverse_functional')


def close(actual, expected):
    assert isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-10), (actual, expected)


def average(values):
    return fsum(values) / len(values)


def bootstrap_interval(values):
    """Enumerate count vectors and multinomial weights, not index draws."""
    n = len(values)
    distribution = []
    for prefix in product(range(n + 1), repeat=n - 1):
        last = n - sum(prefix)
        if last < 0:
            continue
        counts = (*prefix, last)
        weight = factorial(n)
        for count in counts:
            weight //= factorial(count)
        value = fsum(c * v for c, v in zip(counts, values)) / n
        distribution.extend([value] * weight)
    assert len(distribution) == n ** n
    distribution.sort()
    output = []
    for probability in (0.025, 0.975):
        position = (len(distribution) - 1) * probability
        lower = int(position)
        fraction = position - lower
        output.append(distribution[lower] * (1 - fraction)
                      + distribution[min(lower + 1, len(distribution) - 1)] * fraction)
    return output


def run():
    summary_path = ROOT / 'summary.json'
    summary = json.loads(summary_path.read_text())
    plan = json.loads(Path('configs/overnight_inverse_functional.json').read_text())
    records = {(r['replicate'], r['condition'], r['step']): r for r in summary['records']}
    assert len(records) == 186
    for row in records.values():
        geometry = row['geometry']
        for key in ('correct_cka', 'matched_correct_cka', 'matched_wrong_cka',
                    'answer_residual_correct_cka', 'answer_residual_wrong_cka'):
            close(geometry[key], average([cell[key] for cell in row['per_length_geometry']]))
        close(geometry['matched_contrast'], geometry['matched_correct_cka'] - geometry['matched_wrong_cka'])
        close(geometry['answer_residual_contrast'], geometry['answer_residual_correct_cka'] - geometry['answer_residual_wrong_cka'])
    for mean in summary['means']:
        rows = [records[(rep['id'], mean['condition'], mean['step'])] for rep in plan['replicates']]
        close(mean['accuracy'], average([row['accuracy'] for row in rows]))
        for group in ('modal', 'nonmodal'):
            close(mean[group + '_accuracy'], average([row[group]['accuracy'] for row in rows]))
        for key, value in mean['geometry'].items():
            close(value, average([row['geometry'][key] for row in rows]))
    assert {(r['step'], r['contrast']) for r in summary['contrasts']} == {
        (step, contrast) for step in plan['checkpoint_updates'] for contrast in plan['primary_contrasts']}
    statistics = 0
    for contrast in summary['contrasts']:
        assert contrast['primary'] == (contrast['step'] == plan['updates'])
        left, right = contrast['contrast'].split('-')
        values = {}
        for rep in plan['replicates']:
            a = records[(rep['id'], left, contrast['step'])]
            b = records[(rep['id'], right, contrast['step'])]
            delta = {'accuracy_pp': 100 * (a['accuracy'] - b['accuracy'])}
            for group in ('modal', 'nonmodal'):
                difference = 100 * (a[group]['accuracy'] - b[group]['accuracy'])
                delta[group + '_accuracy_pp'] = difference
                delta[group + '_contribution_pp'] = difference * a[group]['count'] / summary['test_examples']
            close(delta['accuracy_pp'], delta['modal_contribution_pp'] + delta['nonmodal_contribution_pp'])
            for key, geometry_key in [('cka', 'correct_cka'), ('matched_contrast', 'matched_contrast'),
                                      ('answer_residual_contrast', 'answer_residual_contrast')]:
                delta[key] = a['geometry'][geometry_key] - b['geometry'][geometry_key]
            for key, value in delta.items():
                values.setdefault(key, []).append(value)
        for key, differences in values.items():
            reported = contrast[key]
            close(reported['mean'], average(differences))
            assert reported['positive_repeats'] == sum(value > 0 for value in differences)
            clusters = [average([value for rep, value in zip(plan['replicates'], differences)
                                 if rep['source_seed'] == seed]) for seed in (17, 42, 101)]
            for field, expected in [('paired_values', differences), ('three_pair_means', clusters),
                                    ('paired_bootstrap_95', bootstrap_interval(differences)),
                                    ('three_pair_bootstrap_95', bootstrap_interval(clusters))]:
                assert len(reported[field]) == len(expected)
                for actual, value in zip(reported[field], expected):
                    close(actual, value)
            statistics += 1
    output = {'status': 'complete', 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'audit_performed_after_outcomes': True, 'new_models_or_analysis_choices': False,
              'summary_sha256': sha256(summary_path.read_bytes()).hexdigest(),
              'verifier_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
              'record_geometry_aggregates_checked': len(records),
              'condition_endpoint_means_checked': len(summary['means']),
              'paired_contrasts_checked': len(summary['contrasts']),
              'paired_statistics_checked': statistics,
              'multinomial_bootstrap_intervals_checked': statistics * 2}
    (ROOT / 'statistics_verification.json').write_text(json.dumps(output, indent=2) + '\n')
    (ROOT / 'statistics_verifier_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(output))


if __name__ == '__main__':
    run()
