"""Exact two-member collision statistics and within-pair orientation controls."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .longrun_engine import atomic_json
from .until_10_verify import read, digest

ROOT = Path('results/collision_pair_followup')
CASES = {'base_CI': ('ci', 5), 'base_ICI': ('ici', 2),
    'observed_C_then_I': ('ci', 5), 'observed_I_then_CI': ('ici', 2)}


def now():
    return datetime.now(timezone.utc).isoformat()


def pair_statistics(predictions, labels, pair_ids):
    pairs = np.array([np.flatnonzero(pair_ids == p) for p in np.unique(pair_ids)])
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError('Each collision pair must have exactly two members')
    answers, truth = predictions[pairs], labels[pairs]
    if not np.all(truth[:, 0] != truth[:, 1]):
        raise ValueError('Paired hidden answers must differ')
    hit = answers == truth
    counts = np.bincount(hit.sum(1), minlength=3)
    both = hit.all(1)
    swapped_both = (answers == truth[:, ::-1]).all(1)
    assert not np.any(both & swapped_both)
    count = len(pairs)
    n_covered = int(both.sum() + swapped_both.sum())
    row = {'pairs': count, 'both_correct_count': int(counts[2]), 'exactly_one_correct_count': int(counts[1]),
        'neither_correct_count': int(counts[0]), 'both_correct_fraction': float(counts[2] / count),
        'exactly_one_correct_fraction': float(counts[1] / count), 'neither_correct_fraction': float(counts[0] / count),
        'answer_accuracy': float(hit.mean()), 'both_correct_after_swap_count': int(swapped_both.sum()),
        'both_correct_after_swap_fraction': float(swapped_both.mean()),
        'orientation_excess_all_pairs': float((both.sum() - swapped_both.sum()) / count),
        'random_pair_orientation_expected_both_fraction': float((both.sum() + swapped_both.sum()) / (2 * count)),
        'covered_candidate_pairs': n_covered, 'covered_candidate_pair_fraction': n_covered / count,
        'correct_orientation_given_covered_candidates': float(both.sum() / n_covered) if n_covered else None,
        'deterministic_known_scalar_only_both_correct_ceiling': 0.,
        'scope': 'Both correct means both members of the same input collision pair. Swapping preserves the two predictions and targets. Conditional orientation uses only pairs where predictions cover both true candidate answers; it is a diagnostic, not an inference method or unfiltered accuracy.'}
    assert sum(counts) == count
    np.testing.assert_allclose(row['answer_accuracy'], .5 + .5 * (row['both_correct_fraction'] - row['neither_correct_fraction']), atol=1e-15)
    arrays = {'pair_ids': np.unique(pair_ids), 'predicted_answers': answers, 'true_answers': truth,
        'correct': hit, 'both_correct': both, 'swapped_both_correct': swapped_both}
    return row, arrays


def jobs():
    result = []
    # Native and calibration-only posthoc operators, with separate CI/ICI.
    roots = [(Path('results/algebra_hidden_relations'), 'original_3'),
        (Path('results/relation_seed_extension'), 'new_6')]
    roots += [(p, 'new_worlds_3') for p in sorted(Path('results/relation_world_confirmation').glob('world*'))]
    for folder, cohort in roots:
        data = folder / 'probe_dataset.npz' if cohort != 'new_6' else Path('results/algebra_hidden_relations/probe_dataset.npz')
        for path in sorted((folder / 'evaluations').glob('*.json')):
            rec = read(path)
            source = rec.get('source', f"{rec['condition']}_s{rec['seed']}")
            for method in rec['methods']:
                if method.get('view', 'query') != 'query' or method['method'] not in ['native_operators', 'posthoc_correct_generators', 'posthoc_shuffled_generators']:
                    continue
                suffix = '_query_' if cohort == 'original_3' else '_'
                archive = folder / 'arrays' / (source + suffix + method['method'] + '.npz')
                for word, action in [('ci', 5), ('ici', 2)]:
                    metric = next(r for r in method.get('metrics', method.get('results', [])) if r['split'] == 'answer_collisions' and r['word'] == word)
                    result.append({'family': 'native_or_posthoc', 'cohort': cohort, 'source': source, 'condition': rec['condition'],
                        'seed': rec['seed'], 'method': method['method'], 'case': 'base_' + word.upper(), 'action': action,
                        'record_path': str(path), 'archive_path': str(archive), 'archive_expected_sha256': rec['array_sha256'][archive.name],
                        'answer_key': 'answer_collisions_' + word + '_answers', 'data_path': str(data), 'existing_metric': metric})
    # Exactly the k=256 identical-frozen-backbone comparison previously reported.
    for folder, cohort in [(Path('results/frozen_relation_followup'), 'original_3'), (Path('results/relation_frozen_seed_confirmation'), 'new_6')]:
        for path in sorted((folder / 'evaluations').glob('*.json')):
            rec = read(path)
            if rec['fit_anchors_per_length'] != 256:
                continue
            suffix = '_k256_' if cohort == 'original_3' else '_'
            archive = folder / 'maps' / (rec['source'] + suffix + rec['pairing'] + '.npz')
            for word, action in [('ci', 5), ('ici', 2)]:
                metric = next(r for r in rec['metrics'] if r['split'] == 'answer_collisions' and r['word'] == word)
                result.append({'family': 'identical_frozen', 'cohort': cohort, 'source': rec['source'], 'condition': rec['source_condition'],
                    'seed': rec['seed'], 'method': rec['pairing'], 'case': 'base_' + word.upper(), 'action': action,
                    'record_path': str(path), 'archive_path': str(archive), 'archive_expected_sha256': rec['map_archive_sha256'],
                    'answer_key': 'answer_collisions_' + word + '_answers',
                    'data_path': 'results/algebra_hidden_relations/probe_dataset.npz', 'existing_metric': metric})
    # Extra actual known-state forward inputs: a separate, labeled diagnostic.
    observed_roots = [(Path('results/relation_observed_start/evaluations'), None)]
    observed_roots += [(p / 'observed_start', p) for p in sorted(Path('results/relation_world_confirmation').glob('world*'))]
    for folder, world in observed_roots:
        for path in sorted(folder.glob('*.json')):
            rec = read(path)
            cohort = 'new_worlds_3' if world else ('original_3' if rec['scope'] == 'original_exploratory' else 'new_6')
            source, condition = rec['source'], rec.get('condition', rec.get('source_condition'))
            seed = rec['world']['source_seed'] if world else rec['seed']
            data = world / 'probe_dataset.npz' if world else Path('results/algebra_hidden_relations/probe_dataset.npz')
            for method in rec['methods']:
                archive = folder / (source + '_' + method['method'] + '.npz') if world else folder.parent / 'arrays' / (source + '_' + method['method'] + '.npz')
                for metric in method.get('metrics', method.get('rows', [])):
                    if metric['split'] != 'answer_collisions':
                        continue
                    case = metric['case']
                    if case.startswith('base_'):
                        if not world or method['method'] not in ['frozen_full_correct', 'frozen_full_shuffled']:
                            continue
                        family, label = 'identical_frozen', method['method'].removeprefix('frozen_full_')
                    else:
                        family, label = 'observed_known_start', method['method']
                    result.append({'family': family, 'cohort': cohort, 'source': source, 'condition': condition, 'seed': seed,
                        'method': label, 'case': case, 'action': CASES[case][1], 'record_path': str(path),
                        'archive_path': str(archive), 'archive_expected_sha256': method['prediction_archive_sha256'],
                        'answer_key': 'answer_collisions_' + case + '_answers', 'data_path': str(data), 'existing_metric': metric})
    return result


def run():
    planned = jobs()
    for folder in [ROOT, ROOT / 'evaluations', ROOT / 'arrays']:
        folder.mkdir(parents=True, exist_ok=True)
    records = sorted({j['record_path'] for j in planned})
    data_paths = sorted({j['data_path'] for j in planned})
    signature = {'code_sha256': digest(__file__), 'input_records_sha256': {p: digest(p) for p in records},
        'input_data_sha256': {p: digest(p) for p in data_paths},
        'jobs': [{k: v for k, v in j.items() if k != 'existing_metric'} for j in planned],
        'prior_completed_study_sha256': digest('results/until_10_followup/completion.json'),
        'scope': 'Exploratory reanalysis of previously inspected saved answers. No new fitting, source training, pair selection or threshold changes. Report both/one/neither, the exact link to per-permutation accuracy and prediction-preserving within-pair swapping. Positive double correctness alone is not evidence against input-dependent noise; orientation excess and existing wrong-relation controls are separate checks. CI and ICI share a true missing statistic and are averaged within each source before pooled source summaries. The source-conditional pairing contrast uses identical frozen weights. Additional true known-state inputs have a separate scope.'}
    protocol = ROOT / 'protocol.json'
    if protocol.exists():
        assert read(protocol)['signature'] == signature
    else:
        atomic_json(protocol, {'registered_utc': now(), 'new_pair_evaluations': 0, 'signature': signature})
    datasets = {p: dict(np.load(p)) for p in data_paths}
    for index, job in enumerate(planned):
        name = '_'.join([job['cohort'], job['source'], job['family'], job['method'], job['case']])
        dest = ROOT / 'evaluations' / (name + '.json')
        if dest.exists():
            continue
        assert digest(job['archive_path']) == job['archive_expected_sha256']
        data = datasets[job['data_path']]
        mask = data['split'] == 3
        with np.load(job['archive_path']) as archive:
            answers = archive[job['answer_key']]
        ids = data['pair_ids'][mask]
        pair_indices = np.array([np.flatnonzero(ids == p) for p in np.unique(ids)])
        labels = data['labels'][mask]
        np.testing.assert_array_equal(labels[pair_indices[:, 0]][:, [0, 1, 4]], labels[pair_indices[:, 1]][:, [0, 1, 4]])
        np.testing.assert_array_equal(data['lengths'][mask][pair_indices[:, 0]], data['lengths'][mask][pair_indices[:, 1]])
        np.testing.assert_array_equal(labels[:, 5], labels[:, 2])
        metrics, arrays = pair_statistics(answers, labels[:, job['action']], ids)
        for key in ['answer_accuracy', 'pair_both_correct']:
            observed = metrics['answer_accuracy'] if key == 'answer_accuracy' else metrics['both_correct_fraction']
            np.testing.assert_allclose(observed, job['existing_metric'][key], atol=1e-15)
        ap = ROOT / 'arrays' / (name + '.npz')
        np.savez_compressed(ap, **arrays)
        atomic_json(dest, {**{k: v for k, v in job.items() if k != 'existing_metric'}, 'input_record_sha256': digest(job['record_path']),
            'array_path': str(ap), 'array_sha256': digest(ap), 'metrics': metrics, 'completed_utc': now()})
    completed = len(list((ROOT / 'evaluations').glob('*.json')))
    assert completed == len(planned)
    atomic_json(ROOT / 'state.json', {'status': 'complete', 'endpoints': completed,
        'families': dict(Counter(j['family'] for j in planned)), 'updated_utc': now()})
    print({'pair_followup_complete': completed, 'families': dict(Counter(j['family'] for j in planned))}, flush=True)


if __name__ == '__main__':
    run()
