"""Verify all main fit budgets independently while supplements may still run."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from .inverse_functional_alignment import now
from .longrun_engine import atomic_json
from .overnight_inverse_functional import initialize
from .permworld_combinations import sha


def verify(plan, root, signature):
    state = json.loads((root / 'training/state.json').read_text())
    assert state['status'] == 'all_targets_complete' and state['target_models'] == 36
    expected_jobs = {f"{rep['id']}_{condition}.json" for rep in plan['replicates'] for condition in plan['conditions']}
    actual_jobs = {p.name for p in (root / 'training').glob('n*_*.json')}
    assert expected_jobs == actual_jobs
    schedules, pairings, checkpoints = 0, 0, 0
    for rep in plan['replicates']:
        labeled_data = np.load(root / 'dataset' / rep['id'] / 'support/dataset.npz')
        pool = np.load(root / 'dataset' / rep['id'] / 'unlabeled/dataset.npz')
        assert 'labels' not in pool.files
        assert int((labeled_data['split'] == 0).sum()) == 192
        assert int((labeled_data['split'] == 1).sum()) == 64
        lbuckets = {n: np.array([i for i, (length, split) in enumerate(zip(labeled_data['lengths'], labeled_data['split'])) if int(length) == n and split == 0]) for n in plan['lengths']}
        ubuckets = {n: np.array([i for i, length in enumerate(pool['lengths']) if int(length) == n]) for n in plan['lengths']}
        lrng = np.random.default_rng(rep['training_seed'] + 81001)
        urng = np.random.default_rng(rep['unlabeled_seed'] + 81001)
        lrows, urows = [], []
        for _ in range(40000):
            n = int(lrng.choice(plan['lengths']))
            lrows.append(lrng.choice(lbuckets[n], 32, replace=False))
            urows.append(urng.choice(ubuckets[n], 32, replace=False))
        schedule = root / 'training' / f"{rep['id']}_schedule.npz"
        saved = np.load(schedule)
        np.testing.assert_array_equal(saved['labeled'], np.asarray(lrows))
        np.testing.assert_array_equal(saved['unlabeled'], np.asarray(urows))
        lindex, uindex = saved['labeled'], saved['unlabeled']
        assert lindex.shape == uindex.shape == (40000, 32)
        assert np.all(labeled_data['split'][lindex] == 0)
        assert np.all(labeled_data['lengths'][lindex] == pool['lengths'][uindex])
        assert np.all(labeled_data['lengths'][lindex] == labeled_data['lengths'][lindex[:, :1]])
        for rows in [lindex, uindex]:
            assert np.all(np.diff(np.sort(rows, axis=1), axis=1) != 0)
        schedules += 1
        for part, data in [('support', labeled_data), ('unlabeled', pool)]:
            teacher = np.load(root / 'teacher' / f"{rep['id']}_{part}.npz")
            pseudo = teacher['logits'].argmax(-1)
            np.testing.assert_array_equal(pseudo, teacher['predicted_answer'])
            answers = data['labels'] if part == 'support' else pseudo
            candidate = data['split'] == 0 if part == 'support' else np.ones(len(answers), dtype=bool)
            frequency = Counter((int(n), int(y)) for n, y, use in zip(data['lengths'], answers, candidate) if use)
            expected = np.array([bool(use) and frequency[int(n), int(y)] >= 2 for n, y, use in zip(data['lengths'], answers, candidate)])
            pairing = np.load(root / 'teacher' / f"{rep['id']}_{part}_mismatch.npz")
            np.testing.assert_array_equal(expected, pairing['eligible'])
            row = np.flatnonzero(expected)
            partner = pairing['partner'][row]
            assert np.all(row != partner) and len(set(partner)) == len(row)
            assert np.all(candidate[partner])
            np.testing.assert_array_equal(answers[row], answers[partner])
            np.testing.assert_array_equal(data['lengths'][row], data['lengths'][partner])
            pairings += 1
        target = next(r for r in signature['targets'] if r['seed'] == rep['target_pretrain_seed'])
        for condition in plan['conditions']:
            name = rep['id'] + '_' + condition
            record = json.loads((root / 'training' / f'{name}.json').read_text())
            assert record['status'] == 'complete' and record['step'] == 40000
            assert record['replicate'] == rep and record['condition'] == condition
            assert record['schedule_sha256'] == sha(schedule)
            assert record['initialization_sha256'] == target['checkpoint_sha256']
            assert record['forward_rows'] == 40000 * 64
            assert record['training_labels'] == 192 and record['validation_labels'] == 64
            assert record['unlabeled_inputs'] == 16384
            assert record['distinct_unlabeled_exposed'] == len(np.unique(uindex))
            assert [r['step'] for r in record['curve']] == [0, 1200, 5000, 10000, 20000, 40000]
            for step in plan['checkpoint_updates']:
                file = root / 'checkpoints' / f'{name}_u{step}.pt'
                assert sha(file) == record['checkpoint_sha256'][str(step)]
                checkpoints += 1
        print({'independently_verified_main_schedule': rep['id']}, flush=True)
    return {'status': 'complete', 'main_fit_records_checked': 36,
            'paired_main_schedules_reconstructed': schedules,
            'answer_preserving_derangements_checked': pairings,
            'main_checkpoint_hashes_checked': checkpoints,
            'each_fit_updates': 40000, 'each_fit_forward_rows': 2560000,
            'true_U_answers_read': False}


def run():
    plan, root, signature = initialize()
    folder = root / 'main_budget_verification'
    folder.mkdir(exist_ok=True)
    path = folder / 'protocol.json'
    registration = {'code_sha256': sha(__file__), 'main_protocol_sha256': sha(root / 'protocol.json'),
                    'scope': 'Independently reconstruct all six main schedules, twelve pairings,36 record budgets and180 checkpoint hashes. No supplemental training dependencies or new outcome selection.'}
    if path.exists():
        assert json.loads(path.read_text())['signature'] == registration
    else:
        assert not (root / 'test_opened.json').exists()
        atomic_json(path, {'registered_utc': now(), 'signature': registration, 'test_opened': False})
    deadline = datetime.fromisoformat(plan['deadline_utc']).timestamp()
    while True:
        statefile = root / 'training/state.json'
        state = json.loads(statefile.read_text()) if statefile.exists() else {}
        if state.get('status') == 'all_targets_complete' and state.get('target_models') == 36:
            break
        if time.time() >= deadline:
            atomic_json(folder / 'state.json', {'status': 'incomplete', 'updated_utc': now()})
            return
        time.sleep(10)
    result = verify(plan, root, signature)
    atomic_json(folder / 'verification.json', {'completed_utc': now(), **result})
    print(result, flush=True)


if __name__ == '__main__':
    run()
