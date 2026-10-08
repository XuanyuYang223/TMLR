"""Recompute every native geometry endpoint from saved feature artifacts."""
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from .longrun_engine import atomic_json
from .longrun_geometry import analyze, analyze_transforms
from .native_transform_audit import OPERATORS, token_transform
from .permutation_audit import transform
from .permworld_combinations import select_groups, sha


def read_csv(path):
    with Path(path).open() as handle: return list(csv.DictReader(handle))


def verify():
    plan = json.loads(Path('configs/six_hour_session.json').read_text())
    config = json.loads(Path(plan['base_config']).read_text())
    root = Path(plan['output']); output = root/'landmarks'
    groups = select_groups(config)
    arch = json.loads((root/'architecture_selection.json').read_text())['selected']
    meta = json.loads((output/'metadata.json').read_text())
    for key, path in (('code_sha256', 'experiments/longrun_geometry.py'),
                      ('query_extraction_sha256', 'experiments/longrun_transfer.py'),
                      ('data_sha256', root/'dataset/data.npz'),
                      ('transformed_data_sha256', root/'native_transform_audit/transformed_representation.npz')):
        assert meta[key] == sha(path)
    signature = {k:v for k,v in meta.items() if k != 'fingerprint'}
    assert meta['fingerprint'] == hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    audit = json.loads((root/'native_transform_audit/audit.json').read_text())
    assert audit['code_sha256'] == sha('experiments/native_transform_audit.py')
    sys.path.insert(0, str(Path(config['repository'])/'src'))
    from neurips_permutations.math_ops import PROPERTY_FUNCTIONS
    source_witnesses = 0
    by_group = {g['id']:g for g in groups}
    for case in audit['cases']:
        if 'witness' not in case: continue
        witness = case['witness']; tasks = by_group[case['group']]['tasks']
        first, second = witness['permutation_a'], witness['permutation_b']
        assert len(first) == len(second) == witness['n'] and 10 <= witness['n'] <= 30
        answer = lambda p: [PROPERTY_FUNCTIONS[t](p) for t in tasks]
        assert answer(first) == answer(second) == witness['same_source_answers']
        assert answer(transform(first, case['operator'])) == witness['transformed_answers_a']
        assert answer(transform(second, case['operator'])) == witness['transformed_answers_b']
        assert witness['transformed_answers_a'] != witness['transformed_answers_b']
        source_witnesses += 1
    with np.load(root/'dataset/data.npz') as data:
        original = data['representation_input']; lengths = data['representation_lengths'].copy()
        train_keys = {tuple(x) for x in data['train_input']}
        with np.load(root/'native_transform_audit/transformed_representation.npz') as transformed:
            for op in OPERATORS:
                changed = token_transform(original, lengths, op)
                np.testing.assert_array_equal(changed, transformed[op+'_input'])
                np.testing.assert_array_equal(token_transform(changed, lengths, op), original)
                assert not any(tuple(x) in train_keys for x in changed)
    with tempfile.TemporaryDirectory(prefix='native-geometry-check-') as temporary:
        target = Path(temporary)
        os.symlink(root.resolve()/'multi', target/'multi')
        os.symlink(root.resolve()/'initial', target/'initial')
        os.symlink(root.resolve()/'native_transform_audit', target/'native_transform_audit')
        (target/'landmarks').mkdir()
        for path in output.glob('*.npy'):
            os.symlink(path.resolve(), target/'landmarks'/path.name)
        checked_plan = {**plan, 'output': str(target)}
        with contextlib.redirect_stdout(io.StringIO()):
            analyze(checked_plan, groups, arch, lengths)
            analyze_transforms(checked_plan, groups, arch, lengths)
        counts = {}
        for filename in ('cross_seed.csv', 'cross_group.csv', 'transformed_cka.csv'):
            expected, actual = read_csv(output/filename), read_csv(target/'landmarks'/filename)
            assert len(expected) == len(actual)
            for left, right in zip(expected, actual):
                assert set(left) == set(right)
                for key in left:
                    if key in ('trained_cka','initial_cka','change_from_initial','linear_cka','categorical_code_cka'):
                        assert abs(float(left[key])-float(right[key])) < 1e-12
                    else: assert left[key] == right[key]
            counts[filename] = len(expected)
    result = {'status':'passed', 'recomputed_geometry_rows':counts,
              'actual_length_nonclosure_witnesses_recomputed':source_witnesses,
              'transformed_inputs_involutive_and_disjoint_from_source_train':True,
              'extraction_code_data_and_analysis_inputs_fingerprinted':True,
              'feature_files_sha256':{p.name:sha(p) for p in output.glob('*.npy')},
              'verifier_sha256':sha(__file__)}
    atomic_json(output/'verification.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'feature_files_sha256'}, indent=2))


if __name__ == '__main__': verify()
