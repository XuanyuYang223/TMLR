"""Exact task-vector actions and finite witnesses for the eight native groups."""
from itertools import permutations
import json
from pathlib import Path
import sys

import numpy as np

from .analysis import linear_cka
from .longrun_engine import atomic_json
from .permutation_audit import transform
from .permworld_combinations import select_groups, sha
from .permworld_combinations_report import within_length_center
from .six_hour_report import write_rows


OPERATORS = ('complement', 'reverse', 'reverse_complement', 'inverse')


def certificates(group):
    tasks = group['tasks']
    identity = np.eye(len(tasks), dtype=np.int64)
    def mapping(changes):
        result = identity.copy()
        for task, terms in changes.items():
            result[tasks.index(task)] = 0
            for name, coefficient in terms.items(): result[tasks.index(task), tasks.index(name)] = coefficient
        return result
    if group['id'] in ('cycle_three', 'cycle_four'):
        return {op: (identity, 'Inverse preserves cycle lengths; reverse-complement is conjugation. Direct-sum cuts map bijectively.') for op in ('inverse', 'reverse_complement')}
    if group['id'] == 'position_four':
        matrix = mapping({'exceedances': {'deficiencies': 1}, 'deficiencies': {'exceedances': 1}})
        return {op: (matrix, 'Inverse and reverse-complement exchange above/below-diagonal positions and preserve cycle lengths.') for op in ('inverse', 'reverse_complement')}
    if group['id'] == 'position_none':
        matrix = mapping({'double_ascents': {'double_descents': 1}, 'double_descents': {'double_ascents': 1},
                          'successions': {'adjacencies': 1, 'successions': -1}})
        return {'complement': (matrix, 'Complement flips adjacent comparison signs; positive and negative unit steps partition adjacencies.'),
                'reverse': (matrix, 'Reverse flips triple order and unit-step direction; adjacency count is preserved.'),
                'reverse_complement': (identity, 'Compose the identical involutive reverse and complement actions.')}
    if group['id'] == 'interior_four':
        complement = mapping({'peaks': {'valleys': 1}, 'valleys': {'peaks': 1},
                              'double_ascents': {'double_descents': 1}, 'double_descents': {'double_ascents': 1}})
        reverse = mapping({'double_ascents': {'double_descents': 1}, 'double_descents': {'double_ascents': 1}})
        return {'complement': (complement, 'Value complement reverses both local comparison signs.'),
                'reverse': (reverse, 'Sequence reversal exchanges ascending/descending triples and preserves peaks/valleys.'),
                'reverse_complement': (complement@reverse, 'Compose the two commuting comparison-sign actions.')}
    if group['id'] == 'interior_none':
        matrix = mapping({'left_to_right_maxima': {'right_to_left_maxima': 1}, 'right_to_left_maxima': {'left_to_right_maxima': 1}})
        return {'reverse': (matrix, 'Reverse exchanges record direction while preserving interior peaks and valleys.')}
    return {}


def token_transform(inputs, lengths, operator):
    changed = inputs.copy()
    for i, n in enumerate(lengths):
        columns = 4+2*np.arange(n)
        value = tuple(map(int, inputs[i, columns]))
        changed[i, columns] = transform(value, operator)
    return changed


def run():
    plan = json.loads(Path('configs/six_hour_session.json').read_text())
    config = json.loads(Path(plan['base_config']).read_text())
    groups = select_groups(config)
    root = Path(plan['output']); output = root/'native_transform_audit'; output.mkdir(exist_ok=True)
    sys.path.insert(0, str(Path(config['repository'])/'src'))
    from neurips_permutations.math_ops import PROPERTY32_TASK_NAMES, PROPERTY_FUNCTIONS
    from neurips_permutations.passage import TOKEN_TO_ID
    assert all(TOKEN_TO_ID[f'{v:02d}'] == v for v in range(31))
    names = list(PROPERTY32_TASK_NAMES)
    tasks = sorted({t for g in groups for t in g['tasks']})
    known = {(g['id'], op): value for g in groups for op, value in certificates(g).items()}
    witnesses, buckets = {}, {}
    checked = 0
    def check(values, labels, transformed, n):
        for g in groups:
            ids = [tasks.index(t) for t in g['tasks']]
            y = labels[:, ids]
            for op in OPERATORS:
                key = g['id'], op
                yt = transformed[op][:, ids]
                if key in known:
                    assert np.array_equal(y@known[key][0].T, yt), key
                elif key not in witnesses:
                    store = buckets.setdefault((key, n), {})
                    for p, code, other in zip(values, y, yt):
                        code, other = tuple(map(int, code)), tuple(map(int, other))
                        if code in store and store[code][1] != other:
                            previous, previous_other = store[code]
                            witnesses[key] = {'n': n, 'permutation_a': list(previous), 'permutation_b': list(p),
                                              'same_source_answers': list(code), 'transformed_answers_a': list(previous_other),
                                              'transformed_answers_b': list(other)}
                            break
                        store.setdefault(code, (tuple(p), other))
    for n in range(2, 8):
        values = list(permutations(range(1, n+1)))
        lookup = {p: i for i, p in enumerate(values)}
        labels = np.array([[PROPERTY_FUNCTIONS[t](p) for t in tasks] for p in values])
        changed = {op: labels[[lookup[transform(p, op)] for p in values]] for op in OPERATORS}
        check(values, labels, changed, n); checked += len(values)
    small_witnesses = witnesses.copy()
    witnesses, buckets = {}, {}
    with np.load(root/'dataset/data.npz') as data:
        for n in range(10, 31):
            unresolved = [(g['id'], op) for g in groups for op in OPERATORS if (g['id'], op) not in known and (g['id'], op) not in witnesses]
            if not unresolved: break
            ids = np.flatnonzero(data['train_lengths'] == n)[:2000]
            values = [tuple(map(int, x[4+2*np.arange(n)])) for x in data['train_input'][ids]]
            labels = data['train_labels'][ids][:, [names.index(t) for t in tasks]]
            changed = {op: np.array([[PROPERTY_FUNCTIONS[t](transform(p, op)) for t in tasks] for p in values]) for op in OPERATORS}
            check(values, labels, changed, n); checked += len(values)
        original_inputs = data['representation_input'].copy()
        lengths = data['representation_lengths'].copy()
        original_labels = data['representation_labels'].copy()
        archived = {'original_lengths': lengths, 'original_input': original_inputs}
        all_source_inputs = {tuple(x) for x in data['train_input']}
    label_kernels, cases = [], []
    for op in OPERATORS:
        inputs = token_transform(original_inputs, lengths, op)
        values = [tuple(map(int, x[4+2*np.arange(n)])) for x, n in zip(inputs, lengths)]
        labels = np.array([[PROPERTY_FUNCTIONS[t](p) for t in names] for p in values])
        archived[op+'_input'], archived[op+'_labels'] = inputs, labels
        overlap = sum(tuple(x) in all_source_inputs for x in inputs)
        for group in groups:
            ids = [names.index(t) for t in group['tasks']]
            y, yt = original_labels[:, ids], labels[:, ids]
            key = group['id'], op
            if key in known:
                matrix, proof = known[key]
                assert np.array_equal(y@matrix.T, yt)
                unary = np.array_equal(matrix@matrix.T, np.eye(4)) and np.all(matrix >= 0)
                case = {'group': group['id'], 'operator': op, 'status': 'proved linear task-vector closure',
                        'categorical_permutation_action': bool(unary), 'action_matrix': matrix.tolist(), 'proof': proof}
            elif key in witnesses:
                case = {'group': group['id'], 'operator': op, 'status': 'not a function of source answers alone', 'witness': witnesses[key]}
            else:
                case = {'group': group['id'], 'operator': op, 'status': 'unresolved, no closure claim'}
            cases.append(case)
            h, ht = np.eye(31)[y].reshape(len(y), -1), np.eye(31)[yt].reshape(len(y), -1)
            for mode in ('raw', 'within_length'):
                pair = [within_length_center(z, lengths) for z in (h, ht)] if mode == 'within_length' else [h, ht]
                cka = linear_cka(*pair)
                if key in known and case['categorical_permutation_action']: assert np.isclose(cka, 1., atol=1e-12)
                label_kernels.append({'group': group['id'], 'operator': op, 'centering': mode, 'categorical_code_cka': cka,
                                      'closure_status': case['status'], 'source_training_overlap_count': overlap})
    np.savez_compressed(output/'transformed_representation.npz', **archived)
    atomic_json(output/'audit.json', {'code_sha256': sha(__file__), 'upstream_property_sha256': sha(Path(config['repository'])/'src/neurips_permutations/math_ops.py'),
        'cases': cases, 'checked_permutations_for_certificates_and_witnesses': checked,
        'small_domain_nonclosure_witnesses': [{'group': key[0], 'operator': key[1], **value} for key, value in small_witnesses.items()],
        'primary_witness_domain': 'actual source length range 10-30; small-n witnesses retained separately',
        'transformed_data_sha256': sha(output/'transformed_representation.npz'),
        'scope': 'proved actions use combinatorial arguments; one finite counterexample disproves universal source-answer closure',
        'labels_scope': 'exact mathematical source-label kernels; learned hidden ordering is not guaranteed'})
    write_rows(output/'categorical_kernel.csv', label_kernels)
    print(json.dumps({'proved_closures': len(known), 'nonclosure_witnesses': len(witnesses), 'unresolved': 32-len(known)-len(witnesses), 'checked': checked}))


if __name__ == '__main__': run()
