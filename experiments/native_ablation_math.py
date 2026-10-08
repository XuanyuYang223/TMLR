"""Task-set actions from counting/record rules, independent of group names."""
import numpy as np
from .native_transform_audit import OPERATORS, token_transform
from .permutation_audit import IDENTITIES
from .permworld_combinations import sha
from .permworld_combinations_report import within_length_center
from .analysis import linear_cka
from .native_transfer_prediction import label_statistics


def rules():
    c, r, rc, inv = {}, {}, {}, {}
    def swap(table, a, b):
        table[a] = {b: 1}; table[b] = {a: 1}
    for table in (c, r):
        swap(table, 'double_ascents', 'double_descents')
        swap(table, 'fixed_points', 'anti_fixed_points')
        table['adjacencies'] = {'adjacencies': 1}
        table['successions'] = {'adjacencies': 1, 'successions': -1}
    swap(c, 'peaks', 'valleys')
    r.update({t: {t: 1} for t in ('peaks', 'valleys')})
    swap(c, 'left_to_right_maxima', 'left_to_right_minima')
    swap(c, 'right_to_left_maxima', 'right_to_left_minima')
    swap(r, 'left_to_right_maxima', 'right_to_left_maxima')
    swap(r, 'left_to_right_minima', 'right_to_left_minima')
    for t, terms in c.items():
        if all(u in r for u in terms):
            result = {}
            for u, a in terms.items():
                for v, b in r[u].items(): result[v] = result.get(v, 0) + a*b
            rc[t] = {v: a for v, a in result.items() if a}
    for table in (rc, inv):
        swap(table, 'exceedances', 'deficiencies')
        for t in ('fixed_points', 'anti_fixed_points', 'cycle_count', 'two_cycle_count',
                  'three_cycle_count', 'even_cycle_count', 'odd_cycle_count', 'longest_cycle',
                  'shortest_cycle', 'nontrivial_cycle_count', 'components'):
            table[t] = {t: 1}
    # Transposing a permutation diagram preserves southwest/northeast corner
    # records and exchanges northwest with southeast corner records.
    swap(inv, 'left_to_right_maxima', 'right_to_left_minima')
    for t in ('left_to_right_minima', 'right_to_left_maxima'): inv[t] = {t: 1}
    return dict(zip(OPERATORS, (c, r, rc, inv)))


def actions(tasks):
    result = {}
    for op, table in rules().items():
        if all(t in table and set(table[t]) <= set(tasks) for t in tasks):
            result[op] = np.array([[table[t].get(u, 0) for u in tasks] for t in tasks], dtype=np.int64)
    return result


def descriptor(tasks, raw, names, functions):
    ids = [names.index(t) for t in tasks]
    x, n, y = (raw['representation_'+k] for k in ('input', 'lengths', 'labels'))
    centered = within_length_center(np.eye(31)[y[:, ids]].reshape(len(y), -1), n)
    matrices = actions(tasks); ckas = []
    for op in OPERATORS:
        changed = token_transform(x, n, op)
        other = np.array([[functions[t](tuple(map(int, row[4+2*np.arange(int(length))])))
                           for t in tasks] for row, length in zip(changed, n)])
        if op in matrices: np.testing.assert_array_equal(y[:, ids]@matrices[op].T, other)
        ckas.append(linear_cka(centered, within_length_center(np.eye(31)[other].reshape(len(y), -1), n)))
    supports = [len(i['terms']) for i in IDENTITIES if set(i['terms']) <= set(tasks)]
    minimum = min(supports) if supports else 5
    unary = sum(bool(np.isin(m, [0, 1]).all() and (m.sum(0)==1).all() and (m.sum(1)==1).all()) for m in matrices.values())
    math = [minimum, float(minimum==5), float(np.mean(ckas)), len(matrices)/4, unary/4]
    stats = label_statistics(raw['train_labels'][:, ids], raw['train_labels'][:, names.index('lis_length')], raw['train_lengths'])
    return {'math': math, 'label_stats': stats, 'certified_operators': list(matrices),
            'scope': 'covered exact rules; missing closure does not establish absence of all relations'}
