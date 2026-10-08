"""Small prime-field utilities; all ranks and dependencies are exact."""
from itertools import combinations, product
import numpy as np


def validate_prime(p):
    if p < 3 or any(p % k == 0 for k in range(2, int(p ** .5) + 1)):
        raise ValueError('p must be an odd prime')


def rank_mod(matrix, p):
    a = np.array(matrix, dtype=np.int64, copy=True) % p
    if a.ndim != 2:
        raise ValueError('expected a matrix')
    row = 0
    for col in range(a.shape[1]):
        pivots = np.flatnonzero(a[row:, col])
        if not len(pivots):
            continue
        pivot = row + pivots[0]
        a[[row, pivot]] = a[[pivot, row]]
        a[row] = a[row] * pow(int(a[row, col]), -1, p) % p
        for other in range(a.shape[0]):
            if other != row:
                a[other] = (a[other] - a[other, col] * a[row]) % p
        row += 1
        if row == a.shape[0]:
            break
    return row


def projective(vector, p):
    vector = np.asarray(vector, dtype=np.int64) % p
    nonzero = np.flatnonzero(vector)
    if not len(nonzero):
        raise ValueError('zero does not define a task')
    return tuple((vector * pow(int(vector[nonzero[0]]), -1, p) % p).tolist())


def circuits(sources, p):
    result = []
    for count in range(2, len(sources) + 1):
        for ids in combinations(range(len(sources)), count):
            if rank_mod(sources[list(ids)], p) < count and not any(set(c) <= set(ids) for c in result):
                result.append(ids)
    return result


def composition_size(sources, target, p):
    for size in range(1, len(sources) + 1):
        for ids in combinations(range(len(sources)), size):
            subset = sources[list(ids)]
            if rank_mod(np.vstack([subset, target]), p) == rank_mod(subset, p):
                return size
    return None


def scenarios():
    basis = np.eye(3, dtype=np.int64)
    coefficients = [(1, 1, 0), (1, 2, 0), (2, 1, 0), (1, 4, 0),
                    (1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1)]
    return [{'id': ('A' if c[2] == 0 else 'B') + ''.join(map(str, c)),
             'family': 'A' if c[2] == 0 else 'B',
             'sources': np.vstack([basis, c])} for c in coefficients]


def target_pool(groups, p, limit):
    excluded = {projective(v, p) for g in groups for v in g['sources']}
    candidates = sorted({projective(v, p) for v in product(range(p), repeat=3) if any(v)} - excluded)
    # Select on algebra alone, before any network is trained. Prefer diverse
    # composition-size profiles, then fill deterministically.
    profiles, chosen = set(), []
    for v in candidates:
        profile = tuple(composition_size(g['sources'], v, p) for g in groups)
        if profile not in profiles:
            chosen.append(v)
            profiles.add(profile)
    chosen += [v for v in candidates if v not in chosen]
    if limit > len(candidates):
        raise ValueError('requested more eligible targets than exist')
    return np.array(chosen[:limit], dtype=np.int64)


def world(p, dimension, seed):
    validate_prime(p)
    if dimension < 3:
        raise ValueError('dimension must be at least three')
    rng = np.random.default_rng(seed)
    while True:
        basis = rng.integers(p, size=(dimension, dimension))
        if rank_mod(basis, p) == dimension:
            break
    inputs = np.array(list(product(range(p), repeat=dimension)), dtype=np.int64)
    latent = inputs @ basis[:3].T % p
    encoded = np.eye(p, dtype=np.float32)[inputs].reshape(len(inputs), -1)
    return inputs, latent, encoded, basis


def categorical_mi(left, right):
    pairs, count = np.unique(np.column_stack([left, right]), axis=0, return_counts=True)
    _, lc = np.unique(left, return_counts=True)
    _, rc = np.unique(right, return_counts=True)
    lp = dict(zip(np.unique(left), lc / len(left)))
    rp = dict(zip(np.unique(right), rc / len(right)))
    prob = count / len(left)
    return float(sum(q * np.log(q / (lp[a] * rp[b])) for (a, b), q in zip(pairs, prob)))


def audit(groups, targets, p):
    latent = np.array(list(product(range(p), repeat=3)), dtype=np.int64)
    rows = []
    for g in groups:
        y = latent @ g['sources'].T % p
        cs = circuits(g['sources'], p)
        rows.append({'id': g['id'], 'family': g['family'], 'sources': g['sources'].tolist(),
                     'rank': rank_mod(g['sources'], p), 'circuits': [list(c) for c in cs],
                     'minimum_circuit_size': min(map(len, cs)),
                     'joint_support': len(np.unique(y, axis=0)),
                     'joint_entropy_nats': 3 * float(np.log(p)),
                     'uniform_marginals': all(len(set(np.unique(y[:, j], return_counts=True)[1])) == 1 for j in range(4)),
                     'max_pairwise_mi_nats': max(categorical_mi(y[:, a], y[:, b]) for a, b in combinations(range(4), 2)),
                     'target_composition_sizes': [composition_size(g['sources'], v, p) for v in targets]})
    return {'p': p, 'targets': targets.tolist(), 'scenarios': rows,
            'target_policy': 'exclude every scalar-equivalent source task across all scenarios'}
