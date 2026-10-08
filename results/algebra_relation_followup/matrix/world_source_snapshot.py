"""Exact finite matrix and polynomial worlds with audited observable splits."""
from collections import defaultdict
from functools import lru_cache
from itertools import product

import numpy as np


def matrix_product(a, b, p):
    a, b = np.asarray(a).reshape(2, 2), np.asarray(b).reshape(2, 2)
    return tuple(map(int, (a @ b % p).ravel()))


def generators(p):
    # A^2=B^3=e, ABA=B^2: the noncommutative six-element matrix group.
    return (0, 1, 1, 0), (0, p - 1, 1, p - 1)


@lru_cache(maxsize=None)
def group(p):
    identity = (1, 0, 0, 1)
    a, b = generators(p)
    found = {identity}
    pending = [identity]
    while pending:
        x = pending.pop()
        for g in [a, b]:
            y = matrix_product(g, x, p)
            if y not in found:
                found.add(y); pending.append(y)
    assert len(found) == 6
    assert matrix_product(a, b, p) != matrix_product(b, a, p)
    return tuple(sorted(found))


def action(x, letter, domain, p):
    x = tuple(map(int, x))
    if domain == 'matrix':
        return matrix_product(generators(p)[letter == 'b'], x, p)
    if letter == 'a':
        c0, c1, c2, c3 = x
        return ((c0 + c1 + c2 + c3) % p, (c1 + 2*c2 + 3*c3) % p,
                (c2 + 3*c3) % p, c3)
    return (x[1], 2*x[2] % p, 3*x[3] % p, 0)


def transform(x, word, domain, p):
    x = tuple(map(int, x))
    for letter in word:
        x = action(x, letter, domain, p)
    return x


def answer(x, domain, p):
    return (int(x[0]) + int(x[3])) % p if domain == 'matrix' else int(x[0])


@lru_cache(maxsize=65000)
def key(x, domain, p):
    x = tuple(map(int, x))
    if domain == 'matrix':
        return min(matrix_product(g, x, p) for g in group(p))
    # Derivative quadratic's leading coefficient and discriminant.
    # This preserves every translation and the complete single-D orbit.
    assert x[3] != 0
    return x[3], (4*x[2]*x[2] - 12*x[1]*x[3]) % p


def encode(anchors, domain, p, words=('', 'a', 'b')):
    x = np.asarray([[transform(anchor, w, domain, p) for w in words] for anchor in anchors], dtype=np.int64)
    y = np.asarray([[answer(state, domain, p) for state in row] for row in x], dtype=np.int64)
    return {'x': x, 'labels': y}


def feature(x, p):
    x = np.asarray(x)
    onehot = np.eye(p, dtype=np.float32)[x].reshape(*x.shape[:-1], 4*p)
    return np.concatenate([x.astype(np.float32) / (p - 1), onehot], axis=-1)


def choose_collision_pairs(rows, count, rng, domain, p, used_keys):
    buckets = defaultdict(list)
    for x in rows:
        if key(x, domain, p) in used_keys:
            continue
        visible = tuple(answer(transform(x, w, domain, p), domain, p) for w in ['', 'a', 'b'])
        buckets[visible].append(x)
    selected = []
    keys = list(buckets); rng.shuffle(keys)
    for visible in keys:
        candidates = buckets[visible]; rng.shuffle(candidates)
        for i, left in enumerate(candidates):
            kl = key(left, domain, p)
            if kl in used_keys:
                continue
            yl = answer(transform(left, 'ab', domain, p), domain, p)
            for right in candidates[i + 1:]:
                kr = key(right, domain, p)
                if kr == kl or kr in used_keys:
                    continue
                if answer(transform(right, 'ab', domain, p), domain, p) == yl:
                    continue
                selected.extend([left, right]); used_keys.update([kl, kr]); break
            if len(selected) == 2*count:
                return selected
            if key(left, domain, p) in used_keys:
                break
    raise ValueError(f'Not enough disjoint collision pairs: {len(selected)//2}/{count}')


def paired_fit(rows, count, rng, domain, p):
    buckets = defaultdict(list)
    for x in rows:
        labels = tuple(answer(transform(x, w, domain, p), domain, p) for w in ['', 'a', 'b'])
        buckets[labels].append(x)
    order = list(buckets); rng.shuffle(order)
    chosen, used = [], set()
    for labels in order:
        candidates = buckets[labels]; rng.shuffle(candidates)
        local, keys = [], set()
        for x in candidates:
            k = key(x, domain, p) if domain == 'matrix' else x
            if k in used or k in keys:
                continue
            keys.add(k); local.append(x)
            if len(local) == 4:
                chosen.extend(local); used.update(keys); break
        if len(chosen) == count:
            return chosen
    raise ValueError(f'Not enough four-way eligible fit anchors: {len(chosen)}/{count}')


def create_world(domain, p=13, seed=261077001, fit_count=1024, source_count=512,
                 validation_count=128, iid_count=256, pair_count=128):
    rng = np.random.default_rng(seed)
    rows = [tuple(x) for x in product(range(p), repeat=4)
            if ((x[0]*x[3]-x[1]*x[2]) % p != 0 if domain == 'matrix' else x[3] != 0)]
    if domain == 'matrix':
        test_used = set()
        collision = choose_collision_pairs(rows, pair_count, rng, domain, p, test_used)
        rng.shuffle(rows); iid = []
        for x in rows:
            k = key(x, domain, p)
            if k not in test_used:
                iid.append(x); test_used.add(k)
                if len(iid) == iid_count:break
        available = defaultdict(list)
        for x in rows:
            if key(x, domain, p) not in test_used:available[key(x, domain, p)].append(x)
        keys = list(available); rng.shuffle(keys)
        val_keys = set(keys[:validation_count+64])
        source_keys = set(keys[validation_count+64:validation_count+64+source_count])
        source = [available[k][0] for k in source_keys]
        validation = [available[k][0] for k in keys[:validation_count]]
        pilot_validation = [available[k][0] for k in keys[validation_count:validation_count+64]]
        fit_pool = [x for x in rows if key(x, domain, p) not in test_used | val_keys | source_keys]
    else:
        blocks = list({key(x, domain, p) for x in rows}); rng.shuffle(blocks)
        test_keys = set(blocks[:40]); val_keys = set(blocks[40:52]); pilot_keys = set(blocks[52:56])
        source_keys = set(blocks[56:96]); fit_keys = set(blocks[96:])
        test_rows = [x for x in rows if key(x, domain, p) in test_keys]
        # Within the test partition, collision members may share a split block:
        # they must be distinct actual inputs, not independent derivative orbits.
        buckets = defaultdict(list)
        for x in test_rows:
            labels = tuple(answer(transform(x, w, domain, p), domain, p) for w in ['', 'a', 'b'])
            buckets[labels].append(x)
        collision, used = [], set()
        names = list(buckets); rng.shuffle(names)
        for labels in names:
            candidates = buckets[labels]; rng.shuffle(candidates)
            for left in candidates:
                if left in used:continue
                for right in candidates:
                    if right in used or right == left:continue
                    if answer(transform(left, 'ab', domain, p), domain, p) != answer(transform(right, 'ab', domain, p), domain, p):
                        collision.extend([left,right]);used.update([left,right]);break
                if len(collision) == 2*pair_count:break
                if left in used:break
            if len(collision) == 2*pair_count:break
        assert len(collision) == 2*pair_count
        remaining = [x for x in test_rows if x not in used]; rng.shuffle(remaining); iid=remaining[:iid_count]
        def sample(keys, count):
            pool=[x for x in rows if key(x, domain, p) in keys];rng.shuffle(pool);return pool[:count]
        source=sample(source_keys,source_count);validation=sample(val_keys,validation_count)
        pilot_validation=sample(pilot_keys,64);fit_pool=[x for x in rows if key(x,domain,p) in fit_keys]
    supports=[]
    for i in range(7):
        local = np.random.default_rng(seed + 101 + i)
        supports.append(encode(paired_fit(list(fit_pool),fit_count,local,domain,p),domain,p))
    words=('', 'a', 'b', 'ab', 'ba', 'aa', 'bb', 'bbb', 'aba', 'bbbb')
    test=encode(iid+collision,domain,p,words)
    test.update(split=np.asarray([0]*iid_count+[1]*(2*pair_count)),
                pair_ids=np.asarray([-1]*iid_count+[i for i in range(pair_count) for _ in range(2)]))
    source=encode(source,domain,p);validation=encode(validation,domain,p)
    pilot_validation=encode(pilot_validation,domain,p)
    train_inputs=set(tuple(map(int,x)) for d in [source]+supports for x in d['x'].reshape(-1,4))
    val_inputs=set(tuple(map(int,x)) for d in [validation,pilot_validation] for x in d['x'].reshape(-1,4))
    test_inputs=set(tuple(map(int,x)) for x in test['x'].reshape(-1,4))
    assert not train_inputs & val_inputs and not train_inputs & test_inputs and not val_inputs & test_inputs
    for pair in range(pair_count):
        ids=np.flatnonzero(test['pair_ids']==pair)
        assert np.array_equal(test['labels'][ids[0],:3],test['labels'][ids[1],:3])
        assert test['labels'][ids[0],3]!=test['labels'][ids[1],3]
    audit={'domain':domain,'p':p,'actual_training_inputs':len(train_inputs),'actual_validation_inputs':len(val_inputs),
           'actual_test_inputs':len(test_inputs),'actual_observed_paths_disjoint':True,
           'collision_pairs':pair_count,'visible_answer_code_ceiling':.5,
           'split_rule':'Entire six-element matrix-group orbits' if domain=='matrix' else
               'Derivative translation-orbit blocks (c3, discriminant); all translations and one derivative. Higher derivatives converge to shared constants, so no full-semigroup-closure claim.',
           'polynomial_degree_restriction':None if domain=='matrix' else 'exactly3, nonzero cubic coefficient',
           'polynomial_collisions_are_not_independent_blocks':domain=='polynomial'}
    return {'source':source,'validation':validation,'pilot_validation':pilot_validation,
            'supports':supports,'test':test,'audit':audit,'words':words}
