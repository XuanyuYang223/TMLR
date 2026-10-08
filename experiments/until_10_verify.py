"""Independent numerical replay of the deadline-bounded relation studies.

This verifier does not call the fitting, composition, or scoring functions
that produced the results. Completed immutable records are cached by hashes;
unfinished source models are never treated as completed repetitions.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch


ROOT = Path('results/until_10_followup')
DEADLINE = datetime(2026, 10, 6, 17, tzinfo=timezone.utc).timestamp()
_DIGEST_CACHE = {}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    path = Path(path)
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key in _DIGEST_CACHE:
        return _DIGEST_CACHE[key]
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    result = h.hexdigest()
    _DIGEST_CACHE[key] = result
    return result


def archive(path):
    with np.load(path) as a:
        return {k: a[k] for k in a.files}


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def orbit(p):
    p = np.asarray(p)
    n = len(p)
    inv = np.argsort(p) + 1
    return np.stack([p, n + 1 - p, p[::-1], n + 1 - p[::-1],
        inv, inv[::-1], n + 1 - inv, n + 1 - inv[::-1]])


def action_order(word):
    # An explicitly calculated faithful eight-state orbit; no producer table.
    ref = orbit([2, 6, 1, 4, 7, 3, 5])
    lookup = {tuple(p): j for j, p in enumerate(ref)}
    output = []
    for p in ref:
        for g in word:
            p = orbit(p)[{'c': 1, 'r': 2, 'i': 4}[g]]
        output.append(lookup[tuple(p)])
    return np.array(output)


def counts(p, tasks):
    p = list(map(int, p))
    result = []
    for task in tasks:
        if task in ['peaks', 'valleys']:
            above = task == 'peaks'
            value = sum((p[i] > p[i-1] and p[i] > p[i+1]) if above else
                (p[i] < p[i-1] and p[i] < p[i+1]) for i in range(1, len(p)-1))
        else:
            sequence = p if task.startswith('left_to_right') else p[::-1]
            upper = task.endswith('maxima')
            threshold = float('-inf') if upper else float('inf')
            value = 0
            for x in sequence:
                if (x > threshold) if upper else (x < threshold):
                    value += 1
                    threshold = x
        result.append(value)
    return result


def pair_rows(data, mask=None):
    ids = data['pair_ids'] if mask is None else data['pair_ids'][mask]
    rows = np.stack([np.flatnonzero(ids == i) for i in np.unique(ids)])
    assert rows.shape[1] == 2
    return rows


def check_data(folder, tasks, partial):
    data_path = folder / ('probe_dataset.npz' if partial else 'dataset.npz')
    data = archive(data_path)
    label_key = 'union_labels' if 'union_labels' in data else 'labels'
    all_inputs = set()
    for j, n in enumerate(data['lengths']):
        raw = data['permutations'][j, :, :n]
        np.testing.assert_array_equal(raw, orbit(raw[0]))
        assert len(set(map(tuple, raw))) == 8
        for k, p in enumerate(raw):
            actual = data['input'][j, k, 4 + 2 * np.arange(n)]
            np.testing.assert_array_equal(actual, p)
            expected = counts(p, tasks)
            label = data[label_key][j, k]
            np.testing.assert_array_equal(np.atleast_1d(label), expected)
            key = tuple(map(int, p))
            assert key not in all_inputs
            all_inputs.add(key)
    mask = data['split'] == 3 if partial else np.ones(len(data['lengths']), bool)
    pairs = pair_rows(data, mask)
    labels = data[label_key][mask]
    ns = data['lengths'][mask]
    np.testing.assert_array_equal(ns[pairs[:, 0]], ns[pairs[:, 1]])
    if partial:
        np.testing.assert_array_equal(labels[pairs[:, 0]][:, [0, 1, 4]], labels[pairs[:, 1]][:, [0, 1, 4]])
        assert np.all(labels[pairs[:, 0], 5] != labels[pairs[:, 1], 5])
        np.testing.assert_array_equal(labels[:, 5], labels[:, 2])
        source = archive(folder / 'source_data.npz')
        audit = archive(folder / 'training_orbit_audit.npz')
        train_seen = set()
        for j, n in enumerate(audit['lengths']):
            raw = audit['permutations'][j, :, :n]
            np.testing.assert_array_equal(raw, orbit(raw[0]))
            train_seen.update(map(tuple, raw))
            for k, a in enumerate([0, 1, 4]):
                np.testing.assert_array_equal(source['train_input'][j, k, 4 + 2 * np.arange(n)], raw[a])
                assert source['train_labels'][j, k] == counts(raw[a], tasks)[0]
        assert not (train_seen & all_inputs)
        val = data['split'] == 1
        np.testing.assert_array_equal(source['validation_input'], data['input'][val][:, [0, 1, 4]])
        np.testing.assert_array_equal(source['validation_labels'], data['labels'][val][:, [0, 1, 4]])
    else:
        np.testing.assert_array_equal(labels[pairs[:, 0]], labels[pairs[:, 1]])
    return {'orbits': len(data['lengths']), 'pairs': len(pairs), 'independent_label_and_orbit_checks': True,
        'collision_ceiling_applies_to_CI_and_ICI_only': partial,
        'source_only_e_C_I_and_all_training_orbits_disjoint': partial}


def homogeneous(operators, word, width):
    product = np.eye(width + 1)
    for g in word:
        block = np.eye(width + 1)
        block[:width, :width] = operators[f'rho_{g}']
        block[-1, :width] = operators[f'bias_{g}']
        product = product @ block
    return product


def replay(hidden, data, maps, rows, predictions, observed=False):
    width = hidden.shape[-1]
    means = dict(zip(map(int, maps.get('mean_lengths', [])), maps.get('mean_vectors', [])))
    checked = 0
    for row in rows:
        split = 3 if row['split'] == 'answer_collisions' else 2
        use = data['split'] == split
        h = hidden[use].astype(np.float64)
        if observed:
            start, word = {'base_CI': (0, 'ci'), 'base_ICI': (0, 'ici'),
                'observed_C_then_I': (1, 'i'), 'observed_I_then_CI': (4, 'ci')}[row['case']]
            key = row['split'] + '_' + row['case']
        else:
            start, word = 0, row['word']
            key = row['split'] + '_' + word
        target_action = int(action_order(word)[start])
        if 'action' in row:
            assert target_action == row['action']
        center = np.array([means.get(int(n), np.zeros(width)) for n in data['lengths'][use]])
        product = homogeneous(maps, word, width)
        augmented = np.column_stack([h[:, start] - center, np.ones(len(h))])
        pred = (augmented @ product)[:, :width] + center
        np.testing.assert_allclose(pred, predictions[key + '_hidden'], rtol=1e-6, atol=2e-5)
        target = h[:, target_action]
        denom = np.square(target - h[:, start]).sum()
        if row.get('displacement_nmse') is not None:
            np.testing.assert_allclose(np.square(pred-target).sum()/denom, row['displacement_nmse'], rtol=1e-9, atol=1e-10)
        if row.get('answer_accuracy') is not None:
            answers = (pred @ maps['readout_weight'].T + maps['readout_bias']).argmax(-1)
            np.testing.assert_array_equal(answers, predictions[key + '_answers'])
            truth = data['labels'][use, target_action]
            np.testing.assert_allclose(np.mean(answers == truth), row['answer_accuracy'], atol=1e-12)
            if split == 3:
                pairs = pair_rows(data, use)
                both = np.mean((answers[pairs] == truth[pairs]).all(1))
                np.testing.assert_allclose(both, row['pair_both_correct'], atol=1e-12)
        checked += 1
    return {'independently_replayed_endpoints': checked, 'affine_homogeneous_product': True,
        'prediction_inputs_only_known_states': True, 'labels_used_only_for_scoring': True}


def verify_relation_record(path, folder):
    rec = read(path)
    name = rec.get('source', f"{rec['condition']}_s{rec['seed']}")
    data_folder = Path('results/algebra_hidden_relations') if folder.name == 'relation_seed_extension' else folder
    data = archive(data_folder / 'probe_dataset.npz')
    h = archive(folder / 'features' / f'{name}.npz')['source_query_concat'][:, :, -1]
    total = 0
    for method in rec['methods']:
        if method.get('view', 'query') != 'query':
            continue
        filename = f"{name}_{method['method']}.npz" if folder.name != 'algebra_hidden_relations' else f"{name}_query_{method['method']}.npz"
        ap = folder / 'arrays' / filename
        assert digest(ap) == rec['array_sha256'][filename]
        maps = archive(ap)
        total += replay(h, data, maps, method.get('metrics', method.get('results')), maps)['independently_replayed_endpoints']
    return {'source': name, 'independently_replayed_endpoints': total}


def source_check(path, folder, ordinary=False):
    r = read(path)
    assert r['status'] == 'complete'
    cp = folder / ('source/checkpoints' if ordinary else 'checkpoints') / (path.stem + '.pt')
    assert digest(cp) == r['checkpoint_sha256']
    state = torch.load(cp, weights_only=True, map_location='cpu')
    assert state['step'] == 20000
    if not ordinary:
        assert r['steps'] == 20000 and r['source_label_exposures'] == 1920000 and r['source_input_actions'] == [0, 1, 4]
        data = archive(folder / 'source_data.npz')
        buckets = {n: np.flatnonzero(data['train_lengths'] == n) for n in range(10, 31)}
        rng, wrong = np.random.default_rng(r['seed'] + 2026100662), np.random.default_rng(r['seed'] + 2026100663)
        samples = hashlib.sha256()
        seen = set()
        for _ in range(20000):
            n = int(rng.integers(10, 31))
            ids = rng.choice(buckets[n], 32, replace=False)
            wrong.integers(1, 32)
            samples.update(np.array([n], dtype=np.int64).tobytes() + ids.tobytes())
            seen.update(ids.tolist())
        assert samples.hexdigest() == r['sample_sha256'] == state['sample_sha256']
        assert rng.bit_generator.state == r['core_rng'] == state['core_rng']
        assert wrong.bit_generator.state == r['wrong_rng'] == state['wrong_rng']
        assert len(seen) == r['unique_source_anchors_seen']
        exposures = r['source_label_exposures']
    else:
        assert r['job']['steps'] == 20000 and r['job']['batch'] == 32
        assert r['total_labels'] == 2560000 and r['labels_per_update'] == 128
        data = archive(folder / 'dataset/data.npz')
        buckets = {n: np.flatnonzero(data['train_lengths'] == n) for n in range(10, 31)}
        rng = np.random.default_rng(r['job']['seed'] + 20261005)
        extra = np.random.default_rng(r['job']['seed'] + 202610061)
        samples = hashlib.sha256()
        seen = set()
        for _ in range(20000):
            n = int(rng.integers(10, 31))
            ids = rng.choice(buckets[n], 32, replace=True)
            samples.update(np.array([n], dtype=np.int64).tobytes() + ids.tobytes())
            seen.update(ids.tolist())
        assert samples.hexdigest() == r['sample_sha256'] == state['sample_sha256']
        assert rng.bit_generator.state == r['core_state'] == state['core_state']
        assert extra.bit_generator.state == r['extra_state'] == state['extra_state']
        assert len(seen) == r['unique_source_inputs_seen']
        exposures = r['total_labels']
    from .longrun_transfer import make_model
    if ordinary:
        from .algebra_structure_replication import load_plan
        plan, config, _ = load_plan()
        seed = r['job']['seed']
    else:
        from .hidden_relation_train import load_plan
        plan, config, _ = load_plan()
        seed = r['seed']
    initial_model = make_model(config, plan['architecture'], seed, 'cpu')
    initial = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in initial_model.parameters())).hexdigest()
    assert initial == r['initial_parameter_sha256']
    return {'source': path.stem, 'checkpoint_step': 20000, 'source_label_exposures': exposures,
        'sampler_20000_updates_replayed': True, 'seed_initialization_independently_recreated': True, 'initial_parameter_sha256': r['initial_parameter_sha256'],
        'sample_sha256': r['sample_sha256']}


def frozen_check(path):
    r = read(path)
    root = path.parent.parent
    parent = Path('results/algebra_hidden_relations')
    mp = root / 'maps' / (path.stem + '.npz')
    assert digest(mp) == r['map_archive_sha256']
    maps = archive(mp)
    data = archive(parent / 'probe_dataset.npz')
    probe_feature = Path(r.get('probe_feature_path', str(parent / 'features' / (r['source'] + '.npz'))))
    hidden = archive(probe_feature)['source_query_concat'][:, :, -1]
    out = replay(hidden, data, maps, r['metrics'], maps)
    known_path = Path(r.get('known_feature_path', str(root / 'features' / (r['source'] + '.npz'))))
    known = archive(known_path)
    source = archive(parent / 'source_data.npz')
    rng = np.random.default_rng(r['seed'] + 2026100671)
    ids = np.concatenate([rng.permutation(np.flatnonzero(source['train_lengths'] == n))[:r['fit_anchors_per_length']]
        for n in np.unique(source['train_lengths'])])
    np.testing.assert_array_equal(ids, maps['source_anchor_ids'])
    train = known['train_hidden'][ids].astype(np.float64)
    ns = source['train_lengths'][ids]
    val = known['validation_hidden'].astype(np.float64)
    vn = source['validation_lengths']
    for n, center in zip(maps['mean_lengths'], maps['mean_vectors']):
        np.testing.assert_allclose(train[ns == n].mean((0, 1)), center, atol=1e-12)
        train[ns == n] -= center
        val[vn == n] -= center
    grid = read('configs/frozen_relation_followup.json')['ridge_grid']
    def edge(h, lengths, k):
        changed = h[:, k]
        if r['pairing'] == 'shuffled':
            order = np.arange(len(h))
            for n in np.unique(lengths):
                index = np.flatnonzero(lengths == n)
                order[index] = np.roll(index, int(rng.integers(1, len(index))))
            changed = changed[order]
        return np.concatenate([h[:, 0], changed]), np.concatenate([changed, h[:, 0]])
    for g, k in [('c', 1), ('i', 2)]:
        x, y = edge(train, ns, k)
        xv, yv = edge(val, vn, k)
        a = np.column_stack([x, np.ones(len(x))])
        gram = a.T @ a
        rhs = a.T @ (y - x)
        scale = np.trace(gram[:-1, :-1]) / x.shape[1]
        losses = []
        fitted = None
        for alpha in grid:
            regularized = gram + np.diag([alpha * max(scale, 1e-20)] * x.shape[1] + [0.])
            fitted_candidate = np.linalg.solve(regularized, rhs)
            losses.append(float(np.square(xv + np.column_stack([xv, np.ones(len(xv))]) @ fitted_candidate - yv).sum()))
            if alpha == r['selected_generator_alphas'][g]:
                fitted = fitted_candidate
        assert grid[int(np.argmin(losses))] == r['selected_generator_alphas'][g]
        # Independent trace accumulation and BLAS reduction order change two
        # ill-conditioned fits by about 2e-9. Check the normal equations as
        # well as coefficients rather than requiring bitwise-equivalent fits.
        np.testing.assert_allclose(maps[f'rho_{g}'], np.eye(x.shape[1]) + fitted[:-1], atol=1e-8, rtol=1e-7)
        np.testing.assert_allclose(maps[f'bias_{g}'], fitted[-1], atol=1e-8, rtol=1e-7)
        stored = np.vstack([maps[f'rho_{g}'] - np.eye(x.shape[1]), maps[f'bias_{g}']])
        selected = r['selected_generator_alphas'][g]
        normal = gram + np.diag([selected * max(scale, 1e-20)] * x.shape[1] + [0.])
        relative_residual = np.linalg.norm(normal @ stored - rhs) / max(np.linalg.norm(rhs), 1e-20)
        assert relative_residual < 1e-10, relative_residual
    return {**out, 'known_only_ridge_normal_equations_and_validation_selection': True}


def run():
    ROOT.mkdir(exist_ok=True, parents=True)
    cache_path = ROOT / 'verification_cache.json'
    cache = read(cache_path) if cache_path.exists() else {}
    own_hash = digest(__file__)
    results, failures = [], []
    def check(key, dependencies, operation):
        signature = hashlib.sha256((own_hash + ''.join(digest(p) for p in dependencies)).encode()).hexdigest()
        if key in cache and cache[key]['signature'] == signature:
            results.append(cache[key]['result'])
            return
        try:
            result = {'check': key, 'status': 'passed', **operation()}
            cache[key] = {'signature': signature, 'result': result}
            results.append(result)
            save(cache_path, cache)
            print(json.dumps({'verified': key, 'checks': len(results)}), flush=True)
        except Exception as error:
            failure = {'check': key, 'status': 'failed', 'error': repr(error)}
            failures.append(failure)
            print(json.dumps(failure), flush=True)
    for parent in ['algebra_structure_replication', 'algebra_hidden_relations']:
        root = Path('results') / parent
        cp = root / 'completion.json'
        def preserved(root=root, cp=cp):
            manifest = read(cp)['artifact_sha256']
            for name, expected in manifest.items():
                assert digest(root / name) == expected, name
            return {'immutable_prior_artifacts': len(manifest)}
        check(parent + '_preserved', [cp, *[root / p for p in read(cp)['artifact_sha256']]], preserved)
    partial_roots = [Path('results/algebra_hidden_relations'), *sorted(Path('results/relation_world_confirmation').glob('world*'))]
    for root in partial_roots:
        deps = [root / p for p in ['probe_dataset.npz', 'source_data.npz', 'training_orbit_audit.npz']]
        check(str(root) + '_data', deps, lambda root=root: check_data(root, ['left_to_right_maxima'], True))
    for root in [Path('results/answer_matched_geometry'), Path('results/joint_answer_matched_geometry')]:
        tasks = read(root / 'dataset_audit.json').get('union_tasks', read('configs/answer_matched_geometry.json')['tasks'])
        check(str(root) + '_data', [root / 'dataset.npz', root / 'dataset_audit.json'], lambda root=root, tasks=tasks: check_data(root, tasks, False))
    source_roots = [Path('results/relation_seed_extension'), *partial_roots[1:]]
    for root in source_roots:
        for path in sorted((root / 'source').glob('*.json')):
            check(str(path) + '_training', [path, root / 'checkpoints' / (path.stem + '.pt'), root / 'source_data.npz'],
                lambda path=path, root=root: source_check(path, root))
        for path in sorted((root / 'evaluations').glob('*.json')):
            rec = read(path)
            deps = [path, root / 'features' / (path.stem + '.npz'), *[root / 'arrays' / p for p in rec['array_sha256']]]
            check(str(path) + '_numerics', deps, lambda path=path, root=root: verify_relation_record(path, root))
    for folder in ['frozen_relation_followup', 'relation_frozen_seed_confirmation']:
      for path in sorted((Path('results') / folder / 'evaluations').glob('*.json')):
        rec = read(path)
        root = path.parent.parent
        probe_feature = Path(rec.get('probe_feature_path', str(Path('results/algebra_hidden_relations/features') / (rec['source'] + '.npz'))))
        deps = [path, root / 'maps' / (path.stem + '.npz'),
            Path(rec.get('known_feature_path', str(root / 'features' / (rec['source'] + '.npz')))), probe_feature]
        check(str(path) + '_known_only_fit_and_numerics', deps, lambda path=path: frozen_check(path))
    ordinary_root = Path('results/ordinary_relation_seed_confirmation')
    for path in sorted((ordinary_root / 'source').glob('*.json')):
        check(str(path) + '_training', [path, ordinary_root / 'source/checkpoints' / (path.stem + '.pt'), ordinary_root / 'dataset/data.npz'],
            lambda path=path: source_check(path, ordinary_root, ordinary=True))
    paired = defaultdict(list)
    for result in results:
        if 'initial_parameter_sha256' in result:
            paired[(str(Path(result['check']).parent), int(result['source'].split('_s')[-1]))].append(result)
    for key, group in paired.items():
        if len(group) > 1:
            assert len(set(r['initial_parameter_sha256'] for r in group)) == 1
            assert len(set(r['sample_sha256'] for r in group)) == 1
    summary = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed_available_completed_records' if not failures else 'failed',
        'verifier_code_sha256': own_hash, 'checks_completed': len(results), 'independently_replayed_endpoints': sum(r.get('independently_replayed_endpoints', 0) for r in results),
        'paired_initialization_and_sampling_groups_checked': len(paired), 'checks': results, 'failures': failures,
        'scope': 'Completed records only. Manual permutation/label formulas, homogeneous affine composition, readout and scoring replay, independent known-only frozen ridge fits. Additional controls may be added with separately visible verifier versions; no scientific producer results are modified.'}
    save(ROOT / 'verification.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    while True:
        summary = run()
        print(json.dumps({'verification_status': summary['status'], 'checks': summary['checks_completed'], 'failures': len(summary['failures'])}), flush=True)
        if not args.watch or time.time() >= DEADLINE:
            break
        time.sleep(min(45, max(0, DEADLINE - time.time())))
