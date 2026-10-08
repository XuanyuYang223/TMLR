"""Independent replay of observed-start, stability and conditional geometry."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .until_10_verify import ROOT, DEADLINE, read, archive, digest, save, replay, action_order, pair_rows


def location_for(name):
    for folder in [Path('results/algebra_hidden_relations'), Path('results/relation_seed_extension'),
            *sorted(Path('results/relation_world_confirmation').glob('world*'))]:
        if (folder / 'source' / (name + '.json')).exists():
            return folder
    raise FileNotFoundError(name)


def observation_check(path, world=False):
    r = read(path)
    folder = location_for(r['source'])
    data_folder = folder if folder.name.startswith('world') else Path('results/algebra_hidden_relations')
    data = archive(data_folder / 'probe_dataset.npz')
    h = archive(folder / 'features' / (r['source'] + '.npz'))['source_query_concat'][:, :, -1]
    total = 0
    for method in r['methods']:
        if world:
            mp = folder / 'arrays' / (r['source'] + '_' + method['method'] + '.npz')
            pp = folder / 'observed_start' / (r['source'] + '_' + method['method'] + '.npz')
        else:
            mp = Path(method['map_archive_path'])
            assert digest(mp) == method['map_sha256']
            pp = path.parent.parent / 'arrays' / (r['source'] + '_' + method['method'] + '.npz')
        assert digest(pp) == method['prediction_archive_sha256']
        total += replay(h, data, archive(mp), method.get('rows', method.get('metrics')), archive(pp), observed=True)['independently_replayed_endpoints']
    return {'independently_replayed_endpoints': total, 'extra_known_input_conditions_kept_separate': True}


def stability_check(path):
    r = read(path)
    folder = location_for(r['source'])
    h = archive(folder / 'features' / (r['source'] + '.npz'))['source_query_concat'][:, :, -1]
    data = archive('results/algebra_hidden_relations/probe_dataset.npz')
    mp = path.parent.parent / 'maps' / (path.stem + '.npz')
    assert digest(mp) == r['map_archive_sha256']
    maps = archive(mp)
    return replay(h, data, maps, r['metrics'], maps)


def head(rec, root):
    from .longrun_transfer import make_model
    from .algebra_structure_replication import load_plan
    plan, config, _ = load_plan()
    if root.name == 'ordinary_relation_seed_confirmation':
        cp = root / 'source/checkpoints' / (rec['source'] + '.pt')
        model = torch.load(cp, map_location='cpu', weights_only=True)['model']
        b = model.get('lm_head.bias')
        return model['lm_head.weight'][:31].numpy().astype(np.float64), np.zeros(31) if b is None else b[:31].numpy().astype(np.float64)
    if rec['status'] == 'trained':
        cp = Path('results/algebra_structure_replication/source/checkpoints') / f"{rec['group']}_s{rec['seed']}.pt"
        model = torch.load(cp, map_location='cpu', weights_only=True)['model']
        b = model.get('lm_head.bias')
        return model['lm_head.weight'][:31].numpy().astype(np.float64), np.zeros(31) if b is None else b[:31].numpy().astype(np.float64)
    model = make_model(config, plan['architecture'], rec['seed'], 'cpu')
    b = model.lm_head.bias
    return model.lm_head.weight[:31].detach().numpy().astype(np.float64), np.zeros(31) if b is None else b[:31].detach().numpy().astype(np.float64)


def geometry_check(path):
    rec = read(path)
    root = path.parent.parent
    new = root.name in ['ordinary_relation_seed_confirmation', 'ordinary_initialization_control']
    dataset = Path('results/joint_answer_matched_geometry/dataset.npz') if new else root / 'dataset.npz'
    data = archive(dataset)
    fp = root / 'features' / (rec['source'] + ('_joint_test.npz' if new else '.npz'))
    features = archive(fp)
    w, bias = head(rec, root)
    # Compute the numeric contrast projection from a Moore-Penrose inverse,
    # independently of the producer's SVD basis construction.
    centered = w - w.mean(0)
    projection = centered.T @ np.linalg.pinv(centered @ centered.T, rcond=1e-12) @ centered
    block = features['source_query_concat'].reshape(len(data['lengths']), 8, 4, -1).astype(np.float64)
    null = (block - block @ projection).reshape(features['source_query_concat'].shape)
    np.testing.assert_allclose(null.reshape(block.shape) @ centered.T, 0, atol=1e-8)
    pairs = pair_rows(data)
    count = 0
    max_prediction_discrepancy = 0.
    views = ['source_query_concat', 'source_query_numeric_null']
    for view in views:
        h = null if view.endswith('numeric_null') else features['source_query_concat']
        differences = (h[pairs[:, 0]] - h[pairs[:, 1]]).astype(np.float64)
        d = differences.reshape(-1, differences.shape[-1]).T
        mp = root / 'maps' / (rec['source'] + '_' + view + '.npz') if new else Path('results/algebra_structure_replication/arrays') / (rec['source'] + '_' + view + '.npz')
        maps = archive(mp)
        q = maps['basis'].astype(np.float64)
        latent = q.T @ d
        ap = root / 'arrays' / (rec['source'] + '_' + view + '_all_joint_answer_matched_pairs.npz')
        if root.name == 'answer_matched_geometry':
            ap = root / 'arrays' / (rec['source'] + '_' + view + '_all_answer_matched_pairs.npz')
        predictions = archive(ap)
        for row in rec['results']:
            if row['view'] != view or not row['subset'].startswith('all_'):
                continue
            product = np.eye(q.shape[1])
            for g in row['word']:
                product = maps['map_' + g].astype(np.float64).T @ product
            predicted = (d + q @ ((product - np.eye(q.shape[1])) @ latent)).T
            target = differences[:, action_order(row['word'])].reshape(predicted.shape)
            source = d.T
            error = np.square(predicted - target).sum()
            denominator = np.square(target - source).sum()
            np.testing.assert_allclose(error / denominator, row['pair_action_displacement_nmse'], rtol=1e-8, atol=1e-9)
            np.testing.assert_allclose(error / np.square(target).sum(), row['pair_target_nmse'], rtol=1e-8, atol=1e-9)
            delta = float(np.max(np.abs(predicted - predictions[row['word'] + '_pair_prediction'])))
            assert delta < 2e-5
            max_prediction_discrepancy = max(max_prediction_discrepancy, delta)
            dot = (predicted * target).sum(-1)
            score = np.mean(np.where(dot > 1e-12, 1., np.where(dot < -1e-12, 0., .5)))
            np.testing.assert_allclose(score, row['pair_assignment_accuracy'], atol=1e-9)
            if 'paired_projection_energy_fraction' in row:
                # Producer metadata sums the raw feature energy in float32;
                # geometry scores themselves use double precision. Allow its
                # single-precision accumulation error only for this coverage.
                np.testing.assert_allclose(np.square(latent).sum() / np.square(d).sum(), row['paired_projection_energy_fraction'], rtol=5e-7, atol=1e-8)
            count += 1
    return {'independently_replayed_endpoints': count, 'all_pairs_replayed_raw64_and_numeric_null64': True,
        'numeric_null_computed_by_independent_pseudoinverse': True, 'maximum_float32_archive_discrepancy': max_prediction_discrepancy,
        'full_width_and_prefix_secondary_views_not_replayed_here': True}


def readout_check(path):
    r = read(path)
    folder = location_for(r['source'])
    data_folder = folder if folder.name.startswith('world') else Path('results/algebra_hidden_relations')
    probe = archive(data_folder / 'probe_dataset.npz')
    source = archive(folder / 'source_data.npz')
    h = archive(folder / 'features' / (r['source'] + '.npz'))['source_query_concat'][:, :, -1]
    old = folder.name == 'algebra_hidden_relations'
    mp = folder / 'arrays' / (r['source'] + ('_query_native_operators.npz' if old else '_native_operators.npz'))
    maps = archive(mp)
    total = 0
    for cal in r['calibrations']:
        rp = path.parent.parent / 'readouts' / (r['source'] + '_' + cal['readout'] + '.npz')
        weights = archive(rp)
        replacement = {**maps, 'readout_weight': weights['weight'], 'readout_bias': weights['bias']}
        pp = path.parent.parent / 'arrays' / (r['source'] + '_' + cal['readout'] + '.npz')
        # Producer metadata used the same basename for arrays/ and readouts/,
        # so the second entry overwrote the prediction checksum. Preserve that
        # metadata and add an explicit path-keyed provenance sidecar below.
        assert digest(rp) == r['archive_sha256'][rp.name]
        total += replay(h, probe, replacement, cal['rows'], archive(pp), observed=True)['independently_replayed_endpoints']
        if cal['readout'] == 'original_lm_head':
            np.testing.assert_array_equal(weights['weight'], maps['readout_weight'])
            np.testing.assert_array_equal(weights['bias'], maps['readout_bias'])
    return {'independently_replayed_endpoints': total, 'stored_readout_archive_hashes_checked': True,
        'original_head_unchanged': True, 'replacement_readout_fitting_not_replayed_here': True}


def run():
    cp = ROOT / 'control_verification_cache.json'
    cache = read(cp) if cp.exists() else {}
    signature = digest(__file__) + digest('experiments/until_10_verify.py')
    results, failures = [], []
    jobs = []
    jobs.extend((p, observation_check) for p in Path('results/relation_observed_start/evaluations').glob('*.json'))
    for world in Path('results/relation_world_confirmation').glob('world*'):
        jobs.extend((p, lambda p: observation_check(p, world=True)) for p in (world / 'observed_start').glob('*.json'))
    jobs.extend((p, stability_check) for p in Path('results/relation_operator_stability/evaluations').glob('*.json'))
    for folder in ['joint_answer_matched_geometry', 'ordinary_relation_seed_confirmation', 'ordinary_initialization_control']:
        jobs.extend((p, geometry_check) for p in (Path('results') / folder / 'evaluations').glob('*.json'))
    jobs.extend((p, readout_check) for p in Path('results/relation_readout_diagnostics/evaluations').glob('*.json'))
    for path, operation in jobs:
        key = str(path)
        fingerprint = signature + digest(path)
        if key in cache and cache[key]['signature'] == fingerprint:
            results.append(cache[key]['result'])
            continue
        try:
            result = {'check': key, 'status': 'passed', **operation(path)}
            cache[key] = {'signature': fingerprint, 'result': result}
            results.append(result)
            save(cp, cache)
            print(json.dumps({'verified_control': key}), flush=True)
        except Exception as error:
            result = {'check': key, 'status': 'failed', 'error': repr(error)}
            failures.append(result)
            print(json.dumps(result), flush=True)
    provenance = {}
    for path in Path('results/relation_readout_diagnostics/evaluations').glob('*.json'):
        rec = read(path)
        for cal in rec['calibrations']:
            for directory in ['arrays', 'readouts']:
                p = path.parent.parent / directory / (rec['source'] + '_' + cal['readout'] + '.npz')
                provenance[str(p)] = digest(p)
    save(ROOT / 'readout_path_provenance.json', {'created_or_updated_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Post hoc audit correction: producer basename-only checksums overwrote arrays/ checksums with readouts/ checksums. Original records and all predictions are preserved. This path-keyed manifest records both archives; predictions are separately independently replayed from saved operators and readouts.',
        'artifact_sha256': provenance})
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed_available_completed_records' if not failures else 'failed',
        'verifier_code_sha256': digest(__file__), 'checks_completed': len(results), 'failures': failures,
        'independently_replayed_endpoints': sum(r.get('independently_replayed_endpoints', 0) for r in results), 'checks': results,
        'scope': 'All completed observed-start and operator-stability endpoints; raw64 and numeric-null64 ordinary geometry on all pairs; calibrated readout scoring. Producer fitting and score functions are not called. Secondary full-width/prefix geometry and readout refitting are not independently replayed by this file.'}
    save(ROOT / 'control_verification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    while True:
        r = run()
        print(json.dumps({'control_verification_status': r['status'], 'checks': r['checks_completed'], 'failures': len(r['failures'])}), flush=True)
        if not args.watch or time.time() >= DEADLINE:
            break
        time.sleep(min(45, max(0, DEADLINE-time.time())))
