"""Post-confirmation frozen-state controls. Never overwrite the parent study."""
import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch

from . import algebra_relation_worlds as worlds
from .algebra_relation_feasibility import apply_encoding
from .final_mechanism_evaluate import predicted_states
from .longrun_engine import atomic_json
from .operator_capacity_confirmation import crossed_interval, load_backbone
from .permworld_combinations import sha
from .relation_error_localization import row_projection
from .two_step_relation_factorial import now

CONFIG = Path('configs/null_space_review_controls.json')


def symmetric_root(covariance, inverse=False):
    values, vectors = np.linalg.eigh(covariance)
    assert values.min() > 0
    powers = 1 / np.sqrt(values) if inverse else np.sqrt(values)
    return (vectors * powers) @ vectors.T


def covariance_fit(values, shrinkage):
    mean = values.mean(0)
    centered = values - mean
    cov = centered.T @ centered / len(values)
    cov = (1-shrinkage)*cov + shrinkage*np.trace(cov)/len(cov)*np.eye(len(cov))
    return mean, cov


def norm_error_calibrate(reference, truth, direction):
    """Oracle diagnostic: preserve per-row norm and distance, vary orientation.

    Vectors are null-basis coordinates. Matching these two scalars fixes the
    component parallel to truth; the orthogonal orientation remains free.
    Neither compound labels nor compound states enter this construction.
    """
    norm_t = np.linalg.norm(truth, axis=1, keepdims=True)
    axis = np.divide(truth, norm_t, out=np.zeros_like(truth), where=norm_t > 1e-12)
    parallel = (reference*axis).sum(1, keepdims=True)
    radius = np.linalg.norm(reference-parallel*axis, axis=1, keepdims=True)
    orthogonal = direction-(direction*axis).sum(1, keepdims=True)*axis
    norm_o = np.linalg.norm(orthogonal, axis=1, keepdims=True)
    # Deterministic fallback when a proposed direction lies on the truth axis.
    for i in np.flatnonzero(norm_o[:, 0] < 1e-12):
        unit = np.eye(truth.shape[1])[np.argmin(np.abs(axis[i]))]
        orthogonal[i] = unit-unit.dot(axis[i])*axis[i]
        norm_o[i] = np.linalg.norm(orthogonal[i])
    answer = parallel*axis + radius*orthogonal/norm_o
    # Zero truth permits any direction with the reference norm.
    zero = norm_t[:, 0] < 1e-12
    answer[zero] = orthogonal[zero]/norm_o[zero]*np.linalg.norm(reference[zero], axis=1, keepdims=True)
    return answer


def state_distances(prediction, truth, reference, mean, inverse_root):
    distance = np.maximum(np.square(prediction).sum(1)[:, None]
                          + np.square(reference).sum(1)[None, :]
                          - 2*prediction@reference.T, 0)
    nearest = np.partition(distance, 4, axis=1)[:, :5]
    return {'true_state_mse': np.square(prediction-truth).mean(1),
            'null_norm_placeholder': np.zeros(len(prediction)),
            'nearest_reference_rms': np.sqrt(nearest.min(1)/prediction.shape[1]),
            'five_neighbor_rms': np.sqrt(nearest.mean(1)/prediction.shape[1]),
            'mahalanobis_per_dimension': np.square((prediction-mean)@inverse_root).mean(1)}


def setup():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    apply_encoding('numeric_gelu')
    config = json.loads(CONFIG.read_text())
    root, parent = Path(config['output']), Path(config['parent'])
    for name in ['', 'datasets', 'states', 'evaluations', 'probes']:
        (root/name).mkdir(parents=True, exist_ok=True)
    parent_config = json.loads(Path('configs/final_mechanism_confirmation.json').read_text())
    signature = {'config': config, 'code_sha256': sha(__file__), 'config_sha256': sha(CONFIG),
                 'parent_delivery_sha256': sha(parent/'delivery.json'),
                 'dependencies_sha256': {p: sha(p) for p in [
                     'experiments/final_mechanism_evaluate.py',
                     'experiments/relation_error_localization.py',
                     'experiments/operator_capacity_confirmation.py']}}
    protocol = root/'protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text())['signature'] == signature
    else:
        atomic_json(protocol, {'registered_utc': now(), 'parent_test_outcomes_observed': True,
                    'new_diagnostic_predictions_observed': False, 'signature': signature,
                    'claims': ['scalar norm/error explanation', 'marginal state distribution explanation',
                               'fixed linear-readout invisibility, not output independence'],
                    'controls': ['wrong donor covariance transported using known validation',
                                 'wrong donor oracle-calibrated to correct norm/error per input',
                                 'isotropic random donor oracle-calibrated to correct norm/error',
                                 'validation-fitted Gaussian covariance random donor',
                                 'same-input correct/wrong error-and-norm overlap subsets'],
                    'matching_uses_possible_mediators': True,
                    'oracle_calibration_is_diagnostic_only': True,
                    'fresh_split_disjoint_from_all_five_sources_and_known_prior_tests': True})
        (root/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return config, root, parent, parent_config


def prepare(config, root, parent, parent_config):
    audit = root/'data_audit.json'
    if audit.exists():
        return
    key = lambda x: worlds.key(tuple(x), 'matrix', 13)
    excluded_files = list((parent/'dataset').glob('*.npz')) + [
        Path(p) for p in parent_config['excluded_test_datasets']]
    excluded = {key(x) for p in excluded_files for x in np.load(p)['x'][:, 0]}
    rows = [x for x in product(range(13), repeat=4)
            if (x[0]*x[3]-x[1]*x[2]) % 13 and key(x) not in excluded]
    available = {key(x) for x in rows}
    rng = np.random.default_rng(config['data_seed'])
    used = set()
    collisions = worlds.choose_collision_pairs(rows, config['fresh_collision_pairs'], rng, 'matrix', 13, used)
    rng.shuffle(rows)
    iid = []
    for x in rows:
        if key(x) not in used:
            used.add(key(x)); iid.append(x)
            if len(iid) == config['fresh_iid_count']:
                break
    assert len(iid) == config['fresh_iid_count']
    assert not used & excluded
    data = worlds.encode(iid+collisions, 'matrix', 13, ['', 'a', 'b', 'ab'])
    data.update(split=np.array([0]*len(iid)+[1]*len(collisions)),
                pair_ids=np.array([-1]*len(iid)+[i for i in range(config['fresh_collision_pairs']) for _ in range(2)]))
    np.savez_compressed(root/'datasets/fresh.npz', **data)
    atomic_json(audit, {'created_utc': now(), 'previously_unused_orbits_available': len(available),
                'fixed_fresh_orbits': len(used), 'collision_pairs': config['fresh_collision_pairs'],
                'iid': len(iid), 'all_five_training_and_validation_and_prior_test_orbits_excluded': True,
                'excluded_input_sha256': {str(p): sha(p) for p in excluded_files},
                'fresh_dataset_sha256': sha(root/'datasets/fresh.npz')})


def fit_probes(train, validation, train_labels, validation_labels, ridges):
    """Independent label probes; hidden targets never update frozen operators."""
    mean, scale = train.mean(0), np.sqrt(train.var(0).mean())
    x = np.column_stack([(train-mean)/scale, np.ones(len(train))])
    v = np.column_stack([(validation-mean)/scale, np.ones(len(validation))])
    target = np.eye(13)[train_labels]
    u, singular, vt = np.linalg.svd(x, full_matrices=False)
    candidates = []
    for ridge in ridges:
        theta = (vt.T*(singular/(singular**2+len(x)*ridge)))@(u.T@target)
        accuracy = float(((v@theta).argmax(1)==validation_labels).mean())
        candidates.append((accuracy, ridge, theta))
    # Stable declared tie rule: first (smallest) ridge, never use test outcomes.
    index = int(np.argmax([c[0] for c in candidates]))
    accuracy, ridge, theta = candidates[index]
    return mean, scale, theta, {'ridge': ridge, 'validation_accuracy': accuracy,
            'all_validation_accuracies': [{'ridge': c[1], 'accuracy': c[0]} for c in candidates]}


def evaluate(config, root, parent, parent_config):
    assert not (root/'results.json').exists(), 'Use the completed report; do not silently rerun.'
    atomic_json(root/'test_opened.json', {'opened_utc': now(), 'registered_controls_fixed': True})
    datasets = {'previously_inspected': dict(np.load(parent/'dataset/test.npz')),
                'fresh': dict(np.load(root/'datasets/fresh.npz'))}
    rows, audit, matching, probes = [], [], [], []
    hits_by_split = defaultdict(dict)
    for i in range(len(parent_config['source_seeds'])):
        z = dict(np.load(parent/'states'/f's{i}_known.npz'))
        w, bw, b, bb = [z[k].astype(float) for k in ['w', 'bias_w', 'b', 'bias_b']]
        p = row_projection(w); q = np.eye(len(p))-p
        _, singular, vt = np.linalg.svd(w, full_matrices=True)
        rank = int((singular > singular[0]*1e-10).sum())
        basis = vt[rank:].T
        independent = np.linalg.pinv(w, rcond=1e-10)@w
        identities = {'symmetric': float(np.abs(p-p.T).max()),
                      'idempotent': float(np.abs(p@p-p).max()),
                      'readout_annihilation': float(np.abs(q@w.T).max()),
                      'pinv_agreement': float(np.abs(p-independent).max())}
        assert max(identities.values()) < 1e-9
        audit.append({'source': i, 'readout_shape': list(w.shape), 'rank': rank,
                      'singular_values': singular.tolist(), 'svd_cutoff': 1e-10, **identities})
        reference = z['train'][:, 1].astype(float)
        center, covariance = covariance_fit(reference, config['covariance_shrinkage'])
        inverse_root = symmetric_root(covariance, inverse=True)
        val = z['validation'].astype(float)
        vp = predicted_states(parent_config, parent, i, val[:, 0], z)
        vc, vw = [vp[c]@basis for c in ['nonlinear_correct', 'nonlinear_wrong']]
        mc, cc = covariance_fit(vc, config['covariance_shrinkage'])
        mw, cw = covariance_fit(vw, config['covariance_shrinkage'])
        transport = symmetric_root(cw, inverse=True)@symmetric_root(cc)
        train_labels = np.load(parent/'dataset'/f'support{i}.npz')['labels'][:, 1]
        val_labels = np.load(parent/'dataset/validation.npz')['labels'][:, 1]
        def compound_labels(path):
            return np.array([worlds.answer(worlds.transform(tuple(x), 'ab', 'matrix', 13), 'matrix', 13)
                             for x in np.load(path)['x'][:, 0]])
        future_train = compound_labels(parent/'dataset'/f'support{i}.npz')
        future_val = compound_labels(parent/'dataset/validation.npz')
        fitted = {}
        for representation, transform in [('full', np.eye(len(p))), ('row', p), ('null', q)]:
            for task, tr, va in [('current', train_labels, val_labels), ('future', future_train, future_val)]:
                fit = fit_probes(reference@transform, val[:, 1]@transform, tr, va, config['probe_ridges'])
                fitted[(representation, task)] = (*fit, transform)
                np.savez_compressed(root/'probes'/f's{i}_{representation}_{task}.npz',
                                    mean=fit[0], scale=fit[1], theta=fit[2])
        model = None
        for dataset_name, data in datasets.items():
            if dataset_name == 'previously_inspected':
                hidden = np.load(parent/'states'/f's{i}_test.npz')['hidden'].astype(float)
            else:
                if model is None:
                    model, _ = load_backbone(parent, i, 'cuda')
                with torch.no_grad():
                    hidden = model(torch.as_tensor(worlds.feature(data['x'], 13), device='cuda')).cpu().numpy().astype(float)
                np.savez_compressed(root/'states'/f's{i}_fresh.npz', hidden=hidden)
            prediction = predicted_states(parent_config, parent, i, hidden[:, 0], z)
            lc, gc, gw = [prediction[c] for c in ['linear_correct', 'nonlinear_correct', 'nonlinear_wrong']]
            row, correct, wrong, truth = lc@p, gc@basis, gw@basis, hidden[:, 1]@basis
            controls = {k: prediction[k][None] for k in [
                'linear_correct', 'nonlinear_correct', 'nonlinear_wrong',
                'affine_correct_ols_initialization',
                'swap_linear_correct_null_nonlinear_correct',
                'swap_linear_correct_null_nonlinear_wrong']}
            controls['natural_true_intermediate'] = hidden[:, 1][None]
            controls['wrong_covariance_transported'] = (row+((wrong-mw)@transport+mc)@basis.T)[None]
            controls['wrong_norm_error_calibrated'] = (row+norm_error_calibrate(correct, truth, wrong)@basis.T)[None]
            rng = np.random.default_rng(config['control_seed']+101*i+(dataset_name=='fresh'))
            random_error, random_cov = [], []
            cov_root = symmetric_root(cc)
            for _ in range(config['random_draws']):
                direction = rng.normal(size=correct.shape)
                random_error.append(row+norm_error_calibrate(correct, truth, direction)@basis.T)
                random_cov.append(row+(mc+rng.normal(size=correct.shape)@cov_root)@basis.T)
            controls['random_norm_error_calibrated'] = np.array(random_error)
            controls['random_validation_covariance'] = np.array(random_cov)
            same_input = {}
            for condition, draws in controls.items():
                hits, all_metrics, first_errors = [], [], []
                for pred in draws:
                    hit = (((pred@b+bb)@w.T+bw).argmax(1)==data['labels'][:, 3])
                    hits.append(hit)
                    metrics = state_distances(pred, hidden[:, 1], reference, center, inverse_root)
                    metrics.pop('null_norm_placeholder')
                    metrics['null_norm'] = np.linalg.norm(pred@basis, axis=1)
                    metrics['null_state_squared_error'] = np.square((pred-hidden[:, 1])@basis).sum(1)
                    metrics['downstream_score_mse'] = np.square((pred-hidden[:, 1])@b@w.T).mean(1)
                    all_metrics.append(metrics)
                    if condition not in ['linear_correct', 'nonlinear_correct', 'nonlinear_wrong',
                                         'affine_correct_ols_initialization', 'natural_true_intermediate']:
                        first_errors.append(np.abs((pred-lc)@w.T).max(1))
                hits = np.array(hits)
                metrics = {k: np.mean([m[k] for m in all_metrics], axis=0) for k in all_metrics[0]}
                if condition in ['swap_linear_correct_null_nonlinear_correct', 'swap_linear_correct_null_nonlinear_wrong']:
                    same_input[condition] = metrics
                if first_errors:
                    errors = np.array(first_errors).ravel()
                    assert errors.max() < 1e-8
                    audit.append({'source': i, 'dataset': dataset_name, 'condition': condition,
                                  'tested_draw_samples': len(errors),
                                  'first_logit_change_quantiles': dict(zip(['min', 'median', 'p90', 'p99', 'max'],
                                                                         np.quantile(errors, [0, .5, .9, .99, 1]).tolist()))})
                if condition in ['wrong_norm_error_calibrated', 'random_norm_error_calibrated']:
                    baseline = controls['swap_linear_correct_null_nonlinear_correct'][0]
                    err = np.max(np.abs(metrics['null_state_squared_error']-np.square((baseline-hidden[:, 1])@basis).sum(1)))
                    norm_err = np.max(np.abs(metrics['null_norm']-np.linalg.norm(baseline@basis, axis=1)))
                    assert err < 1e-8 and norm_err < 1e-8
                    audit.append({'source': i, 'dataset': dataset_name, 'condition': condition,
                                  'maximum_norm_matching_error': float(norm_err), 'maximum_squared_error_matching_error': float(err)})
                np.savez_compressed(root/'evaluations'/f's{i}_{dataset_name}_{condition}.npz', hits=hits, **metrics)
                for sid, split in [(0, 'iid'), (1, 'collisions')]:
                    use = data['split']==sid
                    ids = sorted(set(data['pair_ids'][use])) if sid else []
                    pair_hit = np.array([hits[:, data['pair_ids']==pid] for pid in ids]) if sid else None
                    both = pair_hit.all(2).mean() if sid else None
                    if sid:
                        hits_by_split[dataset_name][(i, condition)] = pair_hit.mean(1)
                        hits_by_split[dataset_name][(i, condition+'_both')] = pair_hit.all(2).mean(1)
                    rows.append({'source': i, 'dataset': dataset_name, 'condition': condition, 'split': split,
                                 'draws': len(draws), 'accuracy': float(hits[:, use].mean()),
                                 'pair_both_correct': None if both is None else float(both),
                                 **{k: float(v[use].mean()) for k, v in metrics.items()}})
            # Observational overlap restriction uses the same input for both arms.
            cm, wm = [same_input[c] for c in ['swap_linear_correct_null_nonlinear_correct',
                                            'swap_linear_correct_null_nonlinear_wrong']]
            ch, wh = [np.load(root/'evaluations'/f's{i}_{dataset_name}_{c}.npz')['hits'][0] for c in [
                'swap_linear_correct_null_nonlinear_correct', 'swap_linear_correct_null_nonlinear_wrong']]
            for caliper in config['matched_relative_calipers']:
                rel = lambda a, b: np.abs(a-b)/np.maximum((np.abs(a)+np.abs(b))/2, 1e-12)
                select = (rel(cm['null_norm'], wm['null_norm'])<=caliper)&(
                    rel(cm['null_state_squared_error'], wm['null_state_squared_error'])<=caliper)
                for sid, split in [(0, 'iid'), (1, 'collisions')]:
                    use = select & (data['split']==sid)
                    matching.append({'source': i, 'dataset': dataset_name, 'split': split, 'relative_caliper': caliper,
                        'matched_samples': int(use.sum()), 'total_samples': int((data['split']==sid).sum()),
                        'conditional_accuracy_difference_pp': float(100*(ch[use].astype(float)-wh[use]).mean()) if use.any() else None,
                        'post_treatment_conditioning_not_causal_isolation': True})
            for (representation, task), (mean, scale, theta, selection, transform) in fitted.items():
                target = data['labels'][:, 1 if task=='current' else 3]
                features = np.column_stack([(hidden[:, 1]@transform-mean)/scale, np.ones(len(hidden))])
                predicted = (features@theta).argmax(1)
                training_target = train_labels if task=='current' else future_train
                mode = int(np.bincount(training_target, minlength=13).argmax())
                for sid, split in [(0, 'iid'), (1, 'collisions')]:
                    use = data['split']==sid
                    probes.append({'source': i, 'dataset': dataset_name, 'representation': representation,
                        'target': task, 'split': split, 'accuracy': float((predicted[use]==target[use]).mean()),
                        'training_mode_accuracy': float((target[use]==mode).mean()), **selection})
        print(json.dumps({'source_complete': i, 'frozen_diagnostics_only': True}), flush=True)
    means = []
    for dataset_name in datasets:
        for split in ['iid', 'collisions']:
            for condition in sorted({r['condition'] for r in rows}):
                rr = [r for r in rows if (r['dataset'], r['split'], r['condition'])==(dataset_name, split, condition)]
                keys = [k for k, v in rr[0].items() if isinstance(v, float)]
                means.append({'dataset': dataset_name, 'split': split, 'condition': condition,
                              **{k: float(np.mean([r[k] for r in rr])) for k in keys}})
    contrasts = []
    comparisons = [
        ('correct_vs_wrong_donor', 'swap_linear_correct_null_nonlinear_correct', 'swap_linear_correct_null_nonlinear_wrong'),
        ('correct_vs_error_calibrated_wrong', 'swap_linear_correct_null_nonlinear_correct', 'wrong_norm_error_calibrated'),
        ('correct_vs_error_calibrated_random', 'swap_linear_correct_null_nonlinear_correct', 'random_norm_error_calibrated'),
        ('correct_vs_covariance_transported_wrong', 'swap_linear_correct_null_nonlinear_correct', 'wrong_covariance_transported'),
        ('correct_vs_covariance_random', 'swap_linear_correct_null_nonlinear_correct', 'random_validation_covariance')]
    for dataset_name, values in hits_by_split.items():
        for endpoint, suffix in [('accuracy', ''), ('pair_both_correct', '_both')]:
            for name, correct, control in comparisons:
                difference = np.array([100*(values[(i, correct+suffix)]-values[(i, control+suffix)])
                                       for i in range(len(parent_config['source_seeds']))])
                # Individual hits retain the two rows in each collision pair.
                if difference.ndim==3:
                    difference = difference.mean(2)
                contrasts.append({'dataset': dataset_name, 'endpoint': endpoint, 'contrast': name,
                    'mean_pp': float(difference.mean()), 'source_effects_pp': difference.mean(1).tolist(),
                    'bootstrap_95_pp': crossed_interval(difference, config['bootstrap_samples'], config['bootstrap_seed'])})
    atomic_json(root/'results.json', {'status': 'complete', 'completed_utc': now(),
        'scope': config['scope'], 'source_models': len(parent_config['source_seeds']),
        'fresh_collision_pairs': config['fresh_collision_pairs'], 'rows': rows, 'means': means,
        'contrasts': contrasts, 'projection_and_matching_audit': audit,
        'error_conditioned_subsets': matching, 'label_probes': probes,
        'covariance_and_error_matching_are_separate_controls': True,
        'natural_causal_relation_encoding_established': False})
    atomic_json(root/'delivery.json', {'completed_utc': now(), 'status': 'complete',
        'artifact_sha256': {str(p): sha(p) for p in root.rglob('*') if p.is_file() and p.name!='delivery.json'}})


def run(stage):
    config, root, parent, parent_config = setup()
    prepare(config, root, parent, parent_config)
    if stage=='evaluate':
        evaluate(config, root, parent, parent_config)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'evaluate'], default='prepare')
    run(parser.parse_args().stage)
