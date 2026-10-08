"""Independently replay interventions and scores without changing sealed files."""
import json
from pathlib import Path

import numpy as np

from .final_mechanism_evaluate import predicted_states
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .public_reproduction import check_manifest
from .two_step_relation_factorial import now


def run():
    config = json.loads(Path('configs/null_space_review_controls.json').read_text())
    pc = json.loads(Path('configs/final_mechanism_confirmation.json').read_text())
    root, parent = Path(config['output']), Path(config['parent'])
    checks = check_manifest(root/'delivery.json')
    result = json.loads((root/'results.json').read_text())
    replay_checks, quality = 0, []
    datasets = {'previously_inspected': dict(np.load(parent/'dataset/test.npz')),
                'fresh': dict(np.load(root/'datasets/fresh.npz'))}

    def covariance(values):
        mu = values.mean(0); centered = values-mu
        raw = centered.T@centered/len(values)
        s = config['covariance_shrinkage']
        return mu, (1-s)*raw+s*np.trace(raw)/len(raw)*np.eye(len(raw)), raw

    def power(matrix, exponent):
        eig, vec = np.linalg.eigh(matrix)
        return (vec*eig**exponent)@vec.T

    def calibrate(reference, truth, proposed):
        # Different expression for the intersection of the two spheres.
        unit = truth/np.linalg.norm(truth, axis=1, keepdims=True)
        dot = np.einsum('ij,ij->i', reference, unit)[:, None]
        remaining = proposed-np.einsum('ij,ij->i', proposed, unit)[:, None]*unit
        radius = np.sqrt(np.maximum(np.square(reference).sum(1, keepdims=True)-dot**2, 0))
        return dot*unit+radius*remaining/np.linalg.norm(remaining, axis=1, keepdims=True)

    for i in range(len(pc['source_seeds'])):
        known = dict(np.load(parent/'states'/f's{i}_known.npz'))
        w, bw, b, bb = [known[k].astype(float) for k in ['w', 'bias_w', 'b', 'bias_b']]
        _, ss, vt = np.linalg.svd(w, full_matrices=True)
        rank = int((ss>ss[0]*1e-10).sum()); basis = vt[rank:].T
        p = vt[:rank].T@vt[:rank]
        vp = predicted_states(pc, parent, i, known['validation'][:, 0].astype(float), known)
        vc, vw = [vp[k]@basis for k in ['nonlinear_correct', 'nonlinear_wrong']]
        mc, cc, rawc = covariance(vc); mw, cw, _ = covariance(vw)
        transport = power(cw, -.5)@power(cc, .5)
        transported = (vw-mw)@transport+mc
        def relative_covariance_error(values):
            return float(np.linalg.norm(covariance(values)[2]-rawc)/np.linalg.norm(rawc))
        quality.append({'source': i, 'validation_mean_difference_before': float(np.linalg.norm(vw.mean(0)-mc)),
                        'validation_mean_difference_after': float(np.linalg.norm(transported.mean(0)-mc)),
                        'validation_raw_covariance_relative_error_before': relative_covariance_error(vw),
                        'validation_raw_covariance_relative_error_after': relative_covariance_error(transported),
                        'shrunk_covariance_transport_identity_max_error': float(np.abs(transport.T@cw@transport-cc).max()),
                        'empirical_covariance_is_not_exactly_matched': True})
        for name, data in datasets.items():
            h = (np.load(parent/'states'/f's{i}_test.npz')['hidden'] if name=='previously_inspected'
                 else np.load(root/'states'/f's{i}_fresh.npz')['hidden']).astype(float)
            predictions = predicted_states(pc, parent, i, h[:, 0], known)
            row = predictions['linear_correct']@p
            correct = predictions['nonlinear_correct']@basis
            wrong = predictions['nonlinear_wrong']@basis
            truth = h[:, 1]@basis
            controls = {k: predictions[k][None] for k in [
                'linear_correct', 'nonlinear_correct', 'nonlinear_wrong',
                'affine_correct_ols_initialization',
                'swap_linear_correct_null_nonlinear_correct',
                'swap_linear_correct_null_nonlinear_wrong']}
            controls['natural_true_intermediate'] = h[:, 1][None]
            controls['wrong_covariance_transported'] = (row+((wrong-mw)@transport+mc)@basis.T)[None]
            controls['wrong_norm_error_calibrated'] = (row+calibrate(correct, truth, wrong)@basis.T)[None]
            rng = np.random.default_rng(config['control_seed']+101*i+(name=='fresh'))
            error_controls, covariance_controls = [], []
            covroot = power(cc, .5)
            for _ in range(config['random_draws']):
                error_controls.append(row+calibrate(correct, truth, rng.normal(size=correct.shape))@basis.T)
                covariance_controls.append(row+(mc+rng.normal(size=correct.shape)@covroot)@basis.T)
            controls['random_norm_error_calibrated'] = np.array(error_controls)
            controls['random_validation_covariance'] = np.array(covariance_controls)
            for condition, first in controls.items():
                saved = dict(np.load(root/'evaluations'/f's{i}_{name}_{condition}.npz'))
                logits = (first@b+bb)@w.T+bw
                hits = logits.argmax(2)==data['labels'][None, :, 3]
                np.testing.assert_array_equal(hits, saved['hits']); replay_checks += hits.size
                errors = np.square(first-h[None, :, 1]).mean(2).mean(0)
                np.testing.assert_allclose(errors, saved['true_state_mse'], rtol=1e-10, atol=1e-10)
                replay_checks += len(errors)
                if condition.startswith(('swap_', 'wrong_', 'random_')):
                    np.testing.assert_allclose((first-predictions['linear_correct'])@w.T, 0, atol=1e-9)
                    replay_checks += first.shape[0]*first.shape[1]
                if condition.endswith('norm_error_calibrated'):
                    np.testing.assert_allclose(np.linalg.norm(first@basis, axis=2),
                                               np.broadcast_to(np.linalg.norm(correct, axis=1), first.shape[:2]), atol=1e-10)
                    baseline = controls['swap_linear_correct_null_nonlinear_correct'][0]
                    np.testing.assert_allclose(errors, np.square(baseline-h[:, 1]).mean(1), atol=1e-10)
                    replay_checks += 2*len(errors)
                for sid, split in [(0, 'iid'), (1, 'collisions')]:
                    record = next(r for r in result['rows'] if (r['source'], r['dataset'], r['condition'], r['split'])==
                                  (i, name, condition, split))
                    assert abs(hits[:, data['split']==sid].mean()-record['accuracy']) < 1e-12
                    replay_checks += 1
                    if sid:
                        pairhits = np.array([hits[:, data['pair_ids']==pid] for pid in sorted(set(data['pair_ids'][data['split']==1]))])
                        assert abs(pairhits.all(2).mean()-record['pair_both_correct']) < 1e-12
                        replay_checks += 1
    out = Path('results/null_space_review_audit'); out.mkdir(exist_ok=True)
    atomic_json(out/'verification.json', {'status': 'passed', 'completed_utc': now(),
                'manifest': checks, 'independent_replay_checks': replay_checks,
                'covariance_matching_quality': quality,
                'oracle_controls_are_not_usable_predictions': True,
                'parent_and_diagnostic_artifacts_modified': False})
    print(json.dumps({'status': 'passed', 'independent_replay_checks': replay_checks}), flush=True)


if __name__=='__main__':
    run()
