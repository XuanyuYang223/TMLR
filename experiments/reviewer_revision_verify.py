"""Independent audit of the expanded test, donor permutations and ridge gradients."""
import argparse
import json
from pathlib import Path

import numpy as np

from .final_mechanism_evaluate import predicted_states
from .final_mechanism_solver import objective_and_gradient
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .public_reproduction import check_manifest
from .two_step_relation_factorial import now


def run(output=Path('results/reviewer_revision_audit')):
    out=Path(output)
    if (out/'verification.json').exists():
        raise FileExistsError('Preserve the previous audit; choose a new --output directory.')
    root=Path('results/reviewer_revision_diagnostics_v2');parent=Path('results/final_mechanism_confirmation')
    config=json.loads(Path('configs/reviewer_revision_diagnostics.json').read_text())
    pc=json.loads(Path('configs/final_mechanism_confirmation.json').read_text())
    manifest=check_manifest(root/'delivery.json');checks=0;gradients=[]
    result=json.loads((root/'results.json').read_text())
    for i in range(5):
        data=dict(np.load(root/'datasets'/f's{i}.npz'));h=np.load(root/'states'/f's{i}.npz')['hidden'].astype(float)
        z=dict(np.load(parent/'states'/f's{i}_known.npz'));w,bw,b,bb=[z[k].astype(float) for k in ['w','bias_w','b','bias_b']]
        # Independent pseudoinverse projector, rather than the production SVD helper.
        p=np.linalg.pinv(w,rcond=1e-10)@w;q=np.eye(128)-p
        pred=predicted_states(pc,parent,i,h[:,0],z)
        for condition in ['linear_correct','linear_wrong','nonlinear_correct','nonlinear_wrong',
                          'affine_correct_ols_initialization','affine_wrong_ols_initialization']:
            first=pred[condition];hits=((first@b+bb)@w.T+bw).argmax(1)==data['labels'][:,3]
            saved=np.load(root/'evaluations'/f's{i}_{condition}.npz')
            np.testing.assert_array_equal(hits,saved['hits']);checks+=len(hits)
        use=data['exchange_eligible'];natural=h[use,1]
        for name,ids in [('different_future',data['different_donor']),('same_future',data['same_donor'])]:
            ids=ids[use]
            np.testing.assert_array_equal(np.sort(ids),np.flatnonzero(use));checks+=len(ids)
            assert (data['orbit_ids'][ids]!=data['orbit_ids'][use]).all()
            np.testing.assert_array_equal(data['labels'][ids,1],data['labels'][use,1]);checks+=len(ids)
            assert ((data['labels'][ids,3]!=data['labels'][use,3]) if name=='different_future' else
                    (data['labels'][ids,3]==data['labels'][use,3])).all();checks+=len(ids)
            mixed=natural@p+h[ids,1]@q
            np.testing.assert_allclose((mixed-natural)@w.T,0,atol=1e-9);checks+=len(ids)
            cls=((mixed@b+bb)@w.T+bw).argmax(1)
            saved=np.load(root/'evaluations'/f's{i}_exchange_{name}.npz')
            np.testing.assert_array_equal(cls,saved['classes']);checks+=len(cls)
        training=z['train'][:,0].astype(float);correct=z['train'][:,1].astype(float)
        pairing=np.load(parent/'dataset'/f'support{i}.npz')['pc'];teacher=np.load(parent/'states'/f's{i}_fixed_supervision.npz')['teacher_logits']
        scale=json.loads((parent/'predictor_fits'/f's{i}_linear_correct.json').read_text())['geometry_scale']
        for label in ['correct','wrong']:
            target=correct if label=='correct' else correct[pairing]
            for ridge in config['ridge_path'][1:]:
                fit=np.load(root/'ridge'/f's{i}_{label}_{ridge}.npz')
                x=np.column_stack([(training-fit['mean'])/fit['scale'],np.ones(len(training))])
                theta=fit['theta'];prediction=x@theta
                value,derivative=objective_and_gradient(prediction,target,teacher,w,bw,scale,pc)
                penalty=np.diag([ridge]*128+[0.])
                gradient=x.T@derivative+2*penalty@theta/128
                metric=pc['geometry_weight']/scale*x.T@x/len(x)+penalty
                eig,vec=np.linalg.eigh(metric);transform=(vec/eig**.5)@vec.T
                whitened=transform.T@gradient
                independent_value=value+np.sum((penalty@theta)*theta)/128
                rec=next(r for r in result['ridge_path'] if r['source']==i and r['pairing']==label and r.get('ridge')==ridge)
                difference=abs(independent_value-rec['objective'])
                assert difference<1e-5
                gradients.append({'source':i,'pairing':label,'ridge':ridge,'objective_replay_difference':float(difference),
                    'independent_whitened_gradient_norm':float(np.linalg.norm(whitened)),
                    'independent_whitened_gap_diagnostic':float(np.square(whitened).sum()*128/4),
                    'not_reported_as_exact_numerical_certificate':True})
                test=np.column_stack([(h[:,0]-fit['mean'])/fit['scale'],np.ones(len(h))])@theta
                hit=((test@b+bb)@w.T+bw).argmax(1)==data['labels'][:,3]
                saved=np.load(root/'evaluations'/f's{i}_ridge_{label}_{ridge}.npz')
                np.testing.assert_array_equal(hit,saved['hits']);checks+=len(hit)
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'verification.json',{'status':'passed','completed_utc':now(),'checks':checks,
        'manifest':manifest,'ridge_gradient_diagnostics':gradients,
        'parent_artifacts_modified':False,'same_answer_disruption_not_suppressed':True})
    print(json.dumps({'status':'passed','checks':checks}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results/reviewer_revision_audit'))
    run(parser.parse_args().output)
