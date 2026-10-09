"""Expanded orbit census, natural interchange, and numerical ridge sensitivity."""
from collections import defaultdict
from itertools import product
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

from . import algebra_relation_worlds as worlds
from .algebra_relation_feasibility import apply_encoding
from .final_mechanism_evaluate import predicted_states
from .longrun_engine import atomic_json
from .operator_capacity_confirmation import load_backbone
from .permworld_combinations import sha
from .relation_error_localization import row_projection
from .reviewer_revision_statistics import tost, required_n
from .two_step_relation_factorial import now

CONFIG = Path('configs/reviewer_revision_diagnostics.json')


def donor_permutations(labels, orbit_ids, seed):
    """Balanced donor permutations preserve marginal states, without model scores."""
    rng = np.random.default_rng(seed); different = np.full(len(labels), -1); same = different.copy()
    for output, same_future in [(different, False), (same, True)]:
        groups = defaultdict(list)
        for i, row in enumerate(labels):
            groups[(int(row[1]), int(row[3])) if same_future else (int(row[1]),)].append(i)
        for ids in groups.values():
            ids = np.array(ids); n = len(ids)
            allowed = orbit_ids[ids, None]!=orbit_ids[ids][None, :]
            if not same_future:
                allowed &= labels[ids, 3, None]!=labels[ids, 3][None, :]
            cost = rng.uniform(0, 1, (n, n))+100000*(~allowed)
            row, col = linear_sum_assignment(cost)
            assert allowed[row, col].all(), 'Gold-only balanced donor matching is infeasible.'
            output[ids[row]] = ids[col]
        assert np.array_equal(np.sort(output), np.arange(len(labels)))
        assert np.array_equal(labels[output, 1], labels[:, 1])
        assert (orbit_ids[output]!=orbit_ids).all()
        assert ((labels[output, 3]==labels[:, 3]) if same_future else
                (labels[output, 3]!=labels[:, 3])).all()
    return different, same


def regularized_affine(h, target, teacher, w, bw, scale, config, ridge):
    """KD+normalized geometry+ridge; ridge-whitened row solve and exact null solve.

    Ridge penalizes standardized centered input coefficients, excluding intercept.
    This changes the old empirical objective and is a sensitivity analysis.
    """
    mean = h.mean(0); input_scale = np.sqrt(h.var(0).mean())
    x = np.column_stack([(h-mean)/input_scale, np.ones(len(h))])
    g = config['geometry_weight']/scale
    penalty = np.diag([ridge]*(x.shape[1]-1)+[0.])
    metric = g*x.T@x/len(x)+penalty
    eig, vec = np.linalg.eigh(metric); assert eig.min()>0
    transform = (vec/eig**.5)@vec.T
    features = x@transform
    _, singular, vt = np.linalg.svd(w, full_matrices=False)
    row = vt[singular>singular[0]*1e-10]
    optimum = g*transform.T@x.T@target/len(x)
    null = optimum-optimum@row.T@row
    tensor = lambda v: torch.as_tensor(v, dtype=torch.float64, device='cuda')
    fx, ty, nu, rb, wt, bwt, mt, pn = [tensor(v) for v in [features, target, null, row, w, bw, transform, penalty]]
    logits = tensor(teacher); temperature = config['temperature']
    teacher_log = F.log_softmax(logits/temperature, -1); teacher_prob = teacher_log.exp()
    parameter = torch.nn.Parameter(tensor(optimum@row.T))
    solver = torch.optim.LBFGS([parameter], max_iter=1000, tolerance_grad=1e-9,
                              tolerance_change=1e-13, history_size=100, line_search_fn='strong_wolfe')
    calls = 0
    def loss():
        coefficients = nu+parameter@rb
        prediction = fx@coefficients
        student = F.log_softmax((prediction@wt.T+bwt)/temperature, -1)
        kd = (teacher_prob*(teacher_log-student)).sum(-1).mean()*temperature**2
        geometry = (prediction-ty).square().mean()/scale
        theta = mt@coefficients
        regularization = ((pn@theta)*theta).sum()/target.shape[1]
        return config['output_kd_weight']*kd+config['geometry_weight']*geometry+regularization
    def closure():
        nonlocal calls
        calls+=1; solver.zero_grad(set_to_none=True)
        value=loss(); value.backward(); return value
    solver.step(closure); objective=closure()
    coefficients = null+parameter.detach().cpu().numpy()@row
    theta = transform@coefficients
    gradient_norm = float(parameter.grad.norm())
    return (mean, input_scale, theta), {'objective':float(objective.detach()),
        'gradient_norm':gradient_norm, 'whitened_gap_bound':gradient_norm**2*target.shape[1]/4,
        'ridge':ridge, 'whitened_metric_condition':float(eig.max()/eig.min()),
        'coefficient_norm':float(np.linalg.norm(theta[:-1])), 'calls':calls,
        'regularization_excludes_intercept':True}


def apply_affine(fit, h):
    mean, scale, theta=fit
    return np.column_stack([(h-mean)/scale, np.ones(len(h))])@theta


def setup():
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False
    apply_encoding('numeric_gelu')
    config=json.loads(CONFIG.read_text()); root=Path(config['output']); parent=Path(config['parent'])
    pc=json.loads(Path('configs/final_mechanism_confirmation.json').read_text())
    for name in ['', 'datasets', 'states', 'evaluations', 'ridge']:(root/name).mkdir(parents=True,exist_ok=True)
    signature={'config':config,'code_sha256':sha(__file__),'statistics_code_sha256':sha('experiments/reviewer_revision_statistics.py'),
               'parent_delivery_sha256':sha(parent/'delivery.json'),'review_controls_delivery_sha256':sha('results/null_space_review_controls/delivery.json')}
    file=root/'protocol.json'
    if file.exists():assert json.loads(file.read_text())['signature']==signature
    else:
        atomic_json(file,{'registered_utc':now(),'new_outcomes_observed':False,'existing_outcomes_observed':True,'signature':signature})
        (root/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return config,root,parent,pc


def prepare(config,root,parent,pc):
    if (root/'data_audit.json').exists():return
    key=lambda x:worlds.key(tuple(x),'matrix',13)
    hist_files=pc['excluded_test_datasets']+[str(parent/'dataset/test.npz'),'results/null_space_review_controls/datasets/fresh.npz']
    historical={key(x) for file in hist_files for x in np.load(file)['x'][:,0]}
    shared=[parent/'dataset'/f'{n}.npz' for n in ['source','validation','pilot_validation']]
    common=historical|{key(x) for file in shared for x in np.load(file)['x'][:,0]}
    universe=[x for x in product(range(13),repeat=4) if (x[0]*x[3]-x[1]*x[2])%13]
    records=[]
    for i in range(5):
        support=parent/'dataset'/f'support{i}.npz'
        excluded=common|{key(x) for x in np.load(support)['x'][:,0]}
        anchors=[x for x in universe if key(x) not in excluded]
        keys=sorted({key(x) for x in anchors}); lookup={k:j for j,k in enumerate(keys)}
        data=worlds.encode(anchors,'matrix',13,['','a','b','ab'])
        orbit=np.array([lookup[key(x)] for x in anchors])
        different,same=donor_permutations(data['labels'],orbit,config['pairing_seed']+i)
        data.update(orbit_ids=orbit,different_donor=different,same_donor=same)
        np.savez_compressed(root/'datasets'/f's{i}.npz',**data)
        assert len(anchors)==6*len(keys)
        records.append({'source':i,'unexposed_orbits':len(keys),'inputs':len(anchors),
                        'full_six_orientations_are_dependent':True,'unique_donors_per_condition':len(anchors),
                        'excluded_orbits':len(excluded),'dataset_sha256':sha(root/'datasets'/f's{i}.npz')})
    atomic_json(root/'data_audit.json',{'registered_utc':now(),'records':records,
                'all_prior_test_orbits_excluded':True,'per_source_training_and_validation_excluded':True,
                'different_sources_have_partly_overlapping_tests':True,
                'donor_reuse_eliminated_but_recipient_donor_network_dependence_remains':True})


def run(stage):
    config,root,parent,pc=setup();prepare(config,root,parent,pc)
    if stage=='prepare':return
    assert not (root/'results.json').exists()
    atomic_json(root/'test_opened.json',{'opened_utc':now(),'gold_only_donors_fixed':True})
    results=[];exchanges=[];ridge_records=[];paired_noise=[]
    for i in range(5):
        data=dict(np.load(root/'datasets'/f's{i}.npz'));model,_=load_backbone(parent,i,'cuda')
        z=dict(np.load(parent/'states'/f's{i}_known.npz'))
        w,bw,b,bb=[z[k].astype(float) for k in ['w','bias_w','b','bias_b']]
        with torch.no_grad():h=model(torch.as_tensor(worlds.feature(data['x'],13),device='cuda')).cpu().numpy().astype(float)
        np.savez_compressed(root/'states'/f's{i}.npz',hidden=h.astype(np.float32))
        pred=predicted_states(pc,parent,i,h[:,0],z)
        oracle=((h[:,1]@b+bb)@w.T+bw).argmax(1)==data['labels'][:,3]
        hit_by_condition={}
        def metrics(name,first):
            hits=((first@b+bb)@w.T+bw).argmax(1)==data['labels'][:,3]
            first_hit=(first@w.T+bw).argmax(1)==data['labels'][:,1]
            mse=np.square(first-h[:,1]).mean(1)
            hit_by_condition[name]=hits
            np.savez_compressed(root/'evaluations'/f's{i}_{name}.npz',hits=hits,first_hits=first_hit,state_mse=mse)
            results.append({'source':i,'condition':name,'inputs':len(h),'orbits':len(np.unique(data['orbit_ids'])),
                'accuracy':float(hits.mean()),'first_accuracy':float(first_hit.mean()),'first_state_mse':float(mse.mean()),
                'true_intermediate_second_accuracy':float(oracle.mean()),'second_accuracy_drop':float(oracle.mean()-hits.mean()),
                'readout_harmful_error':float(np.square((first-h[:,1])@b@w.T).mean())})
        for c in ['linear_correct','linear_wrong','nonlinear_correct','nonlinear_wrong',
                  'affine_correct_ols_initialization','affine_wrong_ols_initialization']:
            metrics(c,pred[c])
        # Preserve *all* recipient logits, not just the shared gold label.
        p=row_projection(w);q=np.eye(128)-p;natural=h[:,1]
        exchange_hits={}
        for name,ids in [('different_future',data['different_donor']),('same_future',data['same_donor'])]:
            mixed=natural@p+natural[ids]@q
            change=np.abs((mixed-natural)@w.T).max(1);assert change.max()<1e-8
            scores=(mixed@b+bb)@w.T+bw;classes=scores.argmax(1)
            target_hit=classes==data['labels'][ids,3];recipient_hit=classes==data['labels'][:,3]
            baseline=((natural@b+bb)@w.T+bw).argmax(1)
            exchange_hits[name]=target_hit
            np.savez_compressed(root/'evaluations'/f's{i}_exchange_{name}.npz',
                donor_answer_hit=target_hit,recipient_answer_hit=recipient_hit,classes=classes,
                recipient_baseline=baseline,max_first_logit_change=change,
                nearest_recipient_rms=np.sqrt(np.square(mixed-natural).mean(1)),
                nearest_donor_rms=np.sqrt(np.square(mixed-natural[ids]).mean(1)))
            exchanges.append({'source':i,'condition':name,'donor_target_accuracy':float(target_hit.mean()),
                'recipient_target_accuracy':float(recipient_hit.mean()),'prediction_changed_fraction':float((classes!=baseline).mean()),
                'natural_recipient_second_accuracy':float(oracle.mean()),'natural_donor_second_accuracy':float(oracle[ids].mean()),
                'first_logit_change_max':float(change.max()),
                'recipient_distance_rms':float(np.sqrt(np.square(mixed-natural).mean(1)).mean()),
                'donor_distance_rms':float(np.sqrt(np.square(mixed-natural[ids]).mean(1)).mean()),
                'both_natural_correct_subset_count':int((oracle&oracle[ids]).sum()),
                'conditional_donor_target_accuracy':float(target_hit[oracle&oracle[ids]].mean())})
        train=z['train'][:,0].astype(float);target=z['train'][:,1].astype(float)
        teacher=np.load(parent/'states'/f's{i}_fixed_supervision.npz')['teacher_logits'].astype(float)
        scale=json.loads((parent/'predictor_fits'/f's{i}_linear_correct.json').read_text())['geometry_scale']
        val=z['validation'].astype(float);pcids=np.load(parent/'dataset'/f'support{i}.npz')['pc']
        for pairing in ['correct','wrong']:
            target_fit=target if pairing=='correct' else target[pcids]
            validation_losses=[]
            for ridge in config['ridge_path']:
                if ridge==0:
                    theta=np.load(parent/'affine_fits'/f's{i}_{pairing}_ols_initialization.npz')['theta']
                    pv=np.column_stack([val[:,0],np.ones(len(val))])@theta
                    test=pred[f'affine_{pairing}_ols_initialization'];metadata={'ridge':0.,'original_unregularized_fit':True}
                else:
                    fit,metadata=regularized_affine(train,target_fit,teacher,w,bw,scale,pc,ridge)
                    pv=apply_affine(fit,val[:,0]);test=apply_affine(fit,h[:,0])
                    np.savez_compressed(root/'ridge'/f's{i}_{pairing}_{ridge}.npz',mean=fit[0],scale=fit[1],theta=fit[2])
                # Fixed known validation criterion, no compound targets.
                logsoft=lambda a:a-a.max(1,keepdims=True)-np.log(np.exp(a-a.max(1,keepdims=True)).sum(1,keepdims=True))
                lt=logsoft((val[:,1]@w.T+bw)/pc['temperature']);lp=logsoft((pv@w.T+bw)/pc['temperature'])
                criterion=float(pc['output_kd_weight']*pc['temperature']**2*(np.exp(lt)*(lt-lp)).sum(1).mean()+
                                pc['geometry_weight']*np.square(pv-val[:,1]).mean()/scale)
                validation_losses.append(criterion)
                condition=f'ridge_{pairing}_{ridge}';metrics(condition,test)
                ridge_records.append({'source':i,'pairing':pairing,'ridge':ridge,'validation_criterion':criterion,**metadata})
            selected=int(np.argmin(validation_losses))
            ridge_records.append({'source':i,'pairing':pairing,'selected_ridge':config['ridge_path'][selected],
                                  'selection_uses_known_states_and_logits_only':True})
        # Correlated orientations are aggregated to independent finite-world orbits.
        d=(hit_by_condition['affine_correct_ols_initialization'].astype(float)-
           hit_by_condition['nonlinear_correct'].astype(float))
        orbit_means=np.array([d[data['orbit_ids']==j].mean() for j in np.unique(data['orbit_ids'])])
        paired_noise.append({'source':i,'paired_difference_pp':float(100*d.mean()),
            'paired_orbit_standard_error_pp':float(100*orbit_means.std(ddof=1)/np.sqrt(len(orbit_means))),
            'naive_input_standard_error_pp':float(100*d.std(ddof=1)/np.sqrt(len(d))),
            'naive_input_se_is_not_primary':True})
        print(json.dumps({'source':i,'completed':True,'input_count':len(h)}),flush=True)
    old=json.loads((parent/'results.json').read_text())
    effects=next(c['source_effects_pp'] for c in old['contrasts'] if c['endpoint']=='accuracy' and c['contrast']=='converged_linear_vs_gelu')
    sigma=float(np.std(effects,ddof=1))
    power=[{'margin_pp':m,'assumed_sd_pp':sigma,'true_mean_pp':mu,
            'required_independent_world_source_units':required_n(sigma,m,config['target_power'],config['alpha'],mu),
            'power_target':config['target_power'],'alpha':config['alpha'],
            'old_sd_not_a_new_world_variance_estimate':True}
           for m in config['secondary_equivalence_margins_pp'] for mu in [0.,1.]]
    secondary=[{'dataset':'original_shared_test','posthoc_only':True,**tost(effects,m)} for m in config['secondary_equivalence_margins_pp']]
    expanded=np.array([r['paired_difference_pp'] for r in paired_noise])
    secondary.extend({'dataset':'expanded_per_source_tests','posthoc_only':True,**tost(expanded,m)} for m in config['secondary_equivalence_margins_pp'])
    normalized=[]
    for i in range(5):
        get=lambda c:next(r for r in results if r['source']==i and r['condition']==c)
        correct,wrong=get('nonlinear_correct'),get('nonlinear_wrong')
        denominator=correct['true_intermediate_second_accuracy']-wrong['accuracy']
        normalized.append({'source':i,'fraction_of_oracle_gap_closed':(correct['accuracy']-wrong['accuracy'])/denominator,
                           'denominator':denominator,'no_claim_that_floor_is_removed':True})
    atomic_json(root/'results.json',{'status':'complete','completed_utc':now(),'scope':config['scope'],
        'primitive_and_composition':results,'natural_interchanges':exchanges,'ridge_path':ridge_records,
        'paired_sampling_error':paired_noise,'secondary_tost':secondary,'equivalence_power_sensitivity':power,
        'normalized_relation_gains':normalized,'equivalence_primary':False,
        'natural_donors_do_not_guarantee_natural_hybrids':True})
    atomic_json(root/'delivery.json',{'status':'complete','completed_utc':now(),
        'artifact_sha256':{str(p):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='delivery.json'}})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','evaluate'],default='prepare')
    run(parser.parse_args().stage)
