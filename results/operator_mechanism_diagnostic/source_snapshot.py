"""Predicted-state swaps and known-only adequately fitted affine diagnoses."""
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import algebra_relation_worlds as worlds
from .downstream_harm_diagnostic_v2 import metrics
from .longrun_engine import atomic_json
from .operator_capacity_confirmation import crossed_interval, load_backbone
from .operator_capacity_verify import replay
from .algebra_relation_feasibility import apply_encoding
from .permworld_combinations import sha
from .relation_error_localization import row_projection
from .two_step_relation_factorial import now

CONFIG = Path('configs/operator_mechanism_diagnostic.json')


def design(h):
    return np.column_stack([np.asarray(h, dtype=float), np.ones(len(h))])


def swap_null(recipient, donor, p):
    return recipient@p+donor@(np.eye(len(p))-p)


def fit_affine_known(h, target, true_target, w, bw, scale, config):
    """OLS + certified convex data objective + known-validation ridge candidates."""
    x = design(h)
    u, singular, vt = np.linalg.svd(x, full_matrices=False)
    keep = singular > singular[0]*config['solver']['svd_relative_rank_threshold']
    assert keep.all(), 'The declared unrestricted affine diagnosis requires numerical full rank.'
    transform = vt.T*(np.sqrt(len(x))/singular)[None, :]
    features = u*np.sqrt(len(x))
    c_ols = features.T@target/len(x)
    theta_ols = transform@c_ols
    # A fixed orthonormal output-row basis. Null directions have the closed-form OLS optimum.
    _, sr, vr = np.linalg.svd(w, full_matrices=False)
    row_basis = vr[sr>sr[0]*1e-10]
    null = c_ols-c_ols@row_basis.T@row_basis
    tensor = lambda z: torch.as_tensor(z, dtype=torch.float64, device='cuda')
    fx, ty, null_t, rb, wt, bwt = [tensor(z) for z in [features, target, null, row_basis, w, bw]]
    temperature = config['temperature']
    true_logits = tensor(true_target@w.T+bw)
    teacher_log = F.log_softmax(true_logits/temperature, -1)
    teacher_prob = teacher_log.exp()
    parameter = torch.nn.Parameter(tensor(c_ols@row_basis.T))
    opt = torch.optim.LBFGS([parameter], lr=1., max_iter=config['solver']['max_iterations'],
        tolerance_grad=config['solver']['gradient_tolerance'], tolerance_change=config['solver']['change_tolerance'],
        history_size=100, line_search_fn='strong_wolfe')
    calls = 0
    def objective():
        pred = fx@(null_t+parameter@rb)
        student_log = F.log_softmax((pred@wt.T+bwt)/temperature, -1)
        kd = (teacher_prob*(teacher_log-student_log)).sum(-1).mean()*temperature**2
        geometry = (pred-ty).square().mean()/scale
        return config['output_kd_weight']*kd+config['geometry_weight']*geometry
    def closure():
        nonlocal calls
        calls += 1
        opt.zero_grad(set_to_none=True)
        loss = objective(); loss.backward()
        return loss
    started = time.monotonic()
    opt.step(closure)
    loss = closure()
    gradient = parameter.grad.detach().cpu().numpy()
    curvature = 2*config['geometry_weight']/(scale*target.shape[1])
    bound = float(np.square(gradient).sum()/(2*curvature))
    c_convex = null+parameter.detach().cpu().numpy()@row_basis
    theta_convex = transform@c_convex
    normal_residual = float(np.linalg.norm(features.T@(features@c_ols-target)/len(x)))
    metadata = {'data_rank': len(singular), 'singular_values': singular.tolist(),
        'condition_number': float(singular[0]/singular[-1]), 'row_rank': len(row_basis),
        'ols_whitened_normal_residual': normal_residual,
        'convex_objective': float(loss.detach()), 'convex_gradient_norm': float(np.linalg.norm(gradient)),
        'strong_convexity_mu': curvature, 'training_optimality_gap_upper_bound': bound,
        'convergence_certificate_pass': bound<=config['solver']['gap_bound_tolerance'],
        'lbfgs_iterations': int(opt.state[parameter]['n_iter']), 'closure_calls': calls,
        'seconds': time.monotonic()-started, 'train_rows': len(x),
        'no_compound_fit_targets': True, 'unregularized_data_objective_only': True}
    ridge = {}
    for lam in config['ridge_candidates']:
        theta = (vt.T*(singular/(singular**2+len(x)*lam))[None, :])@(u.T@target)
        ridge[str(lam)] = theta
    return theta_ols, theta_convex, ridge, metadata


def single_objective(pred, truth, w, bw, scale, config):
    def logsoft(z):
        z = z-z.max(-1, keepdims=True)
        return z-np.log(np.exp(z).sum(-1, keepdims=True))
    t = config['temperature']
    lt = logsoft((truth@w.T+bw)/t)
    lp = logsoft((pred@w.T+bw)/t)
    kd = float((np.exp(lt)*(lt-lp)).sum(-1).mean()*t*t)
    geom = float(np.square(pred-truth).mean()/scale)
    return {'objective': config['output_kd_weight']*kd+config['geometry_weight']*geom,
            'kd': kd, 'true_state_nmse': geom}


def nested_interval(values, draws, seed):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    source = rng.integers(0, len(values), (draws, len(values)))
    pair = rng.integers(0, values.shape[1], (draws, len(values), values.shape[1]))
    resample = values[source[:, :, None], pair].mean((1, 2))
    return np.quantile(resample, [.025, .975]).tolist()


def setup():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    apply_encoding('numeric_gelu')
    config = json.loads(CONFIG.read_text()); root = Path(config['output']); parent = Path(config['parent'])
    for name in ['', 'fits', 'datasets', 'states', 'evaluations']:
        (root/name).mkdir(parents=True, exist_ok=True)
    old = json.loads((parent/'delivery.json').read_text())
    signature = {'config': config, 'config_sha256': sha(CONFIG), 'code_sha256': sha(__file__),
        'parent_delivery_sha256': sha(parent/'delivery.json'),
        'parent_artifacts_sha256': old['artifact_sha256'],
        'dependencies_sha256': {p:sha(p) for p in ['experiments/operator_capacity_confirmation.py',
            'experiments/operator_capacity_verify.py', 'experiments/downstream_harm_diagnostic_v2.py',
            'experiments/relation_error_localization.py', worlds.__file__]}}
    file = root/'protocol.json'
    if file.exists():
        assert json.loads(file.read_text())['signature'] == signature
    else:
        atomic_json(file, {'registered_utc': now(), 'existing_results_observed': True,
                          'fresh_compound_predictions_observed': False, 'signature': signature})
        (root/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return config, root, parent


def prepare_fresh(config, root, parent):
    if (root/'data_audit.json').exists():
        return
    from itertools import product
    hist_files = ['results/algebra_relation_v3/matrix/dataset/test.npz',
                  'results/readout_null_confirmation/matrix/dataset/test.npz', str(parent/'dataset/test.npz')]
    key = lambda x: worlds.key(tuple(x), 'matrix', 13)
    excluded = {key(x) for p in hist_files for x in np.load(p)['x'][:, 0]}
    for name in ['source', 'validation', 'pilot_validation']:
        excluded |= {key(x) for x in np.load(parent/'dataset'/f'{name}.npz')['x'][:, 0]}
    all_rows = [tuple(x) for x in product(range(13), repeat=4) if (x[0]*x[3]-x[1]*x[2])%13]
    fresh_used = set(); audits = []
    for i in range(3):
        own = {key(x) for x in np.load(parent/'dataset'/f'support{i}.npz')['x'][:, 0]}
        rows = [x for x in all_rows if key(x) not in excluded|own|fresh_used]
        rng = np.random.default_rng(config['fresh_seed']+i)
        used = set()
        collision = worlds.choose_collision_pairs(rows, config['fresh_collision_pairs_per_source'], rng, 'matrix', 13, used)
        rng.shuffle(rows); iid=[]
        for x in rows:
            if key(x) not in used:
                iid.append(x); used.add(key(x))
                if len(iid)==config['fresh_iid_examples_per_source']:
                    break
        assert len(iid)==config['fresh_iid_examples_per_source']
        fresh_used |= used
        data = worlds.encode(iid+collision, 'matrix', 13, ['', 'a', 'b', 'ab'])
        data.update(split=np.array([0]*len(iid)+[1]*len(collision)),
                    pair_ids=np.array([-1]*len(iid)+[j for j in range(len(collision)//2) for _ in range(2)]))
        np.savez_compressed(root/'datasets'/f's{i}_fresh.npz', **data)
        audits.append({'source': i, 'fresh_orbits': len(used), 'candidate_orbits': len({key(x) for x in rows}),
                       'own_training_and_all_seen_test_orbits_excluded': True})
    atomic_json(root/'data_audit.json', {'created_utc':now(), 'status':'passed', 'source_audits':audits,
        'all_three_fresh_test_orbits_mutually_disjoint':True, 'scope':config['fresh_policy'],
        'dataset_sha256': {str(p):sha(p) for p in (root/'datasets').glob('*.npz')}})


def fit_all(config, root, parent):
    records=[]
    for i in range(3):
        z = dict(np.load(parent/'states'/f's{i}_known.npz'))
        h, true = z['train'][:,0].astype(float), z['train'][:,1].astype(float)
        valh, valtrue = z['validation'][:,0].astype(float), z['validation'][:,1].astype(float)
        w, bw = z['w'].astype(float), z['bias_w'].astype(float)
        scale = float(np.var(z['train'][:,1],axis=0).mean())
        support = dict(np.load(parent/'dataset'/f'support{i}.npz'))
        for pairing in ['correct','wrong']:
            path = root/'fits'/f's{i}_{pairing}.json'
            if path.exists():
                records.append(json.loads(path.read_text())); continue
            target = true if pairing=='correct' else true[support['pc']]
            ols, convex, candidates, metadata = fit_affine_known(h,target,true,w,bw,scale,config)
            scored = {k:single_objective(design(valh)@theta,valtrue,w,bw,scale,config) for k,theta in candidates.items()}
            selected = min(scored,key=lambda k:scored[k]['objective'])
            theta = {'state_ols':ols,'convex_kd_geometry':convex,'validation_ridge_state':candidates[selected]}
            file = root/'fits'/f's{i}_{pairing}.npz'
            np.savez_compressed(file, **theta)
            grades={}
            for name, param in theta.items():
                grades[name] = {'train':single_objective(design(h)@param,true,w,bw,scale,config),
                               'validation':single_objective(design(valh)@param,valtrue,w,bw,scale,config),
                               'affine_coefficient_norm':float(np.linalg.norm(param)),
                               'assigned_geometry_train_nmse':float(np.square(design(h)@param-target).mean()/scale)}
            record={'source':i,'pairing':pairing,'completed_utc':now(),'solver':metadata,'scores':grades,
                    'ridge_selected_lambda':float(selected),'ridge_known_validation_candidates':scored,
                    'checkpoint_sha256':sha(file),'known_states_sha256':sha(parent/'states'/f's{i}_known.npz')}
            atomic_json(path,record);records.append(record)
            print(json.dumps({'source':i,'pairing':pairing,'solver':metadata,'validation':{k:v['validation'] for k,v in grades.items()} }),flush=True)
    atomic_json(root/'fit_summary.json',{'status':'complete','completed_utc':now(),'records':records,
        'all_six_convex_gap_certificates_pass':all(r['solver']['convergence_certificate_pass'] for r in records)})


def per_sample(first, truth, b, bb, w, bw, labels, scale):
    p = row_projection(w); q = np.eye(len(p))-p
    error = first-truth
    en, er = error@q, error@p
    z = (truth@b+bb)@w.T+bw
    dn, dr = en@b@w.T, er@b@w.T
    dt = error@b@w.T
    np.testing.assert_allclose(dt,dn+dr,atol=1e-9,rtol=1e-9)
    c = lambda a:a-a.mean(-1,keepdims=True)
    cn, cr, ct = [c(d) for d in [dn,dr,dt]]
    nm = metrics(z,dn,labels)
    total_m = metrics(z,dt,labels)
    return {'hidden_error':np.square(error).mean(1)/scale,'null_error':np.square(en).mean(1)/scale,
        'row_error':np.square(er).mean(1)/scale,'downstream_total_sq':np.square(ct).sum(1),
        'downstream_null_sq':np.square(cn).sum(1),'downstream_row_sq':np.square(cr).sum(1),
        'downstream_cross':2*(cn*cr).sum(1),'oracle_center_sq':np.square(c(z)).sum(1),
        'compound_hit':total_m['perturbed_correct'],'oracle_hit':nm['forced_correct'],
        'null_only_hit':nm['perturbed_correct'],'null_margin_crossing':nm['worst_margin_ratio']>=1,
        'first_scores':first@w.T+bw}


def evaluate(config, root, parent):
    fits = list((root/'fits').glob('*.json')); assert len(fits)==6
    marker = root/'fresh_opened.json'
    assert not marker.exists(), 'Preserve the first fresh inference; do not silently reopen.'
    atomic_json(marker,{'opened_utc':now(),'all_known_only_fits_complete':True,
                       'fit_record_sha256':{str(p):sha(p) for p in fits}})
    rows=[]; pair_stats={}; max_preserved=0.; max_prob=0.
    for test in ['existing','fresh']:
        for i in range(3):
            z = dict(np.load(parent/'states'/f's{i}_known.npz'))
            if test=='existing':
                data=dict(np.load(parent/'dataset/test.npz'));h=np.load(parent/'states'/f's{i}_test.npz')['hidden'].astype(float)
            else:
                data=dict(np.load(root/'datasets'/f's{i}_fresh.npz'))
                model,_=load_backbone(parent,i,'cuda')
                with torch.no_grad():h=model(torch.as_tensor(worlds.feature(data['x'],13),device='cuda')).cpu().numpy().astype(float)
                np.savez_compressed(root/'states'/f's{i}_fresh.npz',hidden=h)
            w,bw,b,bb=[z[k].astype(float) for k in ['w','bias_w','b','bias_b']]
            scale=float(np.var(z['train'][:,1],axis=0).mean());p=row_projection(w)
            predictions={}
            for condition in config['conditions']:
                state=torch.load(parent/'predictors'/f's{i}_{condition}.pt',map_location='cpu',weights_only=True)
                predictions[condition]=replay(h[:,0],state['state_dict'],condition.startswith('nonlinear'))
            for pairing in ['correct','wrong']:
                state=dict(np.load(root/'fits'/f's{i}_{pairing}.npz'))
                for kind,theta in state.items():predictions[kind+'_'+pairing]=design(h[:,0])@theta
            for recipient,donor in config['hybrids']:
                hybrid=swap_null(predictions[recipient],predictions[donor],p)
                delta=(hybrid-predictions[recipient])@w.T
                max_preserved=max(max_preserved,float(np.abs(delta).max()))
                assert np.abs(delta).max()<1e-8
                def soft(a):
                    a=a-a.max(1,keepdims=True);ex=np.exp(a);return ex/ex.sum(1,keepdims=True)
                before=predictions[recipient]@w.T+bw;after=hybrid@w.T+bw
                max_prob=max(max_prob,float(np.abs(soft(before)-soft(after)).max()))
                assert np.array_equal(before.argmax(1),after.argmax(1))
                predictions[f'swap_{recipient}_null_{donor}']=hybrid
            for name,first in predictions.items():
                m=per_sample(first,h[:,1],b,bb,w,bw,data['labels'][:,3],scale)
                np.savez_compressed(root/'evaluations'/f'{test}_s{i}_{name}.npz',first_pred=first,**m)
                for sid,split in [(0,'iid'),(1,'collisions')]:
                    use=data['split']==sid;oracle=use&m['oracle_hit']
                    denom=float(m['oracle_center_sq'][use].mean())
                    row={'test':test,'source':i,'condition':name,'split':split,'accuracy':float(m['compound_hit'][use].mean()),
                        'first_accuracy':float((m['first_scores'].argmax(1)[use]==data['labels'][use,1]).mean()),
                        'hidden_nmse':float(m['hidden_error'][use].mean()),'null_nmse':float(m['null_error'][use].mean()),
                        'row_nmse':float(m['row_error'][use].mean()),'null_only_accuracy':float(m['null_only_hit'][use].mean()),
                        'null_margin_crossing_fraction':float(m['null_margin_crossing'][oracle].mean()),
                        'oracle_accuracy':float(m['oracle_hit'][use].mean()),'pair_both_correct':None}
                    for k in ['total','null','row','cross']:
                        key='downstream_'+k+('_sq' if k!='cross' else '')
                        row['downstream_'+k+'_normalized']=float(m[key][use].mean()/denom)
                    if sid==1:
                        phit=np.asarray([m['compound_hit'][data['pair_ids']==pid] for pid in np.unique(data['pair_ids'][use])])
                        row['pair_both_correct']=float(phit.all(1).mean())
                        pair_stats[(test,i,name)]=phit
                    rows.append(row)
    means=[]
    for key in sorted({(r['test'],r['condition'],r['split']) for r in rows}):
        chosen=[r for r in rows if (r['test'],r['condition'],r['split'])==key]
        numeric=[k for k,v in chosen[0].items() if k not in ['test','source','condition','split'] and isinstance(v,(float,int))]
        means.append(dict(zip(['test','condition','split'],key),**{k:float(np.mean([r[k] for r in chosen])) for k in numeric}))
    comparisons=[('correct_null_swap_gain','swap_linear_correct_null_nonlinear_correct','linear_correct'),
        ('wrong_null_swap_gain','swap_linear_correct_null_nonlinear_wrong','linear_correct'),
        ('correct_vs_wrong_null_donor','swap_linear_correct_null_nonlinear_correct','swap_linear_correct_null_nonlinear_wrong'),
        ('remove_gelu_correct_null','nonlinear_correct','swap_nonlinear_correct_null_linear_correct'),
        ('convex_linear_vs_gelu','convex_kd_geometry_correct','nonlinear_correct'),
        ('ridge_linear_vs_gelu','validation_ridge_state_correct','nonlinear_correct'),
        ('ols_linear_vs_gelu','state_ols_correct','nonlinear_correct')]
    contrasts=[]
    for test in ['existing','fresh']:
        for label,a,c in comparisons:
            for endpoint in ['accuracy','pair_both_correct']:
                def values(name):
                    return np.asarray([pair_stats[(test,i,name)].mean(1) if endpoint=='accuracy' else pair_stats[(test,i,name)].all(1) for i in range(3)],float)
                v=100*(values(a)-values(c))
                interval=(crossed_interval if test=='existing' else nested_interval)(v,config['bootstrap_samples'],config['bootstrap_seed'])
                contrasts.append({'test':test,'contrast':label,'endpoint':endpoint,'mean_pp':float(v.mean()),
                    'source_effects_pp':v.mean(1).tolist(),'bootstrap_95_pp':interval})
    atomic_json(root/'results.json',{'status':'complete','completed_utc':now(),'records':rows,'means':means,'contrasts':contrasts,
        'max_swap_first_logit_change':max_preserved,'max_swap_first_probability_change':max_prob,
        'scope':config['analysis_scope'],'fresh_scope':config['fresh_policy'],'sources':3})


def run():
    config,root,parent=setup();prepare_fresh(config,root,parent)
    atomic_json(root/'state.json',{'status':'running','stage':'known_only_linear_fits','updated_utc':now()})
    fit_all(config,root,parent)
    atomic_json(root/'state.json',{'status':'running','stage':'existing_and_fresh_diagnostics','updated_utc':now()})
    evaluate(config,root,parent)
    atomic_json(root/'state.json',{'status':'evaluated','updated_utc':now()})
    print(json.dumps({'status':'evaluated','output':str(root)}),flush=True)


if __name__=='__main__':run()
