"""Fixed final-confirmation endpoints, opened only after all declared fits."""
import json
from pathlib import Path
import numpy as np
import torch

from .final_mechanism_solver import objective_and_gradient
from .longrun_engine import atomic_json
from .operator_capacity_confirmation import load_backbone,crossed_interval
from .operator_capacity_verify import replay
from .operator_mechanism_diagnostic import per_sample,swap_null
from .relation_error_localization import row_projection
from .permworld_combinations import sha
from .two_step_relation_factorial import now
from . import algebra_relation_worlds as worlds


def predicted_states(config,root,i,h,z):
    output={}
    for c in config['conditions']:
        state=torch.load(root/'predictors'/f's{i}_{c}.pt',map_location='cpu',weights_only=True)['state_dict']
        output[c]=replay(h,state,c.startswith('nonlinear'))
    x=np.column_stack([h,np.ones(len(h))])
    for p in ['correct','wrong']:
        for kind in config['linear_solvers']:
            theta=np.load(root/'affine_fits'/f's{i}_{p}_{kind}.npz')['theta']
            output[f'affine_{p}_{kind}']=x@theta
    q=row_projection(z['w'].astype(float))
    for recipient,donor in config['hybrids']:
        output[f'swap_{recipient}_null_{donor}']=swap_null(output[recipient],output[donor],q)
    return output


def fixed_contrasts(pairs,config):
    coef={
        'gelu_relation_advantage':{'nonlinear_correct':1,'nonlinear_wrong':-1},
        'linear_relation_advantage':{'linear_correct':1,'linear_wrong':-1},
        'same_budget_interaction':{'nonlinear_correct':1,'nonlinear_wrong':-1,'linear_correct':-1,'linear_wrong':1},
        'correct_null_swap_gain':{'swap_linear_correct_null_nonlinear_correct':1,'linear_correct':-1},
        'correct_vs_wrong_null_donor':{'swap_linear_correct_null_nonlinear_correct':1,'swap_linear_correct_null_nonlinear_wrong':-1},
        'wrong_null_swap_gain':{'swap_linear_correct_null_nonlinear_wrong':1,'linear_correct':-1},
        'remove_correct_gelu_null':{'nonlinear_correct':1,'swap_nonlinear_correct_null_linear_correct':-1},
        'converged_linear_relation_advantage':{'affine_correct_ols_initialization':1,'affine_wrong_ols_initialization':-1},
        'remaining_relation_interaction':{'nonlinear_correct':1,'nonlinear_wrong':-1,'affine_correct_ols_initialization':-1,'affine_wrong_ols_initialization':1},
        'converged_linear_vs_gelu':{'affine_correct_ols_initialization':1,'nonlinear_correct':-1},
        'linear_optimization_gain':{'affine_correct_ols_initialization':1,'linear_correct':-1}}
    records=[]
    for endpoint in ['accuracy','pair_both_correct']:
        values={c:np.array([pairs[(i,c)].mean(1) if endpoint=='accuracy' else pairs[(i,c)].all(1)
                            for i in range(len(config['source_seeds']))],float) for c in {c for _,c in pairs}}
        for name,coeff in coef.items():
            v=100*sum(weight*values[c] for c,weight in coeff.items())
            records.append({'endpoint':endpoint,'contrast':name,'mean_pp':float(v.mean()),
                'source_effects_pp':v.mean(1).tolist(),
                'bootstrap_95_pp':crossed_interval(v,config['bootstrap_samples'],config['bootstrap_seed'])})
    return records


def known_check(config,root,i,z):
    teacher=np.load(root/'states'/f's{i}_fixed_supervision.npz')['teacher_logits'].astype(float)
    scale=json.loads((root/'predictor_fits'/f's{i}_linear_correct.json').read_text())['geometry_scale']
    pc=np.load(root/'dataset'/f'support{i}.npz')['pc'];rows=[]
    for split in ['train','validation']:
        h=z[split].astype(float);pred=predicted_states(config,root,i,h[:,0],z)
        if split=='validation':
            with torch.no_grad():
                teacher_split=torch.nn.functional.linear(torch.as_tensor(z['validation'][:,1],device='cuda'),
                    torch.as_tensor(z['w'],device='cuda'),torch.as_tensor(z['bias_w'],device='cuda')).cpu().numpy().astype(float)
        else:teacher_split=teacher
        for name,first in pred.items():
            obj,_=objective_and_gradient(first,h[:,1],teacher_split,z['w'].astype(float),z['bias_w'].astype(float),scale,config)
            assigned=h[pc,1] if split=='train' and ('_wrong' in name and not name.startswith('swap_')) else h[:,1]
            assigned_obj,_=objective_and_gradient(first,assigned,teacher_split,z['w'].astype(float),z['bias_w'].astype(float),scale,config)
            rows.append({'source':i,'condition':name,'split':split,'true_state_nmse':float(np.square(first-h[:,1]).mean()/scale),
                         'correct_target_objective':obj,'assigned_train_objective':assigned_obj if split=='train' else None})
    return rows


def evaluate(config,root):
    n=len(config['source_seeds'])
    records=[root/'predictor_fits'/f's{i}_{c}.json' for i in range(n) for c in config['conditions']]
    affine=[root/'affine_fits'/f's{i}_{p}_{k}.json' for i in range(n) for p in ['correct','wrong'] for k in config['linear_solvers']]
    assert len(records)==len(affine)==20 and all(p.exists() for p in records+affine)
    assert not (root/'test_opened.json').exists(), 'Do not silently repeat final test inference.'
    known=[]
    for i in range(n):known.extend(known_check(config,root,i,dict(np.load(root/'states'/f's{i}_known.npz'))))
    atomic_json(root/'known_only_results.json',{'completed_utc':now(),'records':known})
    atomic_json(root/'test_opened.json',{'opened_utc':now(),'all20_budget_and20_convex_fits_complete':True,
        'evaluation_code_sha256':sha(__file__),'fit_sha256':{str(p):sha(p) for p in records+affine}})
    data=dict(np.load(root/'dataset/test.npz'))
    features=torch.as_tensor(worlds.feature(data['x'],13),device='cuda')
    rows=[];pairs={};precision=[];stability=[];max_logit=0.;max_probability=0.;grades=[]
    for i in range(n):
        model,_=load_backbone(root,i,'cuda')
        with torch.no_grad():h=model(features).cpu().numpy().astype(float)
        np.savez_compressed(root/'states'/f's{i}_test.npz',hidden=h.astype(np.float32))
        z=dict(np.load(root/'states'/f's{i}_known.npz'))
        w,bw,b,bb=[z[k].astype(float) for k in ['w','bias_w','b','bias_b']]
        scale=json.loads((root/'predictor_fits'/f's{i}_linear_correct.json').read_text())['geometry_scale']
        prediction=predicted_states(config,root,i,h[:,0],z)
        for recipient,donor in config['hybrids']:
            hybrid=prediction[f'swap_{recipient}_null_{donor}'];orig=prediction[recipient]
            change=(hybrid-orig)@w.T
            max_logit=max(max_logit,float(np.abs(change).max()))
            def soft(v):
                v=v-v.max(1,keepdims=True);e=np.exp(v);return e/e.sum(1,keepdims=True)
            max_probability=max(max_probability,float(np.abs(soft(hybrid@w.T+bw)-soft(orig@w.T+bw)).max()))
            assert np.abs(change).max()<1e-8
            assert np.array_equal((hybrid@w.T+bw).argmax(1),(orig@w.T+bw).argmax(1))
        for name,first in prediction.items():
            sample=per_sample(first,h[:,1],b,bb,w,bw,data['labels'][:,3],scale)
            np.savez_compressed(root/'evaluations'/f's{i}_{name}.npz',first_pred=first,**sample)
            for sid,split in [(0,'iid'),(1,'collisions')]:
                use=data['split']==sid;oracle=use&sample['oracle_hit']
                denom=float(sample['oracle_center_sq'][use].mean())
                row={'source':i,'source_seed':config['source_seeds'][i],'condition':name,'split':split,
                    'accuracy':float(sample['compound_hit'][use].mean()),
                    'first_accuracy':float((sample['first_scores'].argmax(1)[use]==data['labels'][use,1]).mean()),
                    'hidden_nmse':float(sample['hidden_error'][use].mean()),'null_nmse':float(sample['null_error'][use].mean()),
                    'row_nmse':float(sample['row_error'][use].mean()),'oracle_accuracy':float(sample['oracle_hit'][use].mean()),
                    'null_margin_crossing_fraction':float(sample['null_margin_crossing'][oracle].mean()),'pair_both_correct':None,
                    'downstream_null_normalized':float(sample['downstream_null_sq'][use].mean()/denom),
                    'downstream_total_normalized':float(sample['downstream_total_sq'][use].mean()/denom)}
                if sid==1:
                    hit=np.array([sample['compound_hit'][data['pair_ids']==pid] for pid in range(128)])
                    row['pair_both_correct']=float(hit.all(1).mean());pairs[(i,name)]=hit
                rows.append(row)
        for pairing in ['correct','wrong']:
            original=prediction[f'affine_{pairing}_ols_initialization']
            repeated=prediction[f'affine_{pairing}_budget_linear_initialization']
            changes={}
            for kind in config['linear_solvers']:
                theta=np.load(root/'affine_fits'/f's{i}_{pairing}_{kind}.npz')['theta']
                x=np.column_stack([h[:,0].astype(np.float32),np.ones(len(h),np.float32)])
                first32=x@theta.astype(np.float32)
                assert first32.dtype==np.float32
                logits32=(first32@z['b']+z['bias_b'])@z['w'].T+z['bias_w']
                logits64=(prediction[f'affine_{pairing}_{kind}']@b+bb)@w.T+bw
                hit32=logits32.argmax(1)==data['labels'][:,3];hit64=logits64.argmax(1)==data['labels'][:,3]
                for sid,split in [(0,'iid'),(1,'collisions')]:
                    use=data['split']==sid
                    precision.append({'source':i,'pairing':pairing,'kind':kind,'split':split,
                        'float32_accuracy':float(hit32[use].mean()),'float64_accuracy':float(hit64[use].mean()),
                        'changed_compound_predictions':int((logits32.argmax(1)[use]!=logits64.argmax(1)[use]).sum()),
                        'first_state_max_rounding_change':float(np.abs(first32-prediction[f'affine_{pairing}_{kind}']).max())})
                changes[kind]=logits64.argmax(1)
            stability.append({'source':i,'pairing':pairing,'max_repeat_first_state_difference':float(np.abs(original-repeated).max()),
                'different_compound_predictions':int((changes[config['linear_solvers'][0]]!=changes[config['linear_solvers'][1]]).sum())})
        grade=json.loads((root/'source_preparation'/f's{i}/fits/n0_both_correct.json').read_text())['curve'][-1]
        grades.append({'source':i,'seed':config['source_seeds'][i],**grade})
    means=[]
    for c in sorted({r['condition'] for r in rows}):
        for split in ['iid','collisions']:
            select=[r for r in rows if r['condition']==c and r['split']==split]
            keys=[k for k,v in select[0].items() if k not in ['source','source_seed','condition','split'] and isinstance(v,(float,int))]
            means.append({'condition':c,'split':split,**{k:float(np.mean([r[k] for r in select])) for k in keys}})
    convergence=[json.loads(p.read_text()) for p in affine]
    atomic_json(root/'results.json',{'status':'complete','completed_utc':now(),'source_models':n,
        'same_budget_models':20,'convex_fits':20,'records':rows,'means':means,'contrasts':fixed_contrasts(pairs,config),
        'precision_checks':precision,'solver_repeat_checks':stability,'source_known_grades':grades,
        'all_source_gates_pass':all(min(g['native'])>=config['source_gate']['native'] and min(g['generators'])>=config['source_gate']['generators'] for g in grades),
        'all_convex_certificates_pass':all(r['metadata']['converged'] for r in convergence),
        'max_swap_first_logit_change':max_logit,'max_swap_probability_change':max_probability,
        'ending_rule':config['ending_rule'],'scope':config['scope']})
    atomic_json(root/'state.json',{'status':'evaluated','updated_utc':now()})
    print(json.dumps({'status':'evaluated','all_source_gates_pass':all(min(g['generators'])>=.9 for g in grades)}),flush=True)
