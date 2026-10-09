"""Frozen-protocol independent-world confirmation in previously untrained F17."""
import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy import stats
import torch
from torch.nn import functional as F

from . import algebra_relation_campaign as core
from . import algebra_relation_worlds as worlds
from .algebra_relation_feasibility import apply_encoding
from .longrun_engine import atomic_json, atomic_torch
from .operator_capacity_confirmation import fit_predictor
from .operator_capacity_verify import replay
from .permworld_combinations import sha
from .relation_error_localization import row_projection
from .reviewer_revision_diagnostics import donor_permutations
from .reviewer_revision_statistics import holm
from .two_step_relation_factorial import factorial_pairings, now

CONFIG=Path('configs/reviewer_fresh_confirmation.json')


def setup():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;apply_encoding('numeric_gelu')
    c=json.loads(CONFIG.read_text());root=Path(c['output']);root.mkdir(parents=True,exist_ok=True)
    signature={'config':c,'code_sha256':sha(__file__),'config_sha256':sha(CONFIG),
       'dependencies_sha256':{p:sha(p) for p in ['experiments/algebra_relation_campaign.py',
           'experiments/algebra_relation_worlds.py','experiments/operator_capacity_confirmation.py',
           'experiments/reviewer_revision_diagnostics.py','experiments/reviewer_revision_statistics.py']},
       'preexisting_deliveries':{p:sha(p) for p in ['results/final_mechanism_confirmation/delivery.json',
           'results/null_space_review_controls/delivery.json','results/reviewer_revision_diagnostics_v2/delivery.json']}}
    p=root/'protocol.json'
    if p.exists():assert json.loads(p.read_text())['signature']==signature
    else:
        atomic_json(p,{'registered_utc':now(),'new_F17_models_trained':False,'new_composite_outcomes_seen':False,'signature':signature})
        (root/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return c,root


def unit_config(c,index):
    return {**c,'source_seeds':[c['source_seed_base']+1009*index],
            'predictor_seed':c['source_seed_base']+1009*index+401,
            'schedule_seed':c['source_seed_base']+1009*index+501}


def prepare(c,root,index):
    local=root/f'u{index:02d}'
    for name in ['dataset','sources','models','fits','states','predictors','predictor_fits','seconds','evaluations']:
        (local/name).mkdir(parents=True,exist_ok=True)
    if (local/'data_audit.json').exists():return local
    p=c['field_prime'];seed=c['data_seed_base']+2003*index;rng=np.random.default_rng(seed)
    rows=[x for x in product(range(p),repeat=4) if (x[0]*x[3]-x[1]*x[2])%p]
    key=lambda x:worlds.key(tuple(x),'matrix',p)
    used=set();collision=worlds.choose_collision_pairs(rows,c['collision_pairs'],rng,'matrix',p,used)
    rng.shuffle(rows);iid=[]
    for x in rows:
        if key(x) not in used:
            iid.append(x);used.add(key(x))
            if len(iid)==c['iid_examples']:break
    available=defaultdict(list)
    for x in rows:
        if key(x) not in used:available[key(x)].append(x)
    keys=list(available);rng.shuffle(keys)
    nv,npilot,ns=[c[k] for k in ['validation_anchors','pilot_validation_anchors','source_anchors']]
    parts={name:worlds.encode([available[k][0] for k in selected],'matrix',p)
           for name,selected in [('validation',keys[:nv]),('pilot_validation',keys[nv:nv+npilot]),
                                ('source',keys[nv+npilot:nv+npilot+ns])]}
    known=set(keys[:nv+npilot+ns]);pool=[x for x in rows if key(x) not in used|known]
    anchors=worlds.paired_fit(pool,c['fit_anchors'],np.random.default_rng(seed+31),'matrix',p)
    support=worlds.encode(anchors,'matrix',p)
    pc,pi,eligible=factorial_pairings(np.ones(len(anchors),int),support['labels'],seed+43)
    assert eligible.all();support.update(pc=pc,pi=pi,eligible=eligible);parts['support0']=support
    test=worlds.encode(iid+collision,'matrix',p,['','a','b','ab'])
    test.update(split=np.array([0]*len(iid)+[1]*len(collision)),
                pair_ids=np.array([-1]*len(iid)+[j for j in range(c['collision_pairs']) for _ in range(2)]))
    orbit_lookup={k:j for j,k in enumerate(sorted({key(x) for x in test['x'][:,0]}))}
    test['orbit_ids']=np.array([orbit_lookup[key(x)] for x in test['x'][:,0]])
    different,same,exchange_eligible=donor_permutations(test['labels'],test['orbit_ids'],seed+61)
    test.update(different_donor=different,same_donor=same,exchange_eligible=exchange_eligible)
    parts['test']=test
    sets={name:{key(x) for x in d['x'][:,0]} for name,d in parts.items()}
    for name in sets:
        for other in sets:
            if name!=other:assert not sets[name]&sets[other]
    for name,d in parts.items():np.savez_compressed(local/'dataset'/f'{name}.npz',**d)
    atomic_json(local/'data_audit.json',{'created_utc':now(),'data_seed':seed,'source_seed':c['source_seed_base']+1009*index,
        'field_prime':p,'within_unit_whole_orbit_splits_disjoint':True,'new_mathematical_world':True,
        'exchange_eligible':int(exchange_eligible.sum()),'partition_orbits':{k:len(v) for k,v in sets.items()},
        'dataset_sha256':{str(p):sha(p) for p in (local/'dataset').glob('*.npz')}})
    return local


def backbone(c,local):
    model=core.Encoder(c['field_prime'],c['hidden_width']).to('cuda')
    ops=core.Operators(c['hidden_width']).to('cuda')
    state=torch.load(local/'models/n0_both_correct.pt',map_location='cuda',weights_only=True)
    model.load_state_dict(state['model']);ops.load_state_dict(state['operators']);model.eval();ops.eval()
    for v in list(model.parameters())+list(ops.parameters()):v.requires_grad_(False)
    return model,ops


def train_unit(c,root,index):
    local=prepare(c,root,index);uc=unit_config(c,index)
    plan=json.loads(Path('configs/algebra_relation_common_v3.json').read_text())
    plan.update(field_prime=c['field_prime'],hidden_width=c['hidden_width'],
                source_seeds=[uc['source_seeds'][0]]*3,source_epochs=c['source_epochs'],
                joint_epochs=c['joint_epochs'],batch_size=128,source_batch_size=256)
    atomic_json(root/'state.json',{'stage':'source_and_known_generator_training','unit':index,'updated_utc':now()})
    core.fit(plan,local,0,'both_correct','cuda')
    model,ops=backbone(c,local)
    cache=local/'states/s0_known.npz'
    if not cache.exists():
        with torch.no_grad():
            _,x,_=core.arrays(local/'dataset/support0.npz',c['field_prime'],'cuda')
            _,v,_=core.arrays(local/'dataset/validation.npz',c['field_prime'],'cuda')
            z={'train':model(x).cpu().numpy(),'validation':model(v).cpu().numpy(),
               'w':model.readout.weight.cpu().numpy(),'bias_w':model.readout.bias.cpu().numpy()}
            for j,name in enumerate(['a','b']):
                z[name]=ops.maps[j].weight.cpu().numpy().T;z['bias_'+name]=ops.maps[j].bias.cpu().numpy()
        np.savez_compressed(cache,**z)
    for condition in c['conditions']:
        atomic_json(root/'state.json',{'stage':'first_operator_training','unit':index,'condition':condition,'updated_utc':now()})
        fit_predictor(uc,local,0,condition,'cuda')
    z=dict(np.load(cache));tx=torch.as_tensor(z['train'],device='cuda');vx=torch.as_tensor(z['validation'],device='cuda')
    w,bw=[torch.as_tensor(z[k],device='cuda') for k in ['w','bias_w']]
    teacher=F.linear(tx[:,2],w,bw);scale=float(tx[:,2].var(0,unbiased=False).mean())
    validation_targets=F.linear(vx[:,2],w,bw);selection=[]
    for noise in c['robust_second_noise']:
        record=local/'seconds'/f'noise_{noise}.json'
        if record.exists():selection.append(json.loads(record.read_text()));continue
        torch.manual_seed(uc['predictor_seed']+911)
        second=torch.nn.Linear(c['hidden_width'],c['hidden_width']).to('cuda')
        with torch.no_grad():second.weight.copy_(torch.as_tensor(z['b'].T,device='cuda'));second.bias.copy_(torch.as_tensor(z['bias_b'],device='cuda'))
        opt=torch.optim.AdamW(second.parameters(),lr=.001,weight_decay=0.)
        rng=np.random.default_rng(uc['schedule_seed']+811)
        generator=torch.Generator(device='cuda').manual_seed(uc['predictor_seed']+1011)
        for _ in range(c['robust_second_steps']):
            ids=torch.as_tensor(rng.integers(0,len(tx),c['batch_size']),device='cuda')
            eps=torch.randn((len(ids),c['hidden_width']),generator=generator,device='cuda')*noise*np.sqrt(scale)
            pred=second(tx[ids,0]+eps)
            loss=core.kd(F.linear(pred,w,bw),teacher[ids],c['temperature'])+c['geometry_weight']*(pred-tx[ids,2]).square().mean()/scale
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        vg=torch.Generator(device='cuda').manual_seed(uc['predictor_seed']+2011);scores=[]
        with torch.no_grad():
            for sigma in c['robust_second_noise']:
                eps=torch.randn(vx[:,0].shape,generator=vg,device='cuda')*sigma*np.sqrt(scale)
                pred=second(vx[:,0]+eps)
                value=core.kd(F.linear(pred,w,bw),validation_targets,c['temperature'])+c['geometry_weight']*(pred-vx[:,2]).square().mean()/scale
                scores.append(float(value))
        cp=local/'seconds'/f'noise_{noise}.pt';atomic_torch(cp,{'state_dict':second.state_dict()})
        result={'noise':noise,'known_validation_criterion':float(np.mean(scores)),'known_noise_validation_scores':scores,
                'steps':c['robust_second_steps'],'no_composite_targets':True,'checkpoint_sha256':sha(cp)}
        atomic_json(record,result);selection.append(result)
    chosen=min(selection,key=lambda r:r['known_validation_criterion'])['noise']
    atomic_json(local/'unit_complete.json',{'completed_utc':now(),'selected_second_noise':chosen,
        'source_grade':json.loads((local/'fits/n0_both_correct.json').read_text())['curve'][-1],
        'source_initialization':uc['source_seeds'][0],'test_still_closed':True})
    print(json.dumps({'unit_complete':index,'selected_second_noise':chosen,'test_still_closed':True}),flush=True)


def evaluate(c,root):
    files=[root/f'u{i:02d}/unit_complete.json' for i in range(c['independent_units'])]
    assert all(p.exists() for p in files)
    assert not (root/'test_opened.json').exists()
    atomic_json(root/'test_opened.json',{'opened_utc':now(),'all_declared_units_complete':True,'fit_sha256':{str(p):sha(p) for p in files}})
    rows=[];exchanges=[];primary_effects=[[],[]]
    for i in range(c['independent_units']):
        local=root/f'u{i:02d}';data=dict(np.load(local/'dataset/test.npz'));model,ops=backbone(c,local)
        with torch.no_grad():h=model(torch.as_tensor(worlds.feature(data['x'],c['field_prime']),device='cuda')).cpu().numpy().astype(float)
        z=dict(np.load(local/'states/s0_known.npz'));w,bw=[z[k].astype(float) for k in ['w','bias_w']]
        complete=json.loads((local/'unit_complete.json').read_text());noise=complete['selected_second_noise']
        ss=torch.load(local/'seconds'/f'noise_{noise}.pt',map_location='cpu',weights_only=True)['state_dict']
        seconds={'original':(z['b'].astype(float),z['bias_b'].astype(float)),
                 'known_selected':(ss['weight'].numpy().astype(float).T,ss['bias'].numpy().astype(float))}
        hits={};collision=data['split']==1
        for second_name,(b,bb) in seconds.items():
            oracle=((h[:,1]@b+bb)@w.T+bw).argmax(1)==data['labels'][:,3]
            for condition in c['conditions']:
                state=torch.load(local/'predictors'/f's0_{condition}.pt',map_location='cpu',weights_only=True)['state_dict']
                first=replay(h[:,0],state,condition.startswith('nonlinear'))
                hit=((first@b+bb)@w.T+bw).argmax(1)==data['labels'][:,3];hits[(second_name,condition)]=hit
                np.savez_compressed(local/'evaluations'/f'{second_name}_{condition}.npz',hits=hit)
                for sid,split in [(0,'iid'),(1,'collisions')]:
                    use=data['split']==sid;both=None
                    if sid:both=float(np.array([hit[data['pair_ids']==pid] for pid in range(c['collision_pairs'])]).all(1).mean())
                    rows.append({'unit':i,'second':second_name,'condition':condition,'split':split,
                        'accuracy':float(hit[use].mean()),'pair_both_correct':both,'oracle_accuracy':float(oracle[use].mean()),
                        'first_state_mse':float(np.square(first[use]-h[use,1]).mean()),
                        'first_accuracy':float(((first@w.T+bw).argmax(1)[use]==data['labels'][use,1]).mean())})
            if second_name=='known_selected':
                primary_effects[0].append(100*(hits[(second_name,'nonlinear_correct')][collision].mean()-hits[(second_name,'nonlinear_wrong')][collision].mean()))
                p=row_projection(w);q=np.eye(len(p))-p;use=data['exchange_eligible'];natural=h[use,1]
                values={}
                for name,ids,part in [('null_different',data['different_donor'],'null'),
                                     ('row_different',data['different_donor'],'row'),('null_same',data['same_donor'],'null')]:
                    ids=ids[use];donor=h[ids,1]
                    mixed=natural@p+donor@q if part=='null' else donor@p+natural@q
                    delta=np.abs((mixed-natural)@w.T).max()
                    if part=='null':assert delta<1e-8
                    cls=((mixed@b+bb)@w.T+bw).argmax(1);target=cls==data['labels'][ids,3]
                    base=((natural@b+bb)@w.T+bw).argmax(1)
                    exchanges.append({'unit':i,'condition':name,'donor_target_accuracy':float(target.mean()),
                        'prediction_changed_fraction':float((cls!=base).mean()),'maximum_first_logit_change':float(delta)})
                    values[name]=float(target.mean())
                primary_effects[1].append(100*(values['null_different']-values['row_different']))
        print(json.dumps({'unit_evaluated':i}),flush=True)
    contrasts=[];pvalues=[]
    for name,values in zip(c['primary_contrasts'],primary_effects):
        v=np.asarray(values);se=stats.sem(v);t=stats.ttest_1samp(v,0)
        pvalues.append(float(t.pvalue))
        contrasts.append({'contrast':name,'mean_pp':float(v.mean()),'source_world_effects_pp':v.tolist(),
            'independent_unit_count':len(v),'two_sided_p':float(t.pvalue),
            'ci95_pp':(v.mean()+np.array([-1,1])*stats.t.ppf(.975,len(v)-1)*se).tolist()})
    for r,pvalue in zip(contrasts,holm(pvalues)):r['holm_adjusted_p']=pvalue
    score=np.mean([r['accuracy'] for r in rows if (r['second'],r['condition'],r['split'])==('known_selected','nonlinear_correct','collisions')])
    same_change=np.mean([r['prediction_changed_fraction'] for r in exchanges if r['condition']=='null_same'])
    atomic_json(root/'results.json',{'status':'complete','completed_utc':now(),'scope':c['scope'],'records':rows,
        'interchanges':exchanges,'primary_contrasts':contrasts,'equivalence_primary':False,
        'floor_check':{'accuracy':float(score),'threshold':.4,'passed':bool(score>=.4)},
        'same_answer_stability':{'changed_fraction':float(same_change),'threshold':.05,'passed':bool(same_change<=.05)},
        'all_failed_gates_retained':True})
    atomic_json(root/'delivery.json',{'status':'complete','completed_utc':now(),
        'artifact_sha256':{str(p):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='delivery.json'}})


def run(stage):
    c,root=setup()
    if stage=='prepare':
        for i in range(c['independent_units']):prepare(c,root,i)
    if stage in ['train','run']:
        for i in range(c['independent_units']):train_unit(c,root,i)
    if stage in ['evaluate','run']:evaluate(c,root)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','train','evaluate','run'],default='prepare')
    run(parser.parse_args().stage)
