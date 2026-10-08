"""Deadline-aware competent-target inverse-relation functional study.

Registration precedes generation and any new outcomes. Earlier studies are read-only.
"""
import argparse
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from .inverse_functional_alignment import configure, extract, forward, geometry_loss, matched_derangement, now, output_loss, validation
from .longrun_attention import accelerate
from .longrun_engine import atomic_json, atomic_torch
from .permworld_combinations import sha
from .permutation_audit import TRANSFORMS, transform
from .specialist_cka_controls import api, input_key, old_inputs, original_input_key

CONFIG = Path('configs/overnight_inverse_functional.json')
STOP_REQUESTED = False


def initialize():
    plan = json.loads(CONFIG.read_text()); root = Path(plan['output']); old = Path(plan['previous_study'])
    for name in ['', 'confirmation', 'dataset', 'teacher', 'training', 'checkpoints', 'evaluations']:
        (root/name).mkdir(parents=True, exist_ok=True)
    dest = root/'protocol.json'
    if dest.exists():
        sig = json.loads(dest.read_text())['signature']
        assert sig['plan'] == plan and sig['code_sha256'] == sha(__file__) and sig['config_sha256'] == sha(CONFIG)
        for p,digest in sig['dependencies_sha256'].items(): assert sha(p) == digest, p
        for source in sig['sources'] + sig['targets']: assert sha(source['checkpoint']) == source['checkpoint_sha256']
        return plan, root, sig
    previous = json.loads((old/'protocol.json').read_text())['signature']
    specialists = json.loads(Path('results/specialist_cka_controls/protocol.json').read_text())['signature']['sources']
    targets = [s for s in specialists if s['task'] == plan['target_task']]
    names = {'data.npz', 'probe_dataset.npz', 'source_data.npz', 'training_orbit_audit.npz', 'dataset.npz'}
    archives = []; unique = set()
    for file in sorted(Path('results').rglob('*.npz')):
        if file.name not in names or root in file.parents: continue
        digest = sha(file)
        if digest not in unique: archives.append({'path': str(file), 'sha256': digest}); unique.add(digest)
    dependencies = ['experiments/overnight_inverse_diagnostic.py', 'experiments/inverse_functional_alignment.py',
                    'experiments/longrun_attention.py', 'experiments/permworld_combinations.py',
                    'experiments/permutation_audit.py', 'experiments/specialist_cka_controls.py']
    sig = {'plan': plan, 'sources': previous['sources'], 'targets': targets, 'excluded_archives': archives,
           'code_sha256': sha(__file__), 'config_sha256': sha(CONFIG),
           'prior_completion_sha256': sha(old/'completion.json'),
           'prior_artifacts_sha256': json.loads((old/'completion.json').read_text())['artifact_sha256'],
           'frozen_modes_sha256': sha(old/'length_priors/summary.json'),
           'dependencies_sha256': {p:sha(p) for p in dependencies},
           'upstream_sha256': {p:sha(Path(plan['upstream_python'])/'neurips_permutations'/p) for p in ['models.py', 'training.py', 'math_ops.py', 'passage.py']}}
    # Baseline code/config and all previous artifacts are protected by this boundary.
    atomic_json(dest, {'registered_utc': now(), 'new_confirmation_results_observed': False,
                      'new_training_started': False, 'signature': sig})
    return plan, root, sig


def cohorts(plan):
    result = [{'id':'confirmation/test', 'seed':plan['confirmation_seed'], 'counts':[plan['test_per_length']]*5, 'labeled':True},
              {'id':'test', 'seed':plan['final_test_seed'], 'counts':[plan['test_per_length']]*5, 'labeled':True}]
    counts = [plan['unlabeled_per_repeat']//5 + (j<plan['unlabeled_per_repeat']%5) for j in range(5)]
    for rep in plan['replicates']:
        result.append({'id':rep['id']+'/support', 'seed':rep['support_seed'],
                       'counts':[a+b for a,b in zip(plan['train_counts'],plan['validation_counts'])], 'labeled':True, 'support':True})
        result.append({'id':rep['id']+'/unlabeled', 'seed':rep['unlabeled_seed'], 'counts':counts, 'labeled':False})
    return result


def create_data(plan, root, sig):
    if (root/'dataset/audit.json').exists(): return
    functions,tokens,one_line,_,_ = api(plan); seen = old_inputs(sig['excluded_archives'])
    used = set(); lookup = {}; proposals = {}; specs = cohorts(plan)
    for world,spec in enumerate(specs):
        rng = np.random.default_rng(spec['seed'])
        for n,need in zip(plan['lengths'],spec['counts']):
            count = max(1024,need*(8 if n==10 else 2)); rows=[]
            while len(rows)<count:
                p=tuple(map(int,rng.permutation(n)+1)); keys=[input_key(transform(p,t)) for t in TRANSFORMS]
                if len(set(keys))!=8 or any(k in seen or k in used for k in keys): continue
                anchor=(world,n,len(rows)); rows.append(p)
                for key in keys: used.add(key); lookup[key]=anchor
            proposals[world,n]=rows
        print({'proposed_cohort':spec['id'],'candidate_orbit_states':len(used)},flush=True)
    manifest=Path(plan['repository'])/'data/permutation-properties-16m-v1/manifest.json';parent=json.loads(manifest.read_text());rejected=set();scanned=0
    for j,shard in enumerate(parent['shards']):
        file=manifest.parent/shard['filename'];assert sha(file)==shard['sha256']
        with gzip.open(file,'rb') as handle:
            for line in handle:
                anchor=lookup.get(original_input_key(line))
                if anchor is not None:rejected.add(anchor)
                scanned+=1
        if j%25==0:print({'original_inputs_scanned':scanned,'rejected_orbits':len(rejected)},flush=True)
    records=[]
    for world,spec in enumerate(specs):
        inputs=[];perms=[];lengths=[];labels=[];splits=[];available_counts={}
        for j,(n,need) in enumerate(zip(plan['lengths'],spec['counts'])):
            available=[p for i,p in enumerate(proposals[world,n]) if (world,n,i) not in rejected];assert len(available)>=need,(spec['id'],n,len(available),need)
            available_counts[str(n)]=len(available)
            for k,p in enumerate(available[:need]):
                pair=[]
                for q in [p,transform(p,'inverse')]:
                    prefix=[tokens['<BOS>'],tokens['<SIZE>'],tokens[f'{n:02d}']]+[tokens[v] for v in one_line(q)]
                    pair.append(prefix+[tokens['<PAD>']]*(64-len(prefix)))
                inputs.append(pair);perms.append([list(transform(p,t))+[0]*(30-n) for t in TRANSFORMS]);lengths.append(n)
                if spec['labeled']:labels.append(functions[plan['target_task']](p))
                if spec.get('support'):splits.append(0 if k<plan['train_counts'][j] else 1)
        arrays={'input':np.asarray(inputs),'permutations':np.asarray(perms),'lengths':np.asarray(lengths)}
        if spec['labeled']:arrays['labels']=np.asarray(labels)
        if spec.get('support'):arrays['split']=np.asarray(splits)
        base=root/'confirmation/dataset/test' if world==0 else root/'dataset'/spec['id'];base.mkdir(parents=True,exist_ok=True)
        file=base/'dataset.npz';np.savez_compressed(file,**arrays)
        records.append({'cohort':spec['id'],'path':str(file),'sha256':sha(file),'examples':len(lengths),
                        'oracle_labels_present':spec['labeled'],'accepted_candidates':available_counts})
    atomic_json(root/'dataset/audit.json',{'created_utc':now(),'original_manifest_sha256':sha(manifest),
        'original_inputs_scanned':scanned,'prior_local_inputs_excluded':len(seen),'rejected_orbits':len(rejected),'cohorts':records,
        'all_eight_orbit_states_excluded':True,'new_supports':True,'unlabeled_oracle_answers_computed':False})


def load_model(source, plan, device):
    _,_,_,TrainConfig,factory=api(plan);cp=torch.load(source['checkpoint'],map_location='cpu',weights_only=True)
    model=accelerate(factory(TrainConfig.from_value(cp['config'])));model.load_state_dict(cp['model']);del cp
    return model.to(device)


def confirm_previous(plan, root, sig):
    from .overnight_inverse_diagnostic import analyze
    folder=root/'confirmation';old=Path(plan['previous_study']);data=dict(np.load(folder/'dataset/test/dataset.npz'))
    for name in ['teacher','evaluations']:(folder/name).mkdir(exist_ok=True)
    registration=folder/'protocol.json'
    if not registration.exists():atomic_json(registration,{'registered_utc':json.loads((root/'protocol.json').read_text())['registered_utc'],
        'protocol_sha256':sha(root/'protocol.json'),'frozen_modes_sha256':sig['frozen_modes_sha256'],
        'all_original_conditions':True,'endpoint':'fixed1200','new_outcomes_inspected':False})
    device=configure();_,tokens,_,_,_=api(plan);previous=json.loads((old/'protocol.json').read_text())['signature']
    for source in sig['sources']:
        file=folder/'teacher'/f"test_s{source['seed']}.npz"
        if not file.exists():
            model=load_model(source,plan,device);np.savez_compressed(file,**extract(model,data,plan['source_task'],tokens,1));del model
        for rep in previous['replicates']:
            if rep['source_seed']!=source['seed']:continue
            for condition in previous['plan']['conditions']:
                name=rep['id']+'_'+condition;file=folder/'evaluations'/f'{name}.npz'
                if file.exists():continue
                model=load_model(source,plan,device);cp=torch.load(old/'checkpoints'/f'{name}_u1200.pt',map_location='cpu',weights_only=True);model.load_state_dict(cp['model']);del cp
                out=extract(model,data,plan['target_task'],tokens,0);np.savez_compressed(file,final_logits=out['logits'],final_hidden=out['hidden']);del model
        print({'confirmed_source_group':source['seed']},flush=True)
    analyze(folder/'dataset/test/dataset.npz',folder,folder/'diagnostic',True)
    atomic_json(folder/'state.json',{'status':'complete','completed_utc':now(),'models_refitted':0,'frozen_target_models':36,'new_test_examples':len(data['labels'])})


def prepare_teachers(plan, root, sig):
    device=configure();_,tokens,_,_,_=api(plan)
    for source in sig['sources']:
        model=load_model(source,plan,device);model.eval()
        for p in model.parameters():p.requires_grad_(False)
        for rep in plan['replicates']:
            if rep['source_seed']!=source['seed']:continue
            for part in ['support','unlabeled']:
                data=dict(np.load(root/'dataset'/rep['id']/part/'dataset.npz'))
                if part=='unlabeled':assert 'labels'not in data
                file=root/'teacher'/f"{rep['id']}_{part}.npz"
                if not file.exists():
                    out=extract(model,data,plan['source_task'],tokens,1);out['predicted_answer']=out['logits'].argmax(-1);np.savez_compressed(file,**out)
                teacher=dict(np.load(file));mask=data['split']==0 if part=='support' else np.ones(len(data['lengths']),dtype=bool)
                match_answer=data['labels'] if part=='support' else teacher['predicted_answer']
                partner,eligible=matched_derangement(data['lengths'],match_answer,mask,rep['training_seed']+99001)
                paired=root/'teacher'/f"{rep['id']}_{part}_mismatch.npz"
                if not paired.exists():np.savez_compressed(paired,partner=partner,eligible=eligible)
            print({'prepared_teacher_repeat':rep['id']},flush=True)
        del model
    atomic_json(root/'teacher/state.json',{'status':'ready','completed_utc':now(),'unlabeled_oracle_answers_read':False,'final_test_consulted':False})


def schedules(plan,rep,support,pool):
    rng=np.random.default_rng(rep['training_seed']+81001);ur=np.random.default_rng(rep['unlabeled_seed']+81001);ls=[];us=[]
    for _ in range(plan['updates']):
        n=int(rng.choice(plan['lengths']));a=np.flatnonzero((support['split']==0)&(support['lengths']==n));b=np.flatnonzero(pool['lengths']==n)
        ls.append(rng.choice(a,plan['labeled_batch'],replace=False));us.append(ur.choice(b,plan['unlabeled_batch'],replace=False))
    return np.asarray(ls),np.asarray(us)


def losses(logits,h,labels,correct,wrong,teacher,eligible,condition,plan):
    ce=F.cross_entropy(logits[:plan['labeled_batch']].float(),labels)
    right_loss=geometry_loss(h[eligible],correct[eligible]);wrong_loss=geometry_loss(h[eligible],wrong[eligible]);kd=output_loss(logits,teacher,plan['distillation_temperature'])
    loss=ce+plan['geometry_weight']*(float(condition in ['correct_geometry','distillation_geometry'])*right_loss+float(condition in ['matched_mismatch','distillation_mismatch'])*wrong_loss)
    loss=loss+plan['distillation_weight']*float(condition.startswith('distillation'))*kd
    return loss,{'ce':ce,'correct_geometry_loss':right_loss,'mismatch_geometry_loss':wrong_loss,'kd':kd}


def request_stop(signum,frame):
    global STOP_REQUESTED
    STOP_REQUESTED=True


def train(plan,root,sig):
    assert json.loads((root/'teacher/state.json').read_text())['status']=='ready'
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    stop_at=datetime.fromisoformat(plan['training_stop_utc']).timestamp();device=configure();_,tokens,_,_,_=api(plan)
    for rep in plan['replicates']:
        target=next(s for s in sig['targets'] if s['seed']==rep['target_pretrain_seed'])
        support=dict(np.load(root/'dataset'/rep['id']/'support/dataset.npz'));pool=dict(np.load(root/'dataset'/rep['id']/'unlabeled/dataset.npz'));assert 'labels'not in pool
        teacher=dict(np.load(root/'teacher'/f"{rep['id']}_support.npz"));unteacher=dict(np.load(root/'teacher'/f"{rep['id']}_unlabeled.npz"))
        match=dict(np.load(root/'teacher'/f"{rep['id']}_support_mismatch.npz"));unmatch=dict(np.load(root/'teacher'/f"{rep['id']}_unlabeled_mismatch.npz"))
        schedule=root/'training'/f"{rep['id']}_schedule.npz"
        if not schedule.exists():ls,us=schedules(plan,rep,support,pool);np.savez_compressed(schedule,labeled=ls,unlabeled=us)
        sch=dict(np.load(schedule));ls=sch['labeled'];us=sch['unlabeled']
        for condition in plan['conditions']:
            name=rep['id']+'_'+condition;file=root/'training'/f'{name}.json';latest=root/'checkpoints'/f'{name}_latest.pt'
            if file.exists() and json.loads(file.read_text())['status']=='complete':continue
            if time.time()>=stop_at or STOP_REQUESTED:return
            model=load_model(target,plan,device);optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
            arrays={'x':support['input'][:,0],'n':support['lengths'],'y':support['labels'],'ux':pool['input'][:,0],'un':pool['lengths'],
                    'h':teacher['hidden'],'logits':teacher['logits'],'uh':unteacher['hidden'],'ulogits':unteacher['logits'],
                    'partner':match['partner'],'upartner':unmatch['partner'],'eligible':match['eligible'],'ueligible':unmatch['eligible']}
            a={k:torch.tensor(v,device=device) for k,v in arrays.items()};torch.manual_seed(rep['training_seed'])
            if device=='cuda':torch.cuda.manual_seed_all(rep['training_seed'])
            start=0;curve=[];elapsed=0.
            if latest.exists():
                cp=torch.load(latest,map_location='cpu',weights_only=True);model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);start=cp['step'];curve=cp['curve'];elapsed=cp['elapsed_seconds'];torch.set_rng_state(cp['torch_rng'])
                if device=='cuda':torch.cuda.set_rng_state_all(cp['cuda_rng'])
                del cp
            begun=time.monotonic();parts={};step=start
            if not curve:curve=[{'step':0,**validation(model,support,plan['target_task'],tokens,device)}]
            for step in range(start+1,plan['updates']+1):
                if step%100==1 and (time.time()>=stop_at or STOP_REQUESTED):step-=1;break
                if step<=plan['warmup_updates']:ratio=step/plan['warmup_updates']
                else:ratio=plan['minimum_lr_ratio']+(1-plan['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*(step-plan['warmup_updates'])/(plan['updates']-plan['warmup_updates'])))
                for group in optimizer.param_groups:group['lr']=plan['learning_rate']*ratio
                l=torch.tensor(ls[step-1],device=device);u=torch.tensor(us[step-1],device=device)
                x=torch.cat([a['x'][l],a['ux'][u]]);n=torch.cat([a['n'][l],a['un'][u]])
                right=torch.cat([a['h'][l],a['uh'][u]]);wrong=torch.cat([a['h'][a['partner'][l]],a['uh'][a['upartner'][u]]]);tlogits=torch.cat([a['logits'][l],a['ulogits'][u]]);eligible=torch.cat([a['eligible'][l],a['ueligible'][u]])
                model.train();optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):logits,h=forward(model,x,n,plan['target_task'],tokens)
                loss,parts=losses(logits,h,a['y'][l],right,wrong,tlogits,eligible,condition,plan)
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite {name} {step}')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),plan['gradient_clip']);optimizer.step()
                if step in plan['checkpoint_updates']:
                    grade={'step':step,**validation(model,support,plan['target_task'],tokens,device)};curve.append(grade)
                    atomic_torch(root/'checkpoints'/f'{name}_u{step}.pt',{'model':model.state_dict(),'step':step,'grade':grade})
                if step%plan['save_every']==0 or step in plan['checkpoint_updates']:
                    runtime=elapsed+time.monotonic()-begun
                    atomic_torch(latest,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'curve':curve,'elapsed_seconds':runtime,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else []})
                    record={'status':'complete' if step==plan['updates'] else 'training','replicate':rep,'condition':condition,'step':step,'curve':curve,
                            'elapsed_seconds':runtime,'updated_utc':now(),'initialization_sha256':target['checkpoint_sha256'],'schedule_sha256':sha(schedule),
                            'training_labels':192,'validation_labels':64,'unlabeled_inputs':len(pool['lengths']),'distinct_unlabeled_exposed':len(np.unique(us[:step])),
                            'forward_rows':step*(plan['labeled_batch']+plan['unlabeled_batch']),'latest_sha256':sha(latest),
                            'checkpoint_sha256':{str(g['step']):sha(root/'checkpoints'/f"{name}_u{g['step']}.pt") for g in curve if g['step']>0}}
                    atomic_json(file,record);atomic_json(root/'current_job.json',record)
                    print(json.dumps({'job':name,'step':step,'elapsed_seconds':runtime,'validation':curve[-1],**{k:float(v.detach().cpu()) for k,v in parts.items()}}),flush=True)
            if step<plan['updates']:
                runtime=elapsed+time.monotonic()-begun
                atomic_torch(latest,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'curve':curve,'elapsed_seconds':runtime,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else []})
                record={'status':'partial_deadline','step':step,'condition':condition,'replicate':rep,'curve':curve,'elapsed_seconds':runtime,'updated_utc':now(),'latest_sha256':sha(latest)}
                atomic_json(file,record);atomic_json(root/'current_job.json',record);return
            del model,optimizer,a
            if device=='cuda':torch.cuda.empty_cache()
    atomic_json(root/'training/state.json',{'status':'all_targets_complete','target_models':len(plan['conditions'])*len(plan['replicates']),'completed_utc':now()})


def evaluate(plan,root,sig):
    assert json.loads((root/'training/state.json').read_text())['status']=='all_targets_complete'
    marker=root/'test_opened.json'
    if not marker.exists():atomic_json(marker,{'opened_utc':now(),'all36_fits_complete':True,
        'checkpoint_sha256':{p.stem:sha(p) for p in (root/'checkpoints').glob('*_u*.pt')}})
    device=configure();_,tokens,_,_,_=api(plan);data=dict(np.load(root/'dataset/test/dataset.npz'))
    for source in sig['sources']:
        file=root/'teacher'/f"test_s{source['seed']}.npz"
        if not file.exists():
            model=load_model(source,plan,device);np.savez_compressed(file,**extract(model,data,plan['source_task'],tokens,1));del model
    for rep in plan['replicates']:
        target=next(s for s in sig['targets'] if s['seed']==rep['target_pretrain_seed'])
        initial=root/'evaluations'/f"{rep['id']}_initial.npz"
        if not initial.exists():
            model=load_model(target,plan,device);np.savez_compressed(initial,**extract(model,data,plan['target_task'],tokens,0));del model
        for condition in plan['conditions']:
            name=rep['id']+'_'+condition
            for step in plan['checkpoint_updates']:
                file=root/'evaluations'/f'{name}_u{step}.npz'
                if file.exists():continue
                model=load_model(target,plan,device);cp=torch.load(root/'checkpoints'/f'{name}_u{step}.pt',map_location='cpu',weights_only=True);model.load_state_dict(cp['model']);del cp
                np.savez_compressed(file,**extract(model,data,plan['target_task'],tokens,0));del model
            print({'evaluated':name,'fixed_endpoints':plan['checkpoint_updates']},flush=True)
    atomic_json(root/'evaluation_state.json',{'status':'complete','completed_utc':now(),'model_endpoints':180,'test_examples':len(data['labels'])})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['register','data','confirm','teachers','train','evaluate']);args=parser.parse_args()
    plan,root,sig=initialize()
    if args.phase=='data':create_data(plan,root,sig)
    elif args.phase=='confirm':confirm_previous(plan,root,sig)
    elif args.phase=='teachers':prepare_teachers(plan,root,sig)
    elif args.phase=='train':train(plan,root,sig)
    elif args.phase=='evaluate':evaluate(plan,root,sig)
    else:print({'registered_utc':json.loads((root/'protocol.json').read_text())['registered_utc'],'deadline_utc':plan['deadline_utc']})
