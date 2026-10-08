"""A prospectively fixed within-answer alignment intervention.

All arms share correct output distillation. Geometry differs only in the
specified loss and the paired teacher features. Full matched doses are kept.
"""
import argparse
from datetime import datetime
import gzip
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from .inverse_functional_alignment import configure, extract, forward, geometry_loss, now, output_loss, validation
from .longrun_engine import atomic_json, atomic_torch
from .overnight_inverse_functional import initialize as base_initialize, load_model
from .permworld_combinations import sha
from .permutation_audit import TRANSFORMS, transform
from .specialist_cka_controls import api, input_key, old_inputs, original_input_key

CONFIG=Path('configs/overnight_inverse_residual.json');STOP=False


def initialize():
    plan=json.loads(CONFIG.read_text());root=Path(plan['output']);base_plan,base,base_sig=base_initialize()
    for name in ['','dataset','training','checkpoints','evaluations','teacher']:(root/name).mkdir(parents=True,exist_ok=True)
    file=root/'protocol.json'
    if file.exists():
        sig=json.loads(file.read_text())['signature'];assert sig['plan']==plan and sig['code_sha256']==sha(__file__)and sig['config_sha256']==sha(CONFIG)
        for path,digest in sig['references_sha256'].items():assert sha(path)==digest,path
        return plan,root,base_plan,base,base_sig,sig
    entries=list(base_sig['excluded_archives']);audit=json.loads((base/'dataset/audit.json').read_text())
    entries += [{'path':r['path'],'sha256':r['sha256']}for r in audit['cohorts']]
    references={str(base/'protocol.json'):sha(base/'protocol.json'),str(base/'dataset/audit.json'):sha(base/'dataset/audit.json')}
    for rep in base_plan['replicates']:
        for part in ['support','unlabeled']:
            for path in [base/'dataset'/rep['id']/part/'dataset.npz',base/'teacher'/f"{rep['id']}_{part}.npz",base/'teacher'/f"{rep['id']}_{part}_mismatch.npz"]:references[str(path)]=sha(path)
    sig={'plan':plan,'code_sha256':sha(__file__),'config_sha256':sha(CONFIG),'base_protocol_sha256':sha(base/'protocol.json'),
         'excluded_archives':entries,'references_sha256':references,'sources':base_sig['sources'],'targets':base_sig['targets']}
    atomic_json(file,{'registered_utc':now(),'main_test_opened':(base/'test_opened.json').exists(),'supplementary_test_results_observed':False,'signature':sig})
    return plan,root,base_plan,base,base_sig,sig


def create_data(plan,root,bp,base,bs,sig):
    if (root/'dataset/audit.json').exists():return
    functions,tokens,one_line,_,_=api(bp);seen=old_inputs(sig['excluded_archives']);used=set();lookup={};proposals={};rng=np.random.default_rng(plan['test_seed'])
    for n in bp['lengths']:
        rows=[];count=plan['test_per_length']*(8 if n==10 else 2)
        while len(rows)<count:
            p=tuple(map(int,rng.permutation(n)+1));keys=[input_key(transform(p,t))for t in TRANSFORMS]
            if len(set(keys))!=8 or any(k in seen or k in used for k in keys):continue
            anchor=(n,len(rows));rows.append(p)
            for k in keys:used.add(k);lookup[k]=anchor
        proposals[n]=rows
    manifest=Path(bp['repository'])/'data/permutation-properties-16m-v1/manifest.json';parent=json.loads(manifest.read_text());rejected=set();scanned=0
    for shard in parent['shards']:
        file=manifest.parent/shard['filename'];assert sha(file)==shard['sha256']
        with gzip.open(file,'rb')as handle:
            for line in handle:
                anchor=lookup.get(original_input_key(line))
                if anchor is not None:rejected.add(anchor)
                scanned+=1
    inputs=[];perms=[];lengths=[];labels=[]
    for n in bp['lengths']:
        rows=[p for i,p in enumerate(proposals[n])if(n,i)not in rejected];assert len(rows)>=plan['test_per_length']
        for p in rows[:plan['test_per_length']]:
            pair=[]
            for q in [p,transform(p,'inverse')]:
                prefix=[tokens['<BOS>'],tokens['<SIZE>'],tokens[f'{n:02d}']]+[tokens[v]for v in one_line(q)];pair.append(prefix+[tokens['<PAD>']]*(64-len(prefix)))
            inputs.append(pair);perms.append([list(transform(p,t))+[0]*(30-n)for t in TRANSFORMS]);lengths.append(n);labels.append(functions[bp['target_task']](p))
    folder=root/'dataset/test';folder.mkdir(exist_ok=True);file=folder/'dataset.npz';np.savez_compressed(file,input=inputs,permutations=perms,lengths=lengths,labels=labels)
    atomic_json(root/'dataset/audit.json',{'created_utc':now(),'original_manifest_sha256':sha(manifest),'original_inputs_scanned':scanned,'prior_local_inputs_excluded':len(seen),
        'all_eight_orbit_states_excluded':True,'cohorts':[{'cohort':'test','path':str(file),'sha256':sha(file),'examples':len(lengths),'oracle_labels_present':True}]})


def answer_residual(h,answers,minimum=3):
    # Full-vocabulary teacher argmax IDs are in0..187. Dense grouping avoids
    # a device synchronization for every answer and keeps deterministic sums.
    membership=F.one_hot(answers,num_classes=188).float();counts=membership.sum(0)
    means=(membership.T@h.float())/counts.clamp_min(1)[:,None]
    eligible=counts[answers]>=minimum;centered=h.float()-membership@means
    return torch.where(eligible[:,None],centered,torch.zeros_like(centered)),eligible


def loss_terms(logits,h,labels,right,wrong,teacher,answers,labeled_count,condition,plan):
    ce=F.cross_entropy(logits[:labeled_count].float(),labels);kd=output_loss(logits,teacher,plan['distillation_temperature'])
    u=h[labeled_count:];r=right[labeled_count:];w=wrong[labeled_count:]
    ur,eligible=answer_residual(u,answers);rr,mask=answer_residual(r,answers);wr,mask2=answer_residual(w,answers)
    assert torch.equal(eligible,mask)and torch.equal(mask,mask2)
    raw=geometry_loss(u[eligible],r[eligible]);correct=geometry_loss(ur[eligible],rr[eligible]);mismatch=geometry_loss(ur[eligible],wr[eligible])
    loss=ce+kd+plan['geometry_weight']*(float(condition=='raw_correct')*raw+float(condition=='residual_correct')*correct+float(condition=='residual_mismatch')*mismatch)
    return loss,{'ce':ce,'kd':kd,'raw_geometry_loss':raw,'residual_correct_loss':correct,'residual_mismatch_loss':mismatch,'geometry_rows':int(eligible.sum())}


def schedule(plan,bp,rep,support,pool):
    rng=np.random.default_rng(rep['training_seed']+91001);ur=np.random.default_rng(rep['unlabeled_seed']+91001);ls=[];us=[]
    for _ in range(plan['updates']):
        n=int(rng.choice(bp['lengths']));a=np.flatnonzero((support['split']==0)&(support['lengths']==n));b=np.flatnonzero(pool['lengths']==n)
        ls.append(rng.choice(a,plan['labeled_batch'],replace=False));us.append(ur.choice(b,plan['unlabeled_batch'],replace=False))
    return np.asarray(ls),np.asarray(us)


def stop_signal(signum,frame):
    global STOP
    STOP=True


def train(plan,root,bp,base,bs,sig):
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal);stop_at=datetime.fromisoformat(plan['training_stop_utc']).timestamp();device=configure();_,tokens,_,_,_=api(bp)
    for endpoint in plan['round_endpoints']:
        for rep in bp['replicates']:
            target=next(s for s in bs['targets']if s['seed']==rep['target_pretrain_seed']);support=dict(np.load(base/'dataset'/rep['id']/'support/dataset.npz'));pool=dict(np.load(base/'dataset'/rep['id']/'unlabeled/dataset.npz'));assert 'labels'not in pool
            t=dict(np.load(base/'teacher'/f"{rep['id']}_support.npz"));ut=dict(np.load(base/'teacher'/f"{rep['id']}_unlabeled.npz"));pair=np.load(base/'teacher'/f"{rep['id']}_unlabeled_mismatch.npz")
            file=root/'training'/f"{rep['id']}_schedule.npz"
            if not file.exists():ls,us=schedule(plan,bp,rep,support,pool);np.savez_compressed(file,labeled=ls,unlabeled=us)
            sch=np.load(file);ls=sch['labeled'];us=sch['unlabeled']
            for condition in plan['conditions']:
                name=rep['id']+'_'+condition;record_path=root/'training'/f'{name}.json';latest=root/'checkpoints'/f'{name}_latest.pt'
                if record_path.exists()and json.loads(record_path.read_text())['step']>=endpoint:continue
                if time.time()>=stop_at or STOP:return finalize_training_state(plan,root,bp)
                model=load_model(target,bp,device);optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
                arrays={'x':support['input'][:,0],'n':support['lengths'],'y':support['labels'],'ux':pool['input'][:,0],'un':pool['lengths'],
                    'h':t['hidden'],'logits':t['logits'],'uh':ut['hidden'],'ulogits':ut['logits'],'upartner':pair['partner'],'answers':ut['predicted_answer']}
                a={k:torch.tensor(v,device=device)for k,v in arrays.items()};torch.manual_seed(rep['training_seed']+11001)
                if device=='cuda':torch.cuda.manual_seed_all(rep['training_seed']+11001)
                start=0;curve=[];elapsed=0.
                if latest.exists():
                    cp=torch.load(latest,map_location='cpu',weights_only=True);model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);start=cp['step'];curve=cp['curve'];elapsed=cp['elapsed_seconds'];torch.set_rng_state(cp['torch_rng'])
                    if device=='cuda':torch.cuda.set_rng_state_all(cp['cuda_rng'])
                    del cp
                begun=time.monotonic();step=start;parts={}
                if not curve:curve=[{'step':0,**validation(model,support,bp['target_task'],tokens,device)}]
                for step in range(start+1,endpoint+1):
                    if step%100==1 and (time.time()>=stop_at or STOP):step-=1;break
                    ratio=step/plan['warmup_updates']if step<=plan['warmup_updates']else plan['minimum_lr_ratio']+(1-plan['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*(step-plan['warmup_updates'])/(plan['updates']-plan['warmup_updates'])))
                    for group in optimizer.param_groups:group['lr']=plan['learning_rate']*ratio
                    l=torch.tensor(ls[step-1],device=device);u=torch.tensor(us[step-1],device=device);x=torch.cat([a['x'][l],a['ux'][u]]);n=torch.cat([a['n'][l],a['un'][u]])
                    right=torch.cat([a['h'][l],a['uh'][u]]);wrong=torch.cat([a['h'][l],a['uh'][a['upartner'][u]]]);teacher=torch.cat([a['logits'][l],a['ulogits'][u]])
                    model.train();optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):logits,h=forward(model,x,n,bp['target_task'],tokens)
                    loss,parts=loss_terms(logits,h,a['y'][l],right,wrong,teacher,a['answers'][u],plan['labeled_batch'],condition,plan)
                    if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss {name} {step}')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),plan['gradient_clip'])
                    if not torch.isfinite(norm):raise RuntimeError(f'Nonfinite gradient {name} {step}')
                    optimizer.step()
                    if step==endpoint:
                        grade={'step':step,**validation(model,support,bp['target_task'],tokens,device)};curve.append(grade);atomic_torch(root/'checkpoints'/f'{name}_u{step}.pt',{'model':model.state_dict(),'step':step,'grade':grade})
                    if step%plan['save_every']==0 or step==endpoint:
                        runtime=elapsed+time.monotonic()-begun
                        atomic_torch(latest,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'curve':curve,'elapsed_seconds':runtime,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()if device=='cuda'else[]})
                        record={'status':'complete'if step==plan['updates']else'round_complete'if step==endpoint else'training','replicate':rep,'condition':condition,'step':step,'curve':curve,
                            'initialization_sha256':target['checkpoint_sha256'],'schedule_sha256':sha(file),'elapsed_seconds':runtime,'updated_utc':now(),'forward_rows':step*160,'latest_sha256':sha(latest),
                            'checkpoint_sha256':{str(g['step']):sha(root/'checkpoints'/f"{name}_u{g['step']}.pt")for g in curve if g['step']>0}}
                        atomic_json(record_path,record);atomic_json(root/'current_job.json',record)
                        print(json.dumps({'job':name,'round':endpoint,'step':step,'elapsed_seconds':runtime,'validation':curve[-1],**{k:float(v.detach().cpu())if torch.is_tensor(v)else v for k,v in parts.items()}}),flush=True)
                if step<endpoint:
                    atomic_torch(latest,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'curve':curve,'elapsed_seconds':elapsed+time.monotonic()-begun,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()if device=='cuda'else[]})
                    atomic_json(record_path,{'status':'partial_deadline','replicate':rep,'condition':condition,'step':step,'curve':curve,'updated_utc':now(),'latest_sha256':sha(latest)})
                    return finalize_training_state(plan,root,bp)
                del model,optimizer,a
                if device=='cuda':torch.cuda.empty_cache()
        atomic_json(root/f'round_{endpoint}_complete.json',{'completed_utc':now(),'all24_conditions_complete':True,'fixed_updates':endpoint})
    return finalize_training_state(plan,root,bp)


def finalize_training_state(plan,root,bp):
    complete=[step for step in plan['round_endpoints']if all((root/'checkpoints'/f"{rep['id']}_{condition}_u{step}.pt").exists()for rep in bp['replicates']for condition in plan['conditions'])]
    atomic_json(root/'training/state.json',{'status':'all_complete'if plan['updates']in complete else'partial_high_dose'if plan['primary_update']in complete else'primary_incomplete',
        'completed_utc':now(),'complete_matched_endpoints':complete,'primary_complete':plan['primary_update']in complete,
        'partial_higher_doses_excluded':True})


def evaluate(plan,root,bp,base,bs,sig):
    state=json.loads((root/'training/state.json').read_text());assert state['primary_complete'];device=configure();_,tokens,_,_,_=api(bp);data=dict(np.load(root/'dataset/test/dataset.npz'))
    marker=root/'test_opened.json'
    if not marker.exists():atomic_json(marker,{'opened_utc':now(),'all24_primary_fits_complete':True,'complete_matched_endpoints':state['complete_matched_endpoints']})
    for source in bs['sources']:
        file=root/'teacher'/f"test_s{source['seed']}.npz"
        if not file.exists():model=load_model(source,bp,device);np.savez_compressed(file,**extract(model,data,bp['source_task'],tokens,1));del model
    for rep in bp['replicates']:
        target=next(s for s in bs['targets']if s['seed']==rep['target_pretrain_seed']);file=root/'evaluations'/f"{rep['id']}_initial.npz"
        if not file.exists():model=load_model(target,bp,device);np.savez_compressed(file,**extract(model,data,bp['target_task'],tokens,0));del model
        for condition in plan['conditions']:
            for step in state['complete_matched_endpoints']:
                name=rep['id']+'_'+condition;file=root/'evaluations'/f'{name}_u{step}.npz'
                if file.exists():continue
                model=load_model(target,bp,device);cp=torch.load(root/'checkpoints'/f'{name}_u{step}.pt',map_location='cpu',weights_only=True);model.load_state_dict(cp['model']);del cp
                np.savez_compressed(file,**extract(model,data,bp['target_task'],tokens,0));del model
            print({'evaluated_supplement':rep['id']+'_'+condition},flush=True)
    atomic_json(root/'evaluation_state.json',{'status':'complete','completed_utc':now(),'complete_matched_endpoints':state['complete_matched_endpoints']})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['register','data','train','evaluate']);args=parser.parse_args();values=initialize()
    if args.phase=='data':create_data(*values)
    elif args.phase=='train':train(*values)
    elif args.phase=='evaluate':evaluate(*values)
    else:print({'registered_utc':json.loads((values[1]/'protocol.json').read_text())['registered_utc']})
