"""Pre-inspection supplement matching the geometry-loss row count exactly."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .inverse_alignment_coverage import initialize, losses
from .inverse_functional_alignment import configure, now, forward, validation, extract, geometry_loss
from .longrun_attention import accelerate
from .longrun_engine import atomic_json, atomic_torch
from .permworld_combinations import sha
from .specialist_cka_controls import api


def equal_count_loss(h,teacher,labeled_eligible,labeled_count):
    count=int(labeled_eligible.sum())
    assert 3<=count<=len(h)-labeled_count
    return geometry_loss(h[labeled_count:labeled_count+count],teacher[labeled_count:labeled_count+count]),count


def register(root):
    folder=root/'size_control';folder.mkdir(exist_ok=True);file=folder/'protocol.json'
    if file.exists():assert json.loads(file.read_text())['code_sha256']==sha(__file__);return folder
    atomic_json(file,{'registered_utc':now(),'code_sha256':sha(__file__),
        'base_test_computation_started':(root/'test_opened.json').exists(),'base_numerical_test_outcomes_inspected_by_agent':False,
        'reason':'Base coverage uses up to64 geometry rows versus up to32 for support alignment. This control was chosen from code inspection before reading base-test results.',
        'rule':'Same initialized target, stored labeled/unlabeled schedules,64 forward rows,CE on32 labeled rows,optimizer and1200 updates. Each step align the first m unlabeled rows, m=number of eligible labeled rows used by support_alignment. Thus same actual geometry sample count every step, but relations cover4096 rather than192 available inputs. Correct inverse teacher only; no new oracle labels.',
        'primary':'Fixed1200 geometry and accuracy; validation-selected sensitivity; compare with the existing support_alignment, all six repeats. Supplemental, not part of the original six-condition registration.'})
    return folder


def train(plan,root,sig):
    folder=register(root);device=configure();_,tokens,_,TrainConfig,factory=api(plan);old=Path(plan['previous_study'])
    for rep in sig['replicates']:
        record_file=folder/f"{rep['id']}_training.json"
        if record_file.exists()and json.loads(record_file.read_text())['status']=='complete':continue
        source=next(s for s in sig['sources']if s['seed']==rep['source_seed']);cp=torch.load(source['checkpoint'],weights_only=True,map_location='cpu');cfg=TrainConfig.from_value(cp['config']);del cp
        s=dict(np.load(old/'dataset'/rep['id']/'dataset.npz'));u=dict(np.load(root/'dataset'/rep['id']/'dataset.npz'));assert 'labels'not in u
        st=dict(np.load(old/'teacher'/f"{rep['id']}.npz"));ut=dict(np.load(root/'teacher'/f"{rep['id']}.npz"));sm=dict(np.load(old/'dataset'/rep['id']/'mismatch.npz'));um=dict(np.load(root/'teacher'/f"{rep['id']}_mismatch.npz"));schedule=dict(np.load(root/'training'/f"{rep['id']}_schedule.npz"))
        values={'x':s['input'][:,0],'n':s['lengths'],'y':s['labels'],'ux':u['input'][:,0],'un':u['lengths'],'h':st['hidden'],'uh':ut['hidden'],'logits':st['logits'],'ulogits':ut['logits'],'p':sm['partner'],'up':um['partner'],'eligible':sm['eligible'],'ueligible':um['eligible']}
        a={k:torch.tensor(v,device=device)for k,v in values.items()}
        model=accelerate(factory(cfg));model.load_state_dict(torch.load(old/'initializations'/f"{rep['id']}.pt",weights_only=True,map_location='cpu'));model.to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
        torch.manual_seed(rep['target_seed']+71001)
        if device=='cuda':torch.cuda.manual_seed_all(rep['target_seed']+71001)
        latest=folder/f"{rep['id']}_latest.pt";start=0;curve=[];counts=[]
        if latest.exists():
            cp=torch.load(latest,weights_only=True,map_location='cpu');model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);start=cp['step'];curve=cp['curve'];counts=cp['counts'];torch.set_rng_state(cp['torch_rng'])
            if device=='cuda':torch.cuda.set_rng_state_all(cp['cuda_rng'])
            del cp
        begun=time.monotonic()
        for step in range(start+1,1201):
            ratio=step/100 if step<=100 else .1+.9*.5*(1+math.cos(math.pi*(step-100)/1100))
            for g in optimizer.param_groups:g['lr']=plan['learning_rate']*ratio
            l=torch.tensor(schedule['labeled'][step-1],device=device);uu=torch.tensor(schedule['unlabeled'][step-1],device=device)
            x=torch.cat([a['x'][l],a['ux'][uu]]);n=torch.cat([a['n'][l],a['un'][uu]]);hsource=torch.cat([a['h'][l],a['uh'][uu]])
            wrong=torch.cat([a['h'][a['p'][l]],a['uh'][a['up'][uu]]]);teacher=torch.cat([a['logits'][l],a['ulogits'][uu]]);eligible=torch.cat([a['eligible'][l],a['ueligible'][uu]])
            model.train();optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):logits,h=forward(model,x,n,plan['target_task'],tokens)
            base,_=losses(logits,h,a['y'][l],hsource,wrong,teacher,eligible,32,'ordinary_exposure',plan)
            geom,count=equal_count_loss(h,hsource,a['eligible'][l],32);counts.append(count);loss=base+plan['geometry_weight']*geom
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite equal-count loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if step%400==0:
                grade={'step':step,**validation(model,s,plan['target_task'],tokens,device)};curve.append(grade)
                atomic_torch(folder/f"{rep['id']}_u{step}.pt",{'model':model.state_dict(),'step':step,'grade':grade})
                atomic_torch(latest,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'curve':curve,'counts':counts,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()if device=='cuda'else[]})
                selected=min(curve,key=lambda g:(-g['accuracy'],g['cross_entropy'],g['step']))
                atomic_json(record_file,{'status':'complete'if step==1200 else 'training','curve':curve,'selected':selected,'geometry_rows_per_step':counts,
                    'schedule_sha256':sha(root/'training'/f"{rep['id']}_schedule.npz"),'initialization_sha256':sha(old/'initializations'/f"{rep['id']}.pt"),
                    'candidate_sha256':{str(g['step']):sha(folder/f"{rep['id']}_u{g['step']}.pt")for g in curve},'updated_utc':now()})
                print({'size_control':rep['id'],'step':step,'elapsed_seconds':time.monotonic()-begun,'geometry_loss':float(geom.detach().cpu())},flush=True)
        del model,optimizer,a
    atomic_json(folder/'training_state.json',{'status':'all_six_complete','completed_utc':now()})


def evaluate(plan,root,sig):
    folder=register(root);assert json.loads((folder/'training_state.json').read_text())['status']=='all_six_complete'
    if not (folder/'test_opened.json').exists():atomic_json(folder/'test_opened.json',{'opened_utc':now(),'all_six_supplement_fits_finished':True})
    device=configure();_,tokens,_,TrainConfig,factory=api(plan);data=dict(np.load(root/'dataset/test/dataset.npz'))
    for rep in sig['replicates']:
        file=folder/f"{rep['id']}_evaluation.json"
        if file.exists():continue
        source=next(s for s in sig['sources']if s['seed']==rep['source_seed']);cp=torch.load(source['checkpoint'],weights_only=True,map_location='cpu');cfg=TrainConfig.from_value(cp['config']);del cp
        train_record=json.loads((folder/f"{rep['id']}_training.json").read_text());arrays={};grades={}
        for endpoint,step in [('final',1200),('selected',train_record['selected']['step'])]:
            cp=torch.load(folder/f"{rep['id']}_u{step}.pt",weights_only=True,map_location='cpu');model=accelerate(factory(cfg));model.load_state_dict(cp['model']);del cp;model.to(device)
            out=extract(model,data,plan['target_task'],tokens,0);grades[endpoint]={'step':step,'accuracy':float((out['logits'].argmax(-1)==data['labels']).mean())}
            for name,value in out.items():arrays[endpoint+'_'+name]=value
            del model
        archive=file.with_suffix('.npz');np.savez_compressed(archive,**arrays);atomic_json(file,{'results':grades,'archive_sha256':sha(archive)})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['register','train','evaluate']);args=parser.parse_args().stage;plan,root,sig=initialize()
    if args=='register':register(root)
    elif args=='train':train(plan,root,sig)
    else:evaluate(plan,root,sig)
