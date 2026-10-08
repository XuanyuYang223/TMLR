"""One prospective inverse-task functional assay with equal target labels."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .longrun_attention import accelerate
from .longrun_engine import atomic_json, atomic_torch
from .permworld_combinations import sha, prompts
from .permutation_audit import transform, TRANSFORMS
from .specialist_cka_controls import api, input_key, original_input_key, old_inputs

CONFIG=Path('configs/inverse_functional_alignment.json')


def now():return datetime.now(timezone.utc).isoformat()


def configure():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def initialize():
    plan=json.loads(CONFIG.read_text());root=Path(plan['output'])
    for folder in [root,*[root/p for p in ['dataset','teacher','initializations','training','checkpoints','evaluations']]]:
        folder.mkdir(parents=True,exist_ok=True)
    dest=root/'protocol.json'
    if dest.exists():
        sig=json.loads(dest.read_text())['signature']
        assert sig['plan']==plan and sig['code_sha256']==sha(__file__) and sig['config_sha256']==sha(CONFIG)
        for entry in sig['sources']:assert sha(entry['checkpoint'])==entry['checkpoint_sha256']
        return plan,root,sig
    old=json.loads((Path(plan['previous_study'])/'protocol.json').read_text())['signature']
    sources=[x for x in old['sources'] if x['task']==plan['source_task']]
    assert {x['seed'] for x in sources}==set(plan['source_seeds'])
    for x in sources:assert sha(x['checkpoint'])==x['checkpoint_sha256']
    names={'data.npz','probe_dataset.npz','source_data.npz','training_orbit_audit.npz','dataset.npz'}
    paths=sorted({str(p) for p in Path('results').rglob('*.npz') if p.name in names and root not in p.parents})
    entries=[];hashes=set()
    for p in paths:
        digest=sha(p)
        if digest not in hashes:entries.append({'path':p,'sha256':digest});hashes.add(digest)
    sig={'plan':plan,'code_sha256':sha(__file__),'config_sha256':sha(CONFIG),'sources':sources,'excluded_archives':entries,
        'prior_completion_sha256':sha(Path(plan['previous_study'])/'completion.json'),
        'dependencies_sha256':{p:sha(p) for p in ['experiments/longrun_attention.py','experiments/permworld_combinations.py',
            'experiments/permutation_audit.py','experiments/specialist_cka_controls.py']},
        'upstream_sha256':{p:sha(Path(plan['upstream_python'])/'neurips_permutations'/p) for p in ['models.py','training.py','math_ops.py','passage.py']}}
    assert not list((root/'training').glob('*'))
    atomic_json(dest,{'registered_utc':now(),'new_target_training_runs':0,'new_target_test_results':0,'signature':sig})
    return plan,root,sig


def matched_derangement(n,y,mask,seed):
    rng=np.random.default_rng(seed);partner=np.arange(len(n));eligible=np.zeros(len(n),dtype=bool)
    for nn in np.unique(n[mask]):
        for value in np.unique(y[mask&(n==nn)]):
            ix=np.flatnonzero(mask&(n==nn)&(y==value))
            if len(ix)<2:continue
            shift=int(rng.integers(1,len(ix)));partner[ix]=np.roll(ix,shift);eligible[ix]=True
    assert np.all(partner[eligible]!=np.flatnonzero(eligible))
    np.testing.assert_array_equal(n[eligible],n[partner[eligible]])
    np.testing.assert_array_equal(y[eligible],y[partner[eligible]])
    return partner,eligible


def create_data(plan,root,sig):
    if (root/'dataset/audit.json').exists():return
    functions,tokens,one_line,_,_=api(plan);seen=old_inputs(sig['excluded_archives'])
    proposals={};lookup={};unique=set();rejected=set()
    seeds=[plan['test_data_seed']]+[r['data_seed'] for r in plan['replicates']]
    for world,seed in enumerate(seeds):
        rng=np.random.default_rng(seed)
        for n in plan['lengths']:
            count=plan['candidate_orbits_per_length'] if world==0 else 512
            values=[]
            while len(values)<count:
                p=tuple(map(int,rng.permutation(n)+1));orbit=[transform(p,t) for t in TRANSFORMS]
                keys=[input_key(v) for v in orbit]
                if len(set(keys))!=8 or any(k in seen or k in unique for k in keys):continue
                anchor=(world,n,len(values));values.append(p)
                for k in keys:lookup[k]=anchor;unique.add(k)
            proposals[world,n]=values
    manifest=Path(plan['repository'])/'data/permutation-properties-16m-v1/manifest.json';parent=json.loads(manifest.read_text());scanned=0
    for j,shard in enumerate(parent['shards']):
        path=manifest.parent/shard['filename'];assert sha(path)==shard['sha256']
        with gzip.open(path,'rb') as handle:
            for line in handle:
                anchor=lookup.get(original_input_key(line))
                if anchor is not None:rejected.add(anchor)
                scanned+=1
        if j%25==0:print({'original_inputs_scanned':scanned,'rejected_orbits':len(rejected)},flush=True)
    available={(w,n):[p for j,p in enumerate(values) if (w,n,j) not in rejected] for (w,n),values in proposals.items()}
    def archive(name,points,split=None):
        inputs=[];raw=[];labels=[];source_labels=[];lengths=[]
        for p in points:
            n=len(p);inv=transform(p,'inverse');rows=[]
            for value in [p,inv]:
                row=[tokens['<BOS>'],tokens['<SIZE>'],tokens[f'{n:02d}']]+[tokens[t] for t in one_line(value)]
                rows.append(row+[tokens['<PAD>']]*(64-len(row)))
            inputs.append(rows);raw.append([list(transform(p,t))+[0]*(30-n) for t in TRANSFORMS])
            labels.append(functions[plan['target_task']](p));source_labels.append(functions[plan['source_task']](inv));lengths.append(n)
        np.testing.assert_array_equal(labels,source_labels)
        out={'input':np.asarray(inputs),'permutations':np.asarray(raw),'labels':np.asarray(labels),'lengths':np.asarray(lengths)}
        if split is not None:out['split']=np.asarray(split)
        path=root/'dataset'/name/'dataset.npz';path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,**out)
        return out,path
    audit_points=[];test_points=[]
    for n in plan['lengths']:
        values=available[0,n];need=plan['teacher_audit_per_length']+plan['test_per_length'];assert len(values)>=need
        audit_points+=values[:plan['teacher_audit_per_length']]
        test_points+=values[plan['teacher_audit_per_length']:need]
    _,ap=archive('audit',audit_points);_,tp=archive('test',test_points)
    support_records=[]
    for world,rep in enumerate(plan['replicates'],1):
        train=[];val=[]
        for j,n in enumerate(plan['lengths']):
            count=plan['train_counts'][j];vc=plan['validation_counts'][j];values=available[world,n];assert len(values)>=count+vc
            train+=values[:count];val+=values[count:count+vc]
        d,path=archive(rep['id'],train+val,[0]*len(train)+[1]*len(val))
        partner,eligible=matched_derangement(d['lengths'],d['labels'],d['split']==0,rep['data_seed']+90001)
        mp=path.parent/'mismatch.npz';np.savez_compressed(mp,partner=partner,eligible=eligible)
        support_records.append({'replicate':rep['id'],'dataset_sha256':sha(path),'mismatch_sha256':sha(mp),
            'train_labels':len(train),'validation_labels':len(val),'geometry_eligible_train_examples':int(eligible.sum())})
    atomic_json(root/'dataset/audit.json',{'created_utc':now(),'original_parent_manifest_sha256':sha(manifest),
        'original_inputs_scanned':scanned,'prior_local_inputs_excluded':len(seen),'rejected_orbits':len(rejected),
        'audit_sha256':sha(ap),'test_sha256':sha(tp),'test_examples':len(test_points),'audit_examples':len(audit_points),
        'support_records':support_records,'all_eight_orbit_states_excluded':True,'source_inverse_answer_equals_target':True})


def forward(model,x,n,task,tokens):
    ids,mask,q=prompts(x,n,[task],tokens);ar=torch.arange(len(x),device=x.device)
    h,valid=model._embed_inputs(ids,mask)
    for block in model.blocks:h=block(h,valid)
    h=model.final_norm(h)
    return model.lm_head(h[ar,q]),h[ar,2*n+3]


@torch.inference_mode()
def extract(model,d,task,tokens,action):
    device=next(model.parameters()).device;model.eval();outputs=[];hidden=[]
    for start in range(0,len(d['lengths']),128):
        x=torch.tensor(d['input'][start:start+128,action],device=device)
        n=torch.tensor(d['lengths'][start:start+128],device=device)
        logits,h=forward(model,x,n,task,tokens);outputs.append(logits.float().cpu().numpy());hidden.append(h.float().cpu().numpy())
    return {'logits':np.concatenate(outputs),'hidden':np.concatenate(hidden)}


def prepare_teachers(plan,root,sig):
    device=configure();_,tokens,_,TrainConfig,factory=api(plan)
    audit=dict(np.load(root/'dataset/audit/dataset.npz'));grades=[]
    for source in sig['sources']:
        s=source['seed'];cp=torch.load(source['checkpoint'],weights_only=True,map_location='cpu')
        model=accelerate(factory(TrainConfig.from_value(cp['config'])));model.load_state_dict(cp['model']);del cp
        model.to(device).eval()
        for p in model.parameters():p.requires_grad_(False)
        af=root/'teacher'/f'audit_s{s}.npz'
        if af.exists():features=dict(np.load(af))
        else:
            features=extract(model,audit,plan['source_task'],tokens,1);np.savez_compressed(af,**features)
        hit=features['logits'].argmax(-1)==audit['labels']
        per_length={str(n):float(hit[audit['lengths']==n].mean()) for n in plan['lengths']}
        passing=hit.mean()>=plan['source_accuracy_gate']['overall'] and min(per_length.values())>=plan['source_accuracy_gate']['every_length']
        grades.append({'source_seed':s,'accuracy':float(hit.mean()),'per_length':per_length,'passes':bool(passing),'audit_archive_sha256':sha(af)})
        for rep in plan['replicates']:
            if rep['source_seed']!=s:continue
            path=root/'teacher'/f"{rep['id']}.npz"
            if not path.exists():
                d=dict(np.load(root/'dataset'/rep['id']/'dataset.npz'))
                features=extract(model,d,plan['source_task'],tokens,1);np.savez_compressed(path,**features)
        del model
        if device=='cuda':torch.cuda.empty_cache()
    atomic_json(root/'teacher/gate.json',{'status':'passed' if all(x['passes'] for x in grades) else 'failed',
        'grades':grades,'completed_utc':now(),'source_models_frozen':True,'target_test_data_consulted':False})
    assert all(x['passes'] for x in grades),'Source reliability gate failed; no target fitting authorized under this protocol.'
    print({'teacher_gate':grades},flush=True)


def geometry_loss(x,y):
    x=x.float();y=y.detach().float();x=x-x.mean(0);y=y-y.mean(0)
    cross=(x.T@y).square().sum();denom=torch.sqrt((x.T@x).square().sum()*(y.T@y).square().sum()).clamp_min(1e-12)
    return 1-cross/denom


def output_loss(logits,teacher,temperature):
    return F.kl_div(F.log_softmax(logits.float()/temperature,dim=-1),F.softmax(teacher.detach().float()/temperature,dim=-1),reduction='batchmean')*temperature**2


@torch.inference_mode()
def validation(model,d,task,tokens,device):
    model.eval();use=d['split']==1;ix=np.flatnonzero(use)
    x=torch.tensor(d['input'][ix,0],device=device);n=torch.tensor(d['lengths'][ix],device=device);y=torch.tensor(d['labels'][ix],device=device)
    logits,_=forward(model,x,n,task,tokens)
    return {'accuracy':float((logits.argmax(-1)==y).float().mean().cpu()),'cross_entropy':float(F.cross_entropy(logits.float(),y).cpu())}


def train(plan,root,sig):
    assert json.loads((root/'teacher/gate.json').read_text())['status']=='passed'
    device=configure();_,tokens,_,TrainConfig,factory=api(plan)
    for rep in plan['replicates']:
        source=next(x for x in sig['sources'] if x['seed']==rep['source_seed'])
        cp=torch.load(source['checkpoint'],weights_only=True,map_location='cpu');cfg=TrainConfig.from_value(cp['config']);del cp
        d=dict(np.load(root/'dataset'/rep['id']/'dataset.npz'));teacher=dict(np.load(root/'teacher'/f"{rep['id']}.npz"));pair=dict(np.load(root/'dataset'/rep['id']/'mismatch.npz'))
        initial=root/'initializations'/f"{rep['id']}.pt"
        if not initial.exists():
            torch.manual_seed(rep['target_seed']);m=factory(cfg);atomic_torch(initial,m.state_dict());del m
        rng=np.random.default_rng(rep['data_seed']+81001);schedule=[]
        for _ in range(plan['updates']):
            n=int(rng.choice(plan['lengths']));ix=np.flatnonzero((d['split']==0)&(d['lengths']==n));schedule.append(rng.choice(ix,plan['batch_size'],replace=False))
        schedule=np.asarray(schedule);schedule_sha=hashlib.sha256(schedule.tobytes()).hexdigest()
        for condition in plan['conditions']:
            name=rep['id']+'_'+condition;record=root/'training'/f'{name}.json';latest=root/'checkpoints'/f'{name}_latest.pt';best=root/'checkpoints'/f'{name}_selected.pt'
            if record.exists() and json.loads(record.read_text())['status']=='complete':continue
            model=accelerate(factory(cfg));model.load_state_dict(torch.load(initial,weights_only=True,map_location='cpu'));model.to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
            x=torch.tensor(d['input'][:,0],device=device);n=torch.tensor(d['lengths'],device=device);y=torch.tensor(d['labels'],device=device)
            teacher_h=torch.tensor(teacher['hidden'],device=device);teacher_out=torch.tensor(teacher['logits'],device=device)
            partners=torch.tensor(pair['partner'],device=device);eligible=torch.tensor(pair['eligible'],device=device)
            torch.manual_seed(rep['target_seed']+71001)
            if device=='cuda':torch.cuda.manual_seed_all(rep['target_seed']+71001)
            start=0;curve=[];selected=None
            if latest.exists():
                saved=torch.load(latest,weights_only=True,map_location='cpu');model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
                for state in optimizer.state.values():
                    for k,v in state.items():
                        if isinstance(v,torch.Tensor):state[k]=v.to(device)
                start=saved['step'];curve=saved['curve'];selected=saved['selected'];torch.set_rng_state(saved['torch_rng'])
                if device=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
            begun=time.monotonic();last_losses={}
            for step in range(start+1,plan['updates']+1):
                if step<=plan['warmup_updates']:ratio=step/plan['warmup_updates']
                else:ratio=plan['minimum_lr_ratio']+(1-plan['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*(step-plan['warmup_updates'])/(plan['updates']-plan['warmup_updates'])))
                for group in optimizer.param_groups:group['lr']=plan['learning_rate']*ratio
                ix=torch.tensor(schedule[step-1],device=device);model.train();optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
                    logits,h=forward(model,x[ix],n[ix],plan['target_task'],tokens)
                ce=F.cross_entropy(logits.float(),y[ix]);geom=ce.new_tensor(0.);kd=ce.new_tensor(0.)
                if condition in ['correct_alignment','answer_matched_mismatch','output_distillation_correct_alignment']:
                    use=eligible[ix];target=teacher_h[partners[ix]] if condition=='answer_matched_mismatch' else teacher_h[ix]
                    if use.sum()>=3:geom=geometry_loss(h[use],target[use])
                if condition in ['output_distillation','output_distillation_correct_alignment']:kd=output_loss(logits,teacher_out[ix],plan['distillation_temperature'])
                loss=ce+plan['geometry_weight']*geom+plan['distillation_weight']*kd
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at {name} {step}')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),plan['gradient_clip']);optimizer.step()
                if step%200==0:
                    last_losses={'cross_entropy':float(ce.detach().cpu()),'geometry_loss':float(geom.detach().cpu()),'distillation_loss':float(kd.detach().cpu())}
                    print(json.dumps({'training':name,'step':step,'elapsed_seconds':time.monotonic()-begun,**last_losses}),flush=True)
                if step in plan['validation_updates']:
                    grade={'step':step,**validation(model,d,plan['target_task'],tokens,device)};curve.append(grade)
                    better=selected is None or (-grade['accuracy'],grade['cross_entropy'],grade['step'])<(-selected['accuracy'],selected['cross_entropy'],selected['step'])
                    if better:
                        selected=grade;atomic_torch(best,{'model':model.state_dict(),'step':step,'grade':grade,'replicate':rep,'condition':condition})
                    atomic_torch(latest,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'curve':curve,'selected':selected,
                        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else []})
                    atomic_json(record,{'status':'complete' if step==plan['updates'] else 'training','replicate':rep,'condition':condition,
                        'curve':curve,'selected':selected,'initialization_sha256':sha(initial),'schedule_sha256':schedule_sha,
                        'geometry_eligible_examples':int(pair['eligible'].sum()),'teacher_archive_sha256':sha(root/'teacher'/f"{rep['id']}.npz"),
                        'selected_checkpoint_sha256':sha(best),'latest_checkpoint_sha256':sha(latest),'updated_utc':now()})
            del model,optimizer,x,n,y,teacher_h,teacher_out
            if device=='cuda':torch.cuda.empty_cache()
    atomic_json(root/'training/state.json',{'status':'all_targets_complete','target_models':30,'completed_utc':now()})


def evaluate(plan,root,sig):
    assert json.loads((root/'training/state.json').read_text())['status']=='all_targets_complete'
    marker=root/'test_opened.json'
    if not marker.exists():atomic_json(marker,{'opened_utc':now(),'all_target_runs_complete_before_test':True,
        'selected_checkpoints':{p.stem:sha(p) for p in (root/'checkpoints').glob('*_selected.pt')}})
    device=configure();_,tokens,_,TrainConfig,factory=api(plan);d=dict(np.load(root/'dataset/test/dataset.npz'))
    for source in sig['sources']:
        file=root/'teacher'/f"test_s{source['seed']}.npz"
        if not file.exists():
            cp=torch.load(source['checkpoint'],weights_only=True,map_location='cpu');model=accelerate(factory(TrainConfig.from_value(cp['config'])));model.load_state_dict(cp['model']);del cp
            model.to(device);out=extract(model,d,plan['source_task'],tokens,1);np.savez_compressed(file,**out);del model
    for rep in plan['replicates']:
        source=next(x for x in sig['sources'] if x['seed']==rep['source_seed']);cp=torch.load(source['checkpoint'],weights_only=True,map_location='cpu');cfg=TrainConfig.from_value(cp['config']);del cp
        for condition in plan['conditions']:
            name=rep['id']+'_'+condition;dest=root/'evaluations'/f'{name}.json'
            if dest.exists():continue
            results={};arrays={}
            for endpoint in ['selected','latest']:
                saved=torch.load(root/'checkpoints'/f'{name}_{endpoint}.pt',weights_only=True,map_location='cpu')
                model=accelerate(factory(cfg));model.load_state_dict(saved['model']);del saved;model.to(device)
                out=extract(model,d,plan['target_task'],tokens,0);answers=out['logits'].argmax(-1);hit=answers==d['labels']
                results[endpoint]={'answer_accuracy':float(hit.mean()),'per_length':{str(n):float(hit[d['lengths']==n].mean()) for n in plan['lengths']}}
                arrays[endpoint+'_answers']=answers;arrays[endpoint+'_hidden']=out['hidden'];arrays[endpoint+'_logits']=out['logits'];del model
            archive=dest.with_suffix('.npz');np.savez_compressed(archive,**arrays)
            atomic_json(dest,{'replicate':rep,'condition':condition,'results':results,'archive_sha256':sha(archive),'completed_utc':now()})
            print({'evaluated':name,'selected_accuracy':results['selected']['answer_accuracy'],'final_accuracy':results['latest']['answer_accuracy']},flush=True)
    atomic_json(root/'state.json',{'status':'functional_evaluation_complete','endpoints':60,'completed_utc':now()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['register','data','teachers','train','evaluate','all'])
    stage=parser.parse_args().stage;plan,root,sig=initialize()
    if stage in ['data','all']:create_data(plan,root,sig)
    if stage in ['teachers','all']:prepare_teachers(plan,root,sig)
    if stage in ['train','all']:train(plan,root,sig)
    if stage in ['evaluate','all']:evaluate(plan,root,sig)
