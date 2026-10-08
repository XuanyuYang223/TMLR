"""Prospective leave-one-out, label-budget controls, and unseen task sets."""
import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .longrun_engine import atomic_json, atomic_torch, learning_rate
from .longrun_transfer import make_model
from .permworld_combinations import answer_logits, validate, sha, select_groups
from .native_confirmation import setup, permutation_row, feature_vector, adapt_predictions, CORE_PATHS
from .native_target_repeat import apply_weights
from .native_length_baseline import smooth_prediction
from .native_ablation_math import descriptor, actions

CORE = tuple(dict.fromkeys(CORE_PATHS + ('experiments/native_ablation.py',
    'experiments/native_ablation_analysis.py', 'experiments/native_ablation_math.py',
    'experiments/permutation_audit.py', 'experiments/analysis.py', 'experiments/permworld_combinations_report.py')))


def now(): return datetime.now(timezone.utc).isoformat()


def paths():
    p=json.loads(Path('configs/native_ablation.json').read_text())
    c=json.loads(Path(p['base_config']).read_text());c.update(weight_decay=.01, dropout=0.)
    return p,c,Path(p['output'])


def rid(job): return f"{job['id']}_s{job['seed']}"


def jobs(p):
    groups=[{'id':'full_b96','tasks':p['candidate_tasks'],'batch':24},
            {'id':'full_b32','tasks':p['candidate_tasks'],'batch':8}]
    groups += [{'id':'drop_'+t+'_b96','tasks':[u for u in p['candidate_tasks'] if u!=t],'batch':32} for t in p['candidate_tasks']]
    groups += [{'id':'single_'+t+'_b96','tasks':[t],'batch':96} for t in p['candidate_tasks']]
    groups += [{**g,'batch':32} for g in p['novel_groups']]
    result=[{**g,'seed':s,'steps':p['source_steps'],'kind':'new'} for s in p['seeds'] for g in groups]
    assert len(result)==42
    return result


def references(p):
    root=Path(p['source_root']);result=[]
    for s in p['seeds']:
        for ident,phase,old,tasks in [('full_b128','multi','interior_none',p['candidate_tasks'])]+[
            ('single_'+t+'_b32','single',t,[t]) for t in p['candidate_tasks']]:
            oldid=f"{p['architecture']['id']}_{old}_s{s}"
            result.append({'id':ident,'seed':s,'kind':'reused','tasks':tasks,'batch':32,
                           'record':str(root/phase/f'{oldid}.json'),
                           'checkpoint':str(root/phase/'checkpoints'/f'{oldid}.pt')})
    return result


def sample(core,extra,buckets,batch,lo,hi):
    n=int(core.integers(lo,hi+1));base=core.choice(buckets[n],32,replace=True)
    indices=base[:batch] if batch<=32 else np.concatenate([base,extra.choice(buckets[n],batch-32,replace=True)])
    return n,indices


def prepare():
    p,c,r=paths();r.mkdir(parents=True,exist_ok=True)
    manifest=references(p)
    for x in manifest:
        d=json.loads(Path(x['record']).read_text());assert d['status']=='complete'
        x.update(record_sha256=sha(x['record']),checkpoint_sha256=sha(x['checkpoint']))
        assert x['checkpoint_sha256']==d['checkpoint_sha256']
    signature={'plan_sha256':sha('configs/native_ablation.json'),'effective_base_config':c,
               'core_sha256':{f:sha(f) for f in CORE},'sources':manifest,
               'upstream_sha256':{str(Path(c['repository'])/'src/neurips_permutations'/f):sha(Path(c['repository'])/'src/neurips_permutations'/f) for f in ('models.py','math_ops.py','passage.py')},
               'source_data_sha256':sha(Path(p['source_root'])/'dataset/data.npz'),
               'old_weights_sha256':sha(Path(p['source_root'])/'weights.json')}
    protocol=r/'protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text())['signature']==signature
        if (r/'dataset/data.npz').exists():return
    else:
        assert not list((r/'source').glob('*.json')) and not list((r/'transfer').glob('*.json'))
        atomic_json(protocol,{'registered_utc':now(),'signature':signature,'new_source_records':0,'new_transfer_records':0,
                             'plan':p,'scope':'selected from earlier outcomes, frozen before new source training and new target generation'})
    names,functions,tokens,one_line=setup(c)
    with np.load(Path(p['source_root'])/'dataset/data.npz') as a:old={k:a[k] for k in a.files}
    old_groups=select_groups(c);old_sets={frozenset(g['tasks']) for g in old_groups}
    assert all(frozenset(g['tasks']) not in old_sets for g in p['novel_groups'])
    assert all(not set(g['tasks'])&set(c['target_tasks']) for g in jobs(p))
    descriptions={g['id']:descriptor(g['tasks'],old,names,functions) for g in p['novel_groups']}
    # The generalized operator rules must reproduce all previous group features.
    previous=json.loads((Path(p['source_root'])/'forecast_features.json').read_text())
    for g in old_groups:
        d=descriptor(g['tasks'],old,names,functions)
        expected=next(x['math'] for x in previous if x['group']==g['id'])
        np.testing.assert_allclose(d['math'],expected,rtol=0,atol=1e-12)
    atomic_json(r/'descriptors.json',descriptions);atomic_json(r/'jobs.json',jobs(p))
    weights=json.loads((Path(p['source_root'])/'weights.json').read_text());atomic_json(r/'weights.json',weights)
    old_features=json.loads(Path('results/six_hour_session/transfer_prediction/features.json').read_text())
    learning=np.mean([x['baseline'][:2] for x in old_features],axis=0).tolist()
    early=[{'group':g['id'],'baseline':learning+descriptions[g['id']]['label_stats']+[0.,1.,0.],
            'math':descriptions[g['id']]['math']} for g in p['novel_groups']]
    forecast=[]
    for w in weights['weights']:
        if w['mode']!='shared':continue
        pred=apply_weights(np.array([feature_vector(x,w['variant']) for x in early]),w['weights'])
        forecast.extend({'group':x['group'],'budget':w['budget'],'variant':w['variant'],'predicted_gain':float(y)} for x,y in zip(early,pred))
    atomic_json(r/'prior_forecasts.json',{'frozen_utc':now(),'source_learning_assumption':'pooled old source audit/curve means, identical for all novel groups',
                'new_source_training_not_started':True,'fresh_target_data_not_generated':True,'features':early,'rows':forecast})
    data={k:v for k,v in old.items() if any(k.startswith(s+'_') for s in ('train','validation','representation','source_audit'))}
    seen=set()
    excluded=json.loads(Path('configs/native_confirmation.json').read_text())['excluded_datasets']+[str(Path(p['source_root'])/'dataset/data.npz')]
    for f in excluded:
        with np.load(f) as a:
            for k in a.files:
                if k.endswith('_input'):seen.update(permutation_row(x,int(n)) for x,n in zip(a[k],a[k[:-6]+'_lengths']))
    excluded_count=len(seen);rng=np.random.default_rng(p['target_data_seed'])
    for split,count in (('support_pool',100),('target_test',200)):
        inputs,labels,lengths=[],[],[]
        for n in range(10,31):
            for _ in range(count):
                while True:
                    value=tuple(map(int,rng.permutation(n)+1))
                    if value not in seen:seen.add(value);break
                prefix=[tokens['<BOS>'],tokens['<SIZE>'],n]+[tokens[t] for t in one_line(value)]
                inputs.append(prefix+[tokens['<PAD>']]*(64-len(prefix)));lengths.append(n)
                labels.append([functions[t](value) for t in names])
        for suffix,array in (('input',inputs),('lengths',lengths),('labels',labels)):data[split+'_'+suffix]=np.asarray(array,dtype=np.int64)
    (r/'dataset').mkdir(exist_ok=True);np.savez_compressed(r/'dataset/data.npz',**data)
    order=np.random.default_rng(p['target_data_seed']+50000).permutation(2100)
    atomic_json(r/'dataset/support_indices.json',{str(b):order[:b].tolist() for b in p['budgets']})
    atomic_json(r/'dataset/metadata.json',{'names':names,'data_sha256':sha(r/'dataset/data.npz'),'source_world_reused':True,
        'excluded_distinct_prior_inputs':excluded_count,'excluded_datasets_sha256':{f:sha(f) for f in excluded},'fresh_target_inputs':6300})
    atomic_json(r/'state.json',{'status':'prepared','updated_utc':now(),'planned_new_sources':42})


def load():
    p,c,r=paths();protocol=json.loads((r/'protocol.json').read_text())['signature']
    assert protocol['plan_sha256']==sha('configs/native_ablation.json') and protocol['effective_base_config']==c
    for f,h in protocol['core_sha256'].items():assert sha(f)==h,f
    for f,h in protocol['upstream_sha256'].items():assert sha(f)==h,f
    for x in protocol['sources']:
        assert sha(x['record'])==x['record_sha256'] and sha(x['checkpoint'])==x['checkpoint_sha256']
    assert sha(Path(p['source_root'])/'dataset/data.npz')==protocol['source_data_sha256']
    assert sha(Path(p['source_root'])/'weights.json')==protocol['old_weights_sha256']
    assert sha(r/'weights.json')==protocol['old_weights_sha256']
    assert sha(r/'dataset/data.npz')==json.loads((r/'dataset/metadata.json').read_text())['data_sha256']
    names,functions,tokens,_=setup(c);device='cuda' if torch.cuda.is_available() else 'cpu'
    with np.load(r/'dataset/data.npz') as a:raw={k:a[k] for k in a.files}
    data={k:torch.tensor(v,device=device) for k,v in raw.items()}
    return p,c,r,raw,data,names,functions,tokens,device


def train_one(p,c,r,job,raw,data,names,tokens,device):
    directory=r/'source';directory.mkdir(exist_ok=True);path=directory/f'{rid(job)}.json';checkpoint=directory/'checkpoints'/f'{rid(job)}.pt'
    signature={'job':job,'protocol_sha256':sha(r/'protocol.json'),'data_sha256':sha(r/'dataset/data.npz')}
    fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    if path.exists():
        record=json.loads(path.read_text());assert record['fingerprint']==fingerprint
        if record['status']=='complete':assert sha(checkpoint)==record['checkpoint_sha256'];return
    model=make_model(c,p['architecture'],job['seed'],device)
    initial_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.parameters())).hexdigest()
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['source_learning_rate'],weight_decay=c['weight_decay'])
    core=np.random.default_rng(job['seed']+20261005);extra=np.random.default_rng(job['seed']+202610061)
    buckets={n:np.flatnonzero(raw['train_lengths']==n) for n in range(10,31)}
    consumed=np.zeros(len(raw['train_lengths']),dtype=bool);digest=hashlib.sha256();step=0;elapsed=0.
    curve=[{'step':0,**x} for x in validate(model,data,'validation',job['tasks'],names,tokens)]
    if checkpoint.exists():
        state=torch.load(checkpoint,weights_only=True,map_location=device);assert state['fingerprint']==fingerprint
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);step=state['step'];curve=state['curve'];elapsed=state['elapsed']
        for _ in range(step):
            n,ids=sample(core,extra,buckets,job['batch'],10,30);consumed[ids]=True;digest.update(np.array([n],dtype=np.int64).tobytes()+ids.tobytes())
        assert core.bit_generator.state==state['core_state'] and extra.bit_generator.state==state['extra_state']
        assert digest.hexdigest()==state['sample_sha256']
    started=time.monotonic();task_ids=[names.index(t) for t in job['tasks']]
    while step<job['steps']:
        step+=1;n,ids=sample(core,extra,buckets,job['batch'],10,30);consumed[ids]=True
        digest.update(np.array([n],dtype=np.int64).tobytes()+ids.tobytes())
        model.train();optimizer.zero_grad(set_to_none=True)
        for g in optimizer.param_groups:g['lr']=learning_rate(step,job['steps'],p)
        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
            logits=answer_logits(model,data['train_input'][ids],data['train_lengths'][ids],job['tasks'],tokens)
            loss=F.cross_entropy(logits.float(),data['train_labels'][ids][:,task_ids].T.reshape(-1))
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%p['record_every']==0 or step==job['steps']:
            vals=validate(model,data,'validation',job['tasks'],names,tokens);curve.extend({'step':step,**x} for x in vals)
            atomic_torch(checkpoint,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'curve':curve,
                'core_state':core.bit_generator.state,'extra_state':extra.bit_generator.state,'sample_sha256':digest.hexdigest(),
                'elapsed':elapsed+time.monotonic()-started,'fingerprint':fingerprint})
            progress={'job':rid(job),'step':step,'planned_steps':job['steps'],'validation_mean':float(np.mean([x['accuracy'] for x in vals]))}
            atomic_json(r/'current_job.json',progress);print(json.dumps(progress),flush=True)
    record={'status':'complete','signature':signature,'fingerprint':fingerprint,'job':job,'steps':step,
        'labels_per_update':job['batch']*len(job['tasks']),'total_labels':step*job['batch']*len(job['tasks']),
        'per_task_exposures':step*job['batch'],'unique_source_inputs_seen':int(consumed.sum()),
        'sample_sha256':digest.hexdigest(),'initial_parameter_sha256':initial_hash,'checkpoint_sha256':sha(checkpoint),
        'curve':curve,'source_audit':validate(model,data,'source_audit',job['tasks'],names,tokens),
        'core_state':core.bit_generator.state,'extra_state':extra.bit_generator.state,'completed_utc':now()}
    atomic_json(path,record);del model,optimizer
    if device=='cuda':torch.cuda.empty_cache()


def train():
    p,c,r,raw,data,names,functions,tokens,device=load()
    for index,job in enumerate(jobs(p)):
        atomic_json(r/'state.json',{'status':'source_training','completed':index,'planned':42,'job':job,'updated_utc':now()})
        train_one(p,c,r,job,raw,data,names,tokens,device)
    atomic_json(r/'state.json',{'status':'all_sources_complete','updated_utc':now(),'new_sources':42})


def forecast():
    p,c,r=paths()
    if (r/'forecasts.json').exists():return
    assert not list((r/'transfer').glob('*.json'))
    descriptions=json.loads((r/'descriptors.json').read_text());weights=json.loads((r/'weights.json').read_text());features=[]
    for job in jobs(p):
        if job['id'] not in descriptions:continue
        record=json.loads((r/'source'/f'{rid(job)}.json').read_text());assert record['status']=='complete'
        learning=[float(np.mean([x['accuracy'] for x in record[k]])) for k in ('source_audit','curve')]
        features.append({'group':job['id'],'seed':job['seed'],'baseline':learning+descriptions[job['id']]['label_stats']+[0.,1.,0.],'math':descriptions[job['id']]['math']})
    rows=[]
    for w in weights['weights']:
        if w['mode']!='shared':continue
        predicted=apply_weights(np.array([feature_vector(x,w['variant']) for x in features]),w['weights'])
        rows.extend({'group':x['group'],'seed':x['seed'],'budget':w['budget'],'variant':w['variant'],'predicted_gain':float(y)} for x,y in zip(features,predicted))
    atomic_json(r/'forecast_features.json',features);atomic_json(r/'forecasts.json',{'frozen_utc':now(),'rows':rows,
        'weights_sha256':sha(r/'weights.json'),'features_sha256':sha(r/'forecast_features.json'),'new_transfer_records':0})
    atomic_json(r/'state.json',{'status':'forecasts_saved','updated_utc':now(),'rows':len(rows)})


def catalog(p):return [{'id':'random','kind':'random','seed':s,'tasks':[]} for s in p['seeds']]+references(p)+jobs(p)


def evaluate():
    p,c,r,raw,data,names,functions,tokens,device=load();directory=r/'transfer';directory.mkdir(exist_ok=True)
    support=json.loads((r/'dataset/support_indices.json').read_text());truth=raw['target_test_labels'][:,names.index('lis_length')]
    assert (r/'forecasts.json').exists()
    for index,job in enumerate(catalog(p)):
        path=directory/f'{rid(job)}.json';predpath=directory/f'{rid(job)}_predictions.npz'
        expected={'job':job,'protocol_sha256':sha(r/'protocol.json'),'forecast_sha256':sha(r/'forecasts.json')}
        checkpoint=None
        if job['kind']!='random':
            checkpoint=Path(job['checkpoint']) if job['kind']=='reused' else r/'source/checkpoints'/f'{rid(job)}.pt'
            expected['checkpoint_sha256']=sha(checkpoint)
        if path.exists():
            saved=json.loads(path.read_text());assert saved['signature']==expected
            if saved['status']=='complete':assert saved['predictions_sha256']==sha(predpath);continue
        atomic_json(r/'state.json',{'status':'target_evaluation','completed':index,'planned':60,'job':job,'updated_utc':now()})
        model=make_model(c,p['architecture'],job['seed'],device)
        if checkpoint is not None:model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location=device)['model'])
        rows=[];predictions={}
        for budget,(mode,policy) in product(p['budgets'],p['adaptation'].items()):
            predicted=adapt_predictions(model,c,data,'lis_length',support[str(budget)],job['seed'],policy,names,tokens)
            key=f'{budget}_{mode}';predictions[key]=predicted
            rows.append({'budget':budget,'mode':mode,'accuracy':float(np.mean(predicted==truth))})
        np.savez_compressed(predpath,**predictions)
        atomic_json(path,{'status':'complete','signature':expected,'job':job,'rows':rows,'predictions_sha256':sha(predpath),'evaluated_utc':now()})
        del model
        if device=='cuda':torch.cuda.empty_cache()
        print(json.dumps({'transfer_complete':index+1,'total':60,'job':rid(job)}),flush=True)
    length=[];predictions={}
    for budget in p['budgets']:
        ids=support[str(budget)];predicted=smooth_prediction(raw['support_pool_lengths'][ids],raw['support_pool_labels'][ids,names.index('lis_length')],raw['target_test_lengths'],p['length_bandwidth'],31)
        length.append({'budget':budget,'accuracy':float(np.mean(predicted==truth))});predictions[str(budget)]=predicted
    atomic_json(r/'length.json',length);np.savez_compressed(r/'length_predictions.npz',**predictions)
    atomic_json(r/'state.json',{'status':'all_evaluations_complete','updated_utc':now(),'endpoints':240})


def run():
    prepare();train();forecast();evaluate()
    from .native_ablation_analysis import analyze,verify
    analyze();verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('prepare','train','forecast','evaluate','run'),default='run')
    args=parser.parse_args();globals()[args.phase]()
