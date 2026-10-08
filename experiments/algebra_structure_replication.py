"""New source-seed replication of the ordinary-task structural signal."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from .algebra_structure_direct import direct_probe
from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .native_ablation import train_one
from .native_algebra_structure import extract
from .native_confirmation import setup, permutation_row
from .native_source_subspace import contrast_basis
from .permworld_combinations import sha
from .representation_algebra import ACTION_NAMES, permutation_action_table, apply_word
from .permutation_audit import transform


CONFIG='configs/algebra_structure_replication.json'


def now(): return datetime.now(timezone.utc).isoformat()


def load_plan():
    plan=json.loads(Path(CONFIG).read_text()); config=json.loads(Path(plan['base_config']).read_text())
    config.update(weight_decay=.01,dropout=0.)
    return plan,config,Path(plan['output'])


def collect_prior_inputs(paths):
    seen=set()
    for filename in paths:
        with np.load(filename) as a:
            if 'permutations' in a and 'input' in a:
                seen.update(tuple(map(int,p[:n])) for orbit,n in zip(a['permutations'],a['lengths']) for p in orbit)
            else:
                for key in a.files:
                    if key.endswith('_input'):
                        seen.update(permutation_row(row,int(n)) for row,n in zip(a[key],a[key[:-6]+'_lengths']))
    return seen


def create_orbits(plan,config,seen,tokens,one_line):
    prior_count=len(seen);rng=np.random.default_rng(plan['probe_data_seed'])
    rows,raw,splits,lengths=[],[],[],[];rejected=0
    for split,(name,count) in enumerate(plan['orbits_per_length'].items()):
        for n in range(plan['lengths'][0],plan['lengths'][1]+1):
            for _ in range(count):
                while True:
                    base=tuple(map(int,rng.permutation(n)+1));orbit=[transform(base,a) for a in ACTION_NAMES]
                    if len(set(orbit))!=8 or any(p in seen for p in orbit):rejected+=1;continue
                    seen.update(orbit);break
                prefix_rows=[]
                for p in orbit:
                    prefix=[tokens['<BOS>'],tokens['<SIZE>'],n]+[tokens[t] for t in one_line(p)]
                    prefix_rows.append(prefix+[tokens['<PAD>']]*(2*config['lengths'][1]+4-len(prefix)))
                rows.append(prefix_rows);raw.append([list(p)+[0]*(config['lengths'][1]-n) for p in orbit]);splits.append(split);lengths.append(n)
    return {'input':np.array(rows,dtype=np.int64),'permutations':np.array(raw,dtype=np.int64),
            'split':np.array(splits,dtype=np.int64),'lengths':np.array(lengths,dtype=np.int64)},prior_count,rejected


def initialize():
    plan,config,root=load_plan();root.mkdir(exist_ok=True,parents=True)
    for name in ('dataset','source','features','probes','arrays'):(root/name).mkdir(exist_ok=True)
    core=('experiments/native_ablation.py','experiments/longrun_engine.py','experiments/longrun_transfer.py',
          'experiments/longrun_attention.py','experiments/algebra_structure_direct.py',
          'experiments/representation_algebra.py','experiments/native_algebra_structure.py',
          'experiments/native_source_subspace.py','experiments/permworld_combinations.py')
    signature={'plan':plan,'config_sha256':sha(CONFIG),'replication_code_sha256':sha(__file__),
        'core_sha256':{p:sha(p) for p in core},'source_data_sha256':sha(plan['source_data']),
        'excluded_data_sha256':{p:sha(p) for p in plan['excluded_datasets']},
        'pilot_sources':{g['id']:sha(Path(plan['source_reference'])/'checkpoints'/f"{g['id']}_s{plan['pilot_seed']}.pt") for g in plan['groups']},
        'upstream_sha256':{name:sha(Path(config['repository'])/'src/neurips_permutations'/name) for name in ('models.py','math_ops.py','passage.py')}}
    path=root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())['signature']==signature
    else:
        assert not list((root/'source').glob('*.json')) and not list((root/'probes').glob('*.json'))
        atomic_json(path,{'registered_utc':now(),'signature':signature,'new_source_models':0,'new_geometry_results':0})
    # train_one reads this immutable source archive in its fingerprint.
    source_link=root/'dataset/data.npz'
    if not source_link.exists():source_link.symlink_to(Path(plan['source_data']).resolve())
    assert sha(source_link)==signature['source_data_sha256']
    names,functions,tokens,one_line=setup(config)
    dataset=root/'probe_dataset.npz'
    if not dataset.exists():
        seen=collect_prior_inputs(plan['excluded_datasets']);data,prior,rejected=create_orbits(plan,config,seen,tokens,one_line)
        np.savez_compressed(dataset,**data)
        atomic_json(root/'dataset_audit.json',{'created_utc':now(),'probe_dataset_sha256':sha(dataset),
            'excluded_distinct_inputs':prior,'orbits':len(data['split']),'states':len(data['split'])*8,
            'rejected_orbits':rejected,'all_orbits_size_eight':True,
            'split_orbits':{name:int(np.sum(data['split']==k)) for k,name in enumerate(plan['orbits_per_length'])}})
    return plan,config,root,names,functions,tokens


def train_sources():
    plan,config,root,names,_,tokens=initialize();device='cuda' if torch.cuda.is_available() else 'cpu'
    with np.load(plan['source_data']) as archive:raw={k:archive[k] for k in archive.files}
    data={k:torch.tensor(v,device=device) for k,v in raw.items()}
    completed=0
    for seed in plan['source_seeds']:
        for group in plan['groups']:
            job={**group,'seed':seed,'batch':plan['examples_per_task_per_step'],'steps':plan['source_steps'],'kind':'new'}
            atomic_json(root/'state.json',{'status':'source_training','completed':completed,'planned':9,
                'group':group['id'],'seed':seed,'updated_utc':now()})
            train_one(plan,config,root,job,raw,data,names,tokens,device);completed+=1
    atomic_json(root/'state.json',{'status':'sources_complete','completed':completed,'updated_utc':now()})


def extra_metrics(hidden,data,arrays,result):
    """Full-space II reconstruction and explicit identity baseline."""
    h=np.asarray(hidden,dtype=np.float64).copy()
    for n in np.unique(data['lengths']):
        mean=h[(data['lengths']==n)&(data['split']==0)].reshape(-1,h.shape[-1]).mean(0)
        h[data['lengths']==n]-=mean
    actual=h[data['split']==2].reshape(-1,h.shape[-1]);q=arrays['basis'].astype(np.float64)
    z=actual @ q;rho=arrays['map_i'].astype(np.float64)
    error=(z @ rho @ rho-z) @ q.T
    return {'identity_displacement_nmse':1.,
        'two_inverse_full_space_reconstruction_nmse':float(np.square(error).sum()/np.square(actual).sum()),
        'generator_nmse':float(np.mean([r['full_space_displacement_nmse'] for r in result['generators'] if r['status']=='complete'])),
        'composite_nmse':float(np.mean([r['full_space_displacement_nmse'] for r in result['composites'] if r['status']=='complete'])),
        'generator_shuffled_nmse':float(np.mean([r['shuffled_fit_displacement_nmse'] for r in result['generators'] if r['status']=='complete'])),
        'composite_shuffled_nmse':float(np.mean([r['shuffled_fit_displacement_nmse'] for r in result['composites'] if r['status']=='complete']))}


def evaluate_sources():
    plan,config,root,_,functions,tokens=initialize();data=dict(np.load(root/'probe_dataset.npz'))
    table=permutation_action_table();device='cuda' if torch.cuda.is_available() else 'cpu'
    for seed in [plan['pilot_seed']]+plan['source_seeds']:
        for group in plan['groups']:
            for status in ('random','trained'):
                ident=f"{group['id']}_s{seed}_{status}";dest=root/'probes'/f'{ident}.json'
                if dest.exists():continue
                atomic_json(root/'state.json',{'status':'geometry_evaluation','condition':ident,'updated_utc':now()})
                model=make_model(config,plan['architecture'],seed,device)
                source_record=None;checkpoint=None
                if status=='trained':
                    src=Path(plan['source_reference']) if seed==plan['pilot_seed'] else root/'source'
                    source_record=json.loads((src/f"{group['id']}_s{seed}.json").read_text());checkpoint=src/'checkpoints'/f"{group['id']}_s{seed}.pt"
                    assert source_record['status']=='complete' and sha(checkpoint)==source_record['checkpoint_sha256']
                    model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location=device)['model'])
                fp=root/'features'/f'{ident}.npz'
                if fp.exists():features=dict(np.load(fp))
                else:
                    features=extract(model,data,group['tasks'],tokens,plan['batch_size']);np.savez_compressed(fp,**features)
                weights=model.lm_head.weight[:31].detach().cpu().numpy().astype(np.float64)
                basis,contrasts=contrast_basis(weights)
                full=features['source_query_concat'][:,:,-1].astype(np.float64)
                block=full.reshape(*full.shape[:-1],4,256);projected=((block @ basis) @ basis.T).reshape(full.shape)
                views={'ONE_END':features['ONE_END'][:,:,-1],'source_query_concat':full,
                    'source_query_numeric_contrast':projected,'source_query_numeric_null':full-projected,
                    'source_query_numeric_logits':(block @ contrasts.T).reshape(*full.shape[:-1],-1)}
                results={};array_hashes={}
                for view,hidden in views.items():
                    for dimension in [plan['probe_dimension']]+([hidden.shape[-1]] if view in plan['full_width_sensitivity'] else []):
                        name=view if dimension==plan['probe_dimension'] else view+'_full_width'
                        result,arrays=direct_probe(hidden,data['split'],data['lengths'],table,{'c':1,'r':2,'i':4},
                            [('r','c'),('c','i'),('r','i'),('r','c','i')],dimension,plan['ridge_grid'],seed+9200)
                        result['explicit_checks']=extra_metrics(hidden,data,arrays,result)
                        path=root/'arrays'/f'{ident}_{name}.npz';np.savez_compressed(path,**arrays);array_hashes[path.name]=sha(path)
                        results[name]=result
                atomic_json(dest,{'status':'complete','group':group['id'],'seed':seed,'model_status':status,
                    'replication_role':'pilot_retested' if seed==plan['pilot_seed'] else 'new_source_seed',
                    'feature_sha256':sha(fp),'checkpoint_sha256':sha(checkpoint) if checkpoint else None,
                    'source_accuracy':float(np.mean([r['accuracy'] for r in source_record['source_audit']])) if source_record else None,
                    'results':results,'array_sha256':array_hashes,'completed_utc':now()})
                print(json.dumps({'geometry_completed':ident,'generator_nmse':results['source_query_concat']['explicit_checks']['generator_nmse'],
                    'composite_nmse':results['source_query_concat']['explicit_checks']['composite_nmse']}),flush=True)
                del model
    for group in plan['groups']:
        path=root/'probes'/f"teacher_{group['id']}.json"
        if path.exists():continue
        labels=np.array([[[functions[t](tuple(map(int,p[:n]))) for t in group['tasks']] for p in orbit] for orbit,n in zip(data['permutations'],data['lengths'])])
        results={}
        for view,hidden in [('numeric',labels.astype(np.float64)),('onehot',np.eye(31)[labels].reshape(len(labels),8,-1))]:
            result,arrays=direct_probe(hidden,data['split'],data['lengths'],table,{'c':1,'r':2,'i':4},
                [('r','c'),('c','i'),('r','i'),('r','c','i')],plan['probe_dimension'],plan['ridge_grid'],9200)
            results[view]=result
        atomic_json(path,{'group':group['id'],'tasks':group['tasks'],'results':results})
    atomic_json(root/'state.json',{'status':'geometry_complete','source_models':9,'conditions':24,'updated_utc':now()})


def run():
    train_sources();evaluate_sources()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('initialize','train','evaluate','run'),default='run')
    phase=parser.parse_args().phase
    {'initialize':initialize,'train':train_sources,'evaluate':evaluate_sources,'run':run}[phase]()
