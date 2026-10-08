"""One frozen-source constraint ablation, with visible-only model selection."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from .analysis import write_csv
from .collision_pair_followup import pair_statistics
from .hidden_relation_train import load_plan, collision_pair, query_hidden
from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .native_confirmation import setup
from .permworld_combinations import sha
from .representation_algebra import ACTION_NAMES, permutation_action_table, word_action
from .specialist_cka_controls import old_inputs

CONFIG=Path('configs/frozen_operator_constraint_ablation.json')


def now():return datetime.now(timezone.utc).isoformat()


def register():
    plan=json.loads(CONFIG.read_text());root=Path(plan['output'])
    for folder in [root,root/'maps',root/'features',root/'evaluations']:
        folder.mkdir(parents=True,exist_ok=True)
    sources=[]
    for seed in plan['source_seeds']:
        name=f"{plan['source_condition']}_s{seed}"
        cp=Path(plan['sources'])/'checkpoints'/f'{name}.pt';fp=Path(plan['known_features'])/f'{name}.npz'
        sources.append({'seed':seed,'name':name,'checkpoint':str(cp),'checkpoint_sha256':sha(cp),
            'known_features':str(fp),'known_features_sha256':sha(fp)})
    names={'data.npz','probe_dataset.npz','source_data.npz','training_orbit_audit.npz','dataset.npz'}
    excludes=sorted({str(p) for p in Path('results').rglob('*.npz') if p.name in names and root not in p.parents})
    archives=[];hashes=set()
    for p in excludes:
        h=sha(p)
        if h not in hashes:archives.append({'path':p,'sha256':h});hashes.add(h)
    signature={'plan':plan,'code_sha256':sha(__file__),'config_sha256':sha(CONFIG),'sources':sources,
        'excluded_archives':archives,'parent_protocol_sha256':sha(Path(plan['parent'])/'protocol.json'),
        'cka_protocol_sha256':sha('results/specialist_cka_controls/protocol.json'),
        'dependencies_sha256':{p:sha(p) for p in ['experiments/hidden_relation_train.py','experiments/collision_pair_followup.py',
            'experiments/representation_algebra.py','experiments/longrun_transfer.py','experiments/specialist_cka_controls.py']}}
    path=root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())['signature']==signature
    else:
        assert not list((root/'maps').glob('*')) and not (root/'dataset.npz').exists()
        atomic_json(path,{'registered_utc':now(),'new_compound_evaluations':0,'signature':signature})
    return plan,root,signature


def data(plan,root,sig):
    if (root/'dataset.npz').exists():return
    parent,config,_=load_plan();_,functions,tokens,one_line=setup(config)
    prior=old_inputs(sig['excluded_archives'])
    seen={tuple(map(int,k[1:-1].split(b','))) for k in prior}
    rng=np.random.default_rng(plan['data_seed']);rows=[];perms=[];labels=[];lengths=[];pairs=[]
    f=functions[parent['source_task']];pair_id=0
    for n in range(plan['lengths'][0],plan['lengths'][1]+1):
        pending={}
        for _ in range(plan['collision_pairs_per_length']):
            for orbit in collision_pair(rng,n,seen,pending,f):
                encoded=[]
                for p in orbit:
                    row=[tokens['<BOS>'],tokens['<SIZE>'],n]+[tokens[t] for t in one_line(p)]
                    encoded.append(row+[tokens['<PAD>']]*(64-len(row)))
                rows.append(encoded);perms.append([list(p)+[0]*(30-n) for p in orbit])
                labels.append([f(p) for p in orbit]);lengths.append(n);pairs.append(pair_id)
            pair_id+=1
    out={'input':np.asarray(rows),'permutations':np.asarray(perms),'labels':np.asarray(labels),
        'lengths':np.asarray(lengths),'pair_ids':np.asarray(pairs)}
    np.testing.assert_array_equal(out['labels'][::2][:,[0,1,4]],out['labels'][1::2][:,[0,1,4]])
    assert np.all(out['labels'][::2,5]!=out['labels'][1::2,5])
    np.testing.assert_array_equal(out['labels'][:,5],out['labels'][:,2])
    np.savez_compressed(root/'dataset.npz',**out)
    atomic_json(root/'dataset_audit.json',{'created_utc':now(),'sha256':sha(root/'dataset.npz'),'pairs':pair_id,
        'prior_distinct_inputs_excluded':len(prior),'visible_answers_equal':True,'hidden_answers_different':True,
        'scope':'Fresh pairs and all eight states disjoint from input-bearing local archives, including source data and the fresh specialist CKA data. Native relation assay source training uses these local archives. Original 16M parent is not this assay training population and is not rescanned.'})


def apply_twice(x,matrix,bias):
    return (x@matrix+bias)@matrix+bias


@torch.no_grad()
def project(matrix,cap):
    u,s,vh=torch.linalg.svd(matrix,full_matrices=False)
    matrix.copy_((u*s.clamp(max=cap).unsqueeze(-2))@vh)


def optimize(known,native,plan,cap,weight,seed):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    raw=known['train_hidden'].astype(np.float64);val=known['validation_hidden'].astype(np.float64)
    center=raw.reshape(-1,256).mean(0);scale=float(np.sqrt(np.square(raw-center).mean()))
    train=torch.tensor((raw-center)/scale,dtype=torch.float32,device=device)
    valid=torch.tensor((val-center)/scale,dtype=torch.float32,device=device)
    initial=np.eye(256)[None]+native['offset'].numpy().astype(np.float64)
    initial_bias=(center@initial+native['bias'].numpy()-center)/scale
    matrix=torch.nn.Parameter(torch.tensor(initial,dtype=torch.float32,device=device))
    bias=torch.nn.Parameter(torch.tensor(initial_bias,dtype=torch.float32,device=device))
    opt=torch.optim.Adam([matrix,bias],lr=plan['learning_rate'])
    rng=np.random.default_rng(seed+2026100704)
    for step in range(1,plan['updates']+1):
        ix=torch.tensor(rng.integers(0,len(train),size=plan['batch_anchors']),device=device)
        loss=0.
        for g,a in enumerate([1,2]):
            x=torch.cat([train[ix,0],train[ix,a]]);y=torch.cat([train[ix,a],train[ix,0]])
            loss=loss+torch.square(x@matrix[g]+bias[g]-y).mean()/2
            if weight:loss=loss+weight*torch.square(apply_twice(x,matrix[g],bias[g])-x).mean()/2
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if cap is not None and (step%plan['projection_every']==0 or step==plan['updates']):project(matrix,cap)
    with torch.no_grad():
        gen,rt=0.,0.
        for g,a in enumerate([1,2]):
            x=torch.cat([valid[:,0],valid[:,a]]);y=torch.cat([valid[:,a],valid[:,0]])
            gen+=float(torch.square(x@matrix[g]+bias[g]-y).mean().cpu())/2
            rt+=float(torch.square(apply_twice(x,matrix[g],bias[g])-x).mean().cpu())/2
    rho=matrix.detach().cpu().numpy().astype(np.float64)
    original_bias=scale*bias.detach().cpu().numpy().astype(np.float64)+center-center@rho
    return rho,original_bias,{'norm_cap':cap,'roundtrip_weight':weight,'visible_validation_generator_mse':gen,
        'visible_validation_roundtrip_mse':rt,'spectral_norms':np.linalg.svd(rho,compute_uv=False)[:,0].tolist()}


def fit(plan,root,sig):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    for source in sig['sources']:
        destination=root/'maps'/f"{source['name']}.json"
        if destination.exists():continue
        state=torch.load(source['checkpoint'],map_location='cpu',weights_only=True)
        known=dict(np.load(source['known_features']));native=state['operators'];del state
        candidates=[];arrays={}
        families={'unconstrained_refit':[(None,0.)],
            'norm_constraint':[(cap,0.) for cap in plan['norm_caps']],
            'roundtrip_constraint':[(None,w) for w in plan['roundtrip_weights']],
            'combined_constraints':[(cap,w) for cap in plan['norm_caps'] for w in plan['roundtrip_weights']]}
        selected={}
        arrays['native_frozen_rho']=np.eye(256)[None]+native['offset'].numpy()
        arrays['native_frozen_bias']=native['bias'].numpy()
        for family,grid in families.items():
            values=[]
            for j,(cap,weight) in enumerate(grid):
                rho,bias,meta=optimize(known,native,plan,cap,weight,source['seed'])
                values.append(meta);arrays[f'{family}_{j}_rho']=rho;arrays[f'{family}_{j}_bias']=bias
            best=int(np.argmin([m['visible_validation_generator_mse'] for m in values]))
            selected[family]={'candidate':best,**values[best]};candidates.append({'family':family,'candidates':values})
            print(json.dumps({'operator_family_fit':source['name'],'family':family,'selection':selected[family]}),flush=True)
        np.savez_compressed(destination.with_suffix('.npz'),**arrays,
            readout_weight=known['readout_weight'],readout_bias=known['readout_bias'])
        atomic_json(destination,{'source':source,'selected':selected,'candidates':candidates,
            'map_archive_sha256':sha(destination.with_suffix('.npz')),'completed_utc':now()})


@torch.inference_mode()
def features(plan,root,sig):
    parent,config,_=load_plan();_,_,tokens,_=setup(config)
    data=dict(np.load(root/'dataset.npz'));flat=data['input'].reshape(-1,64);ns=np.repeat(data['lengths'],8)
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    for source in sig['sources']:
        p=root/'features'/f"{source['name']}.npz"
        if p.exists():continue
        model=make_model(config,parent['architecture'],source['seed'],device)
        state=torch.load(source['checkpoint'],map_location='cpu',weights_only=True);model.load_state_dict(state['model']);del state
        model.eval();rows=[]
        for start in range(0,len(flat),plan['batch_size']):
            x=torch.tensor(flat[start:start+plan['batch_size']],device=device);n=torch.tensor(ns[start:start+plan['batch_size']],device=device)
            rows.append(query_hidden(model,x,n,parent['source_task'],tokens).float().cpu().numpy())
        np.savez_compressed(p,hidden=np.concatenate(rows).reshape(len(data['lengths']),8,256))
        del model
        if device=='cuda':torch.cuda.empty_cache()
        print(json.dumps({'collision_features_complete':source['name']}),flush=True)


def evaluate(plan,root,sig):
    data=dict(np.load(root/'dataset.npz'));table=permutation_action_table();letters={'c':1,'i':4};rows=[]
    for source in sig['sources']:
        h=np.load(root/'features'/f"{source['name']}.npz")['hidden'].astype(np.float64)
        rec=json.loads((root/'maps'/f"{source['name']}.json").read_text())
        maps=dict(np.load(root/'maps'/f"{source['name']}.npz"));w=maps['readout_weight'].astype(np.float64);b=maps['readout_bias']
        arrays={}
        for method in plan['methods']:
            prefix='native_frozen' if method=='native_frozen' else f"{method}_{rec['selected'][method]['candidate']}"
            rho,bias=maps[prefix+'_rho'],maps[prefix+'_bias']
            for word in ['ci','ici']:
                pred=h[:,0].copy()
                for letter in word:
                    g=0 if letter=='c' else 1;pred=pred@rho[g]+bias[g]
                action=word_action(word,table,letters);truth=data['labels'][:,action]
                answers=(pred@w.T+b).argmax(-1);metrics,_=pair_statistics(answers,truth,data['pair_ids'])
                row={'source':source['name'],'seed':source['seed'],'method':method,'word':word,'action':int(action),
                    'answer_accuracy':metrics['answer_accuracy'],'both_correct':metrics['both_correct_fraction'],
                    'neither_correct':metrics['neither_correct_fraction'],'orientation_excess':metrics['orientation_excess_all_pairs'],
                    'displacement_nmse':float(np.square(pred-h[:,action]).sum()/np.square(h[:,action]-h[:,0]).sum()),
                    'direct_transformed_input_accuracy':float(np.mean((h[:,action]@w.T+b).argmax(-1)==truth))}
                rows.append(row);arrays[f'{method}_{word}_answers']=answers
        np.savez_compressed(root/'evaluations'/f"{source['name']}.npz",**arrays)
        atomic_json(root/'evaluations'/f"{source['name']}.json",{'rows':rows[-10:],'completed_utc':now()})
    write_csv(root/'per_source_results.csv',rows)
    atomic_json(root/'state.json',{'status':'complete','source_models':3,'methods':5,'endpoints':len(rows),'completed_utc':now()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['register','data','fit','features','evaluate','all'])
    stage=parser.parse_args().stage;plan,root,sig=register()
    if stage in ['data','all']:data(plan,root,sig)
    if stage in ['fit','all']:fit(plan,root,sig)
    if stage in ['features','all']:features(plan,root,sig)
    if stage in ['evaluate','all']:evaluate(plan,root,sig)
