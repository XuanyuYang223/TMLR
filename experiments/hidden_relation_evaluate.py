"""Compose generator maps and read out answers without compound-input access."""
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
import torch

from .field_algebra_structure import affine_fit
from .hidden_relation_train import load_plan
from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .native_algebra_structure import extract
from .native_confirmation import setup
from .permworld_combinations import sha
from .representation_algebra import permutation_action_table,word_action


def transform_features(x,word,maps):
    result=x
    for letter in word:
        rho,bias=maps[letter];result=result @ rho+bias
    return result


def partial_probe(hidden,data,grid,seed,shuffled=False):
    """Use only 0<->C and 0<->I, including for means and validation."""
    h=np.asarray(hidden,dtype=np.float64).copy();fit=data['split']==0;val=data['split']==1
    means={}
    for n in np.unique(data['lengths']):
        mean=h[(data['lengths']==n)&fit][:,[0,1,4]].reshape(-1,h.shape[-1]).mean(0)
        means[int(n)]=mean;h[data['lengths']==n]-=mean
    rng=np.random.default_rng(seed);maps={};alphas={}
    def edges(mask,action):
        base=h[mask,0];changed=h[mask,action]
        if shuffled:
            ns=data['lengths'][mask];order=np.arange(len(base))
            for n in np.unique(ns):
                rows=np.flatnonzero(ns==n);order[rows]=np.roll(rows,int(rng.integers(1,len(rows))))
            changed=changed[order]
        return np.concatenate([base,changed]),np.concatenate([changed,base])
    for letter,action in [('c',1),('i',4)]:
        x,y=edges(fit,action);xv,yv=edges(val,action)
        candidates=[affine_fit(x,y-x,a) for a in grid]
        chosen=int(np.argmin([np.square(xv+xv @ w[:-1]+w[-1]-yv).sum() for w in candidates]))
        w=candidates[chosen];maps[letter]=(np.eye(h.shape[-1])+w[:-1],w[-1]);alphas[letter]=grid[chosen]
    return maps,means,alphas


def pair_scores(correct,pair_ids):
    values=[]
    for pair in np.unique(pair_ids):
        ids=np.flatnonzero(pair_ids==pair);assert len(ids)==2
        values.append([float(np.mean(correct[ids])),float(np.all(correct[ids]))])
    values=np.asarray(values);rng=np.random.default_rng(2026100664)
    estimates=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(1)
    return {'pair_both_correct':float(values[:,1].mean()),'paired_accuracy_interval_95':np.quantile(estimates[:,0],[.025,.975]).tolist(),
        'pair_both_correct_interval_95':np.quantile(estimates[:,1],[.025,.975]).tolist()}


def score_method(h,data,maps,means,weights,bias,view):
    table=permutation_action_table();letters={'c':1,'i':4};result=[];arrays={}
    for split,tag in [(2,'iid'),(3,'answer_collisions')]:
        use=data['split']==split;source=h[use,0].astype(np.float64);ns=data['lengths'][use]
        centers=np.array([means.get(int(n),np.zeros(h.shape[-1])) for n in ns]);x=source-centers
        for word in ['c','i','ci','ici','ic','cc','ii','cici']:
            action=word_action(word,table,letters);pred=transform_features(x,word,maps)+centers
            target=h[use,action].astype(np.float64);delta=target-source;denom=np.square(delta).sum()
            row={'split':tag,'word':word,'action':action,'displacement_nmse':float(np.square(pred-target).sum()/denom) if denom>1e-20 else None,
                'prediction_energy_over_source':float(np.square(pred-source).sum()/np.square(source-source.mean(0)).sum()),
                'answer_accuracy':None,'direct_transformed_input_accuracy':None,'pair_both_correct':None}
            key=tag+'_'+word;arrays[key+'_hidden']=pred.astype(np.float32)
            if view=='query':
                prediction=(pred @ weights.T+bias).argmax(-1);truth=data['labels'][use,action]
                direct=(target @ weights.T+bias).argmax(-1)
                row.update(answer_accuracy=float(np.mean(prediction==truth)),direct_transformed_input_accuracy=float(np.mean(direct==truth)))
                arrays[key+'_answers']=prediction
                if split==3:row.update(pair_scores(prediction==truth,data['pair_ids'][use]))
            if word=='ci':
                wrong=h[use,word_action('ic',table,letters)].astype(np.float64)
                row['wrong_order_minus_correct_gap']=float((np.square(pred-wrong).sum()-np.square(pred-target).sum())/denom)
            result.append(row)
    return result,arrays


def run():
    plan,config,root=load_plan();_,_,tokens,_=setup(config);torch.set_num_threads(4)
    state=json.loads((root/'state.json').read_text());assert state['status'] in ['sources_complete','evaluation','evaluation_complete']
    signature={'code_sha256':sha(__file__),'source_protocol_sha256':sha(root/'protocol.json'),'probe_data_sha256':sha(root/'probe_dataset.npz'),
        'core_sha256':{p:sha(p) for p in ['experiments/field_algebra_structure.py','experiments/native_algebra_structure.py','experiments/representation_algebra.py']},
        'method':'Native learned operators are never refit. Posthoc affine maps and means use only observed fit/validation states and four generator edges. No compound-validation tuning. Fixed numeric readout. Compound inputs forwarded only for direct-input diagnostic and hidden-vector target, not for predictions.'}
    path=root/'evaluation_protocol.json'
    if path.exists():assert json.loads(path.read_text())['signature']==signature
    else:atomic_json(path,{'registered_utc':datetime.now(timezone.utc).isoformat(),'signature':signature,'new_evaluations':0})
    data=dict(np.load(root/'probe_dataset.npz'));device='cuda' if torch.cuda.is_available() else 'cpu';completed=0
    for seed in plan['source_seeds']:
        for condition in plan['conditions']:
            name=f'{condition}_s{seed}';dest=root/'evaluations'/f'{name}.json'
            if dest.exists():completed+=1;continue
            atomic_json(root/'state.json',{'status':'evaluation','condition':name,'completed':completed,'updated_utc':datetime.now(timezone.utc).isoformat()})
            source=json.loads((root/'source'/f'{name}.json').read_text());cp=root/'checkpoints'/f'{name}.pt';assert sha(cp)==source['checkpoint_sha256']
            state=torch.load(cp,weights_only=True,map_location=device);model=make_model(config,plan['architecture'],seed,device);model.load_state_dict(state['model'])
            fp=root/'features'/f'{name}.npz'
            if fp.exists():features=dict(np.load(fp))
            else:features=extract(model,data,[plan['source_task']],tokens,plan['batch_size']);np.savez_compressed(fp,**features)
            weights=model.lm_head.weight[:31].detach().cpu().numpy().astype(np.float64)
            readout_bias=getattr(model.lm_head,'bias',None);bias=np.zeros(31) if readout_bias is None else readout_bias[:31].detach().cpu().numpy().astype(np.float64)
            offsets=state['operators']['offset'].detach().cpu().numpy().astype(np.float64);biases=state['operators']['bias'].detach().cpu().numpy().astype(np.float64)
            native={letter:(np.eye(len(weights[0]))+offsets[j],biases[j]) for j,letter in enumerate(['c','i'])}
            methods=[];archive_hashes={}
            for view,key in [('query','source_query_concat'),('prefix','ONE_END')]:
                h=features[key][:,:,-1].astype(np.float64)
                variants=[]
                if view=='query':variants.append(('native_operators',native,{},{}))
                for shuffled in [False,True]:
                    maps,means,alphas=partial_probe(h,data,plan['ridge_grid'],seed+9300,shuffled)
                    variants.append(('posthoc_shuffled_generators' if shuffled else 'posthoc_correct_generators',maps,means,alphas))
                variants.append(('identity',{'c':(np.eye(h.shape[-1]),np.zeros(h.shape[-1])),'i':(np.eye(h.shape[-1]),np.zeros(h.shape[-1]))},{},{}))
                for method,maps,means,alphas in variants:
                    results,arrays=score_method(h,data,maps,means,weights,bias,view)
                    arrays.update({f'rho_{g}':m[0] for g,m in maps.items()});arrays.update({f'bias_{g}':m[1] for g,m in maps.items()})
                    arrays.update(mean_lengths=np.array(list(means)),mean_vectors=np.array(list(means.values())).reshape(-1,h.shape[-1]),readout_weight=weights,readout_bias=bias)
                    ap=root/'arrays'/f'{name}_{view}_{method}.npz';np.savez_compressed(ap,**arrays);archive_hashes[ap.name]=sha(ap)
                    methods.append({'view':view,'method':method,'generator_alphas':alphas,'results':results})
            atomic_json(dest,{'condition':condition,'seed':seed,'source_validation_accuracy':source['observed_validation_accuracy'],
                'checkpoint_sha256':sha(cp),'feature_sha256':sha(fp),'array_sha256':archive_hashes,'methods':methods})
            completed+=1
            primary=next(r for r in methods if r['view']=='query' and r['method']=='native_operators')
            print(json.dumps({'evaluated':name,'collision_primary':[{k:r[k] for k in ['word','answer_accuracy','pair_both_correct','displacement_nmse']} for r in primary['results'] if r['split']=='answer_collisions' and r['word'] in plan['heldout_primary_words']]}),flush=True)
            del model,state
            if device=='cuda':torch.cuda.empty_cache()
    atomic_json(root/'state.json',{'status':'evaluation_complete','conditions':completed,'updated_utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':run()
