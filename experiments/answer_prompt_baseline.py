"""Oracle answer/task controls for the completed structural replication."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .algebra_structure_direct import direct_probe
from .field_algebra_structure import affine_fit
from .longrun_engine import atomic_json
from .native_confirmation import setup
from .permworld_combinations import sha
from .representation_algebra import permutation_action_table


def answer_features(labels,tasks,seed,width=256):
    blocks=[]
    for j,task in enumerate(tasks):
        salt=int(hashlib.sha256(task.encode()).hexdigest()[:8],16)
        rng=np.random.default_rng(seed+salt)
        answers=rng.normal(size=(31,width));prompt=rng.normal(size=width)
        blocks.append(answers[labels[...,j]]+prompt)
    return np.concatenate(blocks,axis=-1)


def remove_linear_answer_component(hidden,labels,data,grid):
    h=np.asarray(hidden,dtype=np.float64);codes=np.eye(31)[labels].reshape(*labels.shape[:-1],-1)
    fit=data['split']==0;val=data['split']==1
    hc=h.copy();xc=codes.copy();means={}
    for n in np.unique(data['lengths']):
        use=(data['lengths']==n)&fit
        hm=h[use].reshape(-1,h.shape[-1]).mean(0);xm=codes[use].reshape(-1,codes.shape[-1]).mean(0)
        means[int(n)]=(hm,xm);hc[data['lengths']==n]-=hm;xc[data['lengths']==n]-=xm
    x=xc[fit].reshape(-1,xc.shape[-1]);y=hc[fit].reshape(-1,hc.shape[-1])
    candidates=[affine_fit(x,y,a) for a in grid]
    xv=xc[val].reshape(-1,xc.shape[-1]);yv=hc[val].reshape(-1,hc.shape[-1])
    chosen=int(np.argmin([np.square(np.column_stack([xv,np.ones(len(xv))]) @ w-yv).sum() for w in candidates]))
    w=candidates[chosen];prediction=xc @ w[:-1]+w[-1]
    residual=hc-prediction
    test=data['split']==2
    return residual,{'selected_alpha':grid[chosen],
        'heldout_h_reconstruction_nmse':float(np.square(residual[test]).sum()/np.square(hc[test]).sum()),
        'scope':'FIT-only linear additive reconstruction from all four true source answers and task/length context; nonlinear joint answer information is not removed.'},w


def run():
    plan=json.loads(Path('configs/algebra_hidden_relations.json').read_text());root=Path(plan['output'])/'answer_controls';root.mkdir(parents=True,exist_ok=True)
    for name in ['probes','arrays']:(root/name).mkdir(exist_ok=True)
    previous=Path(plan['previous_replication']);oldplan=json.loads(Path('configs/algebra_structure_replication.json').read_text())
    config=json.loads(Path(plan['base_config']).read_text());_,functions,_,_=setup(config)
    signature={'code_sha256':sha(__file__),'previous_protocol_sha256':sha(previous/'protocol.json'),
        'old_dataset_sha256':sha(previous/'probe_dataset.npz'),'plan_sha256':sha('configs/algebra_hidden_relations.json'),
        'dependencies':{p:sha(p) for p in ['experiments/algebra_structure_direct.py','experiments/representation_algebra.py','experiments/field_algebra_structure.py']},
        'scope':'Exploratory oracle control of previously inspected neural results; true answers privileged; no independent neural replication or answer-mechanism exclusion claim.'}
    path=root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())['signature']==signature
    else:atomic_json(path,{'registered_utc':datetime.now(timezone.utc).isoformat(),'signature':signature,'new_controls':0})
    data=dict(np.load(previous/'probe_dataset.npz'));table=permutation_action_table();rows=[]
    for group in oldplan['groups']:
        tasks=group['tasks'];labels=np.array([[[functions[t](tuple(map(int,p[:n]))) for t in tasks] for p in orbit] for orbit,n in zip(data['permutations'],data['lengths'])])
        for seed in plan['answer_baseline_seeds']:
            name=f"{group['id']}_answer_prompt_s{seed}";h=answer_features(labels,tasks,seed)
            result,arrays=direct_probe(h,data['split'],data['lengths'],table,{'c':1,'r':2,'i':4},[('r','c'),('c','i'),('r','i'),('r','c','i')],64,oldplan['ridge_grid'],seed+9100)
            ap=root/'arrays'/f'{name}.npz';np.savez_compressed(ap,**arrays)
            row={'group':group['id'],'kind':'answer_plus_prompt','seed':seed,'result':result,'array_sha256':sha(ap)}
            atomic_json(root/'probes'/f'{name}.json',row);rows.append(row)
        for seed in oldplan['source_seeds']:
            name=f"{group['id']}_answer_residual_s{seed}";fp=previous/'features'/f"{group['id']}_s{seed}_trained.npz"
            h=np.load(fp)['source_query_concat'][:,:,-1]
            residual,audit,w=remove_linear_answer_component(h,labels,data,oldplan['ridge_grid'])
            result,arrays=direct_probe(residual,data['split'],data['lengths'],table,{'c':1,'r':2,'i':4},[('r','c'),('c','i'),('r','i'),('r','c','i')],64,oldplan['ridge_grid'],seed+9100)
            np.save(root/'arrays'/f'{name}_residual.npy',residual)
            np.savez_compressed(root/'arrays'/f'{name}.npz',**arrays,answer_regression=w)
            row={'group':group['id'],'kind':'linear_answer_residual','seed':seed,'result':result,'answer_audit':audit,'source_feature_sha256':sha(fp)}
            atomic_json(root/'probes'/f'{name}.json',row);rows.append(row)
    summaries=[]
    for group in oldplan['groups']:
        for kind in ['answer_plus_prompt','linear_answer_residual']:
            selected=[r for r in rows if r['group']==group['id'] and r['kind']==kind]
            summaries.append({'group':group['id'],'kind':kind,
                'generator_nmse':float(np.mean([a['full_space_displacement_nmse'] for r in selected for a in r['result']['generators'] if a['status']=='complete'])),
                'composite_nmse':float(np.mean([a['full_space_displacement_nmse'] for r in selected for a in r['result']['composites'] if a['status']=='complete'])),
                'wrong_order_gap':float(np.mean([r['result']['wrong_order'][0]['gap_wrong_minus_correct'] for r in selected])),
                'hidden_reconstruction_nmse':float(np.mean([r['answer_audit']['heldout_h_reconstruction_nmse'] for r in selected])) if kind=='linear_answer_residual' else None})
    atomic_json(root/'summary.json',{'rows':summaries,'conditions':len(rows),'scope':signature['scope']})
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':run()
