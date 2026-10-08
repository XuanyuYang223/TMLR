"""Calibrate frozen readouts on generated paths ending at known states only."""
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np

from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .relation_observed_start import observed_start_score
from .representation_algebra import permutation_action_table


ROOT=Path('results/known_route_readout_control')
DEADLINE=datetime(2026,10,6,17,tzinfo=timezone.utc).timestamp()
ROUTES=[(0,''),(1,''),(4,''),(0,'c'),(0,'i'),(1,'c'),(4,'i'),
    (0,'cc'),(0,'ii'),(1,'cc'),(1,'ii'),(4,'cc'),(4,'ii')]
GRID=[1e-6,1e-4,.01,1.]


def now():return datetime.now(timezone.utc).isoformat()


def initialize():
    ROOT.mkdir(exist_ok=True)
    for name in ['evaluations','readouts','arrays']:(ROOT/name).mkdir(exist_ok=True)
    signature={'code_sha256':sha(__file__),'prediction_helper_sha256':sha('experiments/relation_observed_start.py'),
        'routes':ROUTES,'ridge_grid':GRID,'methods':['known_route_generated_ridge','known_route_clean_matched_ridge'],
        'scope':'Exploratory readout diagnosis after six native additional source outcomes and the first two fresh-world outcomes were inspected. Backbone and native operators are fixed. Generated routes start from true known e/C/I states and end only at known e/C/I states; no true compound hidden vector or compound answer is used for fitting or selection. CC and II use supplied involution identities. Clean controls repeat true endpoint states with exactly the same anchors, route counts and labels. Every readout is reported; hidden results do not choose a winner. This externally fitted readout with added known algebraic constraints is not unassisted network inference.'}
    path=ROOT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())['signature']==signature
    else:atomic_json(path,{'registered_utc':now(),'signature':signature,'new_known_route_readouts':0})


def route_features(hidden,labels,maps):
    table=permutation_action_table();columns={0:0,1:1,4:2};letters={'c':1,'i':4}
    generated,clean,truth=[],[],[]
    for start,word in ROUTES:
        pred=hidden[:,columns[start]].astype(np.float64)
        action=start
        for g in word:
            pred=pred@maps['rho_'+g]+maps['bias_'+g]
            action=int(table[action,letters[g]])
        assert action in columns
        generated.append(pred);clean.append(hidden[:,columns[action]].astype(np.float64));truth.append(labels[:,columns[action]])
    return np.concatenate(generated),np.concatenate(clean),np.concatenate(truth)


def ridge_readout(x,labels,val,val_labels):
    mean=x.mean(0);xc=x-mean;xv=val-mean
    design=np.column_stack([xc,np.ones(len(xc))]);validation=np.column_stack([xv,np.ones(len(xv))])
    target=np.eye(31)[labels];validation_target=np.eye(31)[val_labels]
    gram=design.T@design;rhs=design.T@target
    scale=np.trace(gram[:-1,:-1])/xc.shape[1]
    fits,losses=[],[]
    for alpha in GRID:
        fit=np.linalg.solve(gram+np.diag([alpha*max(scale,1e-20)]*xc.shape[1]+[0]),rhs)
        fits.append(fit);losses.append(float(np.square(validation@fit-validation_target).sum()))
    chosen=int(np.argmin(losses));fit=fits[chosen]
    return fit[:-1].T,fit[-1]-mean@fit[:-1],{'selected_alpha':GRID[chosen],'known_route_validation_losses':losses}


def evaluate(path,scope):
    rec=json.loads(path.read_text());folder=path.parent.parent
    name=rec.get('source',f"{rec['condition']}_s{rec['seed']}")
    dest=ROOT/'evaluations'/(name+'.json')
    if dest.exists():return
    known=None
    for candidate in [Path('results/frozen_relation_followup/features')/(name+'.npz'),
            Path('results/relation_operator_stability/features')/(name+'.npz'),Path('results/relation_readout_diagnostics/features')/(name+'.npz')]:
        if candidate.exists():known=candidate;break
    if known is None:return
    values=dict(np.load(known));source=dict(np.load(folder/'source_data.npz'))
    old=folder.name=='algebra_hidden_relations'
    mp=folder/'arrays'/(name+('_query_native_operators.npz' if old else '_native_operators.npz'))
    maps=dict(np.load(mp));assert maps['mean_vectors'].size==0
    xs=route_features(values['train_hidden'],source['train_labels'],maps)
    vs=route_features(values['validation_hidden'],source['validation_labels'],maps)
    data_folder=folder if folder.name.startswith('world') else Path('results/algebra_hidden_relations')
    data=dict(np.load(data_folder/'probe_dataset.npz'))
    fp=folder/'features'/(name+'.npz');hidden=np.load(fp)['source_query_concat'][:,:,-1]
    methods,hashes=[],{}
    true_val=values['validation_hidden'].reshape(-1,hidden.shape[-1]).astype(np.float64)
    true_labels=source['validation_labels'].reshape(-1)
    for method,index in [('known_route_generated_ridge',0),('known_route_clean_matched_ridge',1)]:
        weight,bias,selection=ridge_readout(xs[index],xs[2],vs[index],vs[2])
        revised={**maps,'readout_weight':weight,'readout_bias':bias}
        rows,arrays=observed_start_score(hidden,data,revised)
        rp=ROOT/'readouts'/(name+'_'+method+'.npz');np.savez_compressed(rp,weight=weight,bias=bias)
        ap=ROOT/'arrays'/(name+'_'+method+'.npz');np.savez_compressed(ap,**arrays)
        hashes[str(rp)]=sha(rp);hashes[str(ap)]=sha(ap)
        methods.append({'method':method,'selection':selection,'true_known_validation_accuracy':float(np.mean((true_val@weight.T+bias).argmax(-1)==true_labels)),
            'known_generated_validation_accuracy':float(np.mean((vs[0]@weight.T+bias).argmax(-1)==vs[2])),
            'known_clean_matched_validation_accuracy':float(np.mean((vs[1]@weight.T+bias).argmax(-1)==vs[2])),
            'known_source_anchor_count':len(source['train_lengths']),'readout_fit_rows':len(xs[index]),
            'additional_true_hidden_labels':0,'rows':rows})
    atomic_json(dest,{'source':name,'source_condition':rec['condition'],'seed':rec['seed'],'scope':scope,
        'source_folder':str(folder),'native_map_path':str(mp),'native_map_sha256':sha(mp),'known_feature_path':str(known),'known_feature_sha256':sha(known),
        'probe_feature_sha256':sha(fp),'methods':methods,'artifact_sha256':hashes,'completed_utc':now()})
    print(json.dumps({'known_route_readout_evaluated':name,'base_CI':[{'method':m['method'],'accuracy':next(r['answer_accuracy'] for r in m['rows'] if r['case']=='base_CI' and r['split']=='answer_collisions')} for m in methods]}),flush=True)


if __name__=='__main__':
    initialize()
    while time.time()<DEADLINE:
        locations=[(Path('results/algebra_hidden_relations'),'original_exploratory'),(Path('results/relation_seed_extension'),'additional_six_seeds')]
        locations.extend((p,p.name) for p in Path('results/relation_world_confirmation').glob('world*'))
        for folder,scope in locations:
            for path in (folder/'evaluations').glob('*.json'):
                if time.time()<DEADLINE:evaluate(path,scope)
        count=len(list((ROOT/'evaluations').glob('*.json')))
        atomic_json(ROOT/'state.json',{'status':'complete' if count==36 else 'waiting_or_evaluating','sources':count,'updated_utc':now()})
        if count==36:break
        time.sleep(min(45,max(0,DEADLINE-time.time())))
