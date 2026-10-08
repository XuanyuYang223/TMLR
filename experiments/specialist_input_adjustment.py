"""Within-length invariant-input Gram regression and PSD projection."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .analysis import write_csv
from .longrun_engine import atomic_json
from .permworld_combinations import sha

CONFIG=Path('configs/specialist_input_adjustment.json')


def center(x,keys):
    z=np.asarray(x,dtype=np.float64).copy()
    _,inv=np.unique(keys,axis=0,return_inverse=True)
    for g in np.unique(inv):
        ix=inv==g;z[ix]-=z[ix].mean(0)
    return z


def adjust(k,g):
    alpha=float(np.sum(k*g)/np.sum(g*g))
    residual=(k-alpha*g);residual=(residual+residual.T)/2
    e,u=np.linalg.eigh(residual)
    psd=(u*np.maximum(e,0))@u.T
    return psd,alpha,float(np.maximum(-e,0).sum())


def alignment(a,b):
    den=float(np.sqrt(np.square(a).sum()*np.square(b).sum()))
    return float(np.sum(a*b)/den) if den>1e-20 else None


def run():
    plan=json.loads(CONFIG.read_text());root=Path(plan['output']);root.mkdir(exist_ok=True,parents=True)
    parent=Path(plan['parent']);sig=json.loads((parent/'protocol.json').read_text())['signature']
    signature={'plan':plan,'config_sha256':sha(CONFIG),'code_sha256':sha(__file__),
        'parent_protocol_sha256':sha(parent/'protocol.json'),'parent_dataset_sha256':sha(parent/'dataset.npz')}
    p=root/'protocol.json'
    if p.exists():assert json.loads(p.read_text())['signature']==signature
    else:atomic_json(p,{'registered_utc':datetime.now(timezone.utc).isoformat(),'new_adjusted_scores':0,'signature':signature})
    d=dict(np.load(parent/'dataset.npz'));test=d['split']==2;tasks=list(d['tasks']);rows=[];summary=[]
    for r in sig['relations']:
        ya=d['labels'][test,0,tasks.index(r['left'])];yb=d['labels'][test,:,tasks.index(r['right'])]
        ns=d['lengths'][test];keys=np.column_stack([ns,ya,yb]);_,inv,count=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
        retained=count[inv]>=sig['plan']['minimum_stratum_size'];inputs=d['permutations'][test]
        correct=sig['plan']['transforms'].index(r['input_transform']);wrong=1 if correct==2 else 2
        input_grams={}
        for control in plan['controls']:
            for n in range(10,31):
                use=(ns==n)&(retained if control=='answer_strata' else True)
                mats=[]
                groupkeys=keys[use] if control=='answer_strata' else ns[use,None]
                for a in range(3):
                    x=np.eye(n)[inputs[use,a,:n]-1].reshape(use.sum(),n*n)
                    x=center(x,groupkeys);mats.append(x@x.T)
                np.testing.assert_allclose(mats[0],mats[1],atol=1e-12)
                np.testing.assert_allclose(mats[0],mats[2],atol=1e-12)
                input_grams[control,n]=(use,groupkeys,mats[0])
        for seed in sig['plan']['model_seeds']:
            for condition in plan['conditions']:
                ac='matched_init' if condition=='independent_init' else condition
                a=dict(np.load(parent/'features'/f"{ac}_{r['left']}_s{seed}.npz"))
                b=dict(np.load(parent/'features'/f"{condition}_{r['right']}_s{seed}.npz"))
                for branch in plan['branches']:
                    for control in plan['controls']:
                        values=[]
                        for n in range(10,31):
                            use,key,g=input_grams[control,n]
                            x=center(a[branch][test,0][use],key);k=x@x.T;pk,ax,negx=adjust(k,g)
                            ordinary=[];adjusted=[]
                            for action in range(3):
                                y=center(b[branch][test,action][use],key);l=y@y.T;pl,ay,negy=adjust(l,g)
                                raw=alignment(k,l);value=alignment(pk,pl);ordinary.append(raw);adjusted.append(value)
                                rows.append({'relation':r['pair_id'],'seed':seed,'condition':condition,'branch':branch,'control':control,
                                    'length':n,'anchors':int(use.sum()),'transform':sig['plan']['transforms'][action],
                                    'ordinary_cka':raw,'input_adjusted_cka':value,'input_regression_alpha_left':ax,'input_regression_alpha_right':ay,
                                    'removed_negative_eigenvalue_sum_left':negx,'removed_negative_eigenvalue_sum_right':negy})
                            values.append({'ordinary_contrast':ordinary[correct]-ordinary[wrong],
                                'adjusted_contrast':adjusted[correct]-adjusted[wrong]})
                        summary.append({'relation':r['pair_id'],'seed':seed,'condition':condition,'branch':branch,'control':control,
                            'ordinary_correct_minus_wrong':float(np.mean([v['ordinary_contrast'] for v in values])),
                            'input_adjusted_correct_minus_wrong':float(np.mean([v['adjusted_contrast'] for v in values]))})
            print({'input_adjustment_complete':r['pair_id'],'seed':seed},flush=True)
    write_csv(root/'per_length_scores.csv',rows);write_csv(root/'per_source_contrasts.csv',summary)
    atomic_json(root/'state.json',{'status':'complete','score_rows':len(rows),'source_contrasts':len(summary),
        'all_fixed_length_input_grams_verified_invariant':True,'completed_utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':run()
