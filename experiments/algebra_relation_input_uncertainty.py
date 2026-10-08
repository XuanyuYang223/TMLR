"""Supplement fixed-test seed intervals with source/whole-input cluster resampling."""
import argparse
import json
from pathlib import Path
import numpy as np
from .algebra_relation_verify import split_key
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now


def run(domain):
    root=Path('results/algebra_relation_v3')/domain
    d=dict(np.load(root/'dataset/test.npz'))
    labels=d['labels'][:,3]
    conditions=['both_correct','a_correct_b_wrong','a_wrong_b_correct','both_wrong','no_geometry','output_space']
    hits={}
    for c in conditions:
        hits[c]=np.stack([np.load(root/'evaluations'/f'n{i}_{c}.npz')['ab_answers']==labels for i in range(6)])
    records=[]
    for sid,split in [(0,'iid'),(1,'collisions')]:
        use=np.flatnonzero(d['split']==sid)
        for metric in ['accuracy','pair_both_correct']:
            if sid==0 and metric=='pair_both_correct':continue
            if sid==1:
                units=[np.flatnonzero(d['pair_ids']==i) for i in np.unique(d['pair_ids'][use])]
            else:units=[np.array([i]) for i in use]
            blocks=[{split_key(d['x'][i,0],domain,13) for i in unit} for unit in units]
            parent={}
            def find(k):
                parent.setdefault(k,k)
                if parent[k]!=k:parent[k]=find(parent[k])
                return parent[k]
            for block in blocks:
                first=next(iter(block))
                for other in block:
                    a,b=find(first),find(other)
                    if a!=b:parent[a]=b
            group_ids=[find(next(iter(block))) for block in blocks]
            keys=list(set(group_ids));groups=[np.flatnonzero(np.array([x==key for x in group_ids])) for key in keys]
            sizes=np.array([len(g) for g in groups])
            # Keep every collision pair intact. Polynomial pairs can join many blocks;
            # a connected component, rather than an individual pair, is the sampling unit.
            values={c:np.stack([hits[c][:,unit].all(1) if metric=='pair_both_correct' else hits[c][:,unit].mean(1)
                               for unit in units],axis=1).astype(float) for c in conditions}
            for c in conditions[1:]:
                diff=values['both_correct']-values[c]
                by_source=np.stack([(diff[i]+diff[i+3])/2 for i in range(3)])
                sums=np.stack([by_source[:,g].sum(1) for g in groups],axis=1)
                interval=None
                if len(groups)>=10:
                    rng=np.random.default_rng(261078991+sid*100+(metric=='pair_both_correct'))
                    source_ids=rng.integers(3,size=(10000,3));group_draws=rng.integers(len(groups),size=(10000,len(groups)))
                    draws=sums[source_ids[:,:,None],group_draws[:,None,:]].sum((1,2))/(3*sizes[group_draws].sum(1))
                    interval=(100*np.quantile(draws,[.025,.975])).tolist()
                records.append({'split':split,'metric':metric,'comparison':'both_correct-'+c,
                    'mean_pp':float(100*diff.mean()),'whole_input_components':len(groups),
                    'source_and_whole_input_bootstrap95_pp':interval,
                    'reason_no_joint_interval':None if interval is not None else 'Fewer than10 independent block-connected input components; do not treat linked pairs as independent.'})
    output=root.parent/f'{domain}_verification/input_uncertainty.json'
    atomic_json(output,{'status':'complete','completed_utc':now(),'records':records,
        'source_clusters':3,'bootstrap_draws':10000,'code_sha256':sha(__file__),
        'scope':'Supplementary uncertainty analysis, specified after matrix aggregate outcomes but before polynomial compound outcomes. Same endpoints, predictors, test sets and point estimates; no fitting or selection. Source and whole connected-input components resampled jointly.'})
    print(json.dumps({'status':'complete','domain':domain,'intervals':len(records)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('domain',choices=['matrix','polynomial']);args=parser.parse_args()
    run(args.domain)
