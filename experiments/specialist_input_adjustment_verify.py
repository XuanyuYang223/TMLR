"""Torch spectral verification at three fixed lengths, all source conditions."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .longrun_engine import atomic_json

ROOT=Path('results/specialist_input_adjustment')
PARENT=Path('results/specialist_cka_controls')


def within(x,keys):
    y=np.asarray(x,dtype=np.float64).copy();groups={}
    for j,key in enumerate(keys):groups.setdefault(tuple(key),[]).append(j)
    for ids in groups.values():y[ids]-=y[ids].mean(0)
    return y


def adjusted(x,g):
    k=torch.tensor(x,dtype=torch.float64);k=k@k.T
    coeff=(k*g).sum()/g.square().sum()
    z=k-coeff*g;e,q=torch.linalg.eigh(z)
    return (q*e.clamp(min=0).unsqueeze(0))@q.T


def run():
    torch.set_num_threads(2);d=dict(np.load(PARENT/'dataset.npz'));test=d['split']==2
    sig=json.loads((PARENT/'protocol.json').read_text())['signature'];tasks=list(d['tasks']);n=d['lengths'][test]
    with (ROOT/'per_length_scores.csv').open() as f:rows=list(csv.DictReader(f))
    lookup={(r['relation'],int(r['seed']),r['condition'],r['branch'],r['control'],int(r['length']),r['transform']):float(r['input_adjusted_cka']) for r in rows}
    checks=[]
    for rel in sig['relations']:
        labels=np.column_stack([n,d['labels'][test,0,tasks.index(rel['left'])],d['labels'][test,:,tasks.index(rel['right'])]])
        counts={}
        for key in labels:counts[tuple(key)]=counts.get(tuple(key),0)+1
        retained=np.array([counts[tuple(k)]>=3 for k in labels])
        for seed in [17,42,101]:
            for cond in ['trained','matched_init','independent_init']:
                ac='matched_init' if cond=='independent_init' else cond
                a=dict(np.load(PARENT/'features'/f"{ac}_{rel['left']}_s{seed}.npz"))
                b=dict(np.load(PARENT/'features'/f"{cond}_{rel['right']}_s{seed}.npz"))
                for branch in ['prefix_final_norm','query_final_norm']:
                    for control in ['length','answer_strata']:
                        for nn in [10,20,30]:
                            use=(n==nn)&(retained if control=='answer_strata' else True)
                            key=labels[use] if control=='answer_strata' else n[use,None]
                            p=d['permutations'][test,0][use,:nn]
                            mat=np.zeros((len(p),nn,nn))
                            for j,value in enumerate(p):mat[j,np.arange(nn),value-1]=1
                            z=within(mat.reshape(len(p),-1),key);g=torch.tensor(z@z.T,dtype=torch.float64)
                            pk=adjusted(within(a[branch][test,0][use],key),g)
                            for j,t in enumerate(sig['plan']['transforms']):
                                pl=adjusted(within(b[branch][test,j][use],key),g)
                                value=float((pk*pl).sum()/torch.sqrt(pk.square().sum()*pl.square().sum()))
                                expected=lookup[rel['pair_id'],seed,cond,branch,control,nn,t]
                                np.testing.assert_allclose(value,expected,atol=1e-10,rtol=1e-10);checks.append(abs(value-expected))
            print({'input_adjustment_verified':rel['pair_id'],'seed':seed},flush=True)
    atomic_json(ROOT/'verification.json',{'status':'passed','spectral_scores_verified':len(checks),
        'lengths':[10,20,30],'maximum_numpy_vs_torch_spectral_difference':float(max(checks)),
        'scope':'All 24 relation/seed cells, both positions, all initialization conditions and both length/answer controls; three specified lengths and all three transforms. Independently constructed transposed permutation-matrix encoding and Torch float64 eigendecomposition versus NumPy producer.'})


if __name__=='__main__':run()
