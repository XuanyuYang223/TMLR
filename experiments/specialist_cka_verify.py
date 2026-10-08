"""Independent Gram-form verification of the new specialist controls."""
import json
from pathlib import Path

import numpy as np
import torch

from .longrun_engine import atomic_json
from .permworld_combinations import sha

ROOT=Path('results/specialist_cka_controls')


def residual(x,keys):
    x=np.asarray(x,dtype=np.float64).copy()
    groups={}
    for j,key in enumerate(keys):groups.setdefault(tuple(np.atleast_1d(key)),[]).append(j)
    for ids in groups.values():x[ids]-=x[ids].mean(0)
    return x


def gram_cka(x,y):
    x=x-x.mean(0);y=y-y.mean(0)
    if np.square(x).sum()<=1e-20 or np.square(y).sum()<=1e-20:return None
    device='cuda' if torch.cuda.is_available() else 'cpu'
    # Direct sample-by-sample kernels; independent from the producer's covariance form.
    a=torch.tensor(x,dtype=torch.float32,device=device);b=torch.tensor(y,dtype=torch.float32,device=device)
    k=a@a.T;l=b@b.T
    return float(((k*l).sum()/torch.sqrt(k.square().sum()*l.square().sum())).cpu())


def run():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    sig=json.loads((ROOT/'protocol.json').read_text())['signature'];d=dict(np.load(ROOT/'dataset.npz'))
    assert sha(ROOT/'dataset.npz')==json.loads((ROOT/'dataset_audit.json').read_text())['sha256']
    base=d['permutations'][:,0];n=d['lengths']
    for j,nn in enumerate(n):
        np.testing.assert_array_equal(d['permutations'][j,1,:nn],nn+1-base[j,:nn])
        np.testing.assert_array_equal(d['permutations'][j,2,:nn],np.argsort(base[j,:nn])+1)
    test=d['split']==2;tasks=list(d['tasks']);checks=[]
    for r in sig['relations']:
        ya=d['labels'][test,0,tasks.index(r['left'])];yb=d['labels'][test,:,tasks.index(r['right'])]
        keys=np.column_stack([n[test],ya,yb]);groups={}
        for j,key in enumerate(keys):groups.setdefault(tuple(key),[]).append(j)
        keep=np.array([len(groups[tuple(key)])>=sig['plan']['minimum_stratum_size'] for key in keys])
        for seed in sig['plan']['model_seeds']:
            rec=json.loads((ROOT/'evaluations'/f"{r['pair_id']}_s{seed}.json").read_text())
            for condition in ['trained','matched_init','independent_init']:
                ac='matched_init' if condition=='independent_init' else condition
                a=dict(np.load(ROOT/'features'/f"{ac}_{r['left']}_s{seed}.npz"))
                b=dict(np.load(ROOT/'features'/f"{condition}_{r['right']}_s{seed}.npz"))
                for branch in ['prefix_final_norm','query_final_norm','output_prob']:
                    for control in ['raw','length','answer_strata']:
                        for action in range(3):
                            x=a[branch][test,0].astype(np.float64);y=b[branch][test,action].astype(np.float64)
                            if control=='length':x=residual(x,n[test,None]);y=residual(y,n[test,None])
                            elif control=='answer_strata':x=residual(x[keep],keys[keep]);y=residual(y[keep],keys[keep])
                            actual=gram_cka(x,y)
                            expected=next(z['cka'] for z in rec['scores'] if z['condition']==condition and z['branch']==branch and z['control']==control and z['transform']==sig['plan']['transforms'][action])
                            if actual is None or expected is None:
                                assert actual is None and expected is None
                                checks.append(0.)
                            else:
                                np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-6)
                                checks.append(abs(actual-expected))
            print({'verified_relation':r['pair_id'],'seed':seed,'scores_checked':len(checks)},flush=True)
    for source in sig['sources']:assert sha(source['checkpoint'])==source['checkpoint_sha256']
    atomic_json(ROOT/'verification.json',{'status':'passed','gram_scores_checked':len(checks),
        'maximum_abs_covariance_vs_gram_error':float(max(checks)),'all_data_complement_inverse_states_verified':True,
        'source_checkpoint_hashes_verified':48,'scope':'All 24 relation/seed cells, trained and both initialization controls, prefix/query/output branches and raw/length/common-answer controls, all three input transforms. Gram calculation uses float32 with TF32 disabled, producer uses float64 covariance.'})


if __name__=='__main__':run()
