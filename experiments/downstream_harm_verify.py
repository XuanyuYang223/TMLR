"""Least-squares projection replay independent of SVD implementation."""
from pathlib import Path
import json
import numpy as np
from .downstream_harm_diagnostic_v2 import load,CONDITIONS
from .longrun_engine import atomic_json
from .two_step_relation_factorial import now
ROOT=Path('results/downstream_harm_diagnostic_v2')


def run():
    import torch
    torch.set_num_threads(1)
    results=json.loads((ROOT/'results.json').read_text());checks=0;maximum=0.
    for domain in ['permworld','matrix']:
        for i in range(6):
            for c in CONDITIONS:
                h,a,b,ba,bb,w,bw,data,k,_=load(domain,i,c)
                e=h[:,0]@a+ba-h[:,1]
                row=np.linalg.lstsq(w.T,e.T,rcond=1e-10)[0].T@w;null=e-row
                maximum=max(maximum,float(abs(null@w.T).max()))
                z=(h[:,1]@b+bb)@w.T+bw;dn=null@b@w.T;y=data['labels'][:,k]
                saved=dict(np.load(ROOT/f'{domain}_n{i}_{c}.npz'))
                np.testing.assert_allclose(np.square(null).sum(1),saved['null_error_sq'],atol=1e-9,rtol=1e-9)
                np.testing.assert_array_equal((z+dn).argmax(1)==y,saved['perturbed_correct'])
                # Independent explicit rival loop excludes the self class.
                positive=z.argmax(1)==y;ratio=np.zeros(len(y))
                for j in np.flatnonzero(positive):
                    rivals=[t for t in range(w.shape[0]) if t!=y[j]]
                    ratio[j]=max((dn[j,t]-dn[j,y[j]])/(z[j,y[j]]-z[j,t]) for t in rivals)
                np.testing.assert_allclose(ratio[positive],saved['worst_margin_ratio'][positive],atol=1e-8,rtol=1e-8)
                if not np.isfinite(saved['worst_margin_ratio'][positive]).all():raise AssertionError('Nonfinite positive-margin statistic')
                for sid,split in [(0,'iid'),(1,'collisions')]:
                    use=(data['split']==sid)&positive
                    r=next(x for x in results['records'] if (x['domain'],x['replicate'],x['condition'],x['split'])==(domain,i,c,split))
                    assert abs(r['null_margin_crossing_fraction']-np.mean(ratio[use]>=1))<1e-12
                checks+=6
    atomic_json(ROOT/'independent_verification.json',{'status':'complete','completed_utc':now(),'checks':checks,
        'independent_projection':'Least squares, not original SVD helper','maximum_first_logit_change':maximum,
        'self_class_removed_explicitly':True})
    print(json.dumps({'status':'complete','checks':checks,'maximum_first_logit_change':maximum}),flush=True)

if __name__=='__main__':run()
