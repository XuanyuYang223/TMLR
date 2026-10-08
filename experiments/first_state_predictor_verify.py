"""Independently recompute saved residual predictor and compound endpoints."""
import json
from pathlib import Path
import numpy as np
import torch
from .first_state_predictor_diagnostic_v2 import Residual
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now
ROOT=Path('results/first_state_predictor_diagnostic_v2')


def run():
    torch.set_num_threads(1);results=json.loads((ROOT/'results.json').read_text())['records'];checks=0;maximum=0.
    old=json.loads(Path('results/first_state_predictor_diagnostic/results.json').read_text())['records']
    for r in old:
        rr=next(s for s in results if all(s[k]==r[k] for k in ['domain','source','kind','split']))
        for k in r:
            if isinstance(r[k],float):assert abs(r[k]-rr[k])<1e-12;checks+=1
    for domain in ['matrix','permworld']:
        for i in range(3):
            d=dict(np.load(ROOT/f'{domain}_s{i}_states.npz'));z=d['test'];k=3 if domain=='matrix' else 5
            scale=float(np.square(d['train'][:,1]-d['train'][:,1].mean(0)).sum(1).mean())
            for kind in ['original','affine_residual','nonlinear_residual']:
                if kind=='original':pred=z[:,0]@d['rho_a']+d['bias_a']
                else:
                    state=torch.load(ROOT/f'{domain}_s{i}_{kind}.pt',map_location='cpu',weights_only=True)
                    assert state['steps']==2000 and state['batch_size']==128
                    model=Residual(d['rho_a'],d['bias_a'],state['nonlinear']).cuda();model.load_state_dict(state['state_dict']);model.eval()
                    with torch.no_grad():pred=model(torch.as_tensor(z[:,0],dtype=torch.float32,device='cuda')).cpu().numpy().astype(float)
                    assert state['parameters']==sum(p.numel() for p in model.parameters())
                saved=dict(np.load(ROOT/f'{domain}_s{i}_{kind}_predictions.npz'))
                maximum=max(maximum,float(abs(pred-saved['first_pred']).max()))
                np.testing.assert_allclose(pred,saved['first_pred'],atol=1e-8,rtol=1e-8)
                answers=(((pred@d['rho_b']+d['bias_b'])@d['readout_weight'].T)+d['readout_bias']).argmax(1)
                np.testing.assert_array_equal(answers,saved['compound_answers']);use=d['split']==1;hit=answers==d['labels'][:,k]
                row=next(s for s in results if (s['domain'],s['source'],s['kind'],s['split'])==(domain,i,kind,'test'))
                assert row['collision_compound_accuracy']==hit[use].mean()
                both=np.mean([hit[d['pair_ids']==p].all() for p in np.unique(d['pair_ids'][use])])
                assert row['collision_pair_both_correct']==both
                nmse=float(np.square(pred-z[:,1]).sum(1).mean()/scale)
                assert abs(nmse-row['hidden_nmse'])<1e-10;checks+=5
    atomic_json(ROOT/'independent_verification.json',{'status':'complete','completed_utc':now(),'checks':checks,
        'maximum_replay_difference':maximum,'same_fixed_budget_as_initial_diagnostic':True,'original_all_metrics_identical':True,
        'artifacts':{str(p):sha(p) for p in ROOT.glob('*') if p.is_file() and p.name!='independent_verification.json'}})
    print(json.dumps({'status':'complete','checks':checks,'maximum_replay_difference':maximum}),flush=True)

if __name__=='__main__':run()
