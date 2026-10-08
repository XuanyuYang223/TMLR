"""Independent teacher-only variance and readout-nullspace verification."""
import json
from pathlib import Path
import numpy as np
import torch
from .longrun_engine import atomic_json
from .two_step_relation_factorial import now
from .permworld_combinations import sha
ROOT=Path('results/readout_null_confirmation')


def run():
    from . import algebra_relation_campaign as matrix
    from .algebra_relation_feasibility import apply_encoding
    torch.set_num_threads(1);apply_encoding('numeric_gelu');rows=[]
    for domain in ['matrix','permworld']:
        root=ROOT/domain
        for i in range(3):
            if domain=='matrix':
                state=torch.load(root/'sources'/f's{i}.pt',map_location='cpu',weights_only=True)['model']
                model=matrix.Encoder(13,128).cuda();model.load_state_dict(state);model.eval()
                _,x,_=matrix.arrays(root/'dataset'/f'support{i}.npz',13,'cuda')
                with torch.no_grad():h=model(x).cpu().numpy()
                w=state['readout.weight'].numpy().astype(float);ddof=1
            else:
                archive=dict(np.load(root/'features'/f'n{i}_support.npz'))
                pair=dict(np.load(root/'dataset'/f'n{i}/support/pairings.npz'))
                h=archive['train_hidden'][pair['eligible']];w=archive['readout_weight'].astype(float);ddof=0
            # Minimum-norm solution to W*P=W is its orthogonal row projector.
            p=np.linalg.lstsq(w,w,rcond=1e-10)[0];q=np.eye(w.shape[1])-p
            np.testing.assert_allclose(w@q,0,atol=1e-10,rtol=0)
            full=float(h.var(axis=(0,1),ddof=ddof).mean());null=float((h@q).var(axis=(0,1),ddof=ddof).mean())
            for c in ['full_correct','full_wrong','null_correct','null_wrong']:
                name=f'n{i}_{c}' if domain=='matrix' else f'n{i}_hidden_{c}'
                record=json.loads((root/'fits'/f'{name}.json').read_text())
                expected=null if c.startswith('null') else full
                np.testing.assert_allclose(record['normalization_scale'],expected,atol=1e-5,rtol=1e-5)
            rows.append({'domain':domain,'source':i,'full_variance':full,'null_variance':null,
                         'null_weight_relative_to_original_coordinate_penalty':full/null,'rank':int(round(np.trace(p))),
                         'all_four_recorded_scales_verified':True,'maximum_null_logit_change':float(abs(w@q).max())})
    atomic_json(ROOT/'loss_scale_verification.json',{'status':'complete','completed_utc':now(),'records':rows,
        'checks':24,'method':'Independent least-squares projector; teacher-only train data; no compound targets/labels or test-based normalization.'})
    print(json.dumps({'status':'complete','checks':24,'records':rows}),flush=True)

if __name__=='__main__':run()
