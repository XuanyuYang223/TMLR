"""Visible-only encoding checks; compound outcomes stay closed."""
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn

from . import algebra_relation_campaign as campaign
from . import algebra_relation_worlds as worlds
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now


class Sine(nn.Module):
    def forward(self,x):return torch.sin(x)


def apply_encoding(encoding):
    class NumericEncoder(nn.Module):
        def __init__(self,p,width):
            super().__init__()
            self.encoder=nn.Sequential(nn.Linear(4,width),Sine() if encoding=='numeric_sine' else nn.GELU(),
                                       nn.Linear(width,width),nn.LayerNorm(width))
            self.readout=nn.Linear(width,p)

        def forward(self,x):return self.encoder(x)

    def numeric_feature(x,p):
        return (np.asarray(x,dtype=np.float32)-(p-1)/2)*(2*np.pi/p)

    campaign.Encoder=NumericEncoder
    worlds.feature=numeric_feature


def run():
    torch.set_num_threads(1);device='cuda' if torch.cuda.is_available() else 'cpu'
    plan=json.loads(Path('configs/algebra_relation_common.json').read_text())
    base=Path('results/algebra_relation_followup/matrix').resolve()
    root=Path('results/algebra_relation_feasibility');root.mkdir(exist_ok=True)
    signature={'code_sha256':sha(__file__),'original_plan_sha256':sha(campaign.CONFIG),
               'data_sha256':{str(p):sha(p) for p in (base/'dataset').glob('*.npz')},
               'candidates':['numeric_gelu','numeric_sine'],'selection':'First candidate with native>=95% and both generators>=90% on the64 known pilot examples, confirmed on128 separately reserved known examples. Same900-epoch pilot and500-epoch source. Never evaluate compound inputs or labels.',
               'reason':'Original model fits both generators at99.8% on train but inverse-like second generator is7.8% on known heldout inputs. Distinguish generalization from optimizer failure. Remove the one-hot lookup shortcut; retain scalar field values. Sine activation is a generic periodic prior, with no group operations or polynomial transforms in the model.'}
    file=root/'protocol.json'
    if file.exists():assert json.loads(file.read_text())['signature']==signature
    else:
        atomic_json(file,{'registered_utc':now(),'compound_outcomes_observed':False,'signature':signature})
        (root/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    results=[];selected=None
    for candidate in signature['candidates']:
        folder=root/candidate;folder.mkdir(exist_ok=True)
        for name in ['sources','fits','models']:(folder/name).mkdir(exist_ok=True)
        if not (folder/'dataset').exists():(folder/'dataset').symlink_to(base/'dataset',target_is_directory=True)
        apply_encoding(candidate)
        record=campaign.fit(plan,folder,0,'both_correct',device,True)
        grade=record['curve'][-1]
        model=campaign.Encoder(plan['field_prime'],plan['hidden_width']).to(device)
        state=torch.load(folder/'models/pilot.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state['model']);ops=campaign.Operators(plan['hidden_width']).to(device);ops.load_state_dict(state['operators'])
        _,x,y=campaign.arrays(folder/'dataset/validation.npz',plan['field_prime'],device)
        confirm=campaign.known_grade(model,ops,x,y)
        passed=all(min(g['native'])>=.95 and min(g['generators'])>=.9 for g in [grade,confirm])
        results.append({'candidate':candidate,'pilot_grade':grade,'confirmation_grade':confirm,'passed':passed})
        atomic_json(root/'results.json',{'status':'running','results':results,'selected':selected,'compound_test_closed':True})
        if passed:selected=candidate;break
    atomic_json(root/'results.json',{'status':'complete','completed_utc':now(),'results':results,'selected':selected,
                'compound_test_closed':True,'source_data_reused':True,'known_validation_used_for_architecture_selection':True})
    print(json.dumps({'status':'complete','selected':selected,'results':results}),flush=True)


if __name__=='__main__':run()
