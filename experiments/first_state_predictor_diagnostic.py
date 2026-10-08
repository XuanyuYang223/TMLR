"""Fixed-budget capacity diagnostic using old single-step targets only."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from .permworld_combinations import sha
from .longrun_engine import atomic_json
from .two_step_relation_factorial import now
from .downstream_harm_diagnostic_v2 import load
ROOT=Path('results/first_state_predictor_diagnostic')


def primitive_states(domain,i):
    if domain=='matrix':
        from . import algebra_relation_campaign as core
        from .algebra_relation_feasibility import apply_encoding
        apply_encoding('numeric_gelu');root=Path('results/algebra_relation_v3/matrix')
        model=core.Encoder(13,128).cuda();state=torch.load(root/'models'/f'n{i}_both_correct.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state['model']);model.eval()
        with torch.no_grad():
            out=[]
            for p in [root/'dataset'/f'support{i}.npz',root/'dataset/validation.npz']:
                _,x,_=core.arrays(p,13,'cuda');out.append(model(x).cpu().numpy())
        return out
    from . import two_step_relation_factorial as core
    from .hidden_relation_train import load_plan
    from .native_confirmation import setup
    parent,config,_=load_plan();_,_,tokens,_=setup(config)
    root=Path('results/two_step_relation_dose_confirmation');sig=json.loads((root/'protocol.json').read_text())['signature']
    model=core.source_model(sig['sources'][i],parent,config,'cuda')
    state=torch.load(root/'maps'/f'n{i}_hidden_both_correct_e20.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state['model']);raw=dict(np.load(root/'dataset'/f'n{i}/support/dataset.npz'))
    ids=np.flatnonzero(raw['split']==0);rng=np.random.default_rng(261079901+i);ids=rng.choice(ids,4096,replace=False)
    val=np.flatnonzero(raw['split']==1)
    return [core.hidden_features(model,{k:raw[k][ix] for k in ['input','lengths','labels']},parent['source_task'],tokens) for ix in [ids,val]]


class Residual(nn.Module):
    def __init__(self,a,b,nonlinear):
        super().__init__();self.register_buffer('a',torch.as_tensor(a,dtype=torch.float32));self.register_buffer('b',torch.as_tensor(b,dtype=torch.float32))
        d=len(a)
        self.residual=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d)) if nonlinear else nn.Linear(d,d)
        last=self.residual[-1] if nonlinear else self.residual
        nn.init.zeros_(last.weight);nn.init.zeros_(last.bias)
    def forward(self,h):return h@self.a+self.b+self.residual(h)


def run():
    torch.set_num_threads(1);ROOT.mkdir(exist_ok=True,parents=True)
    pp=ROOT/'protocol.json';signature={'code_sha256':sha(__file__),'posthoc_old_data':True,
        'models':'both-correct sources0/1/2 in both domains','fit_targets':'base to first-generator state; no composite targets',
        'predictors':['unchanged original affine','affine residual','one-hidden-layer GELU residual'],
        'budget':'2000 fixed AdamW steps,128 samples/step,lr0.001,wd0.0001; no heldout selection',
        'limits':'Nonlinear residual has more parameters; capacity/optimization diagnostic, not a matched causal control. Test outputs used only after fitting.'}
    if pp.exists():assert json.loads(pp.read_text())['signature']==signature
    else:atomic_json(pp,{'registered_utc':now(),'signature':signature})
    rows=[]
    for domain in ['matrix','permworld']:
        for i in range(3):
            ht,hv=primitive_states(domain,i)
            h,a,b,ba,bb,w,bw,data,k,_=load(domain,i,'both_correct')
            trainx=torch.as_tensor(ht[:,0],device='cuda',dtype=torch.float32);target=torch.as_tensor(ht[:,1],device='cuda',dtype=torch.float32)
            scale=float(np.square(ht[:,1]-ht[:,1].mean(0)).sum(1).mean())
            predictors={'original':None}
            for kind in ['affine_residual','nonlinear_residual']:
                torch.manual_seed(261079501+i);model=Residual(a,ba,kind=='nonlinear_residual').cuda()
                opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);rng=np.random.default_rng(261079801+i)
                for step in range(2000):
                    ids=torch.as_tensor(rng.integers(0,len(ht),128),device='cuda')
                    loss=(model(trainx[ids])-target[ids]).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                model.eval();predictors[kind]=model
            for kind,model in predictors.items():
                for split,z in [('train',ht),('validation',hv),('test',h)]:
                    with torch.no_grad():pred=z[:,0]@a+ba if model is None else model(torch.as_tensor(z[:,0],device='cuda',dtype=torch.float32)).cpu().numpy().astype(float)
                    err=pred-z[:,1]
                    from .relation_error_localization import row_projection
                    en=err@(np.eye(len(a))-row_projection(w))
                    row={'domain':domain,'source':i,'kind':kind,'split':split,'hidden_nmse':float(np.square(err).sum(1).mean()/scale),
                         'null_nmse':float(np.square(en).sum(1).mean()/scale)}
                    if split=='test':
                        use=data['split']==1;truth=data['labels'][:,k]
                        row.update(single_accuracy=float(((pred@w.T+bw).argmax(1)==data['labels'][:,1]).mean()),
                            collision_compound_accuracy=float((((pred@b+bb)@w.T+bw).argmax(1)[use]==truth[use]).mean()))
                    rows.append(row)
            print(json.dumps({'domain':domain,'source':i,'status':'diagnosed'}),flush=True)
    atomic_json(ROOT/'results.json',{'status':'complete','completed_utc':now(),'records':rows,'capacity_matched':False})
    print(json.dumps({'status':'complete','test_records':[r for r in rows if r['split']=='test']}),flush=True)

if __name__=='__main__':run()
