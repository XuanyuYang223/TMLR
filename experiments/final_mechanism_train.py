"""Final five-new-source confirmation; fixed ending, no outcome-driven expansion."""
import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import algebra_relation_campaign as core
from . import algebra_relation_worlds as worlds
from .algebra_relation_feasibility import apply_encoding
from .final_mechanism_solver import solve,affine_from_identity
from .longrun_engine import atomic_json
from .operator_capacity_confirmation import fit_predictor,cache_known_states
from .permworld_combinations import sha
from .two_step_relation_factorial import factorial_pairings,now

CONFIG=Path('configs/final_mechanism_confirmation.json')


def setup():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;apply_encoding('numeric_gelu')
    config=json.loads(CONFIG.read_text());root=Path(config['output'])
    for name in ['', 'dataset','sources','models','states','predictors','predictor_fits','affine_fits','evaluations','source_preparation']:
        (root/name).mkdir(parents=True,exist_ok=True)
    code=[str(CONFIG),__file__,'experiments/final_mechanism_solver.py',core.__file__,worlds.__file__,
          'experiments/operator_capacity_confirmation.py','experiments/algebra_relation_feasibility.py',
          'experiments/two_step_relation_factorial.py']
    signature={'config':config,'code_sha256':{str(p):sha(p) for p in code},
        'historical_test_sha256':{p:sha(p) for p in config['excluded_test_datasets']},
        'previous_delivery_sha256':{p:sha(p) for p in ['results/operator_capacity_confirmation/delivery.json','results/operator_mechanism_diagnostic/delivery.json']}}
    file=root/'protocol.json'
    if file.exists():assert json.loads(file.read_text())['signature']==signature
    else:
        atomic_json(file,{'registered_utc':now(),'new_source_training_started':False,'new_test_predictions_observed':False,'signature':signature})
        (root/'training_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
        (root/'solver_source_snapshot.py').write_bytes(Path('experiments/final_mechanism_solver.py').read_bytes())
    return config,root


def prepare(config,root):
    if (root/'data_audit.json').exists():return
    key=lambda x:worlds.key(tuple(x),'matrix',13)
    historical={key(x) for path in config['excluded_test_datasets'] for x in np.load(path)['x'][:,0]}
    rows=[tuple(x) for x in product(range(13),repeat=4) if (x[0]*x[3]-x[1]*x[2])%13]
    rng=np.random.default_rng(config['data_seed']);fresh=[x for x in rows if key(x) not in historical]
    used=set();collision=worlds.choose_collision_pairs(fresh,128,rng,'matrix',13,used)
    rng.shuffle(fresh);iid=[]
    for x in fresh:
        if key(x) not in used:
            iid.append(x);used.add(key(x))
            if len(iid)==256:break
    assert len(iid)==256
    rng.shuffle(rows);available=defaultdict(list)
    for x in rows:
        if key(x) not in used:available[key(x)].append(x)
    keys=list(available);rng.shuffle(keys)
    validation=worlds.encode([available[k][0] for k in keys[:128]],'matrix',13)
    pilot=worlds.encode([available[k][0] for k in keys[128:192]],'matrix',13)
    source=worlds.encode([available[k][0] for k in keys[192:704]],'matrix',13)
    excluded=used|set(keys[:704]);pool=[x for x in rows if key(x) not in excluded]
    parts={'source':source,'validation':validation,'pilot_validation':pilot}
    for i in range(len(config['source_seeds'])):
        anchors=worlds.paired_fit(pool,1024,np.random.default_rng(config['data_seed']+101+i),'matrix',13)
        d=worlds.encode(anchors,'matrix',13)
        pc,pi,eligible=factorial_pairings(np.ones(len(anchors),int),d['labels'],config['pairing_seed']+i)
        assert eligible.all();d.update(pc=pc,pi=pi,eligible=eligible);parts[f'support{i}']=d
    test=worlds.encode(iid+collision,'matrix',13,['','a','b','ab'])
    test.update(split=np.array([0]*256+[1]*256),pair_ids=np.array([-1]*256+[j for j in range(128) for _ in range(2)]))
    parts['test']=test
    sets={n:{key(x) for x in d['x'][:,0]} for n,d in parts.items()}
    train=set.union(sets['source'],*[sets[f'support{i}'] for i in range(len(config['source_seeds']))])
    val=sets['validation']|sets['pilot_validation']
    assert not train&val and not train&sets['test'] and not val&sets['test'] and not sets['test']&historical
    for n,d in parts.items():np.savez_compressed(root/'dataset'/f'{n}.npz',**d)
    atomic_json(root/'data_audit.json',{'created_utc':now(),'historical_test_orbits_excluded_from_new_test':len(historical),
        'new_test_orbits':len(sets['test']),'new_whole_orbit_partitions_disjoint':True,
        'historical_training_orbits_may_recur':True,'dataset_sha256':{str(p):sha(p) for p in (root/'dataset').glob('*.npz')}})


def sources(config,root):
    common=json.loads(Path('configs/algebra_relation_common_v3.json').read_text())
    for i,seed in enumerate(config['source_seeds']):
        local=root/'source_preparation'/f's{i}'
        for name in ['dataset','sources','fits','models']:(local/name).mkdir(parents=True,exist_ok=True)
        for name in ['source','validation','pilot_validation']:
            target=local/'dataset'/f'{name}.npz'
            if not target.exists():target.symlink_to((root/'dataset'/f'{name}.npz').resolve())
        target=local/'dataset/support0.npz'
        if not target.exists():target.symlink_to((root/'dataset'/f'support{i}.npz').resolve())
        plan={**common,'source_seeds':[seed]*3}
        atomic_json(local/'plan.json',{'logical_source':i,'actual_independent_seed':seed,'local_fit_index':0,'plan':plan})
        atomic_json(root/'state.json',{'status':'running','stage':'independent_source_preparation','source':i,'seed':seed,'updated_utc':now()})
        core.fit(plan,local,0,'both_correct','cuda')
        for target,dest in [(root/'sources'/f's{i}.pt',local/'sources/s0.pt'),
                            (root/'models'/f'n{i}_both_correct.pt',local/'models/n0_both_correct.pt')]:
            if not target.exists():target.symlink_to(dest.resolve())


def fit_all(config,root):
    for i in range(len(config['source_seeds'])):
        for condition in config['conditions']:
            atomic_json(root/'state.json',{'status':'running','stage':'matched_budget_first_operators','source':i,'condition':condition,'updated_utc':now()})
            fit_predictor(config,root,i,condition,'cuda')
    for i in range(len(config['source_seeds'])):
        z=cache_known_states(root,i,'cuda');h=z['train'][:,0].astype(float);target=z['train'][:,1].astype(float)
        with torch.no_grad():logits=F.linear(torch.as_tensor(z['train'][:,1],device='cuda'),torch.as_tensor(z['w'],device='cuda'),torch.as_tensor(z['bias_w'],device='cuda')).cpu().numpy()
        np.savez_compressed(root/'states'/f's{i}_fixed_supervision.npz',teacher_logits=logits)
        scale=json.loads((root/'predictor_fits'/f's{i}_linear_correct.json').read_text())['geometry_scale']
        pc=np.load(root/'dataset'/f'support{i}.npz')['pc']
        for pairing in ['correct','wrong']:
            state=torch.load(root/'predictors'/f's{i}_linear_{pairing}.pt',map_location='cpu',weights_only=True)['state_dict']
            initial=affine_from_identity(state)
            for kind in config['linear_solvers']:
                record=root/'affine_fits'/f's{i}_{pairing}_{kind}.json'
                if record.exists():continue
                atomic_json(root/'state.json',{'status':'running','stage':'same_loss_affine_convergence','source':i,'pairing':pairing,'solver':kind,'updated_utc':now()})
                theta,metadata=solve(h,target if pairing=='correct' else target[pc],logits.astype(float),z['w'].astype(float),z['bias_w'].astype(float),scale,config,
                                     None if kind=='ols_initialization' else initial)
                cp=root/'affine_fits'/f's{i}_{pairing}_{kind}.npz';np.savez_compressed(cp,theta=theta)
                atomic_json(record,{'status':'complete','completed_utc':now(),'source':i,'pairing':pairing,'kind':kind,
                    'metadata':metadata,'geometry_scale':scale,'checkpoint_sha256':sha(cp),
                    'fixed_supervision_sha256':sha(root/'states'/f's{i}_fixed_supervision.npz')})
                print(json.dumps({'source':i,'pairing':pairing,'solver':kind,**{k:metadata[k] for k in ['objective','gap_bound','independent_gap_bound','converged','iterations']}}),flush=True)


def run(stage):
    config,root=setup();prepare(config,root)
    if stage=='prepare':return
    if stage in ['run','sources']:sources(config,root)
    if stage in ['run','fits']:fit_all(config,root)
    if stage in ['run','evaluate']:
        from .final_mechanism_evaluate import evaluate
        evaluate(config,root)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','sources','fits','evaluate','run'],default='run')
    run(parser.parse_args().stage)
