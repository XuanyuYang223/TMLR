"""New ordinary PermWorld sources, without changing archived trainers."""
import json
from pathlib import Path
import numpy as np
import torch
from . import hidden_relation_train as old
from .native_confirmation import setup
from .longrun_engine import atomic_json
from .permworld_combinations import sha
CONFIG=Path('configs/readout_null_confirmation.json')

def run():
    plan=json.loads(CONFIG.read_text());parent,config,_=old.load_plan();root=Path(plan['output'])/'permworld_sources'
    for name in ['','source','checkpoints']:(root/name).mkdir(parents=True,exist_ok=True)
    parent.update(output=str(root),source_seeds=plan['source_seeds'],conditions=['ordinary'],source_steps=plan['permworld']['ordinary_steps'])
    for name in ['source_data.npz','training_orbit_audit.npz']:
        p=root/name
        if not p.exists():p.symlink_to((Path('results/algebra_hidden_relations')/name).resolve())
    signature={'plan':parent,'config_sha256':sha(CONFIG),'code_sha256':sha(__file__),'ordinary_trainer_sha256':sha(old.__file__),
               'source_data_sha256':sha(root/'source_data.npz'),'only_known_states':True,'new_test_outcomes_observed':False}
    pp=root/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())['signature']==signature
    else:atomic_json(pp,{'registered_utc':old.now(),'signature':signature})
    _,_,tokens,_=setup(config)
    torch.set_num_threads(2)
    raw=dict(np.load(root/'source_data.npz'));data={k:torch.as_tensor(v,device='cuda') for k,v in raw.items()}
    for seed in plan['source_seeds']:old.train_one(parent,config,root,tokens,data,raw,seed,'ordinary','cuda')
    atomic_json(root/'completion.json',{'status':'complete','completed_utc':old.now(),'sources':plan['source_seeds']})

if __name__=='__main__':run()
