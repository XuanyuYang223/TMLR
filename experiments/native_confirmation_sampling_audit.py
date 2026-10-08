"""Replay source minibatch RNG and check saved final checkpoints, no target data."""
from datetime import datetime,timezone
import hashlib
import json

import numpy as np
import torch

from .native_confirmation import paths,setup
from .permworld_combinations import new_model,sha
from .longrun_engine import atomic_json


def audit():
    plan,config,root=paths();setup(config)
    with np.load(root/'dataset/data.npz') as data:
        lengths=data['train_lengths']
    buckets={n:np.flatnonzero(lengths==n) for n in np.unique(lengths)}
    schedules={}
    for seed in plan['model_seeds']:
        rng=np.random.default_rng(seed+20261005);digest=hashlib.sha256()
        for _ in range(plan['source_steps']):
            n=int(rng.integers(config['lengths'][0],config['lengths'][1]+1))
            indices=rng.choice(buckets[n],plan['examples_per_task_per_step'],replace=True)
            digest.update(np.array([n],dtype=np.int64).tobytes());digest.update(indices.tobytes())
        model=new_model({**config,**{k:plan['architecture'][k] for k in ('d_model','layers','heads')}},seed,'cpu')
        initial=hashlib.sha256()
        for name,value in sorted(model.state_dict().items()):
            initial.update(name.encode());initial.update(value.detach().numpy().tobytes())
        schedules[str(seed)]={'sample_sequence_sha256':digest.hexdigest(),'final_generator_state':rng.bit_generator.state,
                              'initial_parameter_sha256':initial.hexdigest()}
        del model
    checked=[]
    for phase in ('multi','single'):
        for path in sorted((root/phase).glob('*.json')):
            record=json.loads(path.read_text())
            if record['status']!='complete':continue
            checkpoint=root/phase/'checkpoints'/f"{record['job_id']}.pt"
            assert sha(checkpoint)==record['checkpoint_sha256']
            state=torch.load(checkpoint,weights_only=True,map_location='cpu')
            seed=record['job']['seed']
            assert state['step']==plan['source_steps']
            assert state['generator_state']==schedules[str(seed)]['final_generator_state']
            checked.append({'phase':phase,'job_id':record['job_id'],'seed':seed,'actual_saved_rng_matches_replay':True})
            del state
    result={'status':'passed_for_complete_source_checkpoints','checked_utc':datetime.now(timezone.utc).isoformat(),
            'completed_source_models_checked':len(checked),'data_sha256':sha(root/'dataset/data.npz'),
            'schedule_policy':'same seed + 20261005; uniform length, 32 indices with replacement; same across all source task conditions',
            'initialization_scope':'expected same-architecture initial parameter hashes replayed from seeds; frozen new_model code used by all actual jobs',
            'schedules':schedules,'checkpoint_checks':checked,'verifier_sha256':sha(__file__)}
    atomic_json(root/'source_sampling_verification.json',result)
    print(json.dumps({'status':result['status'],'checked_models':len(checked),'seeds':list(schedules)}))


if __name__=='__main__':audit()
