"""Joint-task transformation types under exactly matched source statistics."""
import argparse
from datetime import datetime
import hashlib
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .algebra import categorical_mi, projective, rank_mod, world
from .analysis import linear_cka, write_csv
from .longrun_engine import atomic_json, atomic_torch
from .models import SourceModel


def group_sources(name, p):
    basis=np.eye(4,dtype=np.int64)
    if name=='P':return basis
    coefficient=int(name[1:])
    return np.vstack([basis[:3],[0,0,1,coefficient]])%p


def audit_group(name,p):
    source=group_sources(name,p)
    latent=np.array(list(product(range(p),repeat=4)),dtype=np.int64)
    labels=latent@source.T%p
    swap=np.eye(4,dtype=np.int64);swap[[2,3]]=swap[[3,2]]
    directions={projective(row,p) for row in source}
    transformed={projective(row@swap,p) for row in source}
    expected=len(directions&transformed)/len(directions)
    features=np.eye(p)[labels].reshape(len(labels),-1)
    changed=np.eye(p)[latent@swap.T@source.T%p].reshape(len(labels),-1)
    observed=linear_cka(features,changed)
    assert rank_mod(source,p)==4
    assert len(np.unique(labels,axis=0))==p**4
    assert max(categorical_mi(labels[:,a],labels[:,b]) for a in range(4) for b in range(a+1,4))<1e-12
    assert np.isclose(observed,expected,atol=1e-12)
    return {'group':name,'sources':source.tolist(),'rank':4,'joint_entropy':4*float(np.log(p)),
            'joint_support':p**4,'max_pairwise_mi':0.,'predicted_categorical_cka':expected,
            'exact_categorical_cka':observed,'directions_retained':len(directions&transformed)}


def spectral_power(representation,latent,p):
    cube=np.zeros((p,p,p,p,representation.shape[1]),dtype=np.float64)
    cube[tuple(latent.T)]=representation
    coefficient=np.fft.fftn(cube,axes=(0,1,2,3))/p**4
    power=np.square(abs(coefficient)).sum(-1);power[0,0,0,0]=0
    return power


def run(config_path='configs/field_symmetry.json'):
    config=json.loads(Path(config_path).read_text())
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    signature={'config':config,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'model_sha256':hashlib.sha256(Path('experiments/models.py').read_bytes()).hexdigest(),
               'algebra_sha256':hashlib.sha256(Path('experiments/algebra.py').read_bytes()).hexdigest()}
    fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    metadata=output/'metadata.json'
    if metadata.exists():assert json.loads(metadata.read_text())['fingerprint']==fingerprint
    atomic_json(metadata,{**signature,'fingerprint':fingerprint,'registered_unix':time.time(),
                          'audit':[audit_group(g,config['p']) for g in config['groups']]})
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    deadline=datetime.fromisoformat(config['deadline_utc']).timestamp()-config['analysis_reserve_seconds']
    all_rows=[]
    for w in config['world_seeds']:
        inputs,_,encoded,basis=world(config['p'],config['dimension'],w)
        latent=inputs@basis.T%config['p']
        lookup={tuple(z):i for i,z in enumerate(latent)}
        changed=latent.copy();changed[:,[2,3]]=changed[:,[3,2]]
        transform_ids=np.array([lookup[tuple(z)] for z in changed])
        assert np.array_equal(transform_ids[transform_ids],np.arange(len(inputs)))
        x=torch.tensor(encoded,device=device)
        for m in config['model_seeds']:
            order=np.random.default_rng(w+40000).permutation(4)
            for g in config['groups']:
                path=output/f'{g}_w{w}_m{m}.json'
                if path.exists():
                    saved=json.loads(path.read_text());assert saved['fingerprint']==fingerprint
                    if saved['status']=='complete':all_rows.extend(saved['curve']);continue
                if time.time()>=deadline:
                    write_csv(output/'endpoints.csv',all_rows)
                    return
                sources=group_sources(g,config['p'])
                y=torch.tensor((latent@sources.T%config['p'])[:,order],device=device)
                torch.manual_seed(m)
                model=SourceModel(encoded.shape[1],config['hidden'],config['features'],config['p']).to(device)
                optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
                curve=[];started=time.monotonic()
                for step in range(config['steps']+1):
                    if step in config['milestones']:
                        with torch.no_grad():
                            h=model.encoder(x).cpu().numpy().astype(np.float64)
                            accuracy=(model(x).argmax(-1)==y).float().mean(0).cpu().numpy()
                        power=spectral_power(h,latent,config['p'])
                        frequency={tuple(a*source%config['p']) for source in sources for a in range(1,config['p'])}
                        source_energy=sum(power[k] for k in frequency)
                        row={'group':g,'world_seed':w,'model_seed':m,'step':step,
                             'source_accuracy':float(accuracy.mean()),'source_min_accuracy':float(accuracy.min()),
                             'transformed_hidden_cka':linear_cka(h,h[transform_ids]),
                             'predicted_categorical_cka':audit_group(g,config['p'])['predicted_categorical_cka'],
                             'source_character_energy_fraction':float(source_energy/power.sum())}
                        curve.append(row)
                        np.save(output/f'{g}_w{w}_m{m}_step{step}_features.npy',h)
                    if step==config['steps']:break
                    if step%100==0 and time.time()>=deadline:
                        atomic_json(path,{'fingerprint':fingerprint,'status':'partial_deadline','step':step,'curve':curve})
                        write_csv(output/'endpoints.csv',all_rows+curve)
                        return
                    optimizer.zero_grad(set_to_none=True)
                    F.cross_entropy(model(x).reshape(-1,config['p']),y.reshape(-1)).backward()
                    optimizer.step()
                checkpoint=output/'checkpoints'/f'{g}_w{w}_m{m}.pt'
                atomic_torch(checkpoint,model.state_dict())
                atomic_json(path,{'fingerprint':fingerprint,'status':'complete','curve':curve,
                                  'source_gate_passed':curve[-1]['source_min_accuracy']>=config['source_accuracy_gate'],
                                  'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                  'seconds':time.monotonic()-started})
                all_rows.extend(curve)
                print(json.dumps(curve[-1]),flush=True)
    write_csv(output/'endpoints.csv',all_rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/field_symmetry.json')
    run(parser.parse_args().config)
