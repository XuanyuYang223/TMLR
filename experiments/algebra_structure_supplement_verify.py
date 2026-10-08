"""Check full-width sensitivity, new-combination forecasts, and extra history."""
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch

from .algebra_structure_verify import check_direct
from .field_algebra_structure import controlled_world,field_orbits,dihedral_table,projection_split
from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .native_confirmation import setup,permutation_row
from .native_source_subspace import contrast_basis
from .permworld_combinations import sha
from .representation_algebra import permutation_action_table


def run():
    plan=json.loads(Path('configs/algebra_structure.json').read_text());root=Path(plan['output'])
    config=json.loads(Path(plan['native_base_config']).read_text());setup(config)
    data=dict(np.load(root/'native/dataset.npz'));table=permutation_action_table();checks=0
    for p in sorted((root/'rank_sensitivity/native').glob('*.json')):
        r=json.loads(p.read_text());features=dict(np.load(root/'native/features'/f'{p.stem}.npz'))
        for landmark in plan['native_landmarks']:
            checks+=check_direct(features[landmark][:,:,-1],data['split'],data['lengths'],table,{'c':1,'r':2,'i':4},
                r['results'][landmark],root/'rank_sensitivity/native'/f'{p.stem}_{landmark}.npz')
    source=Path('results/native_ablation/source');pilot_checks=0
    protocol=json.loads((root/'new_combinations/protocol.json').read_text())['signature']
    for name,digest in protocol['checkpoints'].items():assert sha(source/'checkpoints'/f'{name}_s1009.pt')==digest
    for p in sorted((root/'new_combinations/probes').glob('*.json')):
        r=json.loads(p.read_text());path=root/'new_combinations/features'/f'{p.stem}.npz';assert sha(path)==r['feature_sha256']
        h=dict(np.load(path))
        if r['model_status']=='trained':
            state=torch.load(source/'checkpoints'/f"{r['group']}_s1009.pt",map_location='cpu',weights_only=True,mmap=True)['model']
            basis,_=contrast_basis(state['lm_head.weight'][:31].numpy())
        else:
            model=make_model(config,{'d_model':256,'layers':4,'heads':8},1009,'cpu');basis,_=contrast_basis(model.lm_head.weight[:31].detach().numpy())
        for landmark in plan['native_landmarks']:
            full=h[landmark][:,:,-1].astype(np.float64);blocks=full.reshape(*full.shape[:-1],-1,256)
            numeric=((blocks @ basis) @ basis.T).reshape(full.shape)
            for view,features in [(landmark,full),(landmark+'_numeric_null',full-numeric),(landmark+'_full_width',full)]:
                pilot_checks+=check_direct(features,data['split'],data['lengths'],table,{'c':1,'r':2,'i':4},r['results'][view],
                    root/'new_combinations/arrays'/f'{p.stem}_{view}.npz')
    pilot=json.loads((root/'new_combinations/summary.json').read_text());rows=pilot['rows'];record=next(r for r in rows if r['group']=='novel_records');others=[r for r in rows if r['group']!='novel_records']
    expected={'generator_rank_prediction':all(record['generator_mean_nmse']<r['generator_mean_nmse'] for r in others),
        'inverse_rank_prediction':all(record['i_nmse']<r['i_nmse'] for r in others),
        'noncommuting_order_prediction':record['wrong_order_gap']>0}
    assert expected==pilot['frozen_prediction_outcomes']
    fconfig=json.loads(Path(plan['field_config']).read_text());orbits,split=field_orbits(5,plan['field_split_seed']);table=dihedral_table();field_checks=0;projection_checks=0
    for w in fconfig['world_seeds']:
        x,_,_,b=controlled_world(5,4,w);latent=x @ b.T % 5;lookup={tuple(z):i for i,z in enumerate(latent)}
        ids=np.array([[lookup[tuple(z)] for z in orbit] for orbit in orbits]);psplit=projection_split(latent,plan['field_split_seed']+1)
        values=latent.copy();values[:,2]=0;images=np.array([lookup[tuple(z)] for z in values])
        for g,m,status in product(fconfig['groups'],fconfig['model_seeds'],('random','trained')):
            name=f'{g}_w{w}_s{m}_{status}';step=0 if status=='random' else fconfig['steps']
            h=np.load(Path(plan['field_source'])/f'{g}_w{w}_m{m}_step{step}_features.npy');r=json.loads((root/'rank_sensitivity/field'/f'{name}.json').read_text())
            field_checks+=check_direct(h[ids],split,np.zeros(len(orbits)),table,{'a':1,'s':4},r['results']['hidden'],root/'rank_sensitivity/field'/f'{name}_hidden.npz')
            a=dict(np.load(root/'rank_sensitivity/field'/f'{name}_projection.npz'));mean=h[psplit==0].astype(np.float64).mean(0);basis=a['basis'].astype(np.float64)
            latent_h=(h-mean) @ basis;target=(h[images]-mean) @ basis;test=psplit==2
            pred=latent_h[test] @ a['map'][:-1]+a['map'][-1];denom=np.square(target[test]-target[psplit==0].mean(0)).sum()
            error=float(np.square(pred-target[test]).sum()/denom);np.testing.assert_allclose(error,r['results']['projection']['test_nmse'],rtol=3e-5,atol=3e-5);projection_checks+=1
    probe={tuple(map(int,p[:n])) for orbit,n in zip(data['permutations'],data['lengths']) for p in orbit}
    path=Path('results/permworld_combinations/data.npz')
    with np.load(path) as a:
        extra={permutation_row(row,int(n)) for k in a.files if k.endswith('_input') for row,n in zip(a[k],a[k[:-6]+'_lengths'])}
    assert not probe & extra
    paused=json.loads(Path('results/native_ablation/state.json').read_text());assert paused['status']=='paused_for_research_focus_change'
    result={'status':'passed','verified_utc':datetime.now(timezone.utc).isoformat(),'full_width_native_errors_and_laws_recomputed':checks,
        'full_width_field_errors_and_laws_recomputed':field_checks,'full_width_projection_errors_recomputed':projection_checks,
        'new_combination_errors_and_laws_recomputed':pilot_checks,'forecasts_recomputed':expected,
        'additional_old_pilot_inputs_checked':len(extra),'additional_old_pilot_probe_overlap':0,'additional_old_pilot_sha256':sha(path),
        'lis_ablation_still_paused':True,'scope':'all primary low-rank and full-width new-combination metrics checked against saved matrices; hypotheses registered before new extraction; only one source seed'}
    atomic_json(root/'supplementary_verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':run()
