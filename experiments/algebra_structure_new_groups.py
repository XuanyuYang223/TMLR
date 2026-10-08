"""Structural predictions for three existing but geometry-untested task sets.

Reuse completed source checkpoints from the paused LIS study; do not resume
training or perform target adaptation. All three have four tasks, 32 common
inputs/task/update, 20k updates, and 2.56m labels, matching the main cohort.
One source seed means an exploratory new-combination check, not replication.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from .algebra_structure_direct import direct_probe
from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .native_algebra_structure import extract
from .native_confirmation import setup
from .native_source_subspace import contrast_basis
from .permworld_combinations import sha
from .representation_algebra import permutation_action_table


def run():
    plan=json.loads(Path('configs/algebra_structure.json').read_text()); root=Path(plan['output'])
    output=root/'new_combinations'; output.mkdir(exist_ok=True)
    for subdir in ('features','probes','arrays'): (output/subdir).mkdir(exist_ok=True)
    source=Path('results/native_ablation/source'); names=('novel_records','novel_minima_pair','novel_peak_mixed')
    records={name:json.loads((source/f'{name}_s1009.json').read_text()) for name in names}
    signature={'code_sha256':sha(__file__),'direct_code_sha256':sha('experiments/algebra_structure_direct.py'),
        'checkpoints':{},'dataset_sha256':sha(root/'native/dataset.npz'),
        'scope':'new structural combinations, after old-group endpoints; forecasts before extracting these three models; one source seed only',
        'matched_budget':'all three and original four-task cohort: 32 examples/task/update, 20k updates, 2.56 million labels',
        'forecasts':[
            {'endpoint':'64-d source-query raw-h generator full-space displacement mean',
             'prediction':'novel_records has lower error than each of novel_minima_pair and novel_peak_mixed',
             'reason':'all three generators permute the four directional record tasks; the other sets lack closure'},
            {'endpoint':'64-d source-query raw-h inverse full-space displacement error',
             'prediction':'novel_records has lower error than each of the other two new sets',
             'reason':'inverse swaps LR maxima/RL minima and preserves LR minima/RL maxima'},
            {'endpoint':'64-d source-query predicted CI action versus IC target',
             'prediction':'novel_records has positive wrong-order-minus-correct-order error gap',
             'reason':'C and I induce different noncommuting task permutations in the full record set'}],
        'rules':'No task or model selection based on these geometry outcomes; full width is diagnostic; source learning differs across sets.'}
    for name,record in records.items():
        checkpoint=source/'checkpoints'/f'{name}_s1009.pt'
        assert record['status']=='complete' and record['total_labels']==2560000 and record['steps']==20000
        assert sha(checkpoint)==record['checkpoint_sha256']; signature['checkpoints'][name]=sha(checkpoint)
    protocol=output/'protocol.json'
    if protocol.exists(): assert json.loads(protocol.read_text())['signature']==signature
    else: atomic_json(protocol,{'registered_utc':datetime.now(timezone.utc).isoformat(),'signature':signature,'new_geometry_endpoints_at_registration':0})
    config=json.loads(Path(plan['native_base_config']).read_text()); _,functions,tokens,_=setup(config)
    data=dict(np.load(root/'native/dataset.npz')); device='cuda' if torch.cuda.is_available() else 'cpu'; table=permutation_action_table()
    for name,record in records.items():
        tasks=record['job']['tasks']
        for status in ('random','trained'):
            ident=f'{name}_{status}'; dest=output/'probes'/f'{ident}.json'
            if dest.exists():continue
            model=make_model(config,{'d_model':256,'layers':4,'heads':8},1009,device)
            if status=='trained':model.load_state_dict(torch.load(source/'checkpoints'/f'{name}_s1009.pt',weights_only=True,map_location=device)['model'])
            hidden=extract(model,data,tasks,tokens,plan['batch_size']); np.savez_compressed(output/'features'/f'{ident}.npz',**hidden)
            basis,_=contrast_basis(model.lm_head.weight[:31].detach().cpu().numpy()); results={}
            for landmark in plan['native_landmarks']:
                full=hidden[landmark][:,:,-1].astype(np.float64)
                blocks=full.reshape(*full.shape[:-1],-1,256); numeric=((blocks @ basis) @ basis.T).reshape(full.shape)
                for view,h,dimension in [(landmark,full,64),(landmark+'_numeric_null',full-numeric,64),(landmark+'_full_width',full,full.shape[-1])]:
                    results[view],arrays=direct_probe(h,data['split'],data['lengths'],table,{'c':1,'r':2,'i':4},
                        [('r','c'),('c','i'),('r','i'),('r','c','i')],dimension,plan['ridge_grid'],1009+9200)
                    np.savez_compressed(output/'arrays'/f'{ident}_{view}.npz',**arrays)
            atomic_json(dest,{'group':name,'tasks':tasks,'seed':1009,'model_status':status,'status':'complete','results':results,
                'source_accuracy':float(np.mean([a['accuracy'] for a in record['source_audit']])), 'feature_sha256':sha(output/'features'/f'{ident}.npz')})
            del model
            print(ident,flush=True)
    output_rows=[]
    for name in names:
        record=json.loads((output/'probes'/f'{name}_trained.json').read_text()); r=record['results']['source_query_concat']
        full=record['results']['source_query_concat_full_width']
        output_rows.append({'group':name,'source_accuracy':record['source_accuracy'],
            'generator_mean_nmse':float(np.mean([a['full_space_displacement_nmse'] for a in r['generators']])),
            'c_nmse':r['generators'][0]['full_space_displacement_nmse'],'r_nmse':r['generators'][1]['full_space_displacement_nmse'],
            'i_nmse':r['generators'][2]['full_space_displacement_nmse'],
            'composite_mean_nmse':float(np.mean([a['full_space_displacement_nmse'] for a in r['composites']])),
            'wrong_order_gap':r['wrong_order'][0]['gap_wrong_minus_correct'],
            'full_width_generator_mean_nmse':float(np.mean([a['full_space_displacement_nmse'] for a in full['generators']])),
            'full_width_composite_mean_nmse':float(np.mean([a['full_space_displacement_nmse'] for a in full['composites']]))})
    rec=next(r for r in output_rows if r['group']=='novel_records'); others=[r for r in output_rows if r['group']!='novel_records']
    outcomes={'generator_rank_prediction':all(rec['generator_mean_nmse']<r['generator_mean_nmse'] for r in others),
        'inverse_rank_prediction':all(rec['i_nmse']<r['i_nmse'] for r in others),'noncommuting_order_prediction':rec['wrong_order_gap']>0}
    atomic_json(output/'summary.json',{'rows':output_rows,'frozen_prediction_outcomes':outcomes,'scope':signature['scope'],
        'limitation':'one source seed; mathematical task closure predicts code geometry directly; source grades/joint label statistics are not matched'})
    atomic_json(output/'state.json',{'status':'complete','conditions':6,'source_training_resumed':False,'target_adaptation_ran':False})


if __name__=='__main__':run()
