"""Compare immutable10/20-epoch predictors on the same new dose holdout."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import two_step_relation_joint as joint
from .longrun_engine import atomic_json
from .native_confirmation import setup
from .overnight_inverse_statistics_verify import bootstrap_interval
from .permworld_combinations import sha

ROOT=Path('results/two_step_relation_dose_confirmation')
BASE=Path('results/two_step_relation_coverage_confirmation')
CONFIG=Path('configs/two_step_relation_dose_confirmation.json')


def register():
    signature={'code_sha256':sha(__file__),'dose_config_sha256':sha(CONFIG),
               'base_config_sha256':sha('configs/two_step_relation_coverage_confirmation.json'),
               'words':['ci','ic'],'splits':['iid','collisions'],
               'method':'After all24 cumulative20-epoch fits and their independent verification finish, replay each immutable10-epoch checkpoint on exactly the fresh20-epoch test inputs. Apply only the original learned10-epoch operators to the base hidden vector. Compare accuracy and whole-pair both-correct with20-epoch predictions, in each paired fit and three source means. Exact three-source percentile bootstrap is secondary.',
               'scope':'A dose sensitivity comparison, not an independent replication. Both dose schedules were fixed using visible-only validation, before either new compound test opened. No test-based model/endpoint selection.'}
    file=ROOT/'paired_dose_protocol.json'
    if file.exists():assert json.loads(file.read_text())['signature']==signature
    else:
        atomic_json(file,{'registered_utc':joint.core.now(),'base_compound_test_opened':(BASE/'test_opened.json').exists(),
                          'new_compound_test_opened':(ROOT/'test_opened.json').exists(),'signature':signature})
        (ROOT/'paired_dose_source_snapshot.py').write_bytes(Path(__file__).read_bytes())


def run():
    register()
    while not (ROOT/'completion.json').exists():
        state=json.loads((ROOT/'state.json').read_text()) if (ROOT/'state.json').exists() else {}
        if state.get('status') in ['failed','not_started_visible_feasibility_failed']:raise RuntimeError('Dose incomplete')
        time.sleep(10)
    joint.CONFIG=CONFIG;plan,parent,config,root,sig=joint.initialize()
    _,_,tokens,_=setup(config);device='cuda' if torch.cuda.is_available() else 'cpu'
    data=dict(np.load(root/'dataset/test/dataset.npz'));rows=[]
    for i in range(6):
        for condition in plan['conditions']:
            name=f'n{i}_hidden_{condition}';record=json.loads((BASE/'fits'/f'{name}.json').read_text())
            file=BASE/'maps'/f'{name}_e10.pt';assert sha(file)==record['checkpoint_sha256']
            model=joint.core.source_model(sig['sources'][i%3],parent,config,device)
            state=torch.load(file,map_location='cpu',weights_only=True);model.load_state_dict(state['model'])
            h=joint.core.hidden_features(model,data,parent['source_task'],tokens).astype(float);del model,state
            maps=dict(np.load(BASE/'maps'/f'{name}.npz'));assert sha(BASE/'maps'/f'{name}.npz')==record['map_sha256']
            later=np.load(root/'evaluations'/f'{name}.npz');saved={}
            for word,action in [('ci',5),('ic',6)]:
                prediction=joint.core.apply_numpy(h[:,0],word,maps)
                # Independently reconstruct affine composition in homogeneous coordinates.
                width=h.shape[-1];matrix=np.eye(width+1)
                for letter in word:
                    a=np.eye(width+1);a[:width,:width]=maps['rho_'+letter];a[-1,:width]=maps['bias_'+letter];matrix=matrix@a
                replay=(np.column_stack([h[:,0],np.ones(len(h))])@matrix)[:,:width]
                np.testing.assert_allclose(prediction,replay,atol=1e-10,rtol=1e-10)
                answers=(prediction@maps['readout_weight'].T+maps['readout_bias']).argmax(-1);saved[word+'_e10_answers']=answers
                for split,label in [(0,'iid'),(1,'collisions')]:
                    use=data['split']==split;truth=data['labels'][use,action]
                    hit10=answers[use]==truth;hit20=later[word+'_answers'][use]==truth
                    row={'replicate':f'n{i}','source_seed':sig['sources'][i%3]['seed'],'condition':condition,
                         'word':word,'split':label,'e10_accuracy':float(hit10.mean()),'e20_accuracy':float(hit20.mean()),
                         'accuracy_delta_pp':100*float(hit20.mean()-hit10.mean()),
                         'e10_pair_both_correct':None,'e20_pair_both_correct':None,'pair_delta_pp':None}
                    if split==1:
                        pairs=data['pair_ids'][use]
                        pair10=float(np.mean([hit10[pairs==p].all() for p in np.unique(pairs)]))
                        pair20=float(np.mean([hit20[pairs==p].all() for p in np.unique(pairs)]))
                        row.update(e10_pair_both_correct=pair10,e20_pair_both_correct=pair20,pair_delta_pp=100*(pair20-pair10))
                    rows.append(row)
            np.savez_compressed(root/'evaluations'/f'{name}_paired_dose.npz',**saved)
    means=[]
    for condition in plan['conditions']:
        for word in ['ci','ic']:
            for split in ['iid','collisions']:
                selected=[r for r in rows if (r['condition'],r['word'],r['split'])==(condition,word,split)]
                for metric in ['accuracy_delta_pp']+(['pair_delta_pp'] if split=='collisions' else []):
                    values=[next(r[metric] for r in selected if r['replicate']==f'n{i}') for i in range(6)]
                    sources=[float(np.mean([values[j],values[j+3]])) for j in range(3)]
                    means.append({'condition':condition,'word':word,'split':split,'metric':metric,
                                  'mean':float(np.mean(values)),'paired_values':values,'three_source_means':sources,
                                  'three_source_bootstrap_95':bootstrap_interval(sources)})
    atomic_json(root/'paired_dose.json',{'status':'complete','completed_utc':joint.core.now(),'same_test_inputs':True,
                'primary10_epoch_results_preserved':True,'replayed10_epoch_models':24,'independent_homogeneous_composition_checks':48,
                'means':means,'records':rows,'not_independent_source_replication':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['register','run'])
    register() if parser.parse_args().phase=='register' else run()
