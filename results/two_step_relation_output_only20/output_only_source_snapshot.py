"""Fixed no-geometry baseline on the same source/data/budget as the primary."""
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

CONFIG=Path('configs/two_step_relation_output_only20.json')
BASE=Path('results/two_step_relation_dose_confirmation').resolve()
PREFIX=Path('results/two_step_relation_output_only').resolve()


def register():
    joint.CONFIG=CONFIG;plan,parent,config,root,sig=joint.initialize()
    files=[BASE/'dataset'/f'n{i}'/'support'/f for i in range(6) for f in ['dataset.npz','pairings.npz']]
    files+=[BASE/'features'/f'n{i}_support.npz' for i in range(6)]
    signature={'code_sha256':sha(__file__),'joint_code_sha256':sha(joint.__file__),'plan':plan,
               'shared_support_and_teacher_sha256':{str(p):sha(p) for p in files},
               'difference':'Geometry weight0 instead of0.25; keep original source, frozen readout, native CE/KD, operator-output KD, five-query computation, eligible anchors and exact20-epoch schedule.',
               'prior_main20_outcomes_already_observed':True,
               'prefix_fit_and_resume_sha256':{str(p):sha(p) for p in list((PREFIX/'fits').glob('*.json'))+list((PREFIX/'maps').glob('*_resume.pt'))},
               'reason':'Both-wrong applies incorrect geometry and cannot stand for ordinary output-only training. Separate benefit of correct geometry from harm of wrong geometry.',
               'evaluation':'Only after all six fixed-budget fits finish, evaluate on the exact immutable20-epoch sensitivity test. Compare all four main conditions with this baseline. Registration follows main20 compound grades; preserve the previously fixed no-geometry coefficient and the already fixed20 continuation schedule. No result-based tuning.'}
    file=root/'output_only_protocol.json'
    if file.exists():assert json.loads(file.read_text())['signature']==signature
    else:
        atomic_json(file,{'registered_utc':joint.core.now(),'primary_test_opened_at_registration':(BASE/'test_opened.json').exists(),'signature':signature})
        (root/'output_only_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return plan,parent,config,root,sig


def run(phase):
    plan,parent,config,root,sig=register()
    if phase=='register':return
    try:
        # Preserve priority of completing the originally registered factorial.
        while not (BASE/'completion.json').exists():
            state=json.loads((BASE/'state.json').read_text()) if (BASE/'state.json').exists() else {}
            if state.get('status')=='failed':raise RuntimeError('Primary failed')
            if time.time()>=joint.datetime.fromisoformat(plan['deadline_utc']).timestamp():raise TimeoutError('Primary missed deadline')
            time.sleep(10)
        for i in range(6):
            for name in [f'dataset/n{i}/support',f'features/n{i}_support.npz',f'features/n{i}_support.json']:
                dst=root/name;dst.parent.mkdir(parents=True,exist_ok=True)
                if not dst.exists():dst.symlink_to(BASE/name,target_is_directory=name.endswith('support'))
        _,_,tokens,_=setup(config);device='cuda' if torch.cuda.is_available() else 'cpu'
        import shutil
        for old in (PREFIX/'maps').glob('*_resume.pt'):
            dst=root/'maps'/old.name
            if not dst.exists():shutil.copyfile(old,dst)
        original_load=torch.load
        def cpu_resume(file,*args,**kwargs):
            if Path(file).parent==root/'maps' and str(file).endswith('_resume.pt'):
                kwargs['map_location']='cpu'
            return original_load(file,*args,**kwargs)
        torch.load=cpu_resume
        for i in range(6):
            atomic_json(root/'state.json',{'status':'running','stage':'fixed_output_only_fit','completed_fits':i,'updated_utc':joint.core.now()})
            joint.train_one(plan,parent,config,root,sig,i,'both_correct',tokens,device)
        torch.load=original_load
        atomic_json(root/'test_opened.json',{'opened_utc':joint.core.now(),'all6_output_only_fits_complete':True,
                    'primary_test_dataset_sha256':sha(BASE/'dataset/test/dataset.npz')})
        data=dict(np.load(BASE/'dataset/test/dataset.npz'));rows=[];checks=0
        for i in range(6):
            name=f'n{i}_hidden_both_correct';record=json.loads((root/'fits'/f'{name}.json').read_text())
            main=json.loads((BASE/'fits'/f'{name}.json').read_text())
            assert record['schedule_sha256']==main['schedule_sha256'] and record['updates']==main['updates']
            assert record['anchor_exposures']==main['anchor_exposures'] and record['readout_parameter_sha256']==main['readout_parameter_sha256']
            source=sig['sources'][i%3];model=joint.core.source_model(source,parent,config,device,accelerated=False)
            head=model.lm_head.weight.detach().cpu().clone();state=torch.load(root/'maps'/f'{name}_e20.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(state['model']);torch.testing.assert_close(head,model.lm_head.weight.detach().cpu(),atol=0,rtol=0)
            ops=joint.AffineOperators(parent['architecture']['d_model']).to(device);ops.load_state_dict(state['operators'])
            a=dict(np.load(root/'dataset'/f'n{i}'/'support/dataset.npz'))
            grade=joint.validate(model,ops,{k:a[k][a['split']==1] for k in ['input','lengths','labels']},parent,tokens)
            for key in grade:np.testing.assert_allclose(grade[key],record['curve'][-1][key],atol=1e-8,rtol=1e-8)
            h=joint.core.hidden_features(model,data,parent['source_task'],tokens).astype(float);del model,ops,state
            maps=dict(np.load(root/'maps'/f'{name}.npz'));saved={}
            for word,action in [('c',1),('i',4),('ci',5),('ic',6),('cc',0),('ii',0)]:
                prediction=joint.core.apply_numpy(h[:,0],word,maps)
                width=h.shape[-1];matrix=np.eye(width+1)
                for letter in word:
                    m=np.eye(width+1);m[:width,:width]=maps['rho_'+letter];m[-1,:width]=maps['bias_'+letter];matrix=matrix@m
                replay=(np.column_stack([h[:,0],np.ones(len(h))])@matrix)[:,:width]
                np.testing.assert_allclose(prediction,replay,atol=1e-10,rtol=1e-10);checks+=1
                answers=(prediction@maps['readout_weight'].T+maps['readout_bias']).argmax(-1);saved[word+'_answers']=answers
                for split,label in [(0,'iid'),(1,'collisions')]:
                    use=data['split']==split;hit=answers[use]==data['labels'][use,action]
                    ck=[]
                    for n in plan['lengths']:
                        ids=np.flatnonzero(use&(data['lengths']==n))[:128];ck.append(joint.core.gram_cka(prediction[ids],h[ids,action]))
                    row={'replicate':f'n{i}','source_seed':source['seed'],'word':word,'split':label,'accuracy':float(hit.mean()),
                         'cka':float(np.mean(ck)),'pair_both_correct':None}
                    if split==1:
                        pairs=data['pair_ids'][use];row['pair_both_correct']=float(np.mean([hit[pairs==p].all() for p in np.unique(pairs)]))
                    rows.append(row)
            np.savez_compressed(root/'evaluations'/f'{name}.npz',**saved)
        main_rows=json.loads((BASE/'evaluation_records.json').read_text())['records'];contrasts=[]
        for condition in ['both_correct','c_correct_i_wrong','c_wrong_i_correct','both_wrong']:
            for split in ['iid','collisions']:
                values=[]
                for i in range(6):
                    score=next(r['accuracy'] for r in main_rows if (r['replicate'],r['condition'],r['split'],r['word'])==(f'n{i}',condition,split,'ci'))
                    baseline=next(r['accuracy'] for r in rows if (r['replicate'],r['split'],r['word'])==(f'n{i}',split,'ci'))
                    values.append(100*(score-baseline))
                sources=[float(np.mean([values[j],values[j+3]])) for j in range(3)]
                contrasts.append({'condition':condition,'split':split,'contrast':condition+'-output_only',
                                  'mean_accuracy_pp':float(np.mean(values)),'paired_values':values,'three_source_means':sources,
                                  'three_source_bootstrap_95_pp':bootstrap_interval(sources)})
        atomic_json(root/'results.json',{'status':'complete','completed_utc':joint.core.now(),'fits':6,'source_initializations':3,
                    'geometry_weight':0,'fixed_epochs':20,'same_primary_test':True,'records':rows,'contrasts':contrasts,
                    'all6_original_model_validation_and_readouts_replayed':True,'independent_affine_composition_checks':checks,
                    'all6_exact_schedules_and_exposures_match_primary':True,
                    'limitation':'The no-geometry coefficient was fixed before the10-epoch primary outcomes. This20-epoch extension is registered after main20-epoch outcomes, with no model selection from those outcomes. It uses the same source/data repeats and is not an independent replication.'})
        atomic_json(root/'state.json',{'status':'complete','updated_utc':joint.core.now()})
    except BaseException as error:
        atomic_json(root/'state.json',{'status':'failed','error':repr(error),'updated_utc':joint.core.now()});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['register','run']);run(parser.parse_args().phase)
