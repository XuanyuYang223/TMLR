"""Matched conditional dose extension; preserves the10-epoch primary."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from . import two_step_relation_joint as joint
from . import two_step_relation_joint_verify as replay
from .longrun_engine import atomic_json
from .native_confirmation import setup
from .permworld_combinations import sha

CONFIG=Path('configs/two_step_relation_dose_confirmation.json')
BASE=Path('results/two_step_relation_coverage_confirmation').resolve()
PILOT=Path('results/two_step_relation_dose_pilot')


def register():
    joint.CONFIG=CONFIG
    plan,parent,config,root,sig=joint.initialize()
    support_files=[BASE/'dataset'/f'n{i}'/'support'/f for i in range(6) for f in ['dataset.npz','pairings.npz']]
    teacher_files=[BASE/'features'/f'n{i}_support.npz' for i in range(6)]
    signature={'code_sha256':sha(__file__),'joint_code_sha256':sha(joint.__file__),
               'plan':plan,'pilot_protocol_sha256':sha(PILOT/'dose_pilot_protocol.json'),
               'support_and_teacher_sha256':{str(p):sha(p) for p in support_files+teacher_files},
               'start_gate':'Only if the fixed20-epoch visible-only weaker-source pilot passes native>=0.95 and both generators>=0.90.',
               'budget':'All24 conditions continue their own immutable10-epoch parameters and optimizer for the same additional10 complete epochs; source/readout, input marginal distributions, teacher outputs and geometry assignments remain the same. Epochs11-20 follow the unchanged joint implementation second-half20-epoch cosine. Original10-epoch prefix is retained, not recomputed.',
               'scope':'Registered dose sensitivity, not an independent replication or replacement for the10-epoch primary. New test orbits exclude every prior local archive. All24 cumulative20-epoch endpoints must finish before this new test opens; partial doses are never treated as a matched result.'}
    file=root/'dose_confirmation_protocol.json'
    if file.exists():
        original=json.loads(file.read_text())['signature'];effective=original['code_sha256']
        amendments=root/'dose_implementation_amendments.json'
        if amendments.exists():
            for amendment in json.loads(amendments.read_text()):
                assert amendment['old_code_sha256']==effective
                effective=amendment['new_code_sha256']
                assert sha(root/amendment['source_snapshot'])==effective
        assert effective==sha(__file__)
        signature['code_sha256']=original['code_sha256']
        assert original==signature
    else:
        atomic_json(file,{'registered_utc':joint.core.now(),'base_compound_test_opened_at_registration':(BASE/'test_opened.json').exists(),
                          'new_compound_test_opened_at_registration':False,'signature':signature})
        (root/'dose_confirmation_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return plan,parent,config,root,sig


def prepare(plan,parent,config,root,sig):
    if (root/'data_verification.json').exists():return
    for i in range(6):
        for name in [f'dataset/n{i}/support',f'features/n{i}_support.npz',f'features/n{i}_support.json']:
            dst=root/name;dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():dst.symlink_to(BASE/name,target_is_directory=name.endswith('support'))
    _,functions,tokens,one_line=setup(config);f=functions[parent['source_task']]
    seen=joint.core.old_inputs(sig['excluded_archives']);prior=len(seen)
    rng=np.random.default_rng(plan['test_seed']);orbits=[];lengths=[];splits=[];pair_ids=[];pair=0
    for n in plan['lengths']:
        for _ in range(plan['iid_per_length']):
            orbits.append(joint.core.fresh_orbit(rng,n,seen));lengths.append(n);splits.append(0);pair_ids.append(-1)
        for _ in range(plan['collision_pairs_per_length']):
            for orbit in joint.core.collision_orbits(rng,n,seen,f):
                orbits.append(orbit);lengths.append(n);splits.append(1);pair_ids.append(pair)
            pair+=1
    data=joint.core.encode(orbits,lengths,tuple(range(8)),f,tokens,one_line)
    data.update(split=np.asarray(splits),pair_ids=np.asarray(pair_ids))
    file=root/'dataset/test/dataset.npz';file.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(file,**data)
    # Independently rebuild and verify only the new test; supports are explicit reuse.
    seen=joint.core.old_inputs(sig['excluded_archives'])
    for raw,n,rows,labels in zip(data['permutations'],data['lengths'],data['input'],data['labels']):
        orbit=joint.core.independent_orbit(raw[0,:n]);assert len(set(orbit))==8
        for saved,p,row,y in zip(raw,orbit,rows,labels):
            assert tuple(map(int,saved[:n]))==p
            key=joint.core.input_key(p);assert key not in seen;seen.add(key)
            assert tuple(map(int,row[4:4+2*n:2]))==p and int(y)==f(p)
        assert f(orbit[6])==f(orbit[1])
    for pair in np.unique(data['pair_ids'][data['split']==1]):
        ids=np.flatnonzero(data['pair_ids']==pair);assert len(ids)==2
        assert data['lengths'][ids[0]]==data['lengths'][ids[1]]
        assert np.array_equal(data['labels'][ids[0],[0,1,4]],data['labels'][ids[1],[0,1,4]])
        assert data['labels'][ids[0],5]!=data['labels'][ids[1],5]
    atomic_json(root/'data_manifest.json',{'created_utc':joint.core.now(),'support_origin':str(BASE),'support_reuse':True,
                'test_path':str(file),'test_sha256':sha(file),'prior_local_distinct_inputs':prior,
                'collision_pairs':int((data['split']==1).sum())//2})
    atomic_json(root/'data_verification.json',{'status':'complete','completed_utc':joint.core.now(),
                'supports_reused_from_registered10_epoch_study':True,'support_original_verification_sha256':sha(BASE/'data_verification.json'),
                'new_test_examples_checked':len(lengths),'new_test_full_orbit_states_checked':8*len(lengths),
                'collision_pairs_checked':int((data['split']==1).sum())//2,
                'new_test_excluded_against_all_prior_local_orbits':True,'answer_code_only_ci_accuracy_ceiling':.5})


def run(phase):
    plan,parent,config,root,sig=register()
    if phase=='register':return
    try:
        while not (PILOT/'visible_dose_results.json').exists():
            if time.time()>=joint.datetime.fromisoformat(plan['deadline_utc']).timestamp():raise TimeoutError('Pilot missed deadline')
            time.sleep(10)
        if not json.loads((PILOT/'visible_dose_results.json').read_text())['feasible']:
            atomic_json(root/'state.json',{'status':'not_started_visible_feasibility_failed','updated_utc':joint.core.now()});return
        atomic_json(root/'state.json',{'status':'running','stage':'prepare_new_test_and_shared_supports','updated_utc':joint.core.now()})
        prepare(plan,parent,config,root,sig)
        _,_,tokens,_=setup(config);device='cuda' if torch.cuda.is_available() else 'cpu'
        prefixes=json.loads((root/'prefix_records.json').read_text()) if (root/'prefix_records.json').exists() else {}
        for i in range(6):
            for condition in plan['conditions']:
                name=f'n{i}_hidden_{condition}';original=BASE/'fits'/f'{name}.json'
                while not original.exists():
                    if time.time()>=joint.datetime.fromisoformat(plan['deadline_utc']).timestamp():raise TimeoutError('Prefix missed deadline')
                    time.sleep(10)
                old=json.loads(original.read_text());assert old['status']=='complete' and old['epochs']==10
                resume=BASE/'maps'/f'{name}_resume.pt';target=root/'maps'/resume.name
                if not target.exists():
                    assert sha(BASE/'maps'/f'{name}_e10.pt')==old['checkpoint_sha256']
                    prefixes[name]={'fit_record_path':str(original),'fit_record_sha256':sha(original),'resume_sha256':sha(resume),
                                    'prefix_updates':old['updates'],'prefix_anchor_exposures':old['anchor_exposures']}
                    atomic_json(root/'prefix_records.json',prefixes);shutil.copyfile(resume,target)
                atomic_json(root/'state.json',{'status':'running','stage':'matched_extension','updated_utc':joint.core.now()})
                joint.train_one(plan,parent,config,root,sig,i,condition,tokens,device)
        assert len(prefixes)==24
        for name,p in prefixes.items():
            assert sha(p['fit_record_path'])==p['fit_record_sha256']
            result=json.loads((root/'fits'/f'{name}.json').read_text())
            assert result['updates']==2*p['prefix_updates'] and result['anchor_exposures']==2*p['prefix_anchor_exposures']
        atomic_json(root/'prefix_budget_verification.json',{'status':'complete','verified_utc':joint.core.now(),
                    'all24_prefixes_preserved':True,'all24_total_exposures_exactly_twice10_epoch_prefix':True,
                    'dose_is_not_independent_replication':True})
        joint.evaluate(plan,parent,config,root,sig,tokens,device)
        replay.run()
        from . import two_step_relation_dose_statistics_verify as statistics
        statistics.run()
    except BaseException as error:
        atomic_json(root/'state.json',{'status':'failed','error':repr(error),'updated_utc':joint.core.now()});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['register','run']);run(parser.parse_args().phase)
