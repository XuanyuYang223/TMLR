"""Use leftover time for unchanged optional source controls after all targets."""
from datetime import datetime
import json
from pathlib import Path
import time

from .longrun_engine import atomic_json,load_session,train_job
from .permworld_combinations import sha


def count_complete(root):
    return sum(json.loads(p.read_text())['status']=='complete' for p in (root/'single').glob('*.json'))


def run():
    plan=json.loads(Path('configs/six_hour_session.json').read_text());root=Path(plan['output'])
    stop_at=datetime.fromisoformat(plan['deadline_utc']).timestamp()-plan['analysis_reserve_seconds']
    state={'status':'waiting_for_complete_target_evaluation','code_sha256':sha(__file__),
           'policy':'original task catalog and seed priority; identical source parameters and exposure; start only after target validation and testing finish',
           'source_training_code_sha256':sha('experiments/longrun_engine.py')}
    atomic_json(root/'optional_resume_state.json',state)
    while time.time()<stop_at:
        handoff=json.loads((root/'target_handoff_state.json').read_text())
        if handoff['status']=='scheduled_phases_finished':break
        if handoff['status'] in ('phase_failed','time_limited'):
            state.update({'status':'not_started_primary_incomplete','updated_unix':time.time()})
            atomic_json(root/'optional_resume_state.json',state);return
        time.sleep(10)
    else:return
    plan,config,groups,data,names,token_ids,device=load_session()
    assert sha('experiments/longrun_engine.py')==state['source_training_code_sha256']
    state.update({'status':'running','started_unix':time.time(),'completed_before_resume':count_complete(root)})
    atomic_json(root/'optional_resume_state.json',state)
    arch=json.loads((root/'architecture_selection.json').read_text())['selected']
    tasks=sorted({t for g in groups for t in g['tasks']},key=names.index)
    for seed in plan['model_seeds']:
        for task in tasks:
            if time.time()>=stop_at:
                state.update({'status':'time_limited','completed_single_models':count_complete(root),'updated_unix':time.time()})
                atomic_json(root/'optional_resume_state.json',state);return
            job={'phase':'single','id':task,'seed':seed,'tasks':[task],'steps':plan['source_steps']}
            state.update({'job':job,'updated_unix':time.time()});atomic_json(root/'optional_resume_state.json',state)
            _,record=train_job(plan,config,arch,job,data,names,token_ids,device,stop_at)
            if record['status']!='complete':
                state.update({'status':'time_limited','last_status':record['status'],'completed_single_models':count_complete(root),'updated_unix':time.time()})
                atomic_json(root/'optional_resume_state.json',state);return
    state.update({'status':'all_optional_controls_completed','completed_single_models':count_complete(root),'updated_unix':time.time()})
    atomic_json(root/'optional_resume_state.json',state)


if __name__=='__main__':run()
