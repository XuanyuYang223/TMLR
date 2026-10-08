"""Allocate the final session window to frozen target validation and testing.

The source training implementation and each completed model remain unchanged.
An optional source model is allowed to finish before handing the GPU over.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .longrun_engine import atomic_json
from .permworld_combinations import sha


def owned(pid, module):
    path=Path(f'/proc/{pid}/cmdline')
    return path.exists() and module in path.read_bytes().decode().replace('\0',' ')


def complete_count(root):
    return sum(json.loads(p.read_text()).get('status')=='complete' for p in (root/'single').glob('*.json'))


def source_child(parent):
    path=Path(f'/proc/{parent}/task/{parent}/children')
    if not path.exists():return None
    candidates=[int(p) for p in path.read_text().split()]
    candidates=[p for p in candidates if owned(p,'experiments.six_hour_controller')]
    assert len(candidates)<=1
    return candidates[0] if candidates else None


def run(parent, plan_path='configs/six_hour_session.json'):
    plan=json.loads(Path(plan_path).read_text());root=Path(plan['output'])
    deadline=datetime.fromisoformat(plan['deadline_utc']).timestamp()
    cutoff=deadline-9000
    assert owned(parent,'experiments.six_hour_marathon')
    state={'status':'waiting_for_source_handoff','source_supervisor_pid':parent,
           'source_cutoff_utc':datetime.fromtimestamp(cutoff,timezone.utc).isoformat(),
           'policy':'finish the active optional single-task model, then preserve partial artifacts and begin unchanged target grid',
           'code_sha256':sha(__file__)}
    baseline=None
    while True:
        count=complete_count(root)
        now=time.time()
        state.update({'completed_single_models':count,'updated_unix':now})
        atomic_json(root/'target_handoff_state.json',state)
        if now>=cutoff:
            if baseline is None:baseline=count
            child=source_child(parent)
            if child is None or count>baseline or now>=cutoff+360:break
        time.sleep(5)
    assert owned(parent,'experiments.six_hour_marathon')
    os.kill(parent,signal.SIGSTOP)
    child=source_child(parent)
    if child is not None:
        assert owned(child,'experiments.six_hour_controller')
        os.kill(child,signal.SIGTERM)
        for _ in range(30):
            path=Path(f'/proc/{child}/stat')
            if not path.exists() or path.read_text().split(') ',1)[1].split()[0]=='Z':break
            time.sleep(.5)
        else:raise RuntimeError('Source worker did not stop; target work not launched')
    os.kill(parent,signal.SIGTERM);os.kill(parent,signal.SIGCONT)
    partial=[]
    for path in (root/'single').glob('*.json'):
        record=json.loads(path.read_text())
        if record.get('status')!='complete':
            atomic_json(root/f'pre_handoff_{path.stem}.json',record)
            record.update({'status':'partial_time_allocation','halted_utc':datetime.now(timezone.utc).isoformat(),
                           'reason':'preserve the final session window for the unchanged target validation and test grid'})
            atomic_json(path,record);partial.append(path.name)
    old=dict(plan);plan['reserve_seconds']=9000
    revision={'revised_utc':datetime.now(timezone.utc).isoformat(), 'previous_reserve_seconds':old['reserve_seconds'],
              'new_reserve_seconds':9000, 'source_completed_before_handoff':complete_count(root),
              'source_partial_records':partial,'source_training_and_completed_models_unchanged':True,
              'target_grid_and_selection_policy_unchanged':True,
              'target_tuning_or_test_not_started':not (root/'target_tuning.json').exists(),
              'reason':'Single-task source update timing implies that the equally budgeted target grid and all-model testing need a larger reserve.'}
    assert revision['target_tuning_or_test_not_started']
    atomic_json(root/'session_plan.before_time_allocation.json',old)
    atomic_json(plan_path,plan);atomic_json(root/'session_plan.json',plan)
    atomic_json(root/'time_allocation_revision.json',revision)
    state.update({'status':'running','completed_phases':['calibration','multi'],
                  'single_task_phase':'time_limited','source_partial_records':partial})
    for phase in ('tune','evaluate'):
        name='target_tuning' if phase=='tune' else 'target_evaluation'
        state.update({'phase':name,'updated_unix':time.time()});atomic_json(root/'target_handoff_state.json',state)
        with (root/f'{name}.log').open('a') as log:
            process=subprocess.Popen([sys.executable,'-m','experiments.longrun_transfer','--plan',plan_path,'--phase',phase],
                                     stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'4'})
            while process.poll() is None:
                if time.time()>=deadline-plan['analysis_reserve_seconds']:
                    process.terminate();process.wait(timeout=30)
                    state.update({'status':'time_limited','phase':name,'updated_unix':time.time()})
                    atomic_json(root/'target_handoff_state.json',state);return
                time.sleep(5)
            if process.returncode:
                state.update({'status':'phase_failed','phase':name,'return_code':process.returncode,'updated_unix':time.time()})
                atomic_json(root/'target_handoff_state.json',state)
                raise RuntimeError(f'{name} failed; inspect its log')
        state['completed_phases'].append(name)
    state.update({'status':'scheduled_phases_finished','phase':'awaiting_analysis','updated_unix':time.time()})
    atomic_json(root/'target_handoff_state.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=int,required=True)
    args=parser.parse_args();run(args.parent)
