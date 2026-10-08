"""Keep authorized phases running sequentially without simultaneous GPU jobs."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

from .longrun_engine import atomic_json


def run(plan_path, wait_for_calibration=False):
    plan=json.loads(Path(plan_path).read_text())
    root=Path(plan['output']);root.mkdir(parents=True,exist_ok=True)
    if wait_for_calibration:
        deadline=datetime.fromisoformat(plan['deadline_utc']).timestamp()
        while not (root/'architecture_selection.json').exists():
            if time.time()>=deadline:
                raise RuntimeError('Calibration did not finish before the deadline')
            atomic_json(root/'marathon_state.json',{'status':'waiting_for_existing_calibration','updated_unix':time.time()})
            time.sleep(5)
    phases=[('calibration','experiments.six_hour_controller','calibration'),
            ('multi','experiments.six_hour_controller','multi'),
            ('single','experiments.six_hour_controller','single'),
            ('target_tuning','experiments.longrun_transfer','tune'),
            ('target_evaluation','experiments.longrun_transfer','evaluate')]
    state={'deadline_utc':plan['deadline_utc'],'completed_phases':[],'status':'running'}
    for name,module,argument in phases:
        state.update({'phase':name,'updated_unix':time.time()})
        atomic_json(root/'marathon_state.json',state)
        with (root/f'{name}.log').open('a') as log:
            result=subprocess.run([sys.executable,'-m',module,'--plan',plan_path,'--phase',argument],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update({'status':'phase_failed','failed_phase':name,'return_code':result.returncode,'updated_unix':time.time()})
            atomic_json(root/'marathon_state.json',state)
            raise RuntimeError(f'{name} failed; inspect {root/name}.log')
        state['completed_phases'].append(name)
        atomic_json(root/'marathon_state.json',state)
    state.update({'status':'scheduled_phases_finished','phase':'awaiting_analysis','updated_unix':time.time()})
    atomic_json(root/'marathon_state.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',default='configs/six_hour_session.json')
    parser.add_argument('--wait-for-calibration',action='store_true')
    args=parser.parse_args()
    run(args.plan,args.wait_for_calibration)
