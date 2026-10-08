"""Run the immutable first overnight assay; keep later autonomy separate."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from .longrun_engine import atomic_json
from .permworld_combinations import sha


def run():
    plan=json.loads(Path('configs/overnight_inverse_functional.json').read_text());root=Path(plan['output'])
    stages=[('confirm','experiments.overnight_inverse_functional','confirm'),
            ('teachers','experiments.overnight_inverse_functional','teachers'),
            ('modes','experiments.overnight_inverse_analysis','modes'),
            ('train','experiments.overnight_inverse_functional','train'),
            ('evaluate','experiments.overnight_inverse_functional','evaluate'),
            ('analysis','experiments.overnight_inverse_analysis','analyze')]
    controller={'started_utc':datetime.now(timezone.utc).isoformat(),'code_sha256':sha(__file__),
                'deadline_utc':plan['deadline_utc'],'stages':stages,'pid':__import__('os').getpid()}
    atomic_json(root/'controller.json',controller)
    for name,module,phase in stages:
        if name=='evaluate' and not (root/'training/state.json').exists():
            atomic_json(root/'controller_state.json',{'status':'training_partial_deadline','updated_utc':datetime.now(timezone.utc).isoformat()});return
        atomic_json(root/'controller_state.json',{'status':'running','stage':name,'updated_utc':datetime.now(timezone.utc).isoformat()})
        with open(root/f'{name}.log','a') as handle:
            result=subprocess.run([sys.executable,'-u','-m',module,phase],stdout=handle,stderr=subprocess.STDOUT)
        if result.returncode:
            atomic_json(root/'controller_state.json',{'status':'failed','stage':name,'exit_code':result.returncode,'updated_utc':datetime.now(timezone.utc).isoformat()})
            raise RuntimeError(f'Stage {name} failed; see {root/name}.log')
    atomic_json(root/'controller_state.json',{'status':'base_assay_complete_timed_goal_still_active','updated_utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':run()
