"""Insert one secondary GPU phase after main sources, then resume supervision."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .longrun_engine import atomic_json


def process_state(pid):
    path=Path(f'/proc/{pid}/stat')
    if not path.exists():return None
    return path.read_text().split(') ',1)[1].split()[0]


def run(parent,child):
    plan=json.loads(Path('configs/six_hour_session.json').read_text())
    root=Path(plan['output'])
    assert 'six_hour_marathon' in Path(f'/proc/{parent}/cmdline').read_bytes().decode().replace('\0',' ')
    assert 'six_hour_controller' in Path(f'/proc/{child}/cmdline').read_bytes().decode().replace('\0',' ')
    def interrupted(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    stopped=False
    try:
        os.kill(parent,signal.SIGSTOP);stopped=True
        atomic_json(root/'secondary_state.json',{'status':'waiting_for_primary_source','supervisor_pid':parent,'source_pid':child,'updated_unix':time.time()})
        deadline=datetime.fromisoformat(plan['deadline_utc']).timestamp()
        while process_state(child) not in (None,'Z'):
            if time.time()>=deadline-2*plan['reserve_seconds']:return
            time.sleep(5)
        queue=json.loads((root/'secondary_queue.json').read_text())
        for phase in queue:
            assert phase['module'].startswith('experiments.')
            atomic_json(root/'secondary_state.json',{'status':phase['name']+'_running','supervisor_pid':parent,'updated_unix':time.time()})
            with (root/(phase['name']+'.log')).open('a') as log:
                result=subprocess.run([sys.executable,'-m',phase['module']],stdout=log,stderr=subprocess.STDOUT,
                                      env={**os.environ,'OPENBLAS_NUM_THREADS':'4'})
            if result.returncode:break
        atomic_json(root/'secondary_state.json',{'status':'complete' if result.returncode==0 else 'failed',
                                                'return_code':result.returncode,'updated_unix':time.time()})
    finally:
        if stopped and process_state(parent) is not None:os.kill(parent,signal.SIGCONT)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=int,required=True);parser.add_argument('--child',type=int,required=True)
    args=parser.parse_args();run(args.parent,args.child)
