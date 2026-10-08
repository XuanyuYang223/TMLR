"""Refresh partial-safe reports when completed artifacts or tuning rows change."""
from datetime import datetime
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from .longrun_engine import atomic_json


def run():
    plan=json.loads(Path('configs/six_hour_session.json').read_text());root=Path(plan['output'])
    deadline=datetime.fromisoformat(plan['deadline_utc']).timestamp()
    stop=False
    def interrupted(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    previous=None
    while not stop and time.time()<deadline and not (root/'stop_report_monitor').exists():
        signature=tuple((phase,tuple((p.name,p.stat().st_mtime_ns) for p in sorted((root/phase).glob('*.json')))) for phase in ('multi','single','transfer'))
        tuning=root/'target_tuning.json';handoff=root/'target_handoff_state.json'
        signature+=(tuning.stat().st_mtime_ns if tuning.exists() else None,)
        # Handoff updates its waiting state often; only substantive phase changes
        # should trigger a full report rather than every polling timestamp.
        if handoff.exists():
            state=json.loads(handoff.read_text());signature+=(state.get('status'),state.get('phase'))
        queue=root/'native_repeat_queue_state.json'
        if queue.exists():
            state=json.loads(queue.read_text());signature+=(state.get('status'),state.get('completed_controls'))
        repeat=root.parent/'native_target_repeat'
        signature+=(tuple((p.name,p.stat().st_mtime_ns) for p in sorted((repeat/'transfer').glob('*.json'))),)
        for path in (root/'transfer_prediction/summary.json', root/'transfer_prediction/indicator_corrected_summary.json', repeat/'summary.json'):
            signature+=(path.stat().st_mtime_ns if path.exists() else None,)
        if signature!=previous:
            atomic_json(root/'report_monitor_state.json',{'status':'updating','updated_unix':time.time()})
            with (root/'report_monitor.log').open('a') as log:
                process=subprocess.run([sys.executable,'-m','experiments.six_hour_report','--skip-geometry'],stdout=log,stderr=subprocess.STDOUT)
            atomic_json(root/'report_monitor_state.json',{'status':'idle' if process.returncode==0 else 'report_failed',
                                                        'return_code':process.returncode,'updated_unix':time.time()})
            previous=signature
        time.sleep(20)
    atomic_json(root/'report_monitor_state.json',{'status':'stopped','updated_unix':time.time()})


if __name__=='__main__':run()
