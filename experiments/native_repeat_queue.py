"""Finish first-seed controls, then temporarily hand GPU to frozen repeat."""
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
from .permworld_combinations import sha


def owned(pid):
    path = Path(f'/proc/{pid}/cmdline')
    return path.exists() and b'experiments.resume_optional_controls' in path.read_bytes()


def count(root):
    return sum(json.loads(p.read_text())['status'] == 'complete' for p in (root/'single').glob('*.json'))


def run(source_pid):
    plan = json.loads(Path('configs/six_hour_session.json').read_text()); root = Path(plan['output'])
    deadline = datetime.fromisoformat(plan['deadline_utc']).timestamp()
    state = {'status': 'waiting_for_first_seed_controls', 'source_pid': source_pid,
             'first_seed_completed_control_threshold': 18, 'code_sha256': sha(__file__),
             'policy': 'finish all 18 original first-seed controls, pause only owned GPU worker, run all 27 frozen target-data repeats, resume optional controls',
             'same_source_model_tasks_and_target_policies': True,
             'timing_note': 'paused source-job elapsed wall time includes this repeat window; update counts and sampling are unchanged'}
    assert owned(source_pid)
    while count(root) < 18:
        state.update({'completed_controls': count(root), 'updated_unix': time.time()})
        atomic_json(root/'native_repeat_queue_state.json', state)
        if time.time() > deadline-2400 or not owned(source_pid):
            state['status'] = 'not_started_insufficient_window'; atomic_json(root/'native_repeat_queue_state.json', state); return
        time.sleep(10)
    assert owned(source_pid)
    state.update({'status': 'pausing_owned_source_worker', 'updated_unix': time.time()})
    atomic_json(root/'native_repeat_queue_state.json', state)
    os.kill(source_pid, signal.SIGSTOP)
    try:
        state.update({'status': 'running_frozen_target_repeat', 'started_unix': time.time()})
        atomic_json(root/'native_repeat_queue_state.json', state)
        with (root/'native_target_repeat.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-m', 'experiments.native_target_repeat', '--phase', 'evaluate'],
                                       stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, 'OPENBLAS_NUM_THREADS': '4', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
            while process.poll() is None:
                if time.time() >= deadline-plan['analysis_reserve_seconds']:
                    process.terminate(); process.wait(timeout=30); break
                time.sleep(5)
            state.update({'return_code': process.returncode, 'finished_unix': time.time()})
    finally:
        if owned(source_pid): os.kill(source_pid, signal.SIGCONT)
    state['status'] = 'repeat_completed_source_resumed' if state.get('return_code') == 0 else 'repeat_incomplete_source_resumed'
    atomic_json(root/'native_repeat_queue_state.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source-pid', type=int, required=True)
    run(parser.parse_args().source_pid)
