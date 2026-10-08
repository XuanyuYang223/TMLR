"""Run the still-unstarted n5 fits concurrently using the frozen trainer.

Only execution order changes. The original numerical training routine, schedules,
optimizer, model initialization, seeds, losses and endpoints are reused. A guard
stops this worker when the serial trainer reaches n4, well before it can reach n5.
Complete checkpoints are skipped by the serial trainer; partial checkpoints retain
optimizer and RNG state and are resumed there. Global completion/current-job
markers are isolated so the serial controller remains the sole completion owner.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import threading
import time

from . import overnight_inverse_functional as trainer
from .inverse_functional_alignment import now
from .longrun_engine import atomic_json
from .permworld_combinations import sha


def run():
    plan, root, signature = trainer.initialize()
    folder = root / 'parallel_n5'
    folder.mkdir(exist_ok=True)
    registration = folder / 'protocol.json'
    assert not registration.exists(), 'This one-use execution wrapper has already run.'
    assert not (root / 'test_opened.json').exists()
    job = json.loads((root / 'current_job.json').read_text())
    assert job['replicate']['id'] in ['n0', 'n1', 'n2', 'n3']
    assert not list((root / 'training').glob('n5_*.json'))
    assert not list((root / 'checkpoints').glob('n5_*.pt'))
    atomic_json(registration, {
        'registered_utc': now(), 'code_sha256': sha(__file__),
        'trainer_sha256': sha(trainer.__file__),
        'main_protocol_sha256': sha(root / 'protocol.json'),
        'pid': os.getpid(), 'execution_scope': 'All six still-unstarted n5 conditions.',
        'numerical_changes': [], 'main_test_opened': False,
        'collision_guard': 'Stop at next100-step checkpoint as soon as serial current_job reaches n4 or n5; original serial controller remains completion owner.',
        'resume_rule': 'Serial trainer skips complete records or resumes latest model+optimizer+CPU/CUDA RNG for partial fits.',
        'ownership': 'Only n5 schedule, checkpoints and fit records are written in main training directories. Worker status/current_job markers remain isolated here.',
    })
    atomic_json(folder / 'state.json', {'status': 'running', 'started_utc': now(), 'pid': os.getpid()})
    done = threading.Event()
    reason = []

    def guard():
        while not done.wait(.5):
            try:
                current = json.loads((root / 'current_job.json').read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            if current.get('replicate', {}).get('id') in ['n4', 'n5']:
                reason.append('serial_trainer_reached_' + current['replicate']['id'])
                trainer.STOP_REQUESTED = True
                return

    # This binding is process-local; the immutable trainer file is untouched.
    original_writer = trainer.atomic_json

    def isolated_writer(path, value):
        path = Path(path)
        if path == root / 'training/state.json':
            path = folder / 'training_state.json'
        elif path == root / 'current_job.json':
            path = folder / 'current_job.json'
        original_writer(path, value)

    trainer.atomic_json = isolated_writer
    selected = deepcopy(plan)
    selected['replicates'] = [rep for rep in plan['replicates'] if rep['id'] == 'n5']
    assert len(selected['replicates']) == 1
    thread = threading.Thread(target=guard, daemon=True)
    thread.start()
    try:
        trainer.train(selected, root, signature)
        completed = []
        partial = []
        for file in sorted((root / 'training').glob('n5_*.json')):
            record = json.loads(file.read_text())
            (completed if record['status'] == 'complete' else partial).append(
                {'file': str(file), 'step': record['step'], 'status': record['status']})
        atomic_json(folder / 'state.json', {
            'status': 'complete' if len(completed) == 6 else 'returned_to_serial_trainer',
            'finished_utc': now(), 'pid': os.getpid(), 'stop_reason': reason,
            'completed': completed, 'partial': partial,
            'worker_never_opened_test': not (root / 'test_opened.json').exists(),
        })
    except BaseException as error:
        atomic_json(folder / 'state.json', {'status': 'failed', 'finished_utc': now(),
                    'pid': os.getpid(), 'error': repr(error)})
        raise
    finally:
        done.set()
        thread.join(timeout=1)
        trainer.atomic_json = original_writer


if __name__ == '__main__':
    run()
