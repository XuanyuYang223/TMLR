"""Coordinate the authorized six-hour research session with durable state."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from .longrun_engine import atomic_json, load_session, train_job


def run(plan_path, phase):
    plan, config, groups, data, names, token_ids, device = load_session(plan_path)
    deadline = datetime.fromisoformat(plan['deadline_utc']).timestamp()
    root = Path(plan['output'])
    if phase == 'calibration':
        for arch in plan['architectures']:
            for group_id in plan['calibration_groups']:
                group = next(g for g in groups if g['id'] == group_id)
                job = {'phase': 'calibration', 'id': group_id, 'seed': plan['calibration_seed'], 'tasks': group['tasks'], 'steps': plan['calibration_steps']}
                _, record = train_job(plan, config, arch, job, data, names, token_ids, device, deadline - plan['reserve_seconds'])
                if record['status'] != 'complete':
                    return
        scores = []
        for arch in plan['architectures']:
            values = []
            for group in plan['calibration_groups']:
                record = json.loads((root / 'calibration' / f"{arch['id']}_{group}_s{plan['calibration_seed']}.json").read_text())
                values.extend(row['accuracy'] for row in record['source_validation'])
            scores.append({'architecture': arch, 'source_validation_macro': float(np.mean(values))})
        selected = max(scores, key=lambda row: (row['source_validation_macro'], -row['architecture']['d_model']))
        atomic_json(root / 'architecture_selection.json', {'scores': scores, 'selected': selected['architecture'],
                                                          'selection_uses': 'source validation only; no target labels/results', 'selected_unix': time.time()})
        print(json.dumps({'architecture_selection': selected}, indent=2), flush=True)
    elif phase in ('multi', 'single'):
        arch = json.loads((root / 'architecture_selection.json').read_text())['selected']
        tasks = sorted({t for g in groups for t in g['tasks']}, key=names.index)
        for seed in plan['model_seeds']:
            designs = [(g['id'], g['tasks']) for g in groups] if phase == 'multi' else [(t, [t]) for t in tasks]
            for group_id, group_tasks in designs:
                job = {'phase': phase, 'id': group_id, 'seed': seed, 'tasks': group_tasks, 'steps': plan['source_steps']}
                _, record = train_job(plan, config, arch, job, data, names, token_ids, device, deadline - plan['reserve_seconds'])
                if record['status'] != 'complete':
                    return
    else:
        raise ValueError(phase)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default='configs/six_hour_session.json')
    parser.add_argument('--phase', choices=['calibration', 'multi', 'single'], default='calibration')
    args = parser.parse_args()
    run(args.plan, args.phase)
