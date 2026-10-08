"""Resumable, time-bounded source learning for the six-hour session."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .longrun_attention import accelerate
from .permworld_combinations import answer_logits, features, generate_data, new_model, select_groups, sha, validate


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def load_session(plan_path='configs/six_hour_session.json'):
    plan = json.loads(Path(plan_path).read_text())
    config = json.loads(Path(plan['base_config']).read_text())
    config.update({'data_seed': plan['data_seed'], 'examples_per_length': plan['examples_per_length'],
                   'pretrain_steps': plan['source_steps'], 'record_every': plan['record_every'],
                   'examples_per_task_per_step': plan['examples_per_task_per_step'],
                   'learning_rate': plan['source_learning_rate']})
    root = Path(plan['output'])
    root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(config['repository']) / 'src'))
    from neurips_permutations.math_ops import PROPERTY32_TASK_NAMES, PROPERTY_FUNCTIONS
    from neurips_permutations.passage import TOKEN_TO_ID, one_line_tokens
    names = list(PROPERTY32_TASK_NAMES)
    groups = select_groups(config)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_dir = root / 'dataset'
    dataset_dir.mkdir(exist_ok=True)
    data_config = {k: config[k] for k in ('data_seed', 'lengths', 'examples_per_length')}
    data_signature = {'config': data_config, 'generator_sha256': sha('experiments/permworld_combinations.py'),
                      'property_sha256': sha(Path(config['repository']) / 'src/neurips_permutations/math_ops.py')}
    existing = dataset_dir / 'metadata.json'
    if existing.exists():
        assert json.loads(existing.read_text())['signature'] == data_signature
    raw = generate_data(config, dataset_dir, names, PROPERTY_FUNCTIONS, TOKEN_TO_ID, one_line_tokens)
    atomic_json(existing, {'signature': data_signature, 'data_sha256': sha(dataset_dir / 'data.npz'),
                           'names': names, 'input_policy': 'globally disjoint fresh permutations across every split'})
    data = {key: torch.tensor(value, device=device) for key, value in raw.items()}
    support = {str(b): np.random.default_rng(plan['data_seed'] + 50000).permutation(len(raw['support_pool_input']))[:b].tolist() for b in config['target_budgets']}
    atomic_json(dataset_dir / 'support_indices.json', support)
    atomic_json(root / 'session_plan.json', plan)
    return plan, config, groups, data, names, TOKEN_TO_ID, device


def job_signature(plan, config, arch, job):
    source_paths = ['experiments/longrun_engine.py', 'experiments/longrun_attention.py', 'experiments/permworld_combinations.py']
    repo = Path(config['repository'])
    return {'job': job, 'architecture': arch,
            'training_policy': {key: plan[key] for key in ('source_learning_rate', 'warmup_steps', 'minimum_learning_rate_ratio', 'examples_per_task_per_step', 'record_every', 'precision')},
            'data_sha256': sha(Path(plan['output']) / 'dataset/data.npz'),
            'source_hashes': {path: sha(path) for path in source_paths},
            'upstream_hashes': {name: sha(repo / 'src/neurips_permutations' / name) for name in ('models.py', 'math_ops.py', 'passage.py')}}


def learning_rate(step, steps, plan):
    if step <= plan['warmup_steps']:
        return plan['source_learning_rate'] * step / plan['warmup_steps']
    progress = min(1., (step - plan['warmup_steps']) / max(1, steps - plan['warmup_steps']))
    ratio = plan['minimum_learning_rate_ratio'] + (1 - plan['minimum_learning_rate_ratio']) * .5 * (1 + math.cos(math.pi * progress))
    return plan['source_learning_rate'] * ratio


def train_job(plan, config, arch, job, data, names, token_ids, device, stop_at):
    root = Path(plan['output']) / job['phase']
    root.mkdir(parents=True, exist_ok=True)
    job_id = f"{arch['id']}_{job['id']}_s{job['seed']}"
    record_path, checkpoint = root / f'{job_id}.json', root / 'checkpoints' / f'{job_id}.pt'
    signature = job_signature(plan, config, arch, job)
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    model_config = {**config, **{k: arch[k] for k in ('d_model', 'layers', 'heads')}}
    model = accelerate(new_model(model_config, job['seed'], device))
    if record_path.exists():
        record = json.loads(record_path.read_text())
        assert record['fingerprint'] == fingerprint
        if record['status'] == 'complete':
            assert sha(checkpoint) == record['checkpoint_sha256']
            model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device)['model'])
            return model, record
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['source_learning_rate'], weight_decay=config['weight_decay'])
    generator = np.random.default_rng(job['seed'] + 20261005)
    curve, step, elapsed = [], 0, 0.
    if checkpoint.exists():
        state = torch.load(checkpoint, weights_only=True, map_location=device)
        assert state['fingerprint'] == fingerprint
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        generator.bit_generator.state = state['generator_state']
        curve, step, elapsed = state['curve'], state['step'], state['elapsed_seconds']
    if not curve:
        curve.extend({'step': 0, **row} for row in validate(model, data, 'validation', job['tasks'], names, token_ids))
    ntrain = data['train_lengths'].cpu().numpy()
    buckets = {n: np.flatnonzero(ntrain == n) for n in np.unique(ntrain)}
    task_ids = [names.index(t) for t in job['tasks']]
    started, stopped = time.monotonic(), False
    while step < job['steps']:
        if step % 100 == 0 and time.time() >= stop_at:
            stopped = True
            break
        step += 1
        n = int(generator.integers(config['lengths'][0], config['lengths'][1] + 1))
        indices = generator.choice(buckets[n], plan['examples_per_task_per_step'], replace=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(step, job['steps'], plan)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            logits = answer_logits(model, data['train_input'][indices], data['train_lengths'][indices], job['tasks'], token_ids)
            labels = data['train_labels'][indices][:, task_ids].T.reshape(-1)
            loss = F.cross_entropy(logits.float(), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % plan['record_every'] == 0 or step == job['steps']:
            rows = validate(model, data, 'validation', job['tasks'], names, token_ids)
            curve.extend({'step': step, **row} for row in rows)
            runtime = elapsed + time.monotonic() - started
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'generator_state': generator.bit_generator.state,
                     'curve': curve, 'step': step, 'elapsed_seconds': runtime, 'fingerprint': fingerprint}
            atomic_torch(checkpoint, state)
            if step in (5000, 10000, 20000):
                atomic_torch(root / 'checkpoints' / f'{job_id}_step{step}.pt', {'model': model.state_dict(), 'step': step, 'fingerprint': fingerprint})
                np.save(root / f'{job_id}_step{step}_features.npy', features(model, data, 'representation', token_ids))
            progress = {'job_id': job_id, 'status': 'running', 'step': step, 'planned_steps': job['steps'],
                        'validation_macro': float(np.mean([r['accuracy'] for r in rows])),
                        'seconds': runtime, 'updated_unix': time.time()}
            atomic_json(Path(plan['output']) / 'current_job.json', progress)
            print(json.dumps(progress), flush=True)
    runtime = elapsed + time.monotonic() - started
    atomic_torch(checkpoint, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'generator_state': generator.bit_generator.state,
                             'curve': curve, 'step': step, 'elapsed_seconds': runtime, 'fingerprint': fingerprint})
    record = {**signature, 'fingerprint': fingerprint, 'job_id': job_id, 'status': 'partial_deadline' if stopped else 'complete',
              'step': step, 'exposures_per_task': step * plan['examples_per_task_per_step'], 'elapsed_seconds': runtime,
              'checkpoint_sha256': sha(checkpoint), 'curve': curve,
              'source_validation': validate(model, data, 'validation', job['tasks'], names, token_ids)}
    if not stopped:
        record['source_train'] = validate(model, data, 'train', job['tasks'], names, token_ids)
        if job['phase'] != 'calibration':
            record['source_audit'] = validate(model, data, 'source_audit', job['tasks'], names, token_ids)
        np.save(root / f'{job_id}_features.npy', features(model, data, 'representation', token_ids))
    atomic_json(record_path, record)
    return model, record
