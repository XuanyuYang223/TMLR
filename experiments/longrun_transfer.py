"""Validation-selected adaptation and frozen readouts for the long-run cohort.

Target test labels are used only after equally budgeted condition-specific
validation tuning. A shared random-selected fine-tuning policy is also reported.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .longrun_attention import accelerate
from .longrun_engine import atomic_json, load_session
from .permworld_combinations import answer_logits, features, new_model, prompts, sha, validate


@torch.no_grad()
def query_features(model, data, split, target, token_ids):
    model.eval()
    rows = []
    for start in range(0, len(data[f'{split}_input']), 128):
        x, n = data[f'{split}_input'][start:start+128], data[f'{split}_lengths'][start:start+128]
        ids, mask, position = prompts(x, n, [target], token_ids)
        hidden, valid = model._embed_inputs(ids, mask)
        for block in model.blocks:
            hidden = block(hidden, valid)
        hidden = model.final_norm(hidden)
        rows.append(hidden[torch.arange(len(ids), device=ids.device), position].cpu().numpy())
    return np.concatenate(rows)


def make_model(config, arch, seed, device):
    return accelerate(new_model({**config, **{k: arch[k] for k in ('d_model', 'layers', 'heads')}}, seed, device))


def normalized_features(train, others):
    mean, sigma = train.mean(0), train.std(0)
    sigma = np.maximum(sigma, 1e-4)
    return (train-mean)/sigma, [(x-mean)/sigma for x in others]


def probe_curve(train, train_y, validation, validation_y, width, classes, lr, milestones, seed, kind, device):
    train, (validation,) = normalized_features(train, [validation])
    torch.manual_seed(seed + 70000)
    head = nn.Linear(width, classes).to(device) if kind == 'linear' else nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, classes)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=.01)
    x, xv = torch.tensor(train, device=device), torch.tensor(validation, device=device)
    values = []
    for step in range(1, max(milestones)+1):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(head(x), train_y).backward()
        optimizer.step()
        if step in milestones:
            with torch.no_grad():
                values.append({'steps': step, 'validation_accuracy': float((head(xv).argmax(-1) == validation_y).float().mean().cpu())})
    return values


def fit_probe(train, train_y, test, truth, width, classes, policy, seed, kind, device):
    train, (test,) = normalized_features(train, [test])
    torch.manual_seed(seed + 70000)
    head = nn.Linear(width, classes).to(device) if kind == 'linear' else nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, classes)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=policy['learning_rate'], weight_decay=.01)
    x, xt = torch.tensor(train, device=device), torch.tensor(test, device=device)
    for _ in range(policy['steps']):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(head(x), train_y).backward()
        optimizer.step()
    with torch.no_grad():
        return {'test_accuracy': float((head(xt).argmax(-1) == truth).float().mean().cpu()),
                'support_accuracy': float((head(x).argmax(-1) == train_y).float().mean().cpu())}


def finetune_curve(model, config, data, target, indices, seed, lr, milestones, names, token_ids, final_test=False):
    learner = copy.deepcopy(model)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(learner.parameters(), lr=lr, weight_decay=config['weight_decay'])
    rng = np.random.default_rng(seed + 70000)
    rows = []
    for step in range(1, max(milestones)+1):
        chosen = rng.choice(indices, min(config['adapt_batch_size'], len(indices)), replace=False)
        learner.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = answer_logits(learner, data['support_pool_input'][chosen], data['support_pool_lengths'][chosen], [target], token_ids)
            labels = data['support_pool_labels'][chosen, names.index(target)]
            F.cross_entropy(logits.float(), labels).backward()
        torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.)
        optimizer.step()
        if step in milestones:
            row = {'steps': step, 'validation_accuracy': validate(learner, data, 'target_validation', [target], names, token_ids)[0]['accuracy']}
            rows.append(row)
    if final_test:
        return {'test_accuracy': validate(learner, data, 'target_test', [target], names, token_ids)[0]['accuracy'],
                'validation_accuracy': rows[-1]['validation_accuracy']}
    return rows


def tune(plan_path):
    plan, config, groups, data, names, token_ids, device = load_session(plan_path)
    root = Path(plan['output'])
    arch = json.loads((root/'architecture_selection.json').read_text())['selected']
    support = json.loads((root/'dataset/support_indices.json').read_text())
    signature = {'plan': plan, 'architecture': arch, 'code_sha256': sha(__file__),
                 'data_sha256': sha(root/'dataset/data.npz'), 'seed': 17,
                 'probe_preprocessing': 'support-only feature mean/std with floor 1e-4; applied identically to pretrained and random models'}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    path = root/'target_tuning.json'
    if path.exists():
        record = json.loads(path.read_text())
        assert record['fingerprint'] == fingerprint
        if record['status'] == 'complete':
            return
    rows = []
    policies = {}
    for condition in ['random']+[g['id'] for g in groups]:
        model = make_model(config, arch, 17, device)
        if condition != 'random':
            run_id=f"{arch['id']}_{condition}_s17"
            record=json.loads((root/'multi'/f'{run_id}.json').read_text())
            assert record['status']=='complete'
            checkpoint=root/'multi'/'checkpoints'/f'{run_id}.pt'
            assert sha(checkpoint)==record['checkpoint_sha256']
            model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location=device)['model'])
        free = {split: features(model, data, split, token_ids) for split in ('support_pool', 'target_validation')}
        for target in config['target_tasks']:
            conditional = {split: query_features(model, data, split, target, token_ids) for split in ('support_pool', 'target_validation')}
            for budget in config['target_budgets']:
                ids = support[str(budget)]
                labels = data['support_pool_labels'][ids, names.index(target)]
                validation_y = data['target_validation_labels'][:, names.index(target)]
                for lr in plan['target_adaptation_lrs']:
                    for row in finetune_curve(model, config, data, target, ids, 17, lr, plan['target_adaptation_steps'], names, token_ids):
                        rows.append({'condition':condition,'mode': 'finetune', 'target': target, 'budget': budget, 'learning_rate': lr, **row})
                for mode in ('linear', 'linear_query', 'mlp'):
                    kind = 'linear' if mode == 'linear_query' else mode
                    selected_features = conditional if mode == 'linear_query' else free
                    for lr in (.003, .01, .03):
                        for row in probe_curve(selected_features['support_pool'][ids], labels, selected_features['target_validation'], validation_y,
                                               arch['d_model'], config['lengths'][1]+1, lr, [200, 1000], 17, kind, device):
                            rows.append({'condition':condition,'mode': mode, 'target': target, 'budget': budget, 'learning_rate': lr, **row})
                atomic_json(path, {**signature, 'fingerprint': fingerprint, 'status': 'running', 'rows': rows})
                print(f'target validation tuning: {condition}/{target}, budget={budget}', flush=True)
        policy={}
        for mode in ('finetune','linear','linear_query','mlp'):
            selected_rows=[r for r in rows if r['condition']==condition and r['mode']==mode]
            pairs=sorted({(r['learning_rate'],r['steps']) for r in selected_rows})
            scores=[{'learning_rate':lr,'steps':steps,'validation_macro':float(np.mean([r['validation_accuracy'] for r in selected_rows if r['learning_rate']==lr and r['steps']==steps]))} for lr,steps in pairs]
            policy[mode]=max(scores,key=lambda r:(r['validation_macro'],-r['steps'],-r['learning_rate']))
        policies[condition]=policy
    atomic_json(path, {**signature, 'fingerprint': fingerprint, 'status': 'complete', 'rows': rows,
                      'policies':policies,'policy':policies['random'],'selection_policy': plan['target_tuning_policy'], 'frozen_unix': time.time()})
    print(json.dumps({'selected_adaptation':policies},indent=2),flush=True)


def evaluate_model(model, record, plan, config, data, names, token_ids, root, support, tuning, group, seed):
    run_id = record['job_id'] if record else f'random_s{seed}'
    directory = root/'transfer'
    directory.mkdir(exist_ok=True)
    path = directory/f'{run_id}.json'
    signature = {'run_id': run_id, 'source_fingerprint': record['fingerprint'] if record else None,
                 'tuning_fingerprint': tuning['fingerprint'], 'code_sha256': sha(__file__),
                 'data_sha256': sha(root/'dataset/data.npz'), 'policy': tuning['policy'], 'seed': seed}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    if path.exists():
        existing = json.loads(path.read_text())
        assert existing['fingerprint'] == fingerprint
        if existing['status'] == 'complete':
            return existing
    free = {split: features(model, data, split, token_ids) for split in ('support_pool', 'target_test')}
    rows = []
    policy=tuning['policies'][group]
    for target in config['target_tasks']:
        conditional = {split: query_features(model, data, split, target, token_ids) for split in ('support_pool', 'target_test')}
        for budget in config['target_budgets']:
            ids = support[str(budget)]
            labels = data['support_pool_labels'][ids, names.index(target)]
            truth = data['target_test_labels'][:, names.index(target)]
            for kind, landmark in (('linear', 'task_free'), ('linear', 'query'), ('mlp', 'task_free')):
                f = free if landmark=='task_free' else conditional
                selected_policy = policy['linear_query'] if landmark == 'query' else policy[kind]
                result = fit_probe(f['support_pool'][ids], labels, f['target_test'], truth,
                                   model.config.d_model, config['lengths'][1]+1, selected_policy, seed, kind, truth.device)
                rows.append({'target': target, 'budget': budget, 'mode': f'{kind}_{landmark}', **result})
            result = finetune_curve(model, config, data, target, ids, seed,
                                    policy['finetune']['learning_rate'],[policy['finetune']['steps']],names,token_ids,final_test=True)
            rows.append({'target': target, 'budget': budget, 'mode': 'finetune', **result})
            shared=tuning['policy']['finetune']
            same=shared['learning_rate']==policy['finetune']['learning_rate'] and shared['steps']==policy['finetune']['steps']
            shared_result=result if same else finetune_curve(model,config,data,target,ids,seed,shared['learning_rate'],[shared['steps']],names,token_ids,final_test=True)
            rows.append({'target':target,'budget':budget,'mode':'finetune_shared',**shared_result})
        atomic_json(path, {**signature, 'fingerprint': fingerprint, 'status': 'running', 'group': group, 'rows': rows})
        print(f'target test evaluation: {run_id} / {target}', flush=True)
    result = {**signature, 'fingerprint': fingerprint, 'status': 'complete', 'group': group, 'rows': rows,
              'test_policy':'condition-specific policies fixed with equal validation search; shared random-selected fine-tuning also reported'}
    atomic_json(path, result)
    return result


def evaluate(plan_path):
    plan, config, groups, data, names, token_ids, device = load_session(plan_path)
    root = Path(plan['output'])
    arch = json.loads((root/'architecture_selection.json').read_text())['selected']
    tuning = json.loads((root/'target_tuning.json').read_text())
    assert tuning['status']=='complete' and tuning['code_sha256']==sha(__file__)
    support = json.loads((root/'dataset/support_indices.json').read_text())
    deadline = datetime.fromisoformat(plan['deadline_utc']).timestamp()
    for seed in plan['model_seeds']:
        if time.time() > deadline-plan['analysis_reserve_seconds']:
            break
        model = make_model(config, arch, seed, device)
        evaluate_model(model, None, plan, config, data, names, token_ids, root, support, tuning, 'random', seed)
        for group in groups:
            if time.time() > deadline-plan['analysis_reserve_seconds']:
                return
            job_id = f"{arch['id']}_{group['id']}_s{seed}"
            record_path = root/'multi'/f'{job_id}.json'
            if not record_path.exists():
                continue
            record = json.loads(record_path.read_text())
            if record['status']!='complete':
                continue
            model = make_model(config, arch, seed, device)
            checkpoint = root/'multi'/'checkpoints'/f'{job_id}.pt'
            assert sha(checkpoint)==record['checkpoint_sha256']
            model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device)['model'])
            evaluate_model(model, record, plan, config, data, names, token_ids, root, support, tuning, group['id'], seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default='configs/six_hour_session.json')
    parser.add_argument('--phase', choices=('tune','evaluate'), default='tune')
    args=parser.parse_args()
    (tune if args.phase=='tune' else evaluate)(args.plan)
