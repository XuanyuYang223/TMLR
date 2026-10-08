"""Train new real PermWorld combinations on fresh, mutually disjoint inputs.

Reuse the upstream causal Transformer, Passage vocabulary and exact property
functions. The deliberately smaller answer-token-only pilot is kept separate
from the original full-scale results. Selection never consults neural outcomes.
"""
import argparse
import copy
import csv
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .analysis import linear_cka, write_csv
from .permutation_audit import IDENTITIES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_groups(config):
    with Path(config['candidate_table']).open() as handle:
        candidates = list(csv.DictReader(handle))
    by_tasks = {frozenset(r['tasks'].split('|')): r for r in candidates}
    fixed = [
        ('position_three', 'position', 3, ['fixed_points', 'exceedances', 'deficiencies', 'right_to_left_minima']),
        ('position_four', 'position', 4, ['exceedances', 'deficiencies', 'cycle_count', 'nontrivial_cycle_count']),
        ('cycle_three', 'cycle', 3, ['fixed_points', 'cycle_count', 'nontrivial_cycle_count', 'components']),
        ('cycle_four', 'cycle', 4, ['fixed_points', 'even_cycle_count', 'odd_cycle_count', 'nontrivial_cycle_count']),
        ('interior_four', 'interior', 4, ['peaks', 'valleys', 'double_ascents', 'double_descents']),
    ]
    groups = []
    for name, family, size, tasks in fixed:
        row = by_tasks[frozenset(tasks)]
        assert int(row['minimum_known_constraint_size']) == size
        assert not set(tasks) & set(config['target_tasks'])
        groups.append({'id': name, 'family': family, 'minimum_constraint_size': size, 'tasks': tasks, 'selection_statistics': row})
    for family in ('position', 'cycle', 'interior'):
        references = [g for g in groups if g['family'] == family]
        edge_counts = {int(g['selection_statistics']['expanded_affine_pair_count']) for g in references}
        assert len(edge_counts) == 1
        eligible = [r for r in candidates if int(r['known_joint_constraint_count']) == 0
                    and int(r['expanded_affine_pair_count']) in edge_counts
                    and not set(r['tasks'].split('|')) & set(config['target_tasks'])
                    and all(set(r['tasks'].split('|')) != set(g['tasks']) for g in groups)]
        def distance(row):
            return float(np.mean([sum(weight * abs(float(row[key]) - float(g['selection_statistics'][key]))
                                     for key, weight in [('mean_abs_within_length_correlation', 1), ('max_abs_within_length_correlation', 1), ('mean_conditional_entropy', .5)])
                                  for g in references]))
        selected = min(eligible, key=lambda row: (distance(row), row['tasks']))
        groups.append({'id': f'{family}_none', 'family': family, 'minimum_constraint_size': 0,
                       'tasks': selected['tasks'].split('|'), 'selection_statistics': selected,
                       'null_match_distance': distance(selected)})
    return sorted(groups, key=lambda g: (g['family'], g['minimum_constraint_size']))


def generate_data(config, output, names, functions, token_ids, one_line_tokens):
    """Exact labels with distinct permutations across every experimental split."""
    path = output / 'data.npz'
    if path.exists():
        return {key: value for key, value in np.load(path).items()}
    rng = np.random.default_rng(config['data_seed'])
    seen, data = set(), {}
    lo, hi = config['lengths']
    for split, per_length in config['examples_per_length'].items():
        inputs, labels, lengths = [], [], []
        for n in range(lo, hi + 1):
            for _ in range(per_length):
                while True:
                    permutation = tuple(map(int, rng.permutation(n) + 1))
                    if permutation not in seen:
                        seen.add(permutation)
                        break
                prefix = [token_ids['<BOS>'], token_ids['<SIZE>'], n]
                prefix += [token_ids[token] for token in one_line_tokens(permutation)]
                padded = prefix + [token_ids['<PAD>']] * (2 * hi + 4 - len(prefix))
                inputs.append(padded)
                labels.append([functions[name](permutation) for name in names])
                lengths.append(n)
        data[f'{split}_input'] = np.array(inputs, dtype=np.int64)
        data[f'{split}_labels'] = np.array(labels, dtype=np.int64)
        data[f'{split}_lengths'] = np.array(lengths, dtype=np.int64)
        print(f'data {split}: {len(inputs)} fresh inputs', flush=True)
    for split in config['examples_per_length']:
        y, n = data[f'{split}_labels'], data[f'{split}_lengths']
        for identity in IDENTITIES:
            total = sum(coefficient * y[:, names.index(task)] for task, coefficient in identity['terms'].items())
            assert np.all(total == identity['n'] * n + identity['constant'])
    np.savez_compressed(path, **data)
    return data


def new_model(config, seed, device):
    from neurips_permutations.models import CausalTransformer
    from neurips_permutations.passage import VOCABULARY
    torch.manual_seed(seed)
    return CausalTransformer(len(VOCABULARY), 2 * config['lengths'][1] + 6,
                             d_model=config['d_model'], layers=config['layers'],
                             dropout=config['dropout'], n_heads=config['heads'],
                             mlp_ratio=config['ff_multiplier'], tie_embeddings=True).to(device)


def prompts(prefixes, lengths, tasks, token_ids):
    """Append task and equals; gather answer logits at the equals position."""
    batch, width = prefixes.shape
    ids = torch.full((batch * len(tasks), width + 2), token_ids['<PAD>'], dtype=torch.long, device=prefixes.device)
    positions = 2 * lengths + 4
    for k, task in enumerate(tasks):
        block = slice(k * batch, (k + 1) * batch)
        ids[block, :width] = prefixes
        indices = torch.arange(batch, device=prefixes.device) + k * batch
        ids[indices, positions] = token_ids[f'<{task.upper()}>']
        ids[indices, positions + 1] = token_ids['=']
    mask = ids != token_ids['<PAD>']
    # Length bucketing at training avoids padded attention computation.
    stop = int(positions.max().item()) + 2
    return ids[:, :stop], mask[:, :stop], (positions + 1).repeat(len(tasks))


def answer_logits(model, prefixes, lengths, tasks, token_ids):
    ids, mask, positions = prompts(prefixes, lengths, tasks, token_ids)
    values = model(ids, mask)
    return values[torch.arange(len(ids), device=ids.device), positions]


@torch.no_grad()
def validate(model, data, split, tasks, names, token_ids):
    model.eval()
    correct, count, ce = np.zeros(len(tasks)), 0, np.zeros(len(tasks))
    for start in range(0, len(data[f'{split}_input']), 64):
        x = data[f'{split}_input'][start:start + 64]
        n = data[f'{split}_lengths'][start:start + 64]
        y = data[f'{split}_labels'][start:start + 64][:, [names.index(t) for t in tasks]].T
        logits = answer_logits(model, x, n, tasks, token_ids)
        labels = y.reshape(-1)
        loss = F.cross_entropy(logits, labels, reduction='none').view(len(tasks), -1)
        prediction = logits.argmax(-1).view(len(tasks), -1)
        correct += (prediction == y).sum(-1).cpu().numpy()
        ce += loss.sum(-1).cpu().numpy()
        count += len(x)
    return [{'task': task, 'accuracy': float(c / count), 'cross_entropy': float(l / count)} for task, c, l in zip(tasks, correct, ce)]


@torch.no_grad()
def features(model, data, split, token_ids):
    """Task-free hidden state at ONE_END; test labels never enter features."""
    model.eval()
    values = []
    for start in range(0, len(data[f'{split}_input']), 128):
        x = data[f'{split}_input'][start:start + 128]
        n = data[f'{split}_lengths'][start:start + 128]
        mask = x != token_ids['<PAD>']
        hidden, valid = model._embed_inputs(x, mask)
        for block in model.blocks:
            hidden = block(hidden, valid)
        hidden = model.final_norm(hidden)
        values.append(hidden[torch.arange(len(x), device=x.device), 2 * n + 3].cpu().numpy())
    return np.concatenate(values)


def source_train(config, group, seed, data, names, token_ids, output, fingerprint, device):
    run_id = f"{group['id']}_s{seed}"
    record = output / f'{run_id}.json'
    checkpoint = output / 'checkpoints' / f'{run_id}.pt'
    model = new_model(config, seed, device)
    if record.exists():
        saved = json.loads(record.read_text())
        assert saved['fingerprint'] == fingerprint and sha(checkpoint) == saved['checkpoint_sha256']
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
        return model, saved
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    generator = np.random.default_rng(seed + 20261005)
    indices = {n: np.flatnonzero(data['train_lengths'].cpu().numpy() == n) for n in range(config['lengths'][0], config['lengths'][1] + 1)}
    tasks, curve = group['tasks'], []
    curve.extend({'step': 0, **row} for row in validate(model, data, 'validation', tasks, names, token_ids))
    started = time.monotonic()
    for step in range(1, config['pretrain_steps'] + 1):
        n = int(generator.integers(config['lengths'][0], config['lengths'][1] + 1))
        chosen = generator.choice(indices[n], config['examples_per_task_per_step'], replace=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = answer_logits(model, data['train_input'][chosen], data['train_lengths'][chosen], tasks, token_ids)
        labels = data['train_labels'][chosen][:, [names.index(t) for t in tasks]].T.reshape(-1)
        F.cross_entropy(logits, labels).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % config['record_every'] == 0 or step == config['pretrain_steps']:
            rows = validate(model, data, 'validation', tasks, names, token_ids)
            curve.extend({'step': step, **row} for row in rows)
            print(f"{run_id} step={step} validation={np.mean([r['accuracy'] for r in rows]):.3f} seconds={time.monotonic()-started:.1f}", flush=True)
    checkpoint.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    saved = {'run_id': run_id, 'group': group['id'], 'seed': seed, 'fingerprint': fingerprint,
             'checkpoint_sha256': sha(checkpoint), 'curve': curve,
             'source_train': validate(model, data, 'train', tasks, names, token_ids),
             'source_validation': validate(model, data, 'validation', tasks, names, token_ids),
             'exposures_per_task': config['pretrain_steps'] * config['examples_per_task_per_step']}
    record.write_text(json.dumps(saved, indent=2))
    return model, saved


def adapt(model, config, data, target, ids, seed, token_ids, names):
    learner = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(learner.parameters(), lr=config['adapt_learning_rate'], weight_decay=config['weight_decay'])
    rng = np.random.default_rng(seed + 70000)
    for _ in range(config['adapt_steps']):
        chosen = rng.choice(ids, min(config['adapt_batch_size'], len(ids)), replace=False)
        learner.train()
        optimizer.zero_grad(set_to_none=True)
        logits = answer_logits(learner, data['support_pool_input'][chosen], data['support_pool_lengths'][chosen], [target], token_ids)
        labels = data['support_pool_labels'][chosen, names.index(target)]
        F.cross_entropy(logits, labels).backward()
        torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.)
        optimizer.step()
    return validate(learner, data, 'target_test', [target], names, token_ids)[0]['accuracy']


def probe(train_features, test_features, labels, test_labels, config, seed, device):
    torch.manual_seed(seed + 70000)
    head = nn.Linear(config['d_model'], config['lengths'][1] + 1).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['probe_learning_rate'], weight_decay=config['weight_decay'])
    x = torch.tensor(train_features, device=device)
    xtest = torch.tensor(test_features, device=device)
    for _ in range(config['probe_steps']):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(head(x), labels).backward()
        optimizer.step()
    with torch.no_grad():
        return float((head(xtest).argmax(-1) == test_labels).float().mean().cpu())


def transfer(model, config, data, support, seed, token_ids, names, run_id, output, fingerprint):
    path = output / f'{run_id}_transfer.json'
    if path.exists():
        result = json.loads(path.read_text())
        assert result['fingerprint'] == fingerprint
        return result['rows']
    train_features = features(model, data, 'support_pool', token_ids)
    test_features = features(model, data, 'target_test', token_ids)
    rows = []
    for target in config['target_tasks']:
        for budget in config['target_budgets']:
            ids = support[str(budget)]
            labels = data['support_pool_labels'][ids, names.index(target)]
            truth = data['target_test_labels'][:, names.index(target)]
            for mode in ('probe', 'finetune'):
                accuracy = probe(train_features[ids], test_features, labels, truth, config, seed, next(model.parameters()).device) if mode == 'probe' else adapt(model, config, data, target, ids, seed, token_ids, names)
                rows.append({'run_id': run_id, 'seed': seed, 'target': target, 'budget': budget, 'mode': mode, 'accuracy': accuracy})
    path.write_text(json.dumps({'fingerprint': fingerprint, 'rows': rows}, indent=2))
    print(f'transfer complete: {run_id}', flush=True)
    return rows


def run(config_path, output, pretrain_only=False, max_source_runs=None):
    config = json.loads(Path(config_path).read_text())
    repo = Path(config['repository']).resolve()
    sys.path.insert(0, str(repo / 'src'))
    from neurips_permutations.math_ops import PROPERTY32_TASK_NAMES, PROPERTY_FUNCTIONS
    from neurips_permutations.passage import TOKEN_TO_ID, one_line_tokens
    names = list(PROPERTY32_TASK_NAMES)
    groups = select_groups(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    signature = {'config': config, 'groups': groups, 'code_sha256': sha(__file__),
                 'upstream_commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                 'upstream_files': {name: sha(repo / 'src/neurips_permutations' / name) for name in ('models.py', 'math_ops.py', 'passage.py')},
                 'candidate_sha256': sha(config['candidate_table'])}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    metadata_path = output / 'metadata.json'
    if metadata_path.exists():
        assert json.loads(metadata_path.read_text())['fingerprint'] == fingerprint, 'choose a fresh output directory for modified protocols'
    metadata_path.write_text(json.dumps({**signature, 'fingerprint': fingerprint,
                                        'cka_landmark': 'task-free <ONE_END>; raw and within-length-centered CKA both reported',
                                        'source_objective': 'cross-entropy at answer token; inputs end with task token and equals; EOS not predicted'}, indent=2))
    raw_data = generate_data(config, output, names, PROPERTY_FUNCTIONS, TOKEN_TO_ID, one_line_tokens)
    support_path = output / 'support_indices.json'
    order = np.random.default_rng(config['data_seed'] + 50000).permutation(len(raw_data['support_pool_input']))
    support = {str(b): order[:b].tolist() for b in config['target_budgets']}
    if support_path.exists():
        assert json.loads(support_path.read_text()) == support
    else:
        support_path.write_text(json.dumps(support, indent=2))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = {key: torch.tensor(value, device=device) for key, value in raw_data.items()}
    completed = 0
    for seed in config['model_seeds']:
        initial = new_model(config, seed, device)
        np.save(output / f'initial_s{seed}_features.npy', features(initial, data, 'representation', TOKEN_TO_ID))
        for group in groups:
            if max_source_runs is not None and completed >= max_source_runs:
                break
            model, saved = source_train(config, group, seed, data, names, TOKEN_TO_ID, output, fingerprint, device)
            np.save(output / f"{saved['run_id']}_features.npy", features(model, data, 'representation', TOKEN_TO_ID))
            completed += 1
        if max_source_runs is not None and completed >= max_source_runs:
            break
    if pretrain_only or max_source_runs is not None:
        print(f'completed {completed} source runs; target test not evaluated', flush=True)
        return
    all_rows = []
    for seed in config['model_seeds']:
        initial = new_model(config, seed, device)
        baseline = transfer(initial, config, data, support, seed, TOKEN_TO_ID, names, f'random_s{seed}', output, fingerprint)
        all_rows.extend(baseline)
        for group in groups:
            model, saved = source_train(config, group, seed, data, names, TOKEN_TO_ID, output, fingerprint, device)
            rows = transfer(model, config, data, support, seed, TOKEN_TO_ID, names, saved['run_id'], output, fingerprint)
            all_rows.extend({**row, 'group': group['id'], 'family': group['family'], 'minimum_constraint_size': group['minimum_constraint_size']} for row in rows)
    write_csv(output / 'transfer_endpoints.csv', [{**row, 'group': row.get('group', 'random'), 'family': row.get('family', 'random'), 'minimum_constraint_size': row.get('minimum_constraint_size', '')} for row in all_rows])
    from .permworld_combinations_report import analyze
    analyze(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/permworld_combinations.json')
    parser.add_argument('--output', default='results/permworld_combinations')
    parser.add_argument('--pretrain-only', action='store_true')
    parser.add_argument('--max-source-runs', type=int)
    args = parser.parse_args()
    run(args.config, args.output, args.pretrain_only, args.max_source_runs)
