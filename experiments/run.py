"""Reproducible finite-field pilot with matched budgets and grouped holdouts."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from torch.nn import functional as F
from .algebra import audit, composition_size, projective, scenarios, target_pool, world
from .models import BatchedTargets, Encoder, SourceModel


def balanced_splits(labels, budgets, test_per_class, p, seed):
    rng = np.random.default_rng(seed)
    tests, supports = [], {b: [] for b in budgets}
    for target in range(labels.shape[1]):
        bins = [rng.permutation(np.flatnonzero(labels[:, target] == c)) for c in range(p)]
        if any(len(bin_) < test_per_class + max(budgets) // p for bin_ in bins):
            raise ValueError('not enough examples for disjoint balanced support/test sets')
        tests.append(np.concatenate([bin_[:test_per_class] for bin_ in bins]))
        for budget in budgets:
            supports[budget].append(np.concatenate([bin_[test_per_class:test_per_class + budget // p] for bin_ in bins]))
    return np.array(tests), {b: np.array(ids) for b, ids in supports.items()}


def source_training(model, x, y, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['pretrain_lr'], weight_decay=config['weight_decay'])
    curve = []
    for step in range(config['pretrain_steps'] + 1):
        if step % config['record_every'] == 0 or step == config['pretrain_steps']:
            with torch.no_grad():
                logits = model(x)
                acc = (logits.argmax(-1) == y).float().mean(0).cpu().tolist()
                loss = F.cross_entropy(logits.reshape(-1, config['p']), y.reshape(-1)).item()
            curve.append({'step': step, 'accuracy': acc, 'loss': loss})
        if step == config['pretrain_steps']:
            break
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        F.cross_entropy(logits.reshape(-1, config['p']), y.reshape(-1)).backward()
        optimizer.step()
    return curve


def single_training(encoder, x, labels, config, seed):
    model = BatchedTargets(encoder, labels.shape[1], config['p'], seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['pretrain_lr'], weight_decay=config['weight_decay'])
    truth = labels.T.contiguous()
    curve = []
    for step in range(config['pretrain_steps'] + 1):
        if step % config['record_every'] == 0 or step == config['pretrain_steps']:
            with torch.no_grad():
                acc = (model(x).argmax(-1) == truth).float().mean(-1).cpu().tolist()
            curve.append({'step': step, 'accuracy': acc})
        if step == config['pretrain_steps']:
            break
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        F.cross_entropy(logits.reshape(-1, config['p']), truth.reshape(-1)).backward()
        optimizer.step()
    auc = np.trapezoid(np.array([r['accuracy'] for r in curve]), x=[r['step'] for r in curve], axis=0) / config['pretrain_steps']
    return curve, auc


def adapt(encoder, x, labels, support, test, config, seed, mode):
    frozen = mode == 'probe'
    model = BatchedTargets(encoder, labels.shape[1], config['p'], seed, frozen=frozen)
    if frozen:
        optimizer = torch.optim.AdamW([model.head_weight, model.head_bias], lr=config['probe_lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': config['finetune_encoder_lr']},
            {'params': [model.head_weight, model.head_bias], 'lr': config['finetune_head_lr']}], weight_decay=config['weight_decay'])
    support = torch.as_tensor(support, device=x.device)
    test = torch.as_tensor(test, device=x.device)
    target_ids = torch.arange(labels.shape[1], device=x.device)[:, None]
    train_y = labels[support, target_ids]
    test_y = labels[test, target_ids]
    train_x, test_x = x[support], x[test]
    if frozen:
        with torch.no_grad():
            train_features, test_features = model.encode(train_x), model.encode(test_x)
        def predict(features):
            return torch.bmm(features, model.head_weight.transpose(1, 2)) + model.head_bias[:, None]
    for _ in range(config['adapt_steps']):
        optimizer.zero_grad(set_to_none=True)
        logits = predict(train_features) if frozen else model(train_x)
        F.cross_entropy(logits.reshape(-1, config['p']), train_y.reshape(-1)).backward()
        optimizer.step()
    with torch.no_grad():
        test_logits = predict(test_features) if frozen else model(test_x)
        train_logits = predict(train_features) if frozen else model(train_x)
        acc = (test_logits.argmax(-1) == test_y).float().mean(-1).cpu().tolist()
        train_acc = (train_logits.argmax(-1) == train_y).float().mean(-1).cpu().tolist()
    return acc, train_acc


def run(config_path, output, pretrain_only=False):
    config = json.loads(Path(config_path).read_text())
    for b in config['budgets']:
        if b <= 0 or b % config['p']:
            raise ValueError('every budget must be positive and divisible by p')
    if config['record_every'] <= 0 or config['pretrain_steps'] <= 0:
        raise ValueError('training steps must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = ''.join(Path(__file__).with_name(name).read_text() for name in ['run.py', 'models.py', 'algebra.py'])
    fingerprint = hashlib.sha256((json.dumps(config, sort_keys=True) + source + str(pretrain_only)).encode()).hexdigest()
    metadata_path = output / 'metadata.json'
    if metadata_path.exists() and json.loads(metadata_path.read_text())['fingerprint'] != fingerprint:
        raise ValueError('output belongs to another configuration/code version; choose a fresh output directory')
    device = 'cuda' if config['device'] == 'auto' and torch.cuda.is_available() else ('cpu' if config['device'] == 'auto' else config['device'])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    # GPU matmuls require this setting for deterministic algorithms.
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cuda.matmul.allow_tf32 = False
    groups = scenarios()
    groups = [g for pair in zip(groups[:4], groups[4:]) for g in pair][:config['scenario_limit']]
    targets = target_pool(groups, config['p'], config['target_count'])
    audit_result = audit(groups, targets, config['p'])
    (output / 'algebra_audit.json').write_text(json.dumps(audit_result, indent=2))
    metadata = {'config': config, 'fingerprint': fingerprint, 'python': platform.python_version(),
                'torch': torch.__version__, 'numpy': np.__version__, 'device': device,
                'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
                'evaluation': 'transductive new-task transfer: all inputs exposed during source training; target support and test labels disjoint',
                'replication': 'independent random input bases; initialization seeds crossed within each basis',
                'holdout': 'entire coefficient-defined source combination, across all world/model seeds',
                'pretrain_only': pretrain_only}
    metadata_path.write_text(json.dumps(metadata, indent=2))
    all_metrics = []
    start = time.monotonic()
    for world_seed in config['world_seeds']:
        _, latent, encoded, basis = world(config['p'], config['dimension'], world_seed)
        x = torch.tensor(encoded, device=device)
        target_values = latent @ targets.T % config['p']
        target_labels = torch.tensor(target_values, device=device)
        test, supports = balanced_splits(target_values, config['budgets'], config['test_per_class'], config['p'], world_seed + 10000)
        np.savez(output / f'world_{world_seed}.npz', basis=basis, targets=targets, test=test,
                 **{f'support_{b}': ids for b, ids in supports.items()})
        unique_sources = sorted({tuple(v) for g in groups for v in g['sources']})
        source_lookup = {v: i for i, v in enumerate(unique_sources)}
        for model_seed in config['model_seeds']:
            torch.manual_seed(model_seed)
            initial = SourceModel(x.shape[1], config['hidden'], config['features'], config['p']).to(device)
            initial_state = {k: v.detach().clone() for k, v in initial.state_dict().items()}
            baseline_path = output / f'baseline_w{world_seed}_m{model_seed}.json'
            if baseline_path.exists():
                baseline = json.loads(baseline_path.read_text())
                single_auc = np.array(baseline['single_auc'])
            else:
                single_labels = torch.tensor(latent @ np.array(unique_sources).T % config['p'], device=device)
                single_curve, single_auc = single_training(initial.encoder, x, single_labels, config, model_seed + 30000)
                baseline = {'sources': np.array(unique_sources).tolist(), 'single_curve': single_curve,
                            'single_auc': single_auc.tolist(), 'metrics': []}
                if not pretrain_only:
                    for budget in config['budgets']:
                        for mode in ['probe', 'finetune']:
                            accuracy, train_accuracy = adapt(initial.encoder, x, target_labels, supports[budget], test, config, model_seed + 20000, mode)
                            baseline['metrics'].extend({'target_id': t, 'budget': budget, 'mode': mode,
                                                        'test_accuracy': a, 'support_accuracy': ta} for t, (a, ta) in enumerate(zip(accuracy, train_accuracy)))
                baseline_path.write_text(json.dumps(baseline, indent=2))
            random_metrics = {(r['target_id'], r['budget'], r['mode']): r['test_accuracy'] for r in baseline['metrics']}
            with torch.no_grad():
                init_features = initial.encoder(x).cpu().numpy()
            np.save(output / f'initial_w{world_seed}_m{model_seed}.npy', init_features)
            for g in groups:
                run_id = f"{g['id']}_w{world_seed}_m{model_seed}"
                result_path = output / f'{run_id}.json'
                if result_path.exists():
                    all_metrics.extend(json.loads(result_path.read_text())['metrics'])
                    print(f'resumed {run_id}', flush=True)
                    continue
                model = SourceModel(x.shape[1], config['hidden'], config['features'], config['p']).to(device)
                model.load_state_dict(initial_state)
                # Same head permutation for all source combinations within a basis.
                order = np.random.default_rng(world_seed + 40000).permutation(4)
                source_values = latent @ g['sources'][order].T % config['p']
                labels = torch.tensor(source_values, device=device)
                curve = source_training(model, x, labels, config)
                source_accuracy = float(np.mean(curve[-1]['accuracy']))
                source_min_accuracy = float(min(curve[-1]['accuracy']))
                source_auc = float(np.trapezoid([np.mean(r['accuracy']) for r in curve], x=[r['step'] for r in curve]) / config['pretrain_steps'])
                mean_single_auc = float(np.mean([single_auc[source_lookup[tuple(v)]] for v in g['sources']]))
                structural = next(r for r in audit_result['scenarios'] if r['id'] == g['id'])
                metrics = []
                if not pretrain_only:
                    for budget in config['budgets']:
                        for mode in ['probe', 'finetune']:
                            accuracy, train_accuracy = adapt(model.encoder, x, target_labels, supports[budget], test, config, model_seed + 20000, mode)
                            for t, (a, ta) in enumerate(zip(accuracy, train_accuracy)):
                                random_acc = random_metrics[t, budget, mode]
                                metrics.append({'scenario': g['id'], 'family': g['family'], 'world_seed': world_seed,
                                                'model_seed': model_seed, 'target_id': t, 'target': ','.join(map(str, targets[t])),
                                                'budget': budget, 'mode': mode, 'test_accuracy': a,
                                                'support_accuracy': ta, 'random_accuracy': random_acc,
                                                'transfer_gain': a - random_acc,
                                                'source_accuracy': source_accuracy, 'source_min_accuracy': source_min_accuracy,
                                                'source_auc': source_auc, 'mean_single_auc': mean_single_auc,
                                                'minimum_circuit_size': structural['minimum_circuit_size'],
                                                'target_composition_size': structural['target_composition_sizes'][t]})
                with torch.no_grad():
                    representations = model.encoder(x).cpu().numpy()
                np.save(output / f'{run_id}_features.npy', representations)
                checkpoint_dir = output / 'checkpoints'
                checkpoint_dir.mkdir(exist_ok=True)
                torch.save({k: v.cpu() for k, v in model.state_dict().items()}, checkpoint_dir / f'{run_id}.pt')
                result = {'run_id': run_id, 'source_head_order': order.tolist(), 'source_curve': curve,
                          'source_accuracy': source_accuracy, 'source_min_accuracy': source_min_accuracy,
                          'metrics': metrics, 'fingerprint': fingerprint}
                result_path.write_text(json.dumps(result, indent=2))
                all_metrics.extend(metrics)
                print(f"{run_id}: source={source_accuracy:.4f}, min={source_min_accuracy:.4f}, elapsed={time.monotonic()-start:.1f}s", flush=True)
    if all_metrics:
        from .analysis import summarize
        summarize(output, all_metrics, groups, config)
    print(f'Completed: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/pilot.json')
    parser.add_argument('--output', default='results/pilot')
    parser.add_argument('--pretrain-only', action='store_true')
    args = parser.parse_args()
    run(args.config, args.output, args.pretrain_only)
