"""Frozen readout capacity, learned field relations, and state-repetition controls.

Source checkpoints and target support/test labels are fixed. Field fitting uses
support labels only, with a privileged finite-field affine hypothesis class;
it never receives the true target coefficients.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .algebra import rank_mod, scenarios, world
from .analysis import write_csv
from .models import SourceModel


def fit_field(x, y, p):
    """RREF over Fp. Returns a particular solution and consistency certificate."""
    x, y = np.asarray(x, dtype=np.int64), np.asarray(y, dtype=np.int64)
    a = np.column_stack([x, y]) % p
    pivots, row = [], 0
    for col in range(x.shape[1]):
        candidates = np.flatnonzero(a[row:, col])
        if not len(candidates):
            continue
        pivot = row + candidates[0]
        a[[row, pivot]] = a[[pivot, row]]
        a[row] = a[row] * pow(int(a[row, col]), -1, p) % p
        for other in range(len(a)):
            if other != row:
                a[other] = (a[other] - a[other, col] * a[row]) % p
        pivots.append(col)
        row += 1
        if row == len(a):
            break
    inconsistent = np.any(np.all(a[:, :-1] == 0, axis=1) & (a[:, -1] != 0))
    if inconsistent:
        return None, len(pivots)
    coefficients = np.zeros(x.shape[1], dtype=np.int64)
    for row, col in enumerate(pivots):
        coefficients[col] = a[row, -1]
    return coefficients, len(pivots)


def lookup_predict(train_codes, train_labels, test_codes, p):
    table = {}
    for code, label in zip(train_codes, train_labels):
        key = tuple(code.tolist())
        table.setdefault(key, np.zeros(p, dtype=np.int64))[label] += 1
    majority = int(np.bincount(train_labels, minlength=p).argmax())
    matched = np.array([tuple(code.tolist()) in table for code in test_codes])
    prediction = np.array([int(table[tuple(code.tolist())].argmax()) if hit else majority for code, hit in zip(test_codes, matched)])
    return prediction, matched


class ReadoutBatch(nn.Module):
    """Independent heads per target, on an entirely frozen representation."""
    def __init__(self, targets, features, p, hidden, kind, seed, device):
        super().__init__()
        generator = torch.Generator(device=device).manual_seed(seed)
        self.kind = kind
        if kind == 'linear':
            self.w = nn.Parameter(torch.randn(targets, p, features, generator=generator, device=device) / np.sqrt(features))
            self.b = nn.Parameter(torch.zeros(targets, p, device=device))
        elif kind == 'mlp':
            self.w1 = nn.Parameter(torch.randn(targets, hidden, features, generator=generator, device=device) / np.sqrt(features))
            self.b1 = nn.Parameter(torch.zeros(targets, hidden, device=device))
            self.w2 = nn.Parameter(torch.randn(targets, p, hidden, generator=generator, device=device) / np.sqrt(hidden))
            self.b2 = nn.Parameter(torch.zeros(targets, p, device=device))
        else:
            raise ValueError('unknown head kind')

    def forward(self, features):
        if self.kind == 'linear':
            return torch.bmm(features, self.w.transpose(1, 2)) + self.b[:, None]
        h = F.gelu(torch.bmm(features, self.w1.transpose(1, 2)) + self.b1[:, None])
        return torch.bmm(h, self.w2.transpose(1, 2)) + self.b2[:, None]


def train_readout(features, labels, support, test, plan, p, kind, lr, seed):
    targets = labels.shape[1]
    head = ReadoutBatch(targets, features.shape[1], p, plan['hidden'], kind, seed, features.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=plan['weight_decay'])
    support = torch.tensor(support, device=features.device)
    test = torch.tensor(test, device=features.device)
    target_ids = torch.arange(targets, device=features.device)[:, None]
    xtrain, xtest = features[support], features[test]
    ytrain, ytest = labels[support, target_ids], labels[test, target_ids]
    for _ in range(plan['steps']):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(head(xtrain).reshape(-1, p), ytrain.reshape(-1)).backward()
        optimizer.step()
    with torch.no_grad():
        test_acc = (head(xtest).argmax(-1) == ytest).float().mean(-1).cpu().tolist()
        support_acc = (head(xtrain).argmax(-1) == ytrain).float().mean(-1).cpu().tolist()
    return test_acc, support_acc


def code_diagnostics(codes, labels, supports, test, p, seed):
    design = np.column_stack([codes, np.ones(len(codes), dtype=np.int64)])
    shuffled = design[np.random.default_rng(seed).permutation(len(design))]
    full_rank = rank_mod(design, p)
    rows = []
    for budget, indices in supports.items():
        for target in range(labels.shape[1]):
            train_ids, test_ids = indices[target], test[target]
            truth = labels[test_ids, target]
            majority = int(np.bincount(labels[train_ids, target], minlength=p).argmax())
            for kind, x in [('field_fit', design), ('shuffled_field_fit', shuffled)]:
                coefficients, rank = fit_field(x[train_ids], labels[train_ids, target], p)
                prediction = np.full(len(test_ids), majority) if coefficients is None else x[test_ids] @ coefficients % p
                train_prediction = np.full(len(train_ids), majority) if coefficients is None else x[train_ids] @ coefficients % p
                rows.append({'method': kind, 'target_id': target, 'budget': budget,
                             'test_accuracy': float(np.mean(prediction == truth)),
                             'support_accuracy': float(np.mean(train_prediction == labels[train_ids, target])),
                             'fit_consistent': coefficients is not None, 'support_rank': rank,
                             'full_source_domain_rank': full_rank,
                             'identifies_full_source_span': coefficients is not None and rank == full_rank,
                             'seen_tuple_fraction': None})
            predicted, matched = lookup_predict(codes[train_ids], labels[train_ids, target], codes[test_ids], p)
            rows.append({'method': 'code_lookup', 'target_id': target, 'budget': budget,
                         'test_accuracy': float(np.mean(predicted == truth)), 'support_accuracy': 1.,
                         'fit_consistent': True, 'support_rank': None, 'full_source_domain_rank': full_rank,
                         'identifies_full_source_span': None, 'seen_tuple_fraction': float(matched.mean())})
    return rows


def run(config_path, output, max_runs=None):
    plan = json.loads(Path(config_path).read_text())
    parents = [Path(p) for p in plan['source_runs']]
    source_metadata = json.loads(Path(plan['source_metadata']).read_text())
    config = source_metadata['config']
    parent_metadata = [json.loads((p / 'metadata.json').read_text()) for p in parents]
    checkpoint_parent = {m: p for p, metadata in zip(parents, parent_metadata) for m in metadata['config']['model_seeds']}
    if set(checkpoint_parent) != set(config['model_seeds']):
        raise ValueError('checkpoint seed coverage does not match source metadata')
    source_audits = [json.loads((p / 'algebra_audit.json').read_text()) for p in parents]
    assert all(a == source_audits[0] for a in source_audits)
    targets = np.array(source_audits[0]['targets'])
    for p, metadata in zip(parents, parent_metadata):
        assert {k: v for k, v in metadata['config'].items() if k != 'model_seeds'} == {k: v for k, v in config.items() if k != 'model_seeds'}
    if not set(plan['budgets']) <= set(config['budgets']):
        raise ValueError('new target splits are not allowed in this follow-up')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    signature = {'plan': plan, 'max_runs': max_runs,
                 'source_fingerprints': [m['fingerprint'] for m in parent_metadata],
                 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    if (output / 'metadata.json').exists() and json.loads((output / 'metadata.json').read_text())['fingerprint'] != fingerprint:
        raise ValueError('output has incompatible provenance; choose a new directory')
    metadata = {**signature, 'source_config': config, 'fingerprint': fingerprint,
                'neural_readout_policy': 'same features, support/test labels, steps, learning rate and initialization seed for paired pretrained/random heads; encoder remains frozen',
                'field_policy': 'fit affine coefficients from decoded source values and target support labels only; no true target coefficient supplied',
                'limitations': ['Exploratory diagnostics after observing initial negative transfer.',
                                'Field fitting supplies a privileged algebraic hypothesis class; it is not spontaneous neural composition.',
                                'All source inputs were exposed during source pretraining.',
                                'Lookup control measures repetition of full decoded source states across support and test.']}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    groups = scenarios()
    groups = [g for pair in zip(groups[:4], groups[4:]) for g in pair][:config['scenario_limit']]
    all_neural, all_codes = [], []
    processed, start = 0, time.monotonic()
    for w in config['world_seeds']:
        _, latent, encoded, _ = world(config['p'], config['dimension'], w)
        target_labels = latent @ targets.T % config['p']
        labels = torch.tensor(target_labels, device=device)
        splits = [np.load(p / f'world_{w}.npz') for p in parents]
        for key in splits[0].files:
            assert all(np.array_equal(splits[0][key], split[key]) for split in splits)
        test = splits[0]['test']
        supports = {b: splits[0][f'support_{b}'] for b in plan['budgets']}
        for m in config['model_seeds']:
            if max_runs is not None and processed >= max_runs:
                break
            parent = checkpoint_parent[m]
            initial_features = torch.tensor(np.load(parent / f'initial_w{w}_m{m}.npy'), device=device)
            baseline_path = output / f'random_w{w}_m{m}.json'
            if baseline_path.exists():
                baseline = json.loads(baseline_path.read_text())
            else:
                baseline = []
                for budget, support in supports.items():
                    for kind in ['linear', 'mlp']:
                        for lr in plan['learning_rates']:
                            acc, train_acc = train_readout(initial_features, labels, support, test, plan, config['p'], kind, lr, m + 20000)
                            baseline.extend({'target_id': t, 'budget': budget, 'kind': kind, 'learning_rate': lr,
                                             'test_accuracy': a, 'support_accuracy': ta} for t, (a, ta) in enumerate(zip(acc, train_acc)))
                baseline_path.write_text(json.dumps(baseline, indent=2))
            baseline_lookup = {(r['target_id'], r['budget'], r['kind'], r['learning_rate']): r['test_accuracy'] for r in baseline}
            for g in groups:
                if max_runs is not None and processed >= max_runs:
                    break
                run_id = f"{g['id']}_w{w}_m{m}"
                result_path = output / f'{run_id}.json'
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                else:
                    features = torch.tensor(np.load(parent / f'{run_id}_features.npy'), device=device)
                    assert not features.requires_grad
                    neural = []
                    for budget, support in supports.items():
                        for kind in ['linear', 'mlp']:
                            for lr in plan['learning_rates']:
                                acc, train_acc = train_readout(features, labels, support, test, plan, config['p'], kind, lr, m + 20000)
                                for t, (a, ta) in enumerate(zip(acc, train_acc)):
                                    random_accuracy = baseline_lookup[t, budget, kind, lr]
                                    neural.append({'scenario': g['id'], 'family': g['family'], 'world_seed': w, 'model_seed': m,
                                                   'target_id': t, 'budget': budget, 'kind': kind, 'learning_rate': lr,
                                                   'test_accuracy': a, 'support_accuracy': ta, 'random_accuracy': random_accuracy,
                                                   'transfer_gain': a - random_accuracy})
                    source_model = SourceModel(encoded.shape[1], config['hidden'], config['features'], config['p']).to(device)
                    state = torch.load(parent / 'checkpoints' / f'{run_id}.pt', map_location=device, weights_only=True)
                    source_model.load_state_dict(state)
                    with torch.no_grad():
                        codes = source_model(torch.tensor(encoded, device=device)).argmax(-1).cpu().numpy()
                    diagnostics = code_diagnostics(codes, target_labels, supports, test, config['p'], w + 10000*m + 50000)
                    for row in diagnostics:
                        row.update({'scenario': g['id'], 'family': g['family'], 'world_seed': w, 'model_seed': m})
                    result = {'run_id': run_id, 'neural': neural, 'code_diagnostics': diagnostics, 'fingerprint': fingerprint}
                    result_path.write_text(json.dumps(result, indent=2))
                all_neural.extend(result['neural'])
                all_codes.extend(result['code_diagnostics'])
                processed += 1
                print(f'readout {processed}: {run_id}, elapsed={time.monotonic()-start:.1f}s', flush=True)
    summarize(output, all_neural, all_codes, plan)


def summarize(output, neural, diagnostics, plan):
    write_csv(output / 'neural_endpoints.csv', neural)
    write_csv(output / 'code_diagnostic_endpoints.csv', diagnostics)
    neural_summary, code_summary = [], []
    for family in sorted({r['family'] for r in neural}):
        for budget in plan['budgets']:
            for kind in ['linear', 'mlp']:
                for lr in plan['learning_rates']:
                    rows = [r for r in neural if r['family'] == family and r['budget'] == budget and r['kind'] == kind and r['learning_rate'] == lr]
                    neural_summary.append({'family': family, 'budget': budget, 'kind': kind, 'learning_rate': lr,
                                           'accuracy': float(np.mean([r['test_accuracy'] for r in rows])),
                                           'random_accuracy': float(np.mean([r['random_accuracy'] for r in rows])),
                                           'gain': float(np.mean([r['transfer_gain'] for r in rows]))})
            for method in ['field_fit', 'shuffled_field_fit', 'code_lookup']:
                rows = [r for r in diagnostics if r['family'] == family and r['budget'] == budget and r['method'] == method]
                code_summary.append({'family': family, 'budget': budget, 'method': method,
                                     'accuracy': float(np.mean([r['test_accuracy'] for r in rows])),
                                     'consistent_fraction': float(np.mean([r['fit_consistent'] for r in rows])),
                                     'full_span_identified_fraction': float(np.mean([bool(r['identifies_full_source_span']) for r in rows])) if method != 'code_lookup' else None,
                                     'seen_tuple_fraction': float(np.mean([r['seen_tuple_fraction'] for r in rows])) if method == 'code_lookup' else None})
    primary = [r for r in neural_summary if r['budget'] == plan['primary_budget'] and r['learning_rate'] == plan['primary_learning_rate']]
    contrasts = []
    for family in sorted({r['family'] for r in primary}):
        linear = next(r for r in primary if r['family'] == family and r['kind'] == 'linear')
        mlp = next(r for r in primary if r['family'] == family and r['kind'] == 'mlp')
        contrasts.append({'family': family, 'mlp_minus_linear': mlp['accuracy'] - linear['accuracy'],
                          'difference_in_transfer_gains': mlp['gain'] - linear['gain'], 'mlp_transfer_gain': mlp['gain']})
    summary = {'source_model_count': len({(r['scenario'], r['world_seed'], r['model_seed']) for r in neural}),
               'neural_endpoint_count': len(neural), 'code_diagnostic_endpoint_count': len(diagnostics),
               'primary_budget': plan['primary_budget'], 'primary_learning_rate': plan['primary_learning_rate'],
               'primary_neural_contrasts': contrasts, 'neural_summary': neural_summary, 'code_summary': code_summary,
               'interpretation': 'Learned field fitting estimates target relations from support labels. It uses an explicit finite-field affine hypothesis class, while neural heads use no symbolic algebra. Lookup separates repeated-source-state benefits from algebraic interpolation.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    write_csv(output / 'neural_summary.csv', neural_summary)
    write_csv(output / 'code_summary.csv', code_summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    colors = {'A': '#2166ac', 'B': '#d95f02'}
    for family in sorted({r['family'] for r in neural}):
        for kind, style in [('linear', '--'), ('mlp', '-')]:
            rows = [r for r in neural_summary if r['family'] == family and r['kind'] == kind and r['learning_rate'] == plan['primary_learning_rate']]
            axes[0].plot([r['budget'] for r in rows], [r['accuracy'] for r in rows], style, marker='o', color=colors[family], label=f'{family} {kind}')
            axes[1].plot([r['budget'] for r in rows], [r['gain'] for r in rows], style, marker='o', color=colors[family], label=f'{family} {kind}')
        for method, style in [('field_fit', '-'), ('code_lookup', '--'), ('shuffled_field_fit', ':')]:
            rows = [r for r in code_summary if r['family'] == family and r['method'] == method]
            axes[2].plot([r['budget'] for r in rows], [r['accuracy'] for r in rows], style, marker='o', color=colors[family], label=f'{family} {method}')
    axes[0].set(title='Frozen heads: matched optimizer', ylabel='Target accuracy', ylim=(0, 1))
    axes[1].set(title='Gain over matched random features', ylabel='Accuracy gain')
    axes[1].axhline(0, color='black', linewidth=.7)
    axes[2].set(title='Learned field relations vs state lookup', ylabel='Target accuracy', ylim=(0, 1.03))
    for ax in axes:
        ax.set_xlabel('Labeled target examples')
        ax.legend(fontsize=7)
    fig.savefig(output / 'readouts.png', dpi=180)
    fig.savefig(output / 'readouts.pdf')
    plt.close(fig)
    table = ''.join(f"<tr><td>{r['family']}</td><td>{r['kind']}</td><td>{100*r['accuracy']:.2f}%</td><td>{100*r['random_accuracy']:.2f}%</td><td>{100*r['gain']:+.2f} pp</td></tr>" for r in primary)
    field_rows = [r for r in code_summary if r['budget'] == plan['primary_budget']]
    code_table = ''.join(f"<tr><td>{r['family']}</td><td>{r['method']}</td><td>{100*r['accuracy']:.2f}%</td><td>{100*r['consistent_fraction']:.1f}%</td></tr>" for r in field_rows)
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>继续试验：读出与代数关系拟合</title>
<style>body{{max-width:1150px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}</style>
<h1>读出诊断：信息可恢复以后，如何用少量标签学习目标？</h1>
<p>使用原 {summary['source_model_count']} 个有限域源模型，保持源权重和目标标签划分不变。线性与小 MLP 读出都冻结编码器，训练 {plan['steps']} 步，分别报告学习率 0.01、0.03；主对照固定为 {plan['primary_budget']} 标签、学习率 {plan['primary_learning_rate']}。每种读出都有匹配的随机特征基线。</p>
<p>原 PermWorld 正确输入变换的最终层 CKA 确实在 24/24 个单元上增加。<a href="../cka_review/report.html">CKA 原始数据核验</a>。本页的有限域读出试验不复现那项 Transformer 输入变换对照，也不否定它。</p><img src="readouts.png">
<table><tr><th>组</th><th>冻结读出</th><th>准确率</th><th>随机特征基线</th><th>迁移收益</th></tr>{table}</table>
<p>第二类诊断先读取四个已训练源任务的输出值，再只用目标支持标签求解有限域仿射系数。拟合过程不接收真实目标公式。它额外假设目标属于这个代数函数族，因此使用了特权结构先验，不能解释成普通神经训练自发学会了组合。</p>
<table><tr><th>组</th><th>源答案读出</th><th>准确率</th><th>支持标签可精确拟合比例</th></tr>{code_table}</table>
<p>查表仅在支持集出现相同完整源答案组合时复用标签，否则输出支持集多数类；它量化潜在状态重复。打乱源答案行顺序保持源答案联合分布不变，但破坏与目标的对应；无法拟合时同样使用多数类。</p>
<p>这些是观察初始负迁移后的探索诊断。源训练覆盖全部输入，因此这里只能讨论同实体的新任务标签迁移。所有方法和学习率完整报告，不按目标测试准确率挑选参数或最佳方法。</p>
<p>数据：<a href="neural_endpoints.csv">神经读出端点</a>；<a href="code_diagnostic_endpoints.csv">代数及查表端点</a>；<a href="metadata.json">协议与来源</a>；<a href="summary.json">汇总</a>。</p></html>'''
    (output / 'report.html').write_text(html)
    print(json.dumps({k: v for k, v in summary.items() if k not in ['neural_summary', 'code_summary']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/readout_followup.json')
    parser.add_argument('--output', default='results/readout_followup')
    parser.add_argument('--max-runs', type=int)
    args = parser.parse_args()
    run(args.config, args.output, args.max_runs)
