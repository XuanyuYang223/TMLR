"""Reports and provenance checks for the new PermWorld combination pilot."""
import argparse
import csv
import hashlib
from html import escape
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

from .analysis import linear_cka, write_csv


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def within_length_center(values, lengths):
    centered = values.copy().astype(np.float64)
    for n in np.unique(lengths):
        mask = lengths == n
        centered[mask] -= centered[mask].mean(axis=0)
    return centered


def analyze(output):
    output = Path(output)
    metadata = json.loads((output / 'metadata.json').read_text())
    config, groups = metadata['config'], metadata['groups']
    data = {key: value for key, value in np.load(output / 'data.npz').items()}
    sys.path.insert(0, str(Path(config['repository']) / 'src'))
    from neurips_permutations.math_ops import PROPERTY32_TASK_NAMES
    names = list(PROPERTY32_TASK_NAMES)
    lengths = data['representation_lengths']
    group_order = [g['id'] for g in groups]
    source_rows, curve_rows, cka_rows, consistency = [], [], [], []
    features = {}
    seeds = config['model_seeds']
    for seed in seeds:
        for group in groups:
            run_id = f"{group['id']}_s{seed}"
            run = json.loads((output / f'{run_id}.json').read_text())
            for split in ('source_train', 'source_validation'):
                source_rows.extend({'group': group['id'], 'seed': seed, 'split': split, **row} for row in run[split])
            curve_rows.extend({'group': group['id'], 'seed': seed, **row} for row in run['curve'])
            feature = np.load(output / f'{run_id}_features.npy')
            features[group['id'], seed, 'raw'] = feature
            features[group['id'], seed, 'within_length'] = within_length_center(feature, lengths)
        for mode in ('raw', 'within_length'):
            for left, right in combinations(group_order, 2):
                cka_rows.append({'seed': seed, 'left': left, 'right': right, 'centering': mode,
                                 'linear_cka': linear_cka(features[left, seed, mode], features[right, seed, mode])})
    for group in groups:
        for first, second in combinations(seeds, 2):
            for mode in ('raw', 'within_length'):
                a = np.load(output / f'initial_s{first}_features.npy')
                b = np.load(output / f'initial_s{second}_features.npy')
                if mode == 'within_length':
                    a, b = within_length_center(a, lengths), within_length_center(b, lengths)
                consistency.append({'group': group['id'], 'seed_a': first, 'seed_b': second, 'centering': mode,
                                    'trained_cka': linear_cka(features[group['id'], first, mode], features[group['id'], second, mode]),
                                    'random_cka': linear_cka(a, b)})
    source_summary = []
    prior_rows, label_statistics = [], []
    for group in groups:
        rows = [r for r in source_rows if r['group'] == group['id']]
        prior_accuracies = []
        for task in group['tasks']:
            predictions = np.zeros(len(data['validation_lengths']), dtype=np.int64)
            for n in np.unique(data['train_lengths']):
                training_labels = data['train_labels'][data['train_lengths'] == n, names.index(task)]
                predictions[data['validation_lengths'] == n] = np.bincount(training_labels).argmax()
            accuracy = float(np.mean(predictions == data['validation_labels'][:, names.index(task)]))
            prior_accuracies.append(accuracy)
            prior_rows.append({'group': group['id'], 'task': task, 'validation_accuracy': accuracy,
                               'method': 'train-label majority separately within each length'})
        entropies, joint_entropies, correlations = [], [], []
        for n in np.unique(data['train_lengths']):
            y = data['train_labels'][data['train_lengths'] == n][:, [names.index(t) for t in group['tasks']]]
            for column in y.T:
                _, counts = np.unique(column, return_counts=True)
                probabilities = counts / counts.sum()
                entropies.append(float(-np.sum(probabilities * np.log(probabilities))))
            _, counts = np.unique(y, axis=0, return_counts=True)
            probabilities = counts / counts.sum()
            joint_entropies.append(float(-np.sum(probabilities * np.log(probabilities))))
            sigma = y.std(axis=0)
            z = (y - y.mean(axis=0)) / np.where(sigma > 0, sigma, 1)
            correlation = z.T @ z / len(y)
            correlations.extend(abs(correlation[a, b]) for a, b in combinations(range(4), 2))
        label_statistics.append({'group': group['id'], 'mean_conditional_entropy': float(np.mean(entropies)),
                                 'empirical_conditional_joint_entropy': float(np.mean(joint_entropies)),
                                 'mean_abs_within_length_correlation': float(np.mean(correlations))})
        val_acc = float(np.mean([r['accuracy'] for r in rows if r['split'] == 'source_validation']))
        source_summary.append({'group': group['id'], 'family': group['family'], 'minimum_constraint_size': group['minimum_constraint_size'],
                               'train_accuracy': float(np.mean([r['accuracy'] for r in rows if r['split'] == 'source_train'])),
                               'validation_accuracy': val_acc, 'length_majority_validation_accuracy': float(np.mean(prior_accuracies)),
                               'validation_gain_over_length_majority': val_acc - float(np.mean(prior_accuracies)),
                               'validation_min_task_accuracy': float(min(r['accuracy'] for r in rows if r['split'] == 'source_validation')),
                               'null_match_distance': group.get('null_match_distance')})
    endpoints = read_csv(output / 'transfer_endpoints.csv')
    random = {(r['seed'], r['target'], r['budget'], r['mode']): float(r['accuracy']) for r in endpoints if r['group'] == 'random'}
    transfer_rows = []
    for row in endpoints:
        if row['group'] == 'random':
            continue
        baseline = random[row['seed'], row['target'], row['budget'], row['mode']]
        transfer_rows.append({**row, 'accuracy': float(row['accuracy']), 'random_accuracy': baseline,
                              'gain': float(row['accuracy']) - baseline})
    transfer_summary = []
    for group in groups:
        for budget in config['target_budgets']:
            for mode in ('probe', 'finetune'):
                selected = [r for r in transfer_rows if r['group'] == group['id'] and int(r['budget']) == budget and r['mode'] == mode]
                seed_gain = [np.mean([r['gain'] for r in selected if int(r['seed']) == seed]) for seed in seeds]
                transfer_summary.append({'group': group['id'], 'family': group['family'], 'minimum_constraint_size': group['minimum_constraint_size'],
                                         'budget': budget, 'mode': mode, 'accuracy': float(np.mean([r['accuracy'] for r in selected])),
                                         'random_accuracy': float(np.mean([r['random_accuracy'] for r in selected])),
                                         'gain': float(np.mean(seed_gain)), 'gain_seed_sd': float(np.std(seed_gain, ddof=1)),
                                         'positive_seed_mean_count': int(sum(g > 0 for g in seed_gain))})
    primary = [r for r in transfer_summary if r['budget'] == config['primary_budget'] and r['mode'] == config['primary_mode']]
    relation_contrasts = []
    for family in ('cycle', 'position', 'interior'):
        null = next(r for r in primary if r['group'] == f'{family}_none')
        for row in primary:
            if row['family'] == family and row['minimum_constraint_size']:
                relation_contrasts.append({'family': family, 'group': row['group'],
                                          'gain_minus_null': row['gain'] - null['gain'],
                                          'accuracy_minus_null': row['accuracy'] - null['accuracy']})
    summary = {'source_model_count': len(groups) * len(seeds), 'groups': source_summary,
               'pretrained_transfer_endpoints': len(transfer_rows), 'random_transfer_endpoints': len(random),
               'primary_budget': config['primary_budget'], 'primary_mode': config['primary_mode'],
               'transfer_summary': transfer_summary, 'primary_relation_contrasts': relation_contrasts,
               'scope': 'fresh disjoint permutations within the same length range; different targets but no out-of-length generalization',
               'limitations': config['limitations'], 'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for filename, rows in [('source_accuracy.csv', source_rows), ('source_curves.csv', curve_rows), ('cka.csv', cka_rows),
                           ('cross_seed_cka.csv', consistency), ('source_summary.csv', source_summary),
                           ('source_length_majority.csv', prior_rows), ('source_label_statistics.csv', label_statistics),
                           ('transfer_gains.csv', transfer_rows), ('transfer_summary.csv', transfer_summary),
                           ('primary_relation_contrasts.csv', relation_contrasts)]:
        write_csv(output / filename, rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    figures(output, source_summary, transfer_summary, cka_rows, config, group_order)
    table = ''.join(f"<tr><td>{escape(g['id'])}</td><td>{g['minimum_constraint_size'] or '无已证明约束'}</td><td>{escape(', '.join(g['tasks']))}</td><td>{float(g['selection_statistics']['mean_conditional_entropy']):.3f}</td><td>{float(g['selection_statistics']['mean_abs_within_length_correlation']):.3f}</td></tr>" for g in groups)
    outcomes = ''.join(f"<tr><td>{escape(r['group'])}</td><td>{100*r['accuracy']:.2f}%</td><td>{100*r['random_accuracy']:.2f}%</td><td>{100*r['gain']:+.2f} pp</td><td>{r['positive_seed_mean_count']}/3</td></tr>" for r in primary)
    effects = ''.join(f"<tr><td>{escape(r['group'])}</td><td>{100*r['gain_minus_null']:+.2f} pp</td></tr>" for r in relation_contrasts)
    low, high = min(r['validation_accuracy'] for r in source_summary), max(r['validation_accuracy'] for r in source_summary)
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PermWorld 新任务组合试验</title>
<style>body{{max-width:1200px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui;color:#18202b}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}aside{{background:#f2f4f7;padding:16px;margin:18px 0}}</style>
<h1>PermWorld：8 个新四任务组合，24 个源模型</h1>
<p>源任务数量均为 4，三个初始化种子共享所有输入和训练采样计划，每任务曝光 {config['pretrain_steps']*config['examples_per_task_per_step']:,} 次。使用原仓库的性质函数、Passage 数字词表与因果 Transformer；宽度 {config['d_model']}、{config['layers']} 层，训练 {config['pretrain_steps']} 步，只预测答案 token。这是新组合的小规模筛查，不是原论文完整规模的复现。</p>
<p>三任务关系包含 fixed+exceed+def=n、fixed+nontrivial=cycles；四任务关系来自消去中间性质，或 peaks+valleys+double_ascents+double_descents=n−2。无约束对照仅指没有已证明的同输入有理线性恒等式，不代表没有其他数学关系。对照只根据训练前的新生成标签统计选择，未查看 CKA 或迁移。</p>
<table><tr><th>组合</th><th>最小约束任务数</th><th>任务</th><th>平均条件熵</th><th>平均长度内 |相关|</th></tr>{table}</table>
<aside>上表为先前独立设计样本的选组统计；<a href="source_label_statistics.csv">本轮源训练标签统计</a>另存。候选标签统计是近似匹配，峰谷对照匹配尤其较差。源任务难度、完整输出分布和联合熵没有严格控制。源验证准确率跨组合为 {100*low:.1f}%–{100*high:.1f}%，不能据此把结果差异因果归于恒等式阶数。尚无单任务学习难度对照。</aside>
<h2>源任务学习与表征</h2><img src="source_learning.png"><img src="representation.png">
<p>源验证还与仅用训练标签、按排列长度预测多数类的基线比较，避免把标签频率拟合解释为学会了完整数学性质。逐任务基线见 <a href="source_length_majority.csv">长度多数类</a>，逐组合差值见 <a href="source_summary.csv">源学习汇总</a>。源任务未充分学习时，CKA 和迁移差异不能确认高阶结构被网络利用。</p>
<p>CKA 在相同的 420 个任务无关前缀、ONE_END 位置测量。并列报告原始 CKA 和逐长度减均值后的 CKA，帮助检查长度信息的贡献。不同组合使用配对初始化；相同组合的跨种子结果另存。这里没有重做正确/错误输入变换对照，原实验 24/24 最终层 CKA 增加的结论不受改变。</p>
<h2>共同未见任务迁移</h2>
<p>目标任务为 {escape(', '.join(config['target_tasks']))}，均未出现在任何源组合中。目标支持输入、目标测试输入、源训练输入、源验证输入、表征输入五组全局互不重复；长度均为 10–30。支持样本预算 64、256，按输入均匀采样，嵌套且所有条件共用；标签类别不强行平衡。随机模型使用配对初始化与相同支持集、更新步数和学习率。</p>
<img src="transfer.png"><p>主端点预先固定为 256 标签的全模型微调；下表平均四个目标任务和三个初始化种子。误差条只描述三个初始化均值的样本标准差，不提供独立数据重复或显著性证据。</p>
<table><tr><th>组合</th><th>准确率</th><th>随机模型准确率</th><th>迁移收益</th><th>种子均值为正</th></tr>{outcomes}</table>
<p>含恒等式组合相对同系列无已证明约束对照的差值：</p><table><tr><th>组合</th><th>迁移收益差</th></tr>{effects}</table>
<aside>本轮检验新输入上的新任务迁移，未检验新长度或新的数学领域。只有一个数据生成种子，三个初始化种子不是三个独立数据世界。所有组合、目标、预算和两类读出完整保留；测试准确率未用于选择组合或调参。这些差值用于提出下一轮假设，不能确认普适结构规律。</aside>
<p>数据：<a href="metadata.json">固定协议与组合</a>；<a href="summary.json">汇总</a>；<a href="source_accuracy.csv">源任务准确率</a>；<a href="cka.csv">CKA</a>；<a href="cross_seed_cka.csv">跨种子 CKA</a>；<a href="transfer_gains.csv">迁移端点</a>；<a href="support_indices.json">支持集</a>；<a href="verification.json">核验</a>。</p></html>'''
    (output / 'report.html').write_text(html)
    verify(output)
    print(json.dumps({key: summary[key] for key in ('source_model_count', 'pretrained_transfer_endpoints', 'random_transfer_endpoints', 'primary_relation_contrasts')}, indent=2))


def figures(output, source, transfer, cka, config, order):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    ax.bar(x - .25, [r['train_accuracy'] for r in source], .25, label='Source training')
    ax.bar(x, [r['validation_accuracy'] for r in source], .25, label='Unseen-input source validation')
    ax.bar(x + .25, [r['length_majority_validation_accuracy'] for r in source], .25, label='Length-conditioned label majority')
    ax.set_xticks(x, order, rotation=30, ha='right')
    ax.set(ylabel='Mean source answer accuracy', ylim=(0, 1), title='Equal exposure; source fitting is measured, not matched')
    ax.legend()
    fig.savefig(output / 'source_learning.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, mode in zip(axes, ('raw', 'within_length')):
        matrix = np.eye(len(order))
        for i, left in enumerate(order):
            for j, right in enumerate(order):
                if i < j:
                    matrix[i, j] = matrix[j, i] = np.mean([r['linear_cka'] for r in cka if r['left'] == left and r['right'] == right and r['centering'] == mode])
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(x, order, rotation=50, ha='right', fontsize=8)
        ax.set_yticks(x, order, fontsize=8)
        ax.set_title(f'Task-free final-layer CKA: {mode}')
        fig.colorbar(im, ax=ax, shrink=.65)
    fig.savefig(output / 'representation.png', dpi=180)
    fig.savefig(output / 'representation.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, mode in zip(axes, ('probe', 'finetune')):
        for k, budget in enumerate(config['target_budgets']):
            rows = [next(r for r in transfer if r['group'] == group and r['budget'] == budget and r['mode'] == mode) for group in order]
            ax.bar(x + (k - .5) * .36, [100*r['gain'] for r in rows], .36,
                   yerr=[100*r['gain_seed_sd'] for r in rows], capsize=2, label=f'{budget} labels')
        ax.axhline(0, color='black', linewidth=.8)
        ax.set_xticks(x, order, rotation=35, ha='right', fontsize=8)
        ax.set(title=f'New-input target transfer: {mode}', ylabel='Gain over matched random model (pp)')
        ax.legend()
    fig.savefig(output / 'transfer.png', dpi=180)
    fig.savefig(output / 'transfer.pdf')
    plt.close(fig)


def verify(output):
    output = Path(output)
    metadata = json.loads((output / 'metadata.json').read_text())
    signature = {k: v for k, v in metadata.items() if k not in ('fingerprint', 'cka_landmark', 'source_objective')}
    assert hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest() == metadata['fingerprint']
    assert hashlib.sha256(Path('experiments/permworld_combinations.py').read_bytes()).hexdigest() == metadata['code_sha256']
    repo = Path(metadata['config']['repository'])
    for name, expected in metadata['upstream_files'].items():
        assert hashlib.sha256((repo / 'src/neurips_permutations' / name).read_bytes()).hexdigest() == expected
    assert hashlib.sha256(Path(metadata['config']['candidate_table']).read_bytes()).hexdigest() == metadata['candidate_sha256']
    from .permworld_combinations import select_groups
    assert select_groups(metadata['config']) == metadata['groups']
    source_count = 0
    for seed in metadata['config']['model_seeds']:
        for group in metadata['groups']:
            run_id = f"{group['id']}_s{seed}"
            record = json.loads((output / f'{run_id}.json').read_text())
            assert record['fingerprint'] == metadata['fingerprint']
            assert hashlib.sha256((output / 'checkpoints' / f'{run_id}.pt').read_bytes()).hexdigest() == record['checkpoint_sha256']
            assert record['exposures_per_task'] == metadata['config']['pretrain_steps'] * metadata['config']['examples_per_task_per_step']
            source_count += 1
    with np.load(output / 'data.npz') as data:
        arrays = [data[f'{split}_input'] for split in metadata['config']['examples_per_length']]
        tuples = [tuple(row) for array in arrays for row in array]
        assert len(tuples) == len(set(tuples)) == 6930
        support = json.loads((output / 'support_indices.json').read_text())
        order = np.random.default_rng(metadata['config']['data_seed'] + 50000).permutation(len(data['support_pool_input']))
        for budget in metadata['config']['target_budgets']:
            assert support[str(budget)] == order[:budget].tolist()
    endpoints = read_csv(output / 'transfer_endpoints.csv')
    keys = [(r['run_id'], r['target'], r['budget'], r['mode']) for r in endpoints]
    assert len(keys) == len(set(keys)) == 432
    assert source_count == 24
    assert len([r for r in endpoints if r['group'] == 'random']) == 48
    for row in endpoints:
        assert 0 <= float(row['accuracy']) <= 1
    for path in output.glob('*_transfer.json'):
        assert json.loads(path.read_text())['fingerprint'] == metadata['fingerprint']
    cka = read_csv(output / 'cka.csv')
    assert len(cka) == 168
    assert all(0 <= float(row['linear_cka']) <= 1 + 1e-10 for row in cka)
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.values = []
        def handle_starttag(self, tag, attrs):
            self.values.extend(v for k, v in attrs if k in ('href', 'src'))
    links = Links()
    links.feed((output / 'report.html').read_text())
    result = {'status': 'passed', 'source_models': source_count, 'unique_transfer_endpoints': len(keys),
              'globally_disjoint_inputs': len(tuples), 'nested_shared_support': True,
              'training_and_checkpoint_fingerprints': True, 'group_selection_reproducible': True,
              'equal_source_exposure': True, 'cka_endpoints': len(cka),
              'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'data_sha256': hashlib.sha256((output / 'data.npz').read_bytes()).hexdigest()}
    (output / 'verification.json').write_text(json.dumps(result, indent=2))
    assert all((output / link).is_file() for link in links.values)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/permworld_combinations')
    args = parser.parse_args()
    analyze(args.output)
