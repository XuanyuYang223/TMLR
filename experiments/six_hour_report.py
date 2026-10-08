"""Descriptive, partial-run-safe reporting for the six-hour PermWorld cohort."""
import argparse
import csv
from datetime import datetime, timezone
from html import escape
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .analysis import linear_cka
from .permworld_combinations import select_groups, sha
from .permworld_combinations_report import within_length_center


def write_rows(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def records(directory):
    return [json.loads(p.read_text()) for p in sorted(Path(directory).glob('*.json'))]


def length_majority(data, names, task, split):
    predictions = np.zeros(len(data[f'{split}_lengths']), dtype=np.int64)
    for n in np.unique(data['train_lengths']):
        labels = data['train_labels'][data['train_lengths'] == n, names.index(task)]
        predictions[data[f'{split}_lengths'] == n] = np.bincount(labels).argmax()
    return float(np.mean(predictions == data[f'{split}_labels'][:, names.index(task)]))


def transfer_summaries(endpoints):
    """Only completed, exactly paired cells enter gain summaries."""
    random = {(r['seed'], r['target'], r['budget'], r['mode']): r['test_accuracy']
              for r in endpoints if r['group'] == 'random'}
    gains = []
    for row in endpoints:
        if row['group'] == 'random':
            continue
        key = row['seed'], row['target'], row['budget'], row['mode']
        if key in random:
            gains.append({**row, 'random_accuracy': random[key], 'gain': row['test_accuracy']-random[key]})
    summaries = []
    keys = sorted({(r['group'], r['budget'], r['mode']) for r in gains})
    for group, budget, mode in keys:
        selected = [r for r in gains if (r['group'], r['budget'], r['mode']) == (group, budget, mode)]
        seeds = sorted({r['seed'] for r in selected})
        seed_gains = [np.mean([r['gain'] for r in selected if r['seed'] == seed]) for seed in seeds]
        summaries.append({'group': group, 'budget': budget, 'mode': mode, 'paired_cells': len(selected),
                          'seeds': len(seeds), 'test_accuracy': float(np.mean([r['test_accuracy'] for r in selected])),
                          'random_accuracy': float(np.mean([r['random_accuracy'] for r in selected])),
                          'gain': float(np.mean(seed_gains)),
                          'gain_seed_sd': float(np.std(seed_gains, ddof=1)) if len(seeds) > 1 else None,
                          'positive_seed_means': int(sum(x > 0 for x in seed_gains))})
    return gains, summaries


def analyze(plan_path='configs/six_hour_session.json', geometry=True):
    plan = json.loads(Path(plan_path).read_text())
    config = json.loads(Path(plan['base_config']).read_text())
    groups = select_groups(config)
    root = Path(plan['output'])
    output = root/'analysis'
    output.mkdir(exist_ok=True)
    metadata = json.loads((root/'dataset/metadata.json').read_text())
    names = metadata['names']
    with np.load(root/'dataset/data.npz') as archive:
        data = {key: archive[key] for key in archive.files}
    multi, singles = records(root/'multi'), records(root/'single')
    complete = [r for r in multi if r['status'] == 'complete']
    source, curves = [], []
    priors = {}
    for phase, collection in (('multi', multi), ('single', singles)):
        for record in collection:
            job = record['job']
            curves.extend({'phase': phase, 'group': job['id'], 'seed': job['seed'], **row} for row in record['curve'])
            for split in ('source_train', 'source_validation', 'source_audit'):
                for row in record.get(split, []):
                    prior = None
                    if split != 'source_train':
                        key = row['task'], ('validation' if split == 'source_validation' else 'source_audit')
                        if key not in priors:
                            priors[key] = length_majority(data, names, *key)
                        prior = priors[key]
                    source.append({'phase': phase, 'group': job['id'], 'seed': job['seed'], 'status': record['status'],
                                   'step': record['step'], 'split': split, **row,
                                   'length_majority_accuracy': prior,
                                   'gain_over_length_majority': row['accuracy']-prior if prior is not None else None})
    source_summary = []
    for group in groups:
        chosen = [r for r in source if r['phase'] == 'multi' and r['group'] == group['id'] and r['status'] == 'complete']
        row = {'group': group['id'], 'family': group['family'], 'minimum_known_constraint_size': group['minimum_constraint_size'],
               'completed_seeds': len({r['seed'] for r in chosen})}
        for split in ('source_train', 'source_validation', 'source_audit'):
            cells = [r for r in chosen if r['split'] == split]
            row[f'{split}_accuracy'] = float(np.mean([r['accuracy'] for r in cells])) if cells else None
            if split != 'source_train':
                row[f'{split}_majority'] = float(np.mean([r['length_majority_accuracy'] for r in cells])) if cells else None
        source_summary.append(row)
    single_lookup = {(r['seed'], r['task'], r['split']): r for r in source if r['phase'] == 'single' and r['status'] == 'complete'}
    single_contrasts = []
    for row in source:
        key = row['seed'], row['task'], row['split']
        if row['phase'] == 'multi' and row['status'] == 'complete' and key in single_lookup:
            single_contrasts.append({**row, 'single_task_accuracy': single_lookup[key]['accuracy'],
                                     'multi_minus_single': row['accuracy']-single_lookup[key]['accuracy']})
    geometry_rows, consistency = [], []
    if geometry:
        for step in (5000, 10000, 20000):
            values = {}
            for record in multi:
                path = root/'multi'/f"{record['job_id']}_step{step}_features.npy"
                if path.exists():
                    feature = np.load(path)
                    assert len(feature) == len(data['representation_lengths'])
                    values[record['job']['id'], record['job']['seed'], 'raw'] = feature
                    values[record['job']['id'], record['job']['seed'], 'within_length'] = within_length_center(feature, data['representation_lengths'])
            for seed in plan['model_seeds']:
                for left, right in combinations(groups, 2):
                    for mode in ('raw', 'within_length'):
                        keys = (left['id'], seed, mode), (right['id'], seed, mode)
                        if all(k in values for k in keys):
                            geometry_rows.append({'step': step, 'seed': seed, 'left': left['id'], 'right': right['id'],
                                                  'centering': mode, 'linear_cka': linear_cka(values[keys[0]], values[keys[1]])})
            for group in groups:
                for a, b in combinations(plan['model_seeds'], 2):
                    for mode in ('raw', 'within_length'):
                        keys = (group['id'], a, mode), (group['id'], b, mode)
                        if all(k in values for k in keys):
                            trained = linear_cka(values[keys[0]], values[keys[1]])
                            initial = None
                            initial_paths = [root/'initial'/f's{s}_features.npy' for s in (a, b)]
                            if all(p.exists() for p in initial_paths):
                                initial_values = [np.load(p) for p in initial_paths]
                                if mode == 'within_length':
                                    initial_values = [within_length_center(v, data['representation_lengths']) for v in initial_values]
                                initial = linear_cka(*initial_values)
                            consistency.append({'step': step, 'group': group['id'], 'seed_a': a, 'seed_b': b,
                                                'centering': mode, 'linear_cka': trained, 'initial_cka': initial,
                                                'change_from_initial': trained-initial if initial is not None else None})
    elif (output/'geometry.csv').exists():
        with (output/'geometry.csv').open() as handle:
            geometry_rows = list(csv.DictReader(handle))
        for row in geometry_rows:
            row['step'], row['seed'], row['linear_cka'] = int(row['step']), int(row['seed']), float(row['linear_cka'])
    transfer_records = records(root/'transfer')
    endpoints = []
    for record in transfer_records:
        if record['status'] == 'complete':
            endpoints.extend({'group': record['group'], 'seed': record['seed'], 'run_id': record['run_id'], **row} for row in record['rows'])
    gains, transfer_summary = transfer_summaries(endpoints)
    target_summary = []
    for group, target, budget, mode in sorted({(r['group'], r['target'], r['budget'], r['mode']) for r in gains}):
        cells = [r for r in gains if (r['group'], r['target'], r['budget'], r['mode']) == (group, target, budget, mode)]
        target_summary.append({'group': group, 'target': target, 'budget': budget, 'mode': mode,
                               'paired_seeds': len(cells), 'test_accuracy': float(np.mean([r['test_accuracy'] for r in cells])),
                               'random_accuracy': float(np.mean([r['random_accuracy'] for r in cells])),
                               'gain': float(np.mean([r['gain'] for r in cells])),
                               'positive_seed_gains': sum(r['gain'] > 0 for r in cells)})
    summary = {'reported_utc': datetime.now(timezone.utc).isoformat(), 'deadline_utc': plan['deadline_utc'],
               'completed_multi_models': len(complete), 'planned_multi_models': len(groups)*len(plan['model_seeds']),
               'completed_single_models': sum(r['status'] == 'complete' for r in singles),
               'planned_single_models': len({t for g in groups for t in g['tasks']})*len(plan['model_seeds']),
               'completed_transfer_models': sum(r['status'] == 'complete' for r in transfer_records),
               'transfer_endpoints': len(endpoints), 'paired_pretrained_transfer_endpoints': len(gains),
               'source_summary': source_summary, 'transfer_summary': transfer_summary,
               'transfer_target_summary': target_summary,
               'source_model_status': [{'job_id': r['job_id'], 'status': r['status'], 'step': r['step']} for r in multi+singles],
               'geometry_rows': len(geometry_rows), 'within_length_centering': 'subtract each length-specific feature mean; residualize features before CKA',
               'data_split_counts': {s: len(data[f'{s}_lengths']) for s in plan['examples_per_length']},
               'limitations': plan['limitations'], 'analysis_sha256': sha(__file__)}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    for filename, rows in (('source_accuracy.csv', source), ('source_curves.csv', curves), ('source_summary.csv', source_summary),
                           ('single_task_contrasts.csv', single_contrasts), ('geometry.csv', geometry_rows),
                           ('cross_seed_geometry.csv', consistency), ('transfer_endpoints.csv', endpoints),
                           ('transfer_gains.csv', gains), ('transfer_summary.csv', transfer_summary),
                           ('transfer_target_summary.csv', target_summary)):
        write_rows(output/filename, rows)
    plot(output, source_summary, curves, geometry_rows, transfer_summary, groups)
    if target_summary: plot_targets(output, target_summary, groups, config['target_tasks'])
    html_report(root, output, plan, config, groups, summary, single_contrasts)
    print(json.dumps({k: summary[k] for k in ('completed_multi_models', 'completed_single_models', 'completed_transfer_models', 'transfer_endpoints', 'geometry_rows')}))
    return summary


def plot(output, source, curves, geometry, transfer, groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = [g['id'] for g in groups]
    colors = plt.get_cmap('tab10').colors
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.7), constrained_layout=True)
    for i, group in enumerate(order):
        selected = [r for r in curves if r['phase'] == 'multi' and r['group'] == group]
        steps = sorted({r['step'] for r in selected})
        means = [np.mean([r['accuracy'] for r in selected if r['step'] == step]) for step in steps]
        axes[0].plot(steps, means, label=group, color=colors[i])
    axes[0].set(xlabel='Source training updates', ylabel='Source validation accuracy', ylim=(0, 1.04))
    axes[0].legend(fontsize=8, ncol=2)
    for i, group in enumerate(order):
        r = next(x for x in source if x['group'] == group)
        if r['source_audit_accuracy'] is not None:
            axes[1].bar(i-.18, r['source_audit_accuracy'], .36, color='#2369b1', label='Fresh source audit' if i == 0 else None)
            axes[1].bar(i+.18, r['source_audit_majority'], .36, color='#9aa4ae', label='Per-length majority' if i == 0 else None)
    axes[1].set(xticks=range(len(order)), xticklabels=order, ylabel='Source accuracy', ylim=(0, 1.04))
    axes[1].tick_params(axis='x', rotation=40)
    axes[1].legend(fontsize=8)
    save_figure(fig, output/'source_learning')
    if geometry:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
        for ax, mode in zip(axes, ('raw', 'within_length')):
            matrix = np.full((len(order), len(order)), np.nan)
            np.fill_diagonal(matrix, 1.)
            for i, left in enumerate(order):
                for j, right in enumerate(order[i+1:], i+1):
                    rows = [r for r in geometry if r['step'] == 20000 and r['centering'] == mode and {r['left'], r['right']} == {left, right}]
                    if rows:
                        matrix[i, j] = matrix[j, i] = np.mean([r['linear_cka'] for r in rows])
            im = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis')
            ax.set(xticks=range(len(order)), yticks=range(len(order)), xticklabels=order, yticklabels=order,
                   title='Raw CKA' if mode == 'raw' else 'CKA after within-length centering')
            ax.tick_params(axis='x', rotation=70, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
        fig.colorbar(im, ax=axes, shrink=.8)
        save_figure(fig, output/'representation')
    if transfer:
        modes = ('linear_task_free', 'linear_query', 'mlp_task_free', 'finetune', 'finetune_shared')
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
        for ax, budget in zip(axes, (64, 256)):
            for i, mode in enumerate(modes):
                selected = {(r['group']): r for r in transfer if r['budget'] == budget and r['mode'] == mode}
                x = [j for j, g in enumerate(order) if g in selected]
                y = [100*selected[order[j]]['gain'] for j in x]
                ax.plot(x, y, 'o-', label=mode, color=colors[i])
            ax.axhline(0, color='gray', linewidth=1)
            ax.set(xticks=range(len(order)), xticklabels=order, ylabel='Gain over paired random model (pp)', title=f'{budget} target labels')
            ax.tick_params(axis='x', rotation=40)
            ax.legend(fontsize=8)
        save_figure(fig, output/'transfer')


def save_figure(fig, path):
    import matplotlib.pyplot as plt
    for suffix in ('png', 'pdf'):
        fig.savefig(str(path)+'.'+suffix, dpi=160)
    plt.close(fig)


def plot_targets(output, rows, groups, targets):
    import matplotlib.pyplot as plt
    order = [g['id'] for g in groups]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    limit = max(1., max(abs(100*r['gain']) for r in rows if r['mode'] in ('finetune', 'finetune_shared')))
    for i, mode in enumerate(('finetune', 'finetune_shared')):
        for j, budget in enumerate((64, 256)):
            ax = axes[i, j]; matrix = np.full((len(order), len(targets)), np.nan)
            counts = np.zeros_like(matrix, dtype=int)
            for a, group in enumerate(order):
                for b, target in enumerate(targets):
                    cell = next((r for r in rows if (r['group'], r['target'], r['budget'], r['mode']) == (group, target, budget, mode)), None)
                    if cell:
                        matrix[a, b] = 100*cell['gain']; counts[a, b] = cell['paired_seeds']
            im = ax.imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
            for a, b in np.ndindex(matrix.shape):
                if np.isfinite(matrix[a, b]): ax.text(b, a, f'{matrix[a,b]:+.1f}\n({counts[a,b]} seeds)', ha='center', va='center', fontsize=8)
            ax.set(xticks=range(len(targets)), xticklabels=targets, yticks=range(len(order)), yticklabels=order,
                   title=f'{mode}; {budget} labels')
            ax.tick_params(axis='x', rotation=25, labelsize=8)
    fig.colorbar(im, ax=axes, shrink=.8, label='Accuracy gain over paired random model (pp)')
    save_figure(fig, output/'transfer_by_target')


def html_report(root, output, plan, config, groups, summary, single_contrasts=()):
    selection = json.loads((root/'architecture_selection.json').read_text())
    group_table = ''.join(f"<tr><td>{g['id']}</td><td>{g['minimum_constraint_size'] or '无已证明线性约束'}</td><td>{escape(', '.join(g['tasks']))}</td></tr>" for g in groups)
    source_table = ''.join(f"<tr><td>{r['group']}</td><td>{r['completed_seeds']}/3</td><td>{pct(r['source_train_accuracy'])}</td><td>{pct(r['source_audit_accuracy'])}</td><td>{pct(r['source_audit_majority'])}</td></tr>" for r in summary['source_summary'])
    contrast_rows=[]
    audit_pairs=[r for r in single_contrasts if r['split']=='source_audit']
    for group,task in sorted({(r['group'],r['task']) for r in audit_pairs}):
        selected=[r for r in audit_pairs if (r['group'],r['task'])==(group,task)]
        contrast_rows.append(f"<tr><td>{escape(group)}</td><td>{escape(task)}</td><td>{len(selected)}/3</td><td>{pct(float(np.mean([r['accuracy'] for r in selected])))}</td><td>{pct(float(np.mean([r['single_task_accuracy'] for r in selected])))}</td><td>{100*np.mean([r['multi_minus_single'] for r in selected]):+.2f} pp</td></tr>")
    contrast_table='<details><summary>展开全部同种子单任务对照（源审计）</summary><p>每行只平均完整且同种子的配对，已完成种子数不同。两训练条件的每任务曝光相同，但单任务损失与四任务平均损失的梯度目标不同。</p><table><tr><th>组合</th><th>性质</th><th>完整配对种子</th><th>多任务</th><th>单任务</th><th>多任务减单任务</th></tr>'+''.join(contrast_rows)+'</table></details>' if contrast_rows else ''
    transfer_table = ''.join(f"<tr><td>{r['group']}</td><td>{r['mode']}</td><td>{pct(r['test_accuracy'])}</td><td>{100*r['gain']:+.2f} pp</td><td>{r['paired_cells']}/12</td></tr>" for r in summary['transfer_summary'] if r['budget'] == 256)
    transfer_image = '<img src="analysis/transfer.png">' if (output/'transfer.png').exists() else '<p>目标验证调参与测试尚未完成。</p>'
    geometry_image = '<img src="analysis/representation.png">' if (output/'representation.png').exists() else ''
    tuning_text = '目标参数尚未冻结。'
    tuning_path = root/'target_tuning.json'
    if tuning_path.exists():
        tuning = json.loads(tuning_path.read_text())
        tuning_text = f"目标参数状态：{escape(tuning['status'])}。验证网格已经记录 {len(tuning['rows'])} 个结果；只有完成全部验证搜索后才进入测试。"
    supplementary = supplementary_html(root)
    length_text = ''
    if (root/'length_baselines/summary.json').exists():
        reference = json.loads((root/'length_baselines/summary.json').read_text())
        chosen = {r['budget']: r['test_macro'] for r in reference['macro'] if r['mode'] == 'smooth_length'}
        length_text = f"<p>追加的<a href=\"length_baselines/report.html\">仅长度、仅支持标签基线</a>在验证集选统一平滑带宽，64/256 标签平均测试准确率为 {100*chosen[64]:.2f}%/{100*chosen[256]:.2f}%。它是在部分原生测试之后加入的对照，三个初始化之间完全相同；对随机模型略有收益，不代表超过了这个简单基线。</p>"
    target_image = '<img src="analysis/transfer_by_target.png"><p>数字是同目标、同种子配对差值；每格标明已完成种子数。<a href="analysis/transfer_target_summary.csv">全部五种读出的逐目标结果</a>均保留。</p>' if (output/'transfer_by_target.png').exists() else ''
    allocation_text = ''
    if (root/'target_handoff_state.json').exists():
        handoff = json.loads((root/'target_handoff_state.json').read_text())
        revision_link = '<a href="time_allocation_revision.json">交接时记录的协议修订</a>' if (root/'time_allocation_revision.json').exists() else '协议修订将在实际交接时保存'
        allocation_text = f"<p>时间分配状态：{escape(handoff['status'])}；可选源对照的交接起点为 {escape(handoff['source_cutoff_utc'])}，先完成当前模型，再进入未改变的目标验证网格。<a href=\"target_handoff_state.json\">执行状态</a>；{revision_link}。</p>"
    if (root/'optional_resume_state.json').exists():
        resumed = json.loads((root/'optional_resume_state.json').read_text())
        allocation_text += f"<p>目标验证与测试全部完成后，剩余训练时间按原先任务和种子顺序补充单任务对照。当前状态 {escape(resumed['status'])}；<a href=\"optional_resume_state.json\">执行记录</a>。源参数、曝光和目标选参规则均未改变。</p>"
    if (root/'native_repeat_queue_state.json').exists():
        queued = json.loads((root/'native_repeat_queue_state.json').read_text())
        allocation_text += f"<p>独立目标抽样复核的执行状态为 {escape(queued['status'])}：先补齐种子 17 的 18 个单任务对照，再临时交接 GPU，完成后继续原可选源对照；<a href=\"native_repeat_queue_state.json\">时间分配记录</a>。</p>"
    if (root/'session_completion.json').exists():
        allocation_text += '<p>六小时研究窗口已结束；<a href="session_completion.json">最终完成记录</a>列出实际完成与时间限制下未完成的条件。</p>'
    root.joinpath('report.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>六小时研究：任务组合、表征与迁移</title>
<style>body{{max-width:1200px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui;color:#18202b}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}aside{{background:#f2f4f7;padding:16px;margin:18px 0}}</style>
<h1>八个任务组合：扩大源学习、加入单任务对照，再检验迁移</h1>
<p>研究时段 {escape(plan['start_utc'])} 至 {escape(plan['deadline_utc'])}；本报告更新于 {escape(summary['reported_utc'])}。已完成源组合模型 {summary['completed_multi_models']}/{summary['planned_multi_models']}，单任务模型 {summary['completed_single_models']}/{summary['planned_single_models']}，目标测试模型 {summary['completed_transfer_models']}/27，共 {summary['transfer_endpoints']} 个测试端点。未完成条件不进入配对收益均值。</p>
<aside>当前证据支持继续研究具体源组合与具体目标之间的联系。正确数学变换对齐的最终层 CKA 保持 24/24 正向；改变训练任务集合的 CKA 与迁移收益没有普遍单调规律。八组合的学习程度未严格匹配，有限域严格统计控制的结果也未显示数学关系对普通神经迁移具有稳定预测收益，因此目前不能把宽泛的代数结构预测命题当作已经成立。</aside>
{allocation_text}
<p>仅按三个预先选定组合的源验证成绩，在三种架构中选出 {selection['selected']['d_model']} 维、{selection['selected']['layers']} 层因果 Transformer。使用原仓库性质函数和模型类；等价 SDPA 注意力加速经过数值与因果检查。所有本轮条件使用相同 BF16 训练、FP32 评估协议。</p>
<table><tr><th>组合</th><th>已证明约束的最小任务数</th><th>源任务</th></tr>{group_table}</table>
<aside>“无已证明线性约束”仅描述同输入有理线性恒等式，不意味着没有对称性、条件依赖或其他数学关系。组合选择来自训练前标签审计；输出边际、完整联合分布、学习难度仍未严格匹配。不能将所有组间差异因果归于约束阶数。</aside>
<h2>源任务是否真的学会</h2>
<p>所有四任务模型固定 20,000 更新，每任务每步 32 输入，曝光 640,000 次；所有种子与条件共享输入采样计划。记录 5k、10k、20k 表征，不为某组单独选择最好测试检查点。单任务对照保持同初始化、每任务曝光、更新数与学习率计划，但每次更新只优化一个任务，其梯度目标与四任务平均损失不同。</p>
<img src="analysis/source_learning.png"><table><tr><th>组合</th><th>已完成种子</th><th>源训练</th><th>新输入源审计</th><th>按长度多数类</th></tr>{source_table}</table>
<p>源审计输入既未用于源训练，也未用于架构选择。逐任务和逐长度基线见 <a href="analysis/source_accuracy.csv">源端点</a>；已完成单任务与多任务的同种子差值见 <a href="analysis/single_task_contrasts.csv">单任务对照</a>。当部分性质没有充分学会时，表征差异不能证明网络已经形成完整代数结构。</p>
{contrast_table}
<h2>表征：长度信息与任务组合分别观察</h2>{geometry_image}
<p>CKA 使用同一组 840 个未训练的任务无关前缀，在 ONE_END 位置测量。并列报告原始 CKA 和逐长度减均值后的 CKA，以及三种子一致性和中间检查点。原论文“正确数学变换提高最终层 CKA”的 <a href="../cka_review/report.html">24/24 核对结果</a>是另一项固定模型、改变输入对齐的实验，不能由这里的组合比较替代或否定。</p>
<h2>新任务迁移：验证与测试分开</h2>
<p>四个共同目标为 {escape(', '.join(config['target_tasks']))}，全部未参与源训练。新数据包括源训练 42,000、源验证 2,100、表征 840、支持池 2,100、目标验证 1,050、目标测试 2,100、源审计 2,100 个排列，七组全局互不重复；长度均为 10–30。支持预算 64、256 嵌套，所有条件相同，按输入均匀采样。</p>
<p>{tuning_text} 每个源条件和随机条件使用相同大小的验证网格，以种子 17 的四目标、两预算平均验证成绩选学习率与更新数，然后应用于所有三个种子。冻结读出特征的标准化只使用支持集。任务无关线性、MLP 与任务查询线性读出分别验证选参。全模型微调同时报告各条件选参和统一随机条件策略，帮助区分不同训练预算的影响。</p>
{transfer_image}<table><tr><th>组合</th><th>读出</th><th>256 标签测试</th><th>对配对随机模型收益</th><th>完整配对端点</th></tr>{transfer_table}</table>
{length_text}{target_image}
<aside>只有一个新数据世界，三个种子仅重复初始化。四目标及各预算不是独立世界重复。报告所有条件与读出，不按目标测试挑选组或参数。本轮测试新输入上的新任务，未检验新长度。</aside>
{supplementary}
<p><a href="analysis/summary.json">汇总</a> · <a href="analysis/source_curves.csv">学习曲线</a> · <a href="analysis/geometry.csv">所有 CKA</a> · <a href="analysis/transfer_gains.csv">全部配对迁移</a> · <a href="target_tuning.json">验证网格与选择</a> · <a href="verification.json">核验</a> · <a href="session_plan.json">最终执行协议</a> · <a href="session_plan.initial.json">初始协议</a> · <a href="plan_revision.json">目标验证预算修订</a></p></html>''')


def pct(value):
    return '未完成' if value is None else f'{100*value:.2f}%'


def supplementary_html(root):
    pieces=['<h2>提取位置、严格统计控制及独立预测</h2>']
    landmarks=root/'landmarks/summary.json'
    if landmarks.exists():
        values=json.loads(landmarks.read_text())['cross_seed']
        cells={r['landmark']:r for r in values if r['centering']=='within_length'}
        end,query=cells['ONE_END'],cells['source_query_concat']
        pieces.append(f"<p>相同模型与输入，逐长度去均值后的跨种子 CKA 在 ONE_END 位置有 {end['positive_changes']}/{end['cells']} 个配对上升，在四个源任务查询的拼接中有 {query['positive_changes']}/{query['cells']} 个配对上升。这是初步结果后追加的<a href=\"landmarks/report.html\">提取位置诊断</a>；查询拼接只比较相同源任务集，三个种子的三个配对不是独立重复。</p>")
    field=[]
    for name in ('field_symmetry','field_matched_support'):
        path=root.parent/name/'summary.json'
        if path.exists():
            r=json.loads(path.read_text())
            field.append(f"<tr><td><a href=\"../{name}/report.html\">{name}</a></td><td>{r['complete_models']}/{r['planned_models']}</td><td>{r['source_gate_passed']}/{r['complete_models']}</td><td>{r['ordered_paired_settings']}/{r['paired_settings']}</td></tr>")
    if field:
        pieces.append('<p>有限域的所有源组合都有四个任务、秩四、均匀边际、零两两互信息及完全相同的均匀联合输出分布。独立复核进一步匹配单个源函数的物理系数支撑大小。精确答案核的三层排序不保证隐藏层排序，下面保留原登记检验的全部设置。</p><table><tr><th>实验</th><th>完成模型</th><th>源学习门槛</th><th>满足隐藏排序的配对设置</th></tr>'+''.join(field)+'</table>')
    if (root.parent/'field_kernel_identity/report.html').exists():
        pieces.append('<p><a href="../field_kernel_identity/report.html">精确答案核的代数推导</a>解释了 1、0.75、0.5：在完整均匀有限域上，类别答案核的 CKA 由任务射影方向的重合决定。该恒等式经五个本轮集合与 90 个额外枚举检查；它不是隐藏层定理。各源任务集在变换下都线性闭合，区别是单任务方向保持程度。</p>')
    forecast=root.parent/'field_forecast/summary.json'
    if forecast.exists():
        r=json.loads(forecast.read_text());scores=r['scores']
        selected={(x['mode'],x['budget'],x['variant']):x for x in scores}
        rows=[]
        for mode in ('linear','mlp','categorical_subset'):
            for budget in (25,50):
                base=selected.get((mode,budget,'baseline'));order=selected.get((mode,budget,'plus_composition'))
                if base and order:
                    rows.append(f"<tr><td>{mode}</td><td>{budget}</td><td>{base['r2']:.3f}</td><td>{order['r2']:.3f}</td><td>{order['delta_r2']:+.3f}</td></tr>")
        pieces.append(f"<p>独立输入基的 {r['frozen_forecasts']} 条预测于 {escape(r['forecast_frozen_utc'])} 保存，早于独立目标读出。当前核对 {r['evaluated_forecasts']} 条，五个规则全部保留。下面比较基线与增加最小源答案组合阶数；linear/MLP 的预测目标是配对随机调整后的迁移收益，类别查表的目标是其准确率。三个源公式已在拟合集中出现，新的是输入基与模型运行。</p><table><tr><th>读出</th><th>标签</th><th>独立基：基线 R²</th><th>增加组合阶数 R²</th><th>ΔR²</th></tr>{''.join(rows)}</table><p><a href=\"../field_forecast/report.html\">全部五种预测与冻结证据</a> · <a href=\"../field_symmetry_transfer/report.html\">原有限域行为</a> · <a href=\"../field_matched_support_transfer/report.html\">独立复核行为</a>。类别查表和已知目标公式的分析诊断有额外结构假设，不能用其成功替代普通神经迁移。</p>")
    gauge=root.parent/'field_readout_geometry/summary.json'
    if gauge.exists():
        r=json.loads(gauge.read_text())
        pieces.append(f"<p><a href=\"../field_readout_geometry/report.html\">源读出子空间与尺度诊断</a>是在隐藏排序失败后追加的事后分析。对 {r['models']} 个模型，保持源输出不变的八种正尺度变换，使同一模型原始 CKA 的最大减最小平均达到 {r['mean_within_model_scaling_cka_range']:.3f}。在不触及标准差下限时，支持集逐维标准化会抵消这些尺度变换；因此原始 CKA 变化不必对应读出能力变化。</p>")
    consistency=root/'source_consistency/report.html'
    if consistency.exists():
        pieces.append('<p><a href="source_consistency/report.html">源输出关系诊断</a>将数学一致性、答案正确率与长度多数类预测分开，避免将满足关系的错误答案当作学会任务。</p>')
    if (root.parent/'field_mechanism/report.html').exists():
        pieces.append('<p><a href="../field_mechanism/report.html">此前编码器的状态留出诊断</a>保留了目标依赖阶数与查表覆盖的区别；源预训练见过全部输入，仍属于新任务标签的转导实验。</p>')
    if (root.parent/'factor_kernel_readout/report.html').exists():
        pieces.append('<p><a href="../factor_kernel_readout/report.html">固定的通用因子交互核读出</a>是在普通神经结果之后追加的事后探索：四个核阶数与训练源头、初始源头、直接物理输入因子三个表示全部保留。额外的交互假设类及其条件性收益不能代替原神经迁移结论。</p>')
    if (root.parent/'controlled_target_followup/report.html').exists():
        pieces.append('<p><a href="../controlled_target_followup/report.html">八个新目标的冻结预测复核</a>沿用独立基的 27 个编码器与旧预测权重。目标仅按数学条件选取，使其在三个物理输入基中均依赖至少三个坐标；这一条件是在发现旧目标含容易的单坐标函数后追加，原负结果全部保留。普通神经平均收益约 0.04–1.10 个百分点，但准确率仍低于 20% 机会水平；组合阶数对类别查表的预测更稳定，对神经收益没有一致改善。新目标预测先于其读出结果保存。</p>')
    if (root/'layer_diagnostic/report.html').exists():
        pieces.append('<p><a href="layer_diagnostic/report.html">全部层的前缀 CKA 与实际激活梯度</a>是在位置差异之后追加的事后检查。答案损失对末层前缀输出的激活导数为零，而前一层前缀通过查询注意力参与损失。共享参数仍被训练，不能将零激活导数解释为前缀未训练。报告全部层，没有据此改动原源训练或目标策略。</p>')
    if (root/'numeric_subspace/report.html').exists():
        pieces.append('<p><a href="numeric_subspace/report.html">原生数值答案区分子空间</a>将 0–30 数值类别的输出权重行空间与其正交补分开。去长度的源查询拼接 CKA 在数值区分空间上升 23/24，正交补上升 16/24；训练源查询约 87.5% 的方差位于该数值空间。两空间只保证这些数值类别之间的 logit 差值分解，不保证其他词表概率或将来任务行为不变，不能把正交补称为无用。</p>')
    if (root/'transfer_prediction/report.html').exists():
        pieces.append('<p><a href="transfer_prediction/report.html">整组合留出的迁移收益预测</a>比较学习程度与标签相似性基线、增加数学关系、增加测得几何以及同时增加两者。规则和代码在本轮测试之前固定，所有五种读出和两种预算全部报告。该分析是在先导结果之后追加，只有八个任务组合及一个输入世界，不能称为独立前瞻复制。</p>')
        if (root/'transfer_prediction/indicator_correction.html').exists():
            pieces.append('<p>冻结预测实现中的无约束单独指示变量因空值标记检查而恒为零，最小阶数 3/4/5 仍正确。错误在部分测试之后、预测拟合之前发现；<a href="transfer_prediction/indicator_correction.html">修正版敏感性分析</a>与原版全部并列，未按得分选择版本。</p>')
    elif (root/'native_transfer_prediction_protocol.json').exists():
        pieces.append('<p><a href="native_transfer_prediction_protocol.json">整组合留出的预测规则</a>已在本轮测试前固定，等待全部 27 个目标条件完成后统一分析；不根据测试选预测器、特征或岭惩罚。</p>')
    repeat = root.parent/'native_target_repeat'
    if (repeat/'report.html').exists():
        pieces.append('<p><a href="../native_target_repeat/report.html">独立目标支持集与测试集复核</a>使用原 24 个源编码器、冻结读出策略、相同源任务集和目标，新增 5,250 个全局互不重复的输入。新的只是目标抽样，不能称为新源世界或新任务组合复制。全部原预测器与两个指示变量修正版共 5,760 条预测在新标签生成前保存，所有版本均保留。</p>')
        if (repeat/'comparison_summary.json').exists():
            comparison=json.loads((repeat/'comparison_summary.json').read_text())
            row=next(r for r in comparison['comparisons'] if (r['group'],r['budget'],r['mode'])==('interior_none',256,'finetune'))
            pieces.append(f'<p><a href="../native_target_repeat/comparison.html">两轮逐组合、逐目标对照</a>各包含 {comparison["original_transfer_endpoints"]:,} 个测试端点。256 标签条件选参微调中，interior_none 对随机初始化的平均收益为 {100*row["old_random_gain"]:+.2f}/{100*row["new_random_gain"]:+.2f} 个百分点；正种子均值为 {row["old_positive_seed_means"]}/3 与 {row["new_positive_seed_means"]}/3。第二轮准确率 {100*row["new_accuracy"]:.2f}% 仍低于仅用长度的参照 {abs(100*row["new_gain_over_length"]):.2f} 个百分点。全部八组合、五读出、两预算与四目标均保留，不能将该组合的宏平均收益推广到每个目标。</p>')
    elif (repeat/'protocol.json').exists():
        pieces.append('<p><a href="../native_target_repeat/protocol.json">新的目标抽样复核</a>已准备，沿用全部源模型与目标策略；<a href="../native_target_repeat/forecasts.json">5,760 条冻结预测</a>保存于新标签生成之前。等待源单任务对照完成后按时间窗口运行，不把相同源模型算成独立复制。</p>')
    return ''.join(pieces)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default='configs/six_hour_session.json')
    parser.add_argument('--skip-geometry', action='store_true')
    args = parser.parse_args()
    analyze(args.plan, not args.skip_geometry)
