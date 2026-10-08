"""Paired behavioral summaries for the rank-four finite-field cohort."""
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
from itertools import product
import json
from pathlib import Path

import numpy as np

from .algebra import composition_size, world
from .analysis import write_csv
from .field_symmetry import group_sources
from .field_symmetry_transfer import splits


def summarize(output='results/field_symmetry_transfer'):
    output = Path(output)
    metadata = json.loads((output/'metadata.json').read_text())
    config = metadata['config']
    parent = json.loads(Path(config['source_config']).read_text())
    complete = []
    for w, m, g in product(parent['world_seeds'], parent['model_seeds'], ['random']+parent['groups']):
        path = output/f'{g}_w{w}_m{m}.json'
        if path.exists():
            record = json.loads(path.read_text())
            assert record['fingerprint'] == metadata['fingerprint']
            if record['status'] == 'complete': complete.append(record)
    rows = [row for r in complete for row in r['rows']]
    baseline = {(r['world_seed'], r['model_seed'], r['target_id'], r['budget'], r['mode']): r['accuracy'] for r in rows if r['group'] == 'random'}
    gains = []
    for row in rows:
        key = row['world_seed'], row['model_seed'], row['target_id'], row['budget'], row['mode']
        if row['group'] != 'random' and key in baseline:
            gains.append({**row, 'random_accuracy': baseline[key], 'gain': row['accuracy']-baseline[key]})
    grouped = []
    for order, budget, mode in product((2, 3, 4), config['budgets'], ('linear', 'mlp', 'categorical_subset', 'full_tuple_lookup')):
        selected = [r for r in rows if (r['target_composition_size'], r['budget'], r['mode']) == (order, budget, mode)]
        if selected:
            matched = [r for r in gains if (r['target_composition_size'], r['budget'], r['mode']) == (order, budget, mode)]
            grouped.append({'target_composition_size': order, 'budget': budget, 'mode': mode, 'cells': len(selected),
                            'source_gates_passed': sum(r['source_gate_passed'] for r in selected),
                            'accuracy': float(np.mean([r['accuracy'] for r in selected])),
                            'random_accuracy': float(np.mean([r['random_accuracy'] for r in matched])) if matched else None,
                            'gain': float(np.mean([r['gain'] for r in matched])) if matched else None})
    contrasts = []
    for w, m, t, budget, mode in product(parent['world_seeds'], parent['model_seeds'], range(len(config['targets'])), config['budgets'], ('linear', 'mlp', 'categorical_subset')):
        selected = [r for r in rows if r['group'] != 'random' and (r['world_seed'], r['model_seed'], r['target_id'], r['budget'], r['mode']) == (w, m, t, budget, mode)]
        if len(selected) != len(parent['groups']): continue
        low, high = min(r['target_composition_size'] for r in selected), max(r['target_composition_size'] for r in selected)
        if low == high: continue
        easy = np.mean([r['accuracy'] for r in selected if r['target_composition_size'] == low])
        hard = np.mean([r['accuracy'] for r in selected if r['target_composition_size'] == high])
        contrasts.append({'world_seed': w, 'model_seed': m, 'target_id': t, 'budget': budget, 'mode': mode,
                          'minimum_order': low, 'maximum_order': high, 'lower_order_accuracy': float(easy),
                          'higher_order_accuracy': float(hard), 'lower_minus_higher': float(easy-hard),
                          'all_source_gates_passed': all(r['source_gate_passed'] for r in selected)})
    contrast_summary = []
    for budget, mode in product(config['budgets'], ('linear', 'mlp', 'categorical_subset')):
        selected = [r for r in contrasts if (r['budget'], r['mode']) == (budget, mode)]
        if selected:
            contrast_summary.append({'budget': budget, 'mode': mode, 'cells': len(selected),
                                     'mean_lower_minus_higher': float(np.mean([r['lower_minus_higher'] for r in selected])),
                                     'positive_cells': sum(r['lower_minus_higher'] > 0 for r in selected)})
    planned = len(parent['world_seeds'])*len(parent['model_seeds'])*(len(parent['groups'])+1)
    summary = {'reported_utc': datetime.now(timezone.utc).isoformat(), 'completed_conditions': len(complete),
               'planned_conditions': planned, 'endpoints': len(rows), 'paired_neural_gain_endpoints': len(gains),
               'order_summary': grouped, 'within_target_contrast_summary': contrast_summary,
               'scope': config['generalization_scope'], 'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.joinpath('summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    for filename, values in (('paired_neural_gains.csv', gains), ('order_summary.csv', grouped),
                             ('within_target_contrasts.csv', contrasts), ('within_target_contrast_summary.csv', contrast_summary)):
        if values: write_csv(output/filename, values)
    plot(output, grouped, config)
    table = ''.join(f"<tr><td>{r['budget']}</td><td>{r['mode']}</td><td>{100*r['mean_lower_minus_higher']:+.2f} pp</td><td>{r['positive_cells']}/{r['cells']}</td></tr>" for r in contrast_summary)
    source_label = Path(parent['output']).name
    preregistration = 'matched_field_preregistration.json' if 'world_provider_sha256' in parent else 'secondary_transfer_preregistration.json'
    output.joinpath('report.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>有限域：源任务因子与新任务读出</title>
<style>body{{max-width:1100px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui;color:#18202b}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}aside{{background:#f2f4f7;padding:16px}}</style>
<h1>相同源答案统计下，目标需要组合几个源答案</h1>
<p>已完成 {len(complete)}/{planned} 个条件，共 {len(rows)} 个读出端点。使用 <a href="../{source_label}/report.html">变换结构试验</a>的 {len(parent['groups'])*9} 个训练编码器和 9 个配对初始编码器。八个共同目标均与所有源任务的任何非零标量倍数不同；同一目标因源组合不同，可需要两、三或四列源答案。</p>
<p>每个目标从 625 输入中，按类别随机留出 125 个目标测试输入；支持预算 25、50，类别平衡且嵌套。所有源条件和初始化在同一输入基世界内共用目标支持与测试。源预训练已见全部 625 输入，这属于新任务标签的转导迁移，不检验新实体。</p>
<img src="behavior.png"><p>图按数学分析给出的目标组合阶数汇总。目标和条件组成在各阶数中不同，因此更有辨别力的比较是固定世界、初始化、目标和标签预算，在 {len(parent['groups'])} 个源组合间比较较低阶与较高阶；下表保留全部符合条件的目标。</p>
<table><tr><th>标签预算</th><th>读出规则</th><th>固定目标后：低阶减高阶</th><th>正差值端点</th></tr>{table}</table>
<p>线性与 MLP 头仅接收隐藏特征和目标支持标签，均使用预先固定的 200 更新、学习率 0.01，支持集均值与标准差用于特征标准化；所有源与随机条件相同。神经读出的配对随机基线另存，不用类别查表成绩替代神经迁移。</p>
<p>通用类别子集查表接收已训练源头解码出的类别与目标支持标签，枚举最小支持集一致子集；它不接收目标系数或数学组合阶数。完整四列查表作为另一个基线。类别查表没有神经随机编码器对应项，因此不计算其神经迁移收益。</p>
<aside>组合阶数来自真实代数定义，不能视为模型自动发现了该结构。类别查表的键数和目标依赖关系本就会影响样本效率，该规则上的成功不证明普通神经读出也成功。完整报告源学习门槛；条件间学习难度与轨迹仍可能不同。世界、初始化、目标和预算交叉形成相关重复，未给出独立数据集显著性声明。</aside>
<p><a href="../six_hour_session/{preregistration}">训练前登记</a> · <a href="metadata.json">冻结执行代码与协议</a> · <a href="summary.json">汇总</a> · <a href="endpoints.csv">完整端点</a> · <a href="paired_neural_gains.csv">神经迁移收益</a> · <a href="within_target_contrasts.csv">固定目标的阶数对照</a> · <a href="verification.json">核验</a> · <a href="behavior.pdf">PDF 图</a></p></html>''')
    verify(output, metadata, complete)
    print(json.dumps({k: summary[k] for k in ('completed_conditions', 'endpoints', 'within_target_contrast_summary')}, indent=2))
    return summary


def plot(output, grouped, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, budget in zip(axes, config['budgets']):
        for mode in ('linear', 'mlp', 'categorical_subset', 'full_tuple_lookup'):
            selected = [r for r in grouped if r['budget'] == budget and r['mode'] == mode]
            ax.plot([r['target_composition_size'] for r in selected], [r['accuracy'] for r in selected], 'o-', label=mode)
        ax.axhline(.2, color='gray', linestyle=':')
        ax.set(xlabel='Minimum target composition size', ylabel='Target test accuracy', title=f'{budget} labels', xticks=[2, 3, 4], ylim=(0, 1.04))
        ax.legend(fontsize=8)
    for suffix in ('png', 'pdf'): fig.savefig(output/f'behavior.{suffix}', dpi=160)
    plt.close(fig)


def verify(output, metadata, complete):
    config = metadata['config']
    parent = json.loads(Path(config['source_config']).read_text())
    if 'world_provider_sha256' in parent:
        registration = json.loads(Path('results/six_hour_session/matched_field_preregistration.json').read_text())
        assert registration['source_wrapper_sha256'] == hashlib.sha256(Path('experiments/matched_field.py').read_bytes()).hexdigest()
        assert registration['transfer_wrapper_sha256'] == hashlib.sha256(Path('experiments/matched_field_transfer.py').read_bytes()).hexdigest()
    for key, path in (('code_sha256', 'experiments/field_symmetry_transfer.py'), ('head_code_sha256', 'experiments/longrun_transfer.py'), ('categorical_code_sha256', 'experiments/field_mechanism.py')):
        assert metadata[key] == hashlib.sha256(Path(path).read_bytes()).hexdigest()
    signature = {k: v for k, v in metadata.items() if k != 'fingerprint'}
    assert metadata['fingerprint'] == hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    expected_models = 0
    keys = []
    for record in complete:
        expected = 32 if record['group'] == 'random' else 64
        assert len(record['rows']) == expected
        for row in record['rows']:
            assert 0 <= row['accuracy'] <= 1
            if row['group'] != 'random': assert row['target_composition_size'] == composition_size(group_sources(row['group'], parent['p']), config['targets'][row['target_id']], parent['p'])
            keys.append(tuple(row[k] for k in ('group', 'world_seed', 'model_seed', 'target_id', 'budget', 'mode')))
        expected_models += 1
    assert len(keys) == len(set(keys))
    output.joinpath('verification.json').write_text(json.dumps({'status': 'passed_for_completed_artifacts',
        'completed_conditions': expected_models, 'unique_endpoints': len(keys), 'code_hashes': True,
        'target_composition_orders_recomputed': True, 'shared_disjoint_balanced_splits_unit_checked': True}, indent=2)+'\n')


if __name__ == '__main__':
    summarize()
