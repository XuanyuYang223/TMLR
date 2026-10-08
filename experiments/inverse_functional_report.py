"""Accuracy-first report of every prospective inverse functional comparison."""
import csv
from html import escape
from itertools import product
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .inverse_functional_alignment import initialize, now
from .longrun_engine import atomic_json


LABELS = {
    'ordinary': '普通训练', 'correct_alignment': '正确几何对齐',
    'answer_matched_mismatch': '答案匹配错配', 'output_distillation': '输出蒸馏',
    'output_distillation_correct_alignment': '输出蒸馏＋正确对齐',
}
SHORT = ['Ordinary', 'Correct geometry', 'Matched wrong', 'Distillation', 'Distill + geometry']


def paired_interval(values):
    values = np.asarray(values, dtype=float)
    # Exhaustive enumeration of the empirical bootstrap for n=6 or n=3.
    indices = np.asarray(list(product(range(len(values)), repeat=len(values))))
    draws = values[indices].mean(1)
    lo, hi = np.quantile(draws, [.025, .975])
    return [float(lo), float(hi)]


def statistics(delta, replicates):
    clusters = {str(seed): float(np.mean([delta[i] for i, r in enumerate(replicates) if r['source_seed'] == seed])) for seed in sorted({r['source_seed'] for r in replicates})}
    return {'mean_pp': float(np.mean(delta)), 'paired_deltas_pp': list(map(float, delta)),
            'positive_replicates': int(np.sum(np.asarray(delta) > 0)),
            'paired_conditional_bootstrap_95_pp': paired_interval(delta),
            'teacher_cluster_deltas_pp': clusters,
            'three_teacher_cluster_bootstrap_95_pp': paired_interval(list(clusters.values()))}


def table(headers, rows):
    return '<table><thead><tr>' + ''.join('<th>' + escape(str(h)) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + escape(str(v)) + '</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table>'


def make_summary(plan, root):
    records = []
    for rep in plan['replicates']:
        for condition in plan['conditions']:
            name = rep['id'] + '_' + condition
            evaluation = json.loads((root / 'evaluations' / f'{name}.json').read_text())
            training = json.loads((root / 'training' / f'{name}.json').read_text())
            records.append({'replicate': rep['id'], 'source_seed': rep['source_seed'], 'condition': condition,
                            'selected_step': training['selected']['step'], 'validation_accuracy': training['selected']['accuracy'],
                            **evaluation['results']})
    matrices = {}; conditions = {}; comparisons = {}
    for endpoint in ['selected', 'latest']:
        matrices[endpoint] = np.asarray([[next(row[endpoint]['answer_accuracy'] for row in records if row['replicate'] == rep['id'] and row['condition'] == condition) for condition in plan['conditions']] for rep in plan['replicates']])
        conditions[endpoint] = {condition: {'mean_accuracy': float(matrices[endpoint][:, j].mean()),
            'replicate_accuracies': matrices[endpoint][:, j].tolist(),
            'per_length_mean_accuracy': {str(n): float(np.mean([r[endpoint]['per_length'][str(n)] for r in records if r['condition'] == condition])) for n in plan['lengths']}} for j, condition in enumerate(plan['conditions'])}
        comparisons[endpoint] = {}
        for contrast in plan['primary_contrasts']:
            left, right = contrast.split('-'); a = plan['conditions'].index(left); b = plan['conditions'].index(right)
            comparisons[endpoint][contrast] = statistics(100 * (matrices[endpoint][:, a] - matrices[endpoint][:, b]), plan['replicates'])
    auxiliary = json.loads((root / 'auxiliary.json').read_text()); geometry = {}
    for condition in ['initialization'] + plan['conditions']:
        endpoints = ['initialization'] if condition == 'initialization' else ['selected', 'latest']
        geometry[condition] = {}
        for endpoint in endpoints:
            subset = [r for r in auxiliary['records'] if r['condition'] == condition and r['endpoint'] == endpoint]
            by_rep = []
            for r in subset:
                mean = {k: float(np.mean([s[k] for s in r['scores']])) for k in ['correct_cka', 'matched_correct_cka', 'matched_wrong_cka', 'answer_residual_correct_cka', 'answer_residual_wrong_cka']}
                mean['matched_contrast'] = mean['matched_correct_cka'] - mean['matched_wrong_cka']
                mean['answer_residual_contrast'] = mean['answer_residual_correct_cka'] - mean['answer_residual_wrong_cka']
                by_rep.append({'replicate': r['replicate'], **mean})
            geometry[condition][endpoint] = {'replicates': by_rep, 'mean': {k: float(np.mean([r[k] for r in by_rep])) for k in by_rep[0] if k != 'replicate'}}
    contrasts = comparisons['selected']
    evidence = {key: all(value > 0 for value in stat['teacher_cluster_deltas_pp'].values()) and stat['three_teacher_cluster_bootstrap_95_pp'][0] > 0 for key, stat in contrasts.items()}
    summary = {'created_utc': now(), 'scope': plan['scope'], 'records': records, 'conditions': conditions,
               'primary_and_final_contrasts': comparisons, 'geometry_secondary': geometry,
               'source_gate': json.loads((root / 'teacher/gate.json').read_text()),
               'accuracy_endpoints': 60, 'source_models_retrained': 0, 'target_labels_including_validation': 256,
               'training_labels': 192, 'validation_labels': 64, 'new_test_examples': 2560,
               'positive_at_all_three_teacher_clusters': evidence,
               'uncertainty': 'Exact empirical percentile bootstrap: 6^6 target/support draws conditional on three frozen teachers, and 3^3 teacher-cluster draws. Shared test set is fixed. These descriptive intervals are not multiplicity-adjusted and do not establish population-level significance with three related teachers.',
               'spontaneous_algebra_discovery_tested': False}
    atomic_json(root / 'summary.json', summary)
    return summary, matrices


def run():
    plan, root, sig = initialize(); summary, matrices = make_summary(plan, root)
    with (root / 'accuracy.csv').open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['replicate', 'source_seed', 'condition', 'selected_update', 'selected_accuracy', 'final_accuracy', 'validation_accuracy'])
        for r in summary['records']: writer.writerow([r['replicate'], r['source_seed'], r['condition'], r['selected_step'], r['selected']['answer_accuracy'], r['latest']['answer_accuracy'], r['validation_accuracy']])
    colors = {17: '#3073ae', 42: '#be642f', 101: '#527b42'}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), constrained_layout=True)
    for i, rep in enumerate(plan['replicates']):
        offset = (i - 2.5) * .033
        axes[0].plot(np.arange(5) + offset, 100 * matrices['selected'][i], '.-', alpha=.7, color=colors[rep['source_seed']], label=f"{rep['id']} / teacher {rep['source_seed']}")
    axes[0].plot(np.arange(5), 100 * matrices['selected'].mean(0), 'kD', label='Mean', markersize=6)
    axes[0].set_ylabel('New-test target accuracy (%)'); axes[0].set_title('Primary: validation-selected target models')
    for i, rep in enumerate(plan['replicates']):
        values = [next(r['correct_cka'] for r in summary['geometry_secondary'][c]['selected']['replicates'] if r['replicate'] == rep['id']) for c in plan['conditions']]
        axes[1].plot(np.arange(5), values, '.-', alpha=.7, color=colors[rep['source_seed']])
    baseline = summary['geometry_secondary']['initialization']['initialization']['mean']['correct_cka']
    axes[1].axhline(baseline, linestyle='--', color='gray', label='Mean target initialization')
    axes[1].set_ylabel('Length-averaged prefix CKA'); axes[1].set_title('Secondary: correct teacher correspondence')
    for ax in axes:
        ax.set_xticks(np.arange(5), SHORT, rotation=25, ha='right'); ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=7, ncol=2); axes[1].legend(fontsize=8)
    fig.savefig(root / 'accuracy_geometry.png', dpi=180); fig.savefig(root / 'accuracy_geometry.pdf'); plt.close(fig)
    primary = summary['primary_and_final_contrasts']['selected']; final = summary['primary_and_final_contrasts']['latest']
    def delta_text(stat):
        lo, hi = stat['three_teacher_cluster_bootstrap_95_pp']
        return f"{stat['mean_pp']:+.2f} 个百分点（教师簇描述性区间 {lo:+.2f} 至 {hi:+.2f}；六次重复中 {stat['positive_replicates']}/6 为正）"
    lead = '<p><strong>新测试准确率：</strong>' + '；'.join(LABELS[c] + f" {summary['conditions']['selected'][c]['mean_accuracy']*100:.2f}%" for c in plan['conditions']) + '。</p>'
    lead += '<p>' + '；'.join(escape(LABELS[a] + ' − ' + LABELS[b] + '：' + delta_text(primary[contrast])) for contrast in plan['primary_contrasts'] for a, b in [contrast.split('-')]) + '。</p>'
    if all(summary['positive_at_all_three_teacher_clusters'].values()):
        conclusion = '三个预先固定的比较都在三个教师簇均值上为正。这支持本设置下正确几何约束的局部功能收益；仍需独立训练数据世界及新的任务对验证，不能写成普通训练自发发现取逆代数。'
    else:
        failed = [LABELS[a] + ' 优于 ' + LABELS[b] for c, supported in summary['positive_at_all_three_teacher_clusters'].items() if not supported for a, b in [c.split('-')]]
        conclusion = '尚未同时支持全部预定功能判断：' + '、'.join(failed) + '未在三个教师簇上得到一致正优势。CKA 改善不能替代准确率改善。这个结果限定于本次固定损失权重、标签预算与随机初始化目标模型，不否定其他训练设置。'
    mean_rows = [[LABELS[c], f"{summary['conditions']['selected'][c]['mean_accuracy']*100:.2f}%", f"{summary['conditions']['latest'][c]['mean_accuracy']*100:.2f}%",
                  f"{summary['geometry_secondary'][c]['selected']['mean']['correct_cka']:.4f}", f"{summary['geometry_secondary'][c]['selected']['mean']['answer_residual_contrast']:+.4f}"] for c in plan['conditions']]
    replicate_rows = [[rep['id'] + f" / 教师 {rep['source_seed']}"] + [f'{100*v:.2f}%' for v in matrices['selected'][i]] for i, rep in enumerate(plan['replicates'])]
    final_rows = [[rep['id']] + [f'{100*v:.2f}%' for v in matrices['latest'][i]] for i, rep in enumerate(plan['replicates'])]
    contrast_rows = []
    for c in plan['primary_contrasts']:
        a, b = c.split('-'); stat = primary[c]
        contrast_rows.append([LABELS[a] + ' − ' + LABELS[b], f"{stat['mean_pp']:+.2f}", ', '.join(f'{x:+.2f}' for x in stat['paired_deltas_pp']),
                              ' / '.join(f"{seed}: {value:+.2f}" for seed, value in stat['teacher_cluster_deltas_pp'].items()),
                              f"{stat['paired_conditional_bootstrap_95_pp'][0]:+.2f} 至 {stat['paired_conditional_bootstrap_95_pp'][1]:+.2f}",
                              f"{stat['three_teacher_cluster_bootstrap_95_pp'][0]:+.2f} 至 {stat['three_teacher_cluster_bootstrap_95_pp'][1]:+.2f}", f"{final[c]['mean_pp']:+.2f}"])
    length_rows = [[n] + [f"{100*summary['conditions']['selected'][c]['per_length_mean_accuracy'][str(n)]:.2f}%" for c in plan['conditions']] for n in plan['lengths']]
    source_rows = [[r['source_seed'], f"{100*r['accuracy']:.2f}%"] + [f"{100*r['per_length'][str(n)]:.2f}%" for n in plan['lengths']] for r in summary['source_gate']['grades']]
    selection_rows = [[r['replicate'], LABELS[r['condition']], r['selected_step'], f"{100*r['validation_accuracy']:.2f}%"] for r in summary['records']]
    geometry_rows = []
    for c in ['initialization'] + plan['conditions']:
        endpoint = 'initialization' if c == 'initialization' else 'selected'; values = summary['geometry_secondary'][c][endpoint]['mean']
        geometry_rows.append(['目标初始化' if c == 'initialization' else LABELS[c]] + [f'{values[k]:.4f}' for k in ['correct_cka', 'matched_correct_cka', 'matched_wrong_cka', 'matched_contrast', 'answer_residual_correct_cka', 'answer_residual_wrong_cka', 'answer_residual_contrast']])
    verification_path = root / 'verification.json'; verification = json.loads(verification_path.read_text()) if verification_path.exists() else {'status': 'pending'}
    links = [('protocol.json', '主实验预注册'), ('auxiliary_protocol.json', '辅助CKA规则'), ('dataset/audit.json', '数据审计'), ('teacher/gate.json', '源模型门槛'), ('accuracy.csv', '30次训练的准确率'), ('summary.json', '完整统计'), ('auxiliary.json', '逐长度几何'), ('verification.json', '独立核验'), ('completion.json', '完成与哈希清单'), ('accuracy_geometry.pdf', '可导出图PDF')]
    body = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>取逆任务对：几何约束的功能检验</title><style>body{font-family:system-ui,sans-serif;max-width:1250px;margin:36px auto;padding:0 22px;line-height:1.65;color:#20252b}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background:#eef2f6}img{width:100%}.scroll{overflow:auto}code,pre{background:#f2f4f7}pre{padding:15px;overflow:auto}.callout{background:#edf3f9;padding:15px;border-left:4px solid #3775ab}</style><body>'
    body += '<h1>取逆任务对：正确几何约束能否提高目标准确率？</h1>' + lead + '<p class="callout">' + escape(conclusion) + '</p><img src="accuracy_geometry.png" alt="六次配对重复的目标准确率与辅助CKA">'
    body += '<h2>主要结果与固定最终步数的敏感性</h2>' + table(['条件', '选定检查点准确率', '1200步准确率', '辅助正确CKA', '答案均值去除后正确−错配CKA'], mean_rows)
    body += '<h2>六次配对重复：选定检查点</h2>' + table(['重复 / 源教师'] + list(LABELS.values()), replicate_rows)
    body += '<h2>预先指定的三项准确率差值</h2><p>所有差值单位为百分点。六次重复共享三个冻结教师与同一测试集；不能把六次重复当成六次独立源训练。区间穷举经验bootstrap，分别列出6⁶个目标/支持集重采样及3³个教师簇重采样。它们是未经多重比较校正的描述性区间；三个共享历史数据世界的源教师不足以支持总体显著性结论。</p><div class="scroll">' + table(['比较', '均值', '六次差值', '教师簇均值', '条件区间', '教师簇区间', '1200步均值'], contrast_rows) + '</div>'
    body += '<h2>逐长度准确率</h2>' + table(['长度'] + list(LABELS.values()), length_rows)
    body += '<details><summary>固定1200步全部重复</summary>' + table(['重复'] + list(LABELS.values()), final_rows) + '</details>'
    body += '<h2>实验设置与解释边界</h2><p>数学恒等式 recoils(x)=descents(x⁻¹)。源教师在取逆输入上查询下降数；目标模型在测试时仅接收原输入x并回答recoils，不使用教师或取逆。所有目标从随机初始化训练。每次重复共256个目标标签：192个参与梯度，64个用于400、800、1200步的检查点选择。每种条件共享初始化、批次顺序与随机种子，均训练1200步；最高验证准确率优先，平局按验证交叉熵、较早步数选择。</p>'
    body += '<p>几何项为同长度批次内ONE_END隐藏向量的1−线性CKA，固定权重0.3；蒸馏使用完整词表输出、温度2、权重1。错配仅在训练集中同长度、同真实答案的样本间固定循环交换；单例对三个几何条件均排除。目标监督及蒸馏的训练输入没有扩大：均仅使用192个训练样本。没有超参数搜索或测试反馈。</p>'
    body += '<p>源预训练已经用了大量监督，教师辅助条件获得额外的源知识；正确取逆对应是实验提供的规则。这个实验检验施加几何约束的功能作用，不检验模型自行发现规则。对少标签随机初始化目标的阴性结果也不能推广到预训练目标、其他几何权重或更多标签。</p>'
    body += '<h2>源模型可靠性</h2>' + table(['源种子', '独立审计总体准确率'] + plan['lengths'], source_rows) + '<p>三源均通过预定总体99%、每长度97%的门槛。审计共1280例，与训练/验证/最终测试完全独立；源模型权重未改变。</p>'
    body += '<h2>辅助CKA</h2><p>每长度固定前128例，五长度等权平均。匹配错配保留至少3例的答案层，正确与错配使用相同保留行；答案均值残差也在这些相同行上计算。移除答案均值不能移除全部输出信息，CKA上升也不等于准确率上升。辅助规则在训练已经开始、最终测试未开启时保存；它不参与检查点或模型选择。</p><div class="scroll">' + table(['条件', '全体正确', '匹配正确', '匹配错配', '匹配差值', '残差正确', '残差错配', '残差差值'], geometry_rows) + '</div>'
    body += '<details><summary>全部检查点选择</summary>' + table(['重复', '条件', '选定更新', '验证准确率'], selection_rows) + '</details>'
    body += '<h2>数据与核验</h2><p>新建审计1280例、最终测试2560例及六组各256个支持样本。它们的完整八状态轨道均排除原始1600万条输入与487032个此前本地输入，并且各组互不相交。所有30次训练和检查点选择完成后才开启最终测试。独立核验状态：' + escape(verification['status']) + '。</p><pre>' + escape(json.dumps(verification, ensure_ascii=False, indent=2)) + '</pre>'
    body += '<p>' + ' · '.join('<a href="' + path + '">' + escape(label) + '</a>' for path, label in links) + '</p>'
    body += '<pre>OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_functional_alignment teachers\nOPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_functional_alignment train\nOPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_functional_alignment evaluate\nOPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_functional_auxiliary run\nOPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_functional_verify\nOPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_functional_report\nOPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_functional_finalize</pre></body></html>'
    (root / 'report.html').write_text(body)
    print(json.dumps({'conditions': summary['conditions']['selected'], 'contrasts': primary}, ensure_ascii=False, indent=2))


if __name__ == '__main__': run()
