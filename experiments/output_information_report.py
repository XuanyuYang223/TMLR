"""Source-level summaries of the matched-budget and collision-pair follow-up."""
from collections import defaultdict
import csv
from datetime import datetime, timezone
import html
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .until_10_verify import read, save
from .until_10_figures import estimate

ROOT = Path('results/output_information_followup')
METRICS = ['answer_accuracy', 'both_correct_fraction', 'exactly_one_correct_fraction',
    'neither_correct_fraction', 'both_correct_after_swap_fraction', 'orientation_excess_all_pairs',
    'covered_candidate_pair_fraction', 'correct_orientation_given_covered_candidates']
COHORT_LABELS = {'original_3': '原三个源种子', 'new_6': '新增六个源种子', 'new_worlds_3': '三个新数据世界'}


def aggregate(rows, fields, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in fields)].append(row)
    result = []
    for values, members in sorted(groups.items()):
        result.append({**dict(zip(fields, values)), 'sources': len(members), 'seeds': sorted(r['seed'] for r in members),
            **{k: statistics.mean(r[k] for r in members if r[k] is not None) if any(r[k] is not None for r in members) else None for k in metrics}})
    return result


def budget_source_rows():
    records = [read(p) for p in sorted(Path('results/budget_matched_geometry/evaluations').glob('*.json'))]
    result = []
    for rec in records:
        buckets = defaultdict(list)
        for r in rec['results']:
            if r['word'] == 'ic_against_ci':
                continue
            kind = 'identity' if r['word'] == 'identity' else ('generators' if len(r['word']) == 1 else 'composites')
            if kind == 'identity' and r['pairing'] == 'shuffled':
                continue
            buckets[r['pipeline'], r['pairing'], kind].append(r)
        for (pipeline, pairing, kind), rows in buckets.items():
            result.append({'source': rec['source'], 'group': rec['group'], 'seed': rec['seed'], 'status': rec['status'],
                'pipeline': pipeline, 'pairing': pairing, 'kind': kind, 'words': [r['word'] for r in rows],
                'full_null_pair_target_nmse': statistics.mean(r['full_null_pair_target_nmse'] for r in rows),
                'shared_pca64_pair_target_nmse': statistics.mean(r['shared_pca64_pair_target_nmse'] for r in rows)})
    return result, records


def pair_source_rows():
    records = [read(p) for p in sorted(Path('results/collision_pair_followup/evaluations').glob('*.json'))]
    return [{**{k: r[k] for k in ['cohort', 'source', 'condition', 'seed', 'family', 'method', 'case']}, **r['metrics']} for r in records]


def table(rows, columns, percentages=False):
    cells = ['<table><thead><tr>' + ''.join('<th>' + html.escape(label) + '</th>' for _, label in columns) + '</tr></thead><tbody>']
    for row in rows:
        entries = []
        for key, _ in columns:
            value = row.get(key)
            if value is None:
                value = '—'
            elif isinstance(value, float):
                value = f'{100*value:.2f}%' if percentages else f'{value:.6f}'
            entries.append('<td>' + html.escape(str(value)) + '</td>')
        cells.append('<tr>' + ''.join(entries) + '</tr>')
    return ''.join(cells) + '</tbody></table>'


def points(ax, x, values, color):
    jitter = np.linspace(-.045, .045, len(values))
    ax.scatter(x + jitter, values, s=35, color=color)
    ax.plot([x-.085, x+.085], [statistics.mean(values)]*2, color=color, lw=2.5)


def figures(budget, pairs):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    groups = ['novel_records', 'novel_minima_pair', 'novel_peak_mixed']
    labels = ['Directional records', 'Minima mixed tasks', 'Peak mixed tasks']
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 5.0), layout='constrained')
    conditions = [('random', 'hidden_fixed_random', 'Init: hidden', '#777'), ('trained', 'hidden_fixed_random', 'Trained: hidden', '#2463ad'),
        ('random', 'output_identity', 'Init: output', '#b8a498'), ('trained', 'output_identity', 'Trained: output', '#1b885b')]
    for j, group in enumerate(groups):
        for x, (status, pipeline, label, color) in enumerate(conditions):
            values = [r['full_null_pair_target_nmse'] for r in budget if r['group'] == group and r['status'] == status
                and r['pipeline'] == pipeline and r['pairing'] == 'correct' and r['kind'] == 'composites']
            points(axes[j], x, values, color)
        axes[j].set_xticks(range(4), [r[2] + '\nn=3 sources' for r in conditions], rotation=15, fontsize=9)
        axes[j].set_title(labels[j])
        axes[j].set_ylabel('Original full-null pair-target NMSE')
        axes[j].axhline(1, color='#777', ls='--', lw=1)
        axes[j].set_ylim(.75, 1.2)
        axes[j].grid(axis='y', alpha=.2)
    fig.suptitle('Matched fitted matrix sizes, calibration rows and search budgets', fontsize=13)
    fig.supxlabel('277 reconstruction inputs → 1024 hidden outputs → 64D generator maps; outputs retain all previous features', fontsize=10)
    fig.savefig(ROOT/'matched_budget.png', dpi=180)
    fig.savefig(ROOT/'matched_budget.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.9, 5.2))
    fig.subplots_adjust(left=.06, right=.99, top=.79, bottom=.23, wspace=.18)
    cases = ['base_CI', 'base_ICI']
    colors = ['#1b885b', '#b8a498', '#c66c35']
    for j, case in enumerate(cases):
        selected = [r for r in pairs if r['cohort'] == 'new_6' and r['condition'] == 'correct_relations'
            and r['family'] == 'native_or_posthoc' and r['method'] == 'native_operators' and r['case'] == case]
        selected.sort(key=lambda r: r['seed'])
        base = np.zeros(len(selected))
        for field, label, color in zip(['both_correct_fraction', 'exactly_one_correct_fraction', 'neither_correct_fraction'],
                ['Both correct', 'Exactly one correct', 'Neither correct'], colors):
            vals = np.array([r[field] for r in selected])
            axes[j].bar(range(len(selected)), vals, bottom=base, color=color, label=label)
            base += vals
        axes[j].set_xticks(range(len(selected)), [str(r['seed']) for r in selected], rotation=30, fontsize=9)
        axes[j].set_ylim(0, 1)
        axes[j].set_ylabel('Fraction of collision pairs')
        axes[j].set_title(case.replace('base_', '') + ': original learned operators')
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(.5, .93), ncol=3, frameon=False, fontsize=9)
    fig.suptitle('New six source seeds: input-pair correctness, 1344 pairs per seed', fontsize=13)
    fig.supxlabel('Accuracy = 0.5 + 0.5 × (both-correct fraction − neither-correct fraction)', fontsize=10)
    fig.savefig(ROOT/'collision_pairs.png', dpi=180)
    fig.savefig(ROOT/'collision_pairs.pdf')
    plt.close(fig)


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    budget, raw_budget = budget_source_rows()
    pairs = pair_source_rows()
    budget_summary = aggregate(budget, ['group', 'status', 'pipeline', 'pairing', 'kind'],
        ['full_null_pair_target_nmse', 'shared_pca64_pair_target_nmse'])
    pair_summary = aggregate(pairs, ['cohort', 'condition', 'family', 'method', 'case'], METRICS)
    uncertainty, comparisons = [], []
    for group in ['novel_records', 'novel_minima_pair', 'novel_peak_mixed']:
        sources = defaultdict(dict)
        for r in budget:
            if r['group'] == group and r['status'] == 'trained' and r['pairing'] == 'correct' and r['kind'] == 'composites':
                sources[r['seed']][r['pipeline']] = r
        for seed, variants in sorted(sources.items()):
            for hidden in ['hidden_fixed_random', 'hidden_fit_pca']:
                comparisons.append({'group': group, 'seed': seed, 'comparison': hidden + '_minus_output',
                    'difference': variants[hidden]['full_null_pair_target_nmse'] - variants['output_identity']['full_null_pair_target_nmse']})
    for comparison in sorted({r['comparison'] for r in comparisons}):
        for group in sorted({r['group'] for r in comparisons}):
            rows = [r for r in comparisons if r['comparison'] == comparison and r['group'] == group]
            uncertainty.append({'study': 'budget', 'group': group, 'comparison': comparison, **estimate(rows, 'difference')})
    macro = []
    buckets = defaultdict(list)
    for r in pairs:
        if r['case'] in ['base_CI', 'base_ICI']:
            buckets[r['cohort'], r['condition'], r['family'], r['method'], r['seed']].append(r)
    for (cohort, condition, family, method, seed), rows in sorted(buckets.items()):
        assert len(rows) == 2
        macro.append({'cohort': cohort, 'condition': condition, 'family': family, 'method': method, 'seed': seed,
            **{k: statistics.mean(r[k] for r in rows) if all(r[k] is not None for r in rows) else None for k in METRICS}})
    for cohort in COHORT_LABELS:
        for method in ['native_operators', 'correct', 'shuffled']:
            rows = [r for r in macro if r['cohort'] == cohort and r['condition'] == 'correct_relations'
                and ((r['family'] == 'native_or_posthoc' and method == 'native_operators') or r['family'] == 'identical_frozen') and r['method'] == method]
            if rows:
                for metric in ['answer_accuracy', 'both_correct_fraction', 'neither_correct_fraction', 'orientation_excess_all_pairs']:
                    uncertainty.append({'study': 'pairs', 'cohort': cohort, 'method': method, 'metric': metric, **estimate(rows, metric)})
    frozen_contrasts = []
    source_variants = defaultdict(dict)
    for r in macro:
        if r['family'] == 'identical_frozen':
            source_variants[r['cohort'], r['condition'], r['seed']][r['method']] = r
    for (cohort, condition, seed), variants in sorted(source_variants.items()):
        assert set(variants) == {'correct', 'shuffled'}
        frozen_contrasts.append({'cohort': cohort, 'condition': condition, 'seed': seed,
            **{k: variants['correct'][k] - variants['shuffled'][k] for k in ['answer_accuracy', 'both_correct_fraction', 'orientation_excess_all_pairs']}})
    for cohort in COHORT_LABELS:
        rows = [r for r in frozen_contrasts if r['cohort'] == cohort and r['condition'] == 'correct_relations']
        for metric in ['answer_accuracy', 'both_correct_fraction', 'orientation_excess_all_pairs']:
            uncertainty.append({'study': 'same_frozen_pairing_difference', 'cohort': cohort, 'metric': metric, **estimate(rows, metric)})
    protocol = read('results/budget_matched_geometry/protocol.json')
    plan = protocol['signature']['plan']
    budget_verify = Path('results/budget_matched_geometry/verification.json')
    pair_verify = Path('results/collision_pair_followup/verification.json')
    summary = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'sources_retrained': 0,
        'budget_sources_trained_and_initialization': len(raw_budget), 'pair_endpoints': len(pairs),
        'primary_pipelines': plan['primary_comparison'], 'budget_per_source': budget,
        'budget_per_word': [{**{k: rec[k] for k in ['source', 'group', 'seed', 'status']}, **row} for rec in raw_budget for row in rec['results']],
        'budget_summary': budget_summary, 'pair_per_source_case': pairs, 'pair_summary': pair_summary,
        'CI_ICI_averaged_within_source': macro, 'budget_paired_source_differences': comparisons,
        'same_frozen_pairing_source_differences': frozen_contrasts, 'source_level_descriptive_uncertainty': uncertainty,
        'parameter_budgets': [{**{k: r[k] for k in ['source', 'group', 'status', 'seed']}, **fit} for r in raw_budget for fit in r['fits']],
        'budget_verification': read(budget_verify) if budget_verify.exists() else None,
        'pair_verification': read(pair_verify) if pair_verify.exists() else None,
        'limitations': ['Exploratory after previous geometry/proxy outcomes; not a new blind hypothesis test.',
            'Same matrix counts do not equal identical function families, effective ranks or causal interventions.',
            'One fixed random encoder; PCA input sensitivity has additional fitted hidden loadings.',
            'Decoder and supplied generator probes are external fits to calibration hidden vectors; ordinary calibration uses complete orbits.',
            'Conditional collision tests and multiple words are not independent general-population repetitions.',
            'Same-world source seeds and new-world sources are separate cohorts. No cross-mathematical-domain confirmation.']}
    save(ROOT/'summary.json', summary)
    for name, rows in [('budget_per_source.csv', budget), ('pair_per_source_case.csv', pairs), ('source_differences.csv', frozen_contrasts)]:
        keys = list(rows[0])
        with (ROOT/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader(); writer.writerows(rows)
    figures(budget, pairs)
    body = '<h1>输出信息与配对推断：现有模型复核</h1>'
    body += '<p class="note">没有增加源训练。本轮核对拟合预算与配对双正确率，所有原先完成的协议和结果保持原样。预算控制与交换配对统计属于看到先前结果后的探索性检查。</p>'
    body += '<h2>拟合预算匹配后的结果</h2><p>主比较保留先前全部277个输出特征。隐藏分支先经固定随机投影得到277个坐标；输出分支使用277维原输出编码。两边都只用2688个校准FIT状态拟合278×1024的仿射重构器，用672个校准VAL状态在四个岭参数中选择。然后分别对重构表征做FIT内的长度均值与1024×64 PCA，拟合三个64×64生成元；每个生成元仍只用相同FIT/VAL及四个候选。所有复合词由生成元相乘得到，没有单独拟合。</p>'
    body += '<p>每个正确关系分支的拟合矩阵元素共362,496个：重构器284,672、探针PCA65,536、生成元12,288；中心与尺度统计也同维。错误关系拟合另列，预算在分支间相同。两个分支都只保留自己重构的输入残差，隐藏分支没有额外使用真实隐藏向量的残差。它们预测同一个原数字读出零空间目标，零预测误差为1，数值越低越好。输入类型及有效秩不同，矩阵数相同不是因果机制的严格匹配。</p>'
    main = [r for r in budget_summary if r['pipeline'] in plan['primary_comparison'] and r['kind'] == 'composites' and r['pairing'] == 'correct']
    body += table(main, [('group', '任务组'), ('status', '源模型'), ('pipeline', '输入分支'), ('sources', '源种子数'),
        ('full_null_pair_target_nmse', '原全零空间复合误差'), ('shared_pca64_pair_target_nmse', '共同PCA64目标误差')])
    body += '<p>记录组三个训练种子的隐藏固定投影分支误差为0.8957、0.8917、0.8919，均值0.8931；输出分支为0.8672、0.8566、0.8471，均值0.8570。输出在三个种子都更低，并复现了先前输出代理的误差。额外拟合预算不能完全解释原输出代理优势；当前几何信号仍不能排除输出关联解释。这不证明输出置信度是网络内部计算的唯一原因。</p>'
    body += '<img src="matched_budget.png" alt="拟合预算匹配实验">'
    body += '<h2>配对双正确：有局部信息，但总准确率受双错误影响</h2><p>双正确指同一碰撞配对的两条排列都答对，而不是CI与ICI两个词都答对。每对已知的三个标量答案与长度相同，隐藏答案不同。确定性已知标量编码无法同时答对两者；输入相关噪声也可能产生非零双正确，因此还比较交换两个预测后是否双正确。</p>'
    body += '<p><strong>准确率 = 1/2 + 1/2 ×（双正确比例 − 双错误比例）。</strong>新增六种子的原始CI双正确15.15%、双错误32.48%，ICI为13.58%、33.69%，对应准确率41.34%、39.94%。双正确不是零，双错误仍更多，所以不支持稳定超过50%。CI/ICI预测同一真实统计，汇总时先在源模型内平均，不能增加独立重复数。</p>'
    headline = [r for r in pair_summary if r['condition'] == 'correct_relations'
        and ((r['family'] == 'native_or_posthoc' and r['method'] == 'native_operators') or r['family'] == 'identical_frozen')]
    body += table(headline, [('cohort', '队列'), ('family', '比较'), ('method', '关系来源'), ('case', '词'), ('sources', '源种子数'),
        ('answer_accuracy', '准确率'), ('both_correct_fraction', '双正确'), ('exactly_one_correct_fraction', '单正确'),
        ('neither_correct_fraction', '双错误'), ('both_correct_after_swap_fraction', '交换后双正确'),
        ('orientation_excess_all_pairs', '正确减交换方向优势')], percentages=True)
    body += '<p>新六种子的原始CI交换后双正确为2.29%，ICI为2.16%；原方向减交换方向为12.86、11.42个百分点。相同冻结模型的正确配对CI/ICI平均双正确20.46%，错误关系为1.07%；方向优势分别17.80与0.27个百分点。这更符合部分预测与隐藏答案有正确对应的解释，而不只是任意输入差异。它仍不是稳定整体推断，更不能证明普通任务训练自行发现了代数规则。</p>'
    body += '<p>交换控制保留每对的两个预测和两个真实答案。条件方向正确率只在预测刚好覆盖两种真实候选答案的配对上计算，属于用测试标签评分的诊断，不能当作部署时的两选一能力或替代原准确率。</p>'
    body += '<img src="collision_pairs.png" alt="六种子配对正确情况">'
    body += '<h2>边界与核验</h2><p>旧源种子、新源种子、新数据世界分开。额外真实C／I前向起步的结果另列，因为它们有额外输入信息。隐藏输入PCA277的敏感性分析有额外拟合的输入载荷，不能冒充主预算匹配；共同FIT-PCA64目标也只是次要口径。所有组的正确、打乱及恒等重构基线完整保存，没有用测试结果挑选获胜方法。源任务难度与联合标签统计的原有组间限制仍在。</p>'
    verifications = [{'study': '拟合预算', **(summary['budget_verification'] or {})}, {'study': '碰撞配对', **(summary['pair_verification'] or {})}]
    body += table(verifications, [('study', '独立核验'), ('status', '状态'), ('checks_completed', '已复核记录')])
    body += '<p>预算核验独立重拟合全部重构器及生成元，重算参数选择并检查FIT内PCA，再直接构造全维预测核对误差。配对核验从原始答案文件重新统计。检查记录数量不是独立源种子的数量。</p>'
    body += '<details><summary>全部预算基线和PCA敏感性结果</summary>' + table(budget_summary,
        [('group', '组'), ('status', '模型'), ('pipeline', '分支'), ('pairing', '配对'), ('kind', '类型'), ('sources', '源数'),
         ('full_null_pair_target_nmse', '全目标NMSE'), ('shared_pca64_pair_target_nmse', 'PCA64 NMSE')]) + '</details>'
    links = [('summary.json', '全部数据与源级区间'), ('budget_per_source.csv', '预算逐源结果'), ('pair_per_source_case.csv', '配对逐源逐词结果'),
        ('source_differences.csv', '冻结配对逐源差值'), ('matched_budget.pdf', '预算图PDF'), ('collision_pairs.pdf', '配对图PDF'),
        ('../budget_matched_geometry/protocol.json', '预算协议'), ('../collision_pair_followup/protocol.json', '配对协议'),
        ('../budget_matched_geometry/verification.json', '独立拟合核验'), ('../collision_pair_followup/verification.json', '独立配对核验'),
        ('../until_10_followup/report.html', '前轮完整报告')]
    body += '<p>' + ' · '.join(f'<a href="{p}">{label}</a>' for p, label in links) + '</p>'
    css = 'body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:0 18px;color:#172334}p{line-height:1.65}img{width:100%;height:auto}table{border-collapse:collapse;font-size:12px;margin:18px 0}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.note{background:#eef4fb;padding:16px}'
    (ROOT/'report.html').write_text(f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>输出信息与配对推断复核</title><style>{css}</style>{body}</html>')
    print({'report': str(ROOT/'report.html'), 'budget_sources': len(raw_budget), 'pair_endpoints': len(pairs)}, flush=True)
    return summary


if __name__ == '__main__':
    run()
