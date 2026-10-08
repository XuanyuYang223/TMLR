"""Descriptive two-target-pool comparisons, same pretrained models."""
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .longrun_engine import atomic_json
from .native_target_repeat import paths
from .permworld_combinations import sha
from .six_hour_report import transfer_summaries, write_rows


def summarize(root):
    records = [json.loads(p.read_text()) for p in (root/'transfer').glob('*.json')]
    assert len(records) == 27 and all(r['status'] == 'complete' for r in records)
    rows = [{**row, 'group': record['group'], 'seed': record['seed']} for record in records for row in record['rows']]
    return transfer_summaries(rows)


def plot_comparison(root, comparison):
    groups=sorted({r['group'] for r in comparison})
    modes=('linear_task_free','linear_query','mlp_task_free','finetune','finetune_shared')
    labels=('Linear\nno task query','Linear\ntask query','MLP\nno task query','Fine-tune\ncondition policy','Fine-tune\nshared policy')
    cells={(r['group'],r['budget'],r['mode']):r for r in comparison}
    limit=max(abs(100*r[key]) for r in comparison for key in ('old_random_gain','new_random_gain'))
    fig,axes=plt.subplots(2,2,figsize=(13,11),layout='constrained')
    for row,budget in enumerate((64,256)):
        for col,(pool,key) in enumerate((('Original pool','old_random_gain'),('Fresh pool','new_random_gain'))):
            ax=axes[row,col]
            matrix=np.array([[100*cells[group,budget,mode][key] for mode in modes] for group in groups])
            picture=ax.imshow(matrix,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
            ax.set_xticks(range(len(modes)),labels,fontsize=8)
            ax.set_yticks(range(len(groups)),groups,fontsize=9)
            ax.set_title(f'{pool}, {budget} support labels')
            for i,j in product(range(len(groups)),range(len(modes))):
                ax.text(j,i,f'{matrix[i,j]:+.1f}',ha='center',va='center',fontsize=9,
                        color='white' if abs(matrix[i,j])>.6*limit else '#18202b')
    fig.colorbar(picture,ax=axes.ravel().tolist(),label='Accuracy gain over paired random initialization (pp)',shrink=.8)
    fig.suptitle('All 80 group/budget/readout comparisons; same pretrained sources in both pools',fontsize=12)
    fig.savefig(root/'pool_comparison.png',dpi=180)
    plt.close(fig)


def run():
    _, config, source_root, root = paths()
    old_gains, old_summary = summarize(source_root)
    new_gains, new_summary = summarize(root)
    old = {(r['group'], r['budget'], r['mode']): r for r in old_summary}
    new = {(r['group'], r['budget'], r['mode']): r for r in new_summary}
    old_length = json.loads((source_root/'length_baselines/summary.json').read_text())
    new_length = json.loads((root/'length_baselines.json').read_text())
    baseline = {('old', r['budget']): r['test_macro'] for r in old_length['macro'] if r['mode'] == 'smooth_length'}
    baseline.update({('new', r['budget']): r['test_macro'] for r in new_length['macro'] if r['mode'] == 'smooth_length'})
    comparison = []
    for key in sorted(old):
        a, b = old[key], new[key]
        comparison.append({'group': key[0], 'budget': key[1], 'mode': key[2],
                           'old_accuracy': a['test_accuracy'], 'new_accuracy': b['test_accuracy'],
                           'old_random_gain': a['gain'], 'new_random_gain': b['gain'],
                           'old_positive_seed_means': a['positive_seed_means'], 'new_positive_seed_means': b['positive_seed_means'],
                           'old_gain_over_length': a['test_accuracy']-baseline['old', key[1]],
                           'new_gain_over_length': b['test_accuracy']-baseline['new', key[1]],
                           'same_pretrained_models': True})
    targets = []
    old_baseline = {(r['target'], r['budget']): r['test_accuracy'] for r in old_length['endpoints'] if r['mode'] == 'smooth_length'}
    new_baseline = {(r['target'], r['budget']): r['test_accuracy'] for r in new_length['endpoints'] if r['mode'] == 'smooth_length'}
    for group, budget, mode, target in product(sorted({r['group'] for r in old_gains}), config['target_budgets'],
                                              ('linear_task_free', 'linear_query', 'mlp_task_free', 'finetune', 'finetune_shared'), config['target_tasks']):
        a = [r for r in old_gains if (r['group'], r['budget'], r['mode'], r['target']) == (group, budget, mode, target)]
        b = [r for r in new_gains if (r['group'], r['budget'], r['mode'], r['target']) == (group, budget, mode, target)]
        assert len(a) == len(b) == 3
        targets.append({'group': group, 'budget': budget, 'mode': mode, 'target': target,
                        'old_random_gain': float(np.mean([r['gain'] for r in a])), 'new_random_gain': float(np.mean([r['gain'] for r in b])),
                        'old_gain_over_length': float(np.mean([r['test_accuracy'] for r in a]))-old_baseline[target, budget],
                        'new_gain_over_length': float(np.mean([r['test_accuracy'] for r in b]))-new_baseline[target, budget]})
    write_rows(root/'pool_comparison.csv', comparison); write_rows(root/'target_pool_comparison.csv', targets)
    plot_comparison(root,comparison)
    summary = {'reported_utc': datetime.now(timezone.utc).isoformat(), 'code_sha256': sha(__file__),
               'scope': 'same 24 pretrained sources and 3 random initializations, two globally disjoint target pools; not independent source/model replicas',
               'original_transfer_endpoints': 1080, 'repeat_transfer_endpoints': 1080,
               'length_baseline': [{'pool': pool, 'budget': budget, 'accuracy': accuracy} for (pool, budget), accuracy in baseline.items()],
               'comparisons': comparison}
    atomic_json(root/'comparison_summary.json', summary)
    table = ''.join(f"<tr><td>{r['group']}</td><td>{r['mode']}</td><td>{100*r['old_accuracy']:.2f}%</td><td>{100*r['new_accuracy']:.2f}%</td><td>{100*r['old_random_gain']:+.2f} pp</td><td>{100*r['new_random_gain']:+.2f} pp</td><td>{100*r['new_gain_over_length']:+.2f} pp</td></tr>" for r in comparison if r['budget'] == 256)
    (root/'comparison.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>两轮目标抽样对照</title><style>body{{max-width:1250px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:7px;border-bottom:1px solid #ddd}}aside{{background:#f2f4f7;padding:16px}}</style><h1>同一批源模型，两套目标支持与测试</h1><p>原生迁移各有 1,080 个端点，所有八个源组合、五读出、两预算均完成。只看长度的平滑计数基线，第一池 64/256 标签为 31.02%/33.74%，第二池为 30.80%/35.56%；第二池沿用第一池验证选出的带宽 2，不重新调参。</p><p>256 标签条件选参微调中，interior_none 对随机模型平均收益为 +2.52/+2.47 pp，方向重复，但正种子从 3/3 变为 2/3。绝对准确率为 34.79%/33.85%，第二池低于长度基线 1.70 pp。其他组及所有读出结果见下表与全量 CSV。</p><p>下图并列所有八组合、五读出、两预算，对随机初始化的收益使用统一色标。两列复用相同源模型，不能视作两次独立模型复制。</p><img src="pool_comparison.png" style="max-width:100%"><table><tr><th>组</th><th>读出</th><th>第一池准确率</th><th>第二池准确率</th><th>第一池对随机</th><th>第二池对随机</th><th>第二池对长度</th></tr>{table}</table><aside>两池复用相同源模型和初始化。新输入抽样能检查目标支持/测试的稳健性，不能确认新源世界、新组合或模型重复。四目标宏均值不表示每个目标都受益。预测器在新池中拟合过同名组合的第一池结果，因此新池预测与第一池“整组留出”是不同检验；两者不能直接当作同一个独立推广得分。</aside><p><a href="pool_comparison.csv">两预算所有读出</a> · <a href="target_pool_comparison.csv">全部逐目标比较</a> · <a href="report.html">冻结预测与新行为</a> · <a href="length_baselines.json">冻结长度参照</a> · <a href="verification.json">全量核验</a></p></html>''')
    print(json.dumps({'group_budget_readout_comparisons': len(comparison), 'target_comparisons': len(targets)}))


if __name__ == '__main__': run()
