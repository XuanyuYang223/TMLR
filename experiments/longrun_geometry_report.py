"""Landmark sensitivity and conditional transformation-kernel predictions."""
import csv
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path

import numpy as np

from .field_predictions import heldout_combination_predictions
from .permworld_combinations import sha
from .six_hour_report import write_rows


def register():
    root = Path('results/six_hour_session')
    assert not (root/'landmarks/transformed_cka.csv').exists()
    protocol = {'registered_utc': datetime.now(timezone.utc).isoformat(), 'code_sha256': sha(__file__),
                'scope': 'exploratory after original ONE_END CKA observations, before any transformed hidden-state measurements',
                'ridge_alpha': 1., 'holdout': 'all three seeds and all four operators of one complete source group',
                'baseline': ['initial self-transformed hidden CKA', 'source final validation accuracy', 'source validation learning AUC',
                             'source conditional marginal entropy', 'source conditional joint entropy', 'source within-length absolute label correlation',
                             'input operator and initialization fixed effects'],
                'addition': 'exact source-answer categorical-kernel CKA under the same operator',
                'interpretation': 'conditional prediction of a geometric metric, not proof of transfer or a uniquely learned algebra',
                'selection': 'report all four landmarks and both length treatments; no select-best landmark primary claim'}
    (root/'native_kernel_prediction_preregistration.json').write_text(json.dumps(protocol, indent=2)+'\n')


def read(path):
    with Path(path).open() as handle: return list(csv.DictReader(handle))


def run():
    root = Path('results/six_hour_session'); output = root/'landmarks'
    protocol = json.loads((root/'native_kernel_prediction_preregistration.json').read_text())
    assert protocol['code_sha256'] == sha(__file__)
    cross = read(output/'cross_seed.csv'); transformed = read(output/'transformed_cka.csv')
    for row in cross:
        for key in ('trained_cka', 'initial_cka', 'change_from_initial'): row[key] = float(row[key])
    for row in transformed:
        for key in ('trained_cka', 'initial_cka', 'change_from_initial', 'categorical_code_cka'): row[key] = float(row[key])
        row['seed'] = int(row['seed'])
    plan = json.loads((root/'session_plan.json').read_text())
    dataset_meta = json.loads((root/'dataset/metadata.json').read_text())
    names = dataset_meta['names']; statistics = {}
    source = [json.loads(path.read_text()) for path in (root/'multi').glob('*.json')]
    with np.load(root/'dataset/data.npz') as data:
        for record in source:
            g, seed = record['job']['id'], record['job']['seed']
            times = sorted({r['step'] for r in record['curve']})
            scores = [np.mean([r['accuracy'] for r in record['curve'] if r['step'] == s]) for s in times]
            entropy, joint, correlations = [], [], []
            ids = [names.index(t) for t in record['job']['tasks']]
            for n in np.unique(data['train_lengths']):
                y = data['train_labels'][data['train_lengths'] == n][:, ids]
                for column in y.T:
                    _, counts = np.unique(column, return_counts=True); q = counts/counts.sum(); entropy.append(float(-(q*np.log(q)).sum()))
                _, counts = np.unique(y, axis=0, return_counts=True); q = counts/counts.sum(); joint.append(float(-(q*np.log(q)).sum()))
                sigma = y.std(0); z = (y-y.mean(0))/np.where(sigma > 0, sigma, 1.)
                c = z.T@z/len(z); correlations.extend(abs(c[i, j]) for i in range(4) for j in range(i+1, 4))
            statistics[g, seed] = [float(scores[-1]), float(np.trapezoid(scores, times)/times[-1]),
                                    float(np.mean(entropy)), float(np.mean(joint)), float(np.mean(correlations))]
    operators = ['complement', 'reverse', 'reverse_complement', 'inverse']
    predictions = {}
    for landmark in ('ONE_END', 'heldout_query', 'prefix_mean', 'source_query_concat'):
        for mode in ('raw', 'within_length'):
            cells = [r for r in transformed if (r['landmark'], r['centering']) == (landmark, mode)]
            baseline = np.array([[r['initial_cka'], *statistics[r['group'], r['seed']]]
                + [float(r['operator'] == op) for op in operators[1:]]
                + [float(r['seed'] == s) for s in plan['model_seeds'][1:]] for r in cells])
            extra = np.array([[r['categorical_code_cka']] for r in cells])
            truth, labels = np.array([r['trained_cka'] for r in cells]), np.array([r['group'] for r in cells])
            a = heldout_combination_predictions(baseline, truth, labels, protocol['ridge_alpha'])
            b = heldout_combination_predictions(np.column_stack([baseline, extra]), truth, labels, protocol['ridge_alpha'])
            predictions[f'{landmark}_{mode}'] = {'baseline': a, 'plus_source_kernel': b,
                                                'delta_r2': b['r2']-a['r2'], 'mae_reduction': a['mae']-b['mae']}
    (output/'prediction_scores.json').write_text(json.dumps({'protocol': protocol, 'results': predictions}, indent=2)+'\n')
    summary = []
    for g in sorted({r['group'] for r in cross}):
        for landmark in ('ONE_END', 'heldout_query', 'prefix_mean', 'source_query_concat'):
            for mode in ('raw', 'within_length'):
                cells = [r for r in cross if (r['group'], r['landmark'], r['centering']) == (g, landmark, mode)]
                summary.append({'group': g, 'landmark': landmark, 'centering': mode,
                                'trained_cka': float(np.mean([r['trained_cka'] for r in cells])),
                                'initial_cka': float(np.mean([r['initial_cka'] for r in cells])),
                                'change_from_initial': float(np.mean([r['change_from_initial'] for r in cells])),
                                'positive_seed_pairs': sum(r['change_from_initial'] > 0 for r in cells), 'seed_pairs': len(cells)})
    write_rows(output/'cross_seed_summary.csv', summary)
    plot(output, summary, transformed)
    table = ''.join(f"<tr><td>{escape(name)}</td><td>{r['baseline']['r2']:.3f}</td><td>{r['plus_source_kernel']['r2']:.3f}</td><td>{r['mae_reduction']:+.4f}</td></tr>" for name, r in predictions.items())
    output.joinpath('report.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>表征位置与数学变换核</title>
<style>body{{max-width:1200px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui;color:#18202b}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}aside{{background:#f2f4f7;padding:16px}}</style>
<h1>CKA 对表征提取位置和任务变换结构的敏感性</h1>
<p>这是在初步 ONE_END 跨种子结果之后追加的探索性分析，所有位置、组合、输入算子与种子均保留。ONE_END 沿用原仓库的任务无关前缀测量；另加入前缀有效 token 均值、共同未见任务 descents 的查询位置，以及四个已训练任务查询位置的拼接。</p>
<p>已训练查询拼接只在同一组的不同种子间，或同一模型的原始/变换输入间比较；没有将不同查询任务集直接当作相同输入做跨组比较。特征提取不接收任何答案标签。</p>
<img src="landmarks.png"><p>图中为训练后减初始化的跨种子 CKA。三个种子形成的三个配对并不独立；这些变化也不是原论文正确变换对照的同一端点。</p>
<h2>由任务答案得到的数学参照</h2><img src="transformation_prediction.png">
<p>对每组任务，在 complement、reverse、reverse-complement、inverse 下比较隐藏表征，并计算源任务答案独热编码的精确 CKA。独热核只是指定输出编码下的数学参照；它的排序不保证训练后隐藏几何相同。13 个情况有明确的任务向量线性变换公式；19 个情况在 n=10 找到原答案相同、变换答案不同的反例。无同输入线性恒等式的组仍可具有输入变换闭合关系。</p>
<h2>整组留出的条件预测</h2>
<p>每折排除某一源组合的全部种子与算子，用其他组合预测该组的变换后隐藏 CKA。基线控制初始化 CKA、源最终成绩和学习 AUC、源标签条件熵/联合熵/相关性，以及算子、初始化固定效应；增加源答案变换核作为结构参照。岭惩罚固定为 1，特征标准化只使用训练折。</p>
<table><tr><th>位置与长度处理</th><th>基线 R²</th><th>加入答案核 R²</th><th>MAE 降低</th></tr>{table}</table>
<aside>只有八个源组合折，仍共享一个排列数据世界。源拟合及完整标签统计未严格匹配，数学参照也利用已知的任务定义。这是几何指标的条件预测，不是迁移收益的证明，不能选择其中最好的位置并称其为预先指定主端点。</aside>
<p><a href="../native_transform_audit/audit.json">变换公式和反例</a> · <a href="../native_transform_audit/categorical_kernel.csv">答案核</a> · <a href="cross_seed.csv">跨种子完整测量</a> · <a href="transformed_cka.csv">全部变换 CKA</a> · <a href="prediction_scores.json">逐折预测</a> · <a href="../native_kernel_prediction_preregistration.json">分析登记</a> · <a href="metadata.json">提取代码与数据指纹</a></p></html>''')
    print(json.dumps({key: {'delta_r2': value['delta_r2'], 'mae_reduction': value['mae_reduction']} for key, value in predictions.items()}, indent=2))


def plot(output, summary, transformed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = sorted({r['group'] for r in summary})
    landmarks = ('ONE_END', 'heldout_query', 'prefix_mean', 'source_query_concat')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, mode in zip(axes, ('raw', 'within_length')):
        for landmark in landmarks:
            cells = {r['group']: r for r in summary if (r['landmark'], r['centering']) == (landmark, mode)}
            ax.plot(range(len(groups)), [cells[g]['change_from_initial'] for g in groups], 'o-', label=landmark)
        ax.axhline(0, color='gray', linewidth=1)
        ax.set(xticks=range(len(groups)), xticklabels=groups, ylabel='Cross-seed CKA change from initialization', title=mode)
        ax.tick_params(axis='x', rotation=40); ax.legend(fontsize=8)
    for suffix in ('png', 'pdf'): fig.savefig(output/f'landmarks.{suffix}', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), constrained_layout=True)
    for ax, landmark in zip(axes, landmarks):
        cells = [r for r in transformed if (r['landmark'], r['centering']) == (landmark, 'within_length')]
        for op in ('complement', 'reverse', 'reverse_complement', 'inverse'):
            selected = [r for r in cells if r['operator'] == op]
            ax.scatter([r['categorical_code_cka'] for r in selected], [r['trained_cka'] for r in selected], s=18, alpha=.6, label=op)
        ax.set(xlabel='Source categorical-kernel CKA', ylabel='Learned hidden CKA', title=landmark, xlim=(0, 1.03), ylim=(0, 1.03))
    axes[-1].legend(fontsize=8)
    for suffix in ('png', 'pdf'): fig.savefig(output/f'transformation_prediction.{suffix}', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--register', action='store_true'); args = parser.parse_args()
    (register if args.register else run)()
